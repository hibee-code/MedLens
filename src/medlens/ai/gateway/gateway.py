"""LLMGateway: the only way MedLens calls a model (CLAUDE.md §3, §5). The provider is Google's
Gemini API (providers.py; decision log, "Switched LLM provider from Anthropic to Gemini").

For every call, in order:

1. kill switch (MEDLENS_LLM__ENABLED), then (async, cacheable routes) the answer cache, then
   today's spend against the daily budget, checked before the call: blocked calls raise
   ``LLMUnavailableError`` and are counted, not metered;
2. the route's tier names an ordered chain of models: the primary (MEDLENS_LLM__MODEL_FAST /
   _BALANCED), then the tier's MEDLENS_LLM__FALLBACK_MODELS. Each model is called behind its own
   circuit breaker; its versioned prompt is the system instruction, the variable content
   (images, untrusted text wrapped in tags) follows; structured output is requested with the
   output model's JSON schema;
3. per failure (providers.FailureKind): transient errors retry the same model up to
   MEDLENS_LLM__MAX_RETRIES times with jittered backoff, then move on; rate limits, safety
   blocks, empty answers and other failures move to the next model at once. Before every
   switch the budget is checked again, this call's spend included: the chain never gets around
   the budget. When every model fails, the feature uses its non-AI fallback;
4. the output is validated against the output model, with one repair attempt on the model that
   answered;
5. one ``llm_calls`` row records the requested and the answering model, prompt version, tokens,
   cost, latency, attempts and outcome. Logs name models and reasons only: never prompt text,
   images or personal data.

``LLMGateway`` is synchronous (Celery workers); ``AsyncLLMGateway`` serves the API.
"""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog
from purgatory import AsyncCircuitBreakerFactory, SyncCircuitBreakerFactory
from purgatory.domain.model import OpenedState
from pydantic import BaseModel, ValidationError

from medlens.ai.gateway.cache import AnswerCache, answer_key
from medlens.ai.gateway.ledger import AsyncSpendLedger, CallRecord, SpendLedger
from medlens.ai.gateway.metrics import (
    LLM_CACHE_HITS,
    LLM_CALLS,
    LLM_COST_USD,
    LLM_DAILY_BUDGET,
    LLM_FALLBACKS,
    LLM_LATENCY,
    LLM_SPENT_TODAY,
)
from medlens.ai.gateway.pricing import TokenUsage, cost_usd
from medlens.ai.gateway.providers import (
    AsyncProvider,
    FailureKind,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    SyncProvider,
    image_part,
)
from medlens.ai.gateway.registry import ROUTES, Route
from medlens.ai.guardrails import wrap_untrusted
from medlens.ai.prompts import load_prompt
from medlens.core.config import LLMSettings
from medlens.modules.control_plane.models import LLMCallOutcome
from medlens.ports.llm import LLMCallFailedError, LLMUnavailableError

log = structlog.stdlib.get_logger(__name__)

BREAKER_THRESHOLD = 5
BREAKER_RESET_SECONDS = 60
# Jittered exponential backoff between retries of one model: a random wait up to
# min(BACKOFF_CAP, BACKOFF_BASE * 2**retry) seconds ("full jitter").
BACKOFF_BASE = 0.5
BACKOFF_CAP = 8.0


def backoff(retry: int) -> float:
    return random.uniform(0, min(BACKOFF_CAP, BACKOFF_BASE * 2**retry))  # noqa: S311 - jitter


@dataclass(frozen=True, slots=True)
class ImagePart:
    data: bytes
    media_type: str = "image/jpeg"


@dataclass(frozen=True, slots=True)
class TextPart:
    """Our own instructions or framing (trusted)."""

    text: str


@dataclass(frozen=True, slots=True)
class UntrustedPart:
    """Text from outside (documents, pack text, questions): wrapped and labelled untrusted."""

    kind: str
    text: str
    attrs: Mapping[str, str] = field(default_factory=dict)


Part = ImagePart | TextPart | UntrustedPart


@dataclass(frozen=True, slots=True)
class Subject:
    """What a call is for, recorded with it (e.g. scan id, question hash)."""

    type: str
    id: str


def _content(parts: Sequence[Part]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, ImagePart):
            blocks.append(image_part(part.data, part.media_type))
        elif isinstance(part, UntrustedPart):
            blocks.append({"text": wrap_untrusted(part.text, kind=part.kind, attrs=part.attrs)})
        else:
            blocks.append({"text": part.text})
    return blocks


def _problems(error: ValidationError) -> str:
    issues = error.errors(include_url=False, include_input=False, include_context=False)
    return "; ".join(
        f"{'.'.join(str(part) for part in issue['loc']) or 'output'}: {issue['msg']}"
        for issue in issues[:10]
    )


def _validate[OutputT: BaseModel](
    output_model: type[OutputT], text: str
) -> tuple[OutputT | None, str | None]:
    try:
        return output_model.model_validate_json(text), None
    except ValidationError as exc:
        return None, _problems(exc)


class _CircuitOpen(Exception):
    """The model's circuit breaker is open: skip to the next model."""


@dataclass
class _Attempt:
    """Bookkeeping for one gateway call (every provider request on every model it tried)."""

    route: Route
    model: str  # the tier's primary, as requested
    chain: list[str]
    started: float
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost: Decimal = Decimal(0)
    served_model: str | None = None
    attempts: int = 0

    def add(self, model: str, usage: TokenUsage, settings: LLMSettings) -> None:
        self.attempts += 1
        self.usage = self.usage + usage
        self.cost += cost_usd(settings.prices[model], usage)
        self.served_model = model


@dataclass
class _Failure:
    error: ProviderError | None = None  # the last model failure (None: only open circuits)


class _GatewayCore:
    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings

    def _start(self, route_name: str, clock: float) -> _Attempt:
        route = ROUTES[route_name]
        if not self._settings.enabled:
            raise self._blocked(route, "disabled")
        chain = self._settings.chain(route.tier.value)
        return _Attempt(route, chain[0], chain, clock)

    def _check_budget(self, attempt: _Attempt, spent: Decimal) -> None:
        LLM_SPENT_TODAY.set(float(spent))
        LLM_DAILY_BUDGET.set(self._settings.daily_budget_usd)
        if spent >= Decimal(str(self._settings.daily_budget_usd)):
            raise self._blocked(attempt.route, "budget")

    @staticmethod
    def _blocked(route: Route, reason: str) -> LLMUnavailableError:
        LLM_CALLS.labels(route=route.name, outcome="blocked").inc()
        log.info("llm.blocked", route=route.name, reason=reason)
        return LLMUnavailableError(reason)

    def _request(
        self, attempt: _Attempt, parts: Sequence[Part], output_model: type[BaseModel]
    ) -> ProviderRequest:
        route = attempt.route
        return ProviderRequest(
            model=attempt.model,
            system=load_prompt(route.prompt, route.version),
            messages=({"role": "user", "parts": _content(parts)},),
            schema=output_model.model_json_schema(),
            max_tokens=route.max_tokens,
            effort=route.effort,
        )

    def _retry_or_raise(
        self, attempt: _Attempt, model: str, exc: ProviderError, retry: int
    ) -> bool:
        """Count a failed request; True when the same model should be tried again."""
        attempt.add(model, exc.usage, self._settings)
        if exc.kind is FailureKind.TRANSIENT and retry < self._settings.max_retries:
            log.info(
                "llm.retry",
                route=attempt.route.name,
                model=model,
                reason=exc.error_type,
                retry=retry + 1,
            )
            return True
        return False

    @staticmethod
    def _switch(attempt: _Attempt, model: str, reason: str, kind: str) -> None:
        index = attempt.chain.index(model)
        following = attempt.chain[index + 1] if index + 1 < len(attempt.chain) else None
        route = attempt.route.name
        if following is not None:
            LLM_FALLBACKS.labels(route=route, reason=kind).inc()
            log.warning(
                "llm.fallback", route=route, model=model, reason=reason, kind=kind, next=following
            )
        else:
            log.warning("llm.chain_exhausted", route=route, model=model, reason=reason, kind=kind)

    def _exhausted(self, attempt: _Attempt, failure: _Failure) -> tuple[LLMCallOutcome, str]:
        error = failure.error
        if error is None:  # every model's circuit was open: nothing was called
            raise self._blocked(attempt.route, "circuit_open")
        if error.kind is FailureKind.BLOCKED:
            return LLMCallOutcome.REFUSED, "refused"
        return LLMCallOutcome.ERROR, error.error_type

    def _record(
        self,
        attempt: _Attempt,
        outcome: LLMCallOutcome,
        clock: float,
        *,
        error_type: str | None = None,
        user_id: UUID | None = None,
        subject: Subject | None = None,
    ) -> CallRecord:
        served = attempt.served_model or attempt.model
        latency = clock - attempt.started
        route = attempt.route.name
        LLM_CALLS.labels(route=route, outcome=outcome.value).inc()
        LLM_COST_USD.labels(route=route).inc(float(attempt.cost))
        LLM_LATENCY.labels(route=route).observe(latency)
        log.info(
            "llm.call",
            route=route,
            requested_model=attempt.model,
            model=served,
            prompt_version=attempt.route.prompt_version,
            attempts=attempt.attempts,
            input_tokens=attempt.usage.input,
            output_tokens=attempt.usage.output,
            cache_read_tokens=attempt.usage.cache_read,
            cost_usd=str(attempt.cost),
            latency_ms=round(latency * 1000),
            outcome=outcome.value,
        )
        return CallRecord(
            route=route,
            tier=attempt.route.tier.value,
            requested_model=attempt.model,
            model=served,
            prompt_version=attempt.route.prompt_version,
            usage=attempt.usage,
            cost_usd=attempt.cost,
            latency_ms=round(latency * 1000),
            attempts=attempt.attempts,
            outcome=outcome,
            error_type=error_type,
            user_id=user_id,
            subject_type=subject.type if subject else None,
            subject_id=subject.id if subject else None,
        )


class LLMGateway(_GatewayCore):
    def __init__(
        self,
        settings: LLMSettings,
        provider: SyncProvider,
        ledger: SpendLedger,
        *,
        breakers: SyncCircuitBreakerFactory | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(settings)
        self._provider = provider
        self._ledger = ledger
        self._breakers = breakers or SyncCircuitBreakerFactory(
            default_threshold=BREAKER_THRESHOLD, default_ttl=BREAKER_RESET_SECONDS
        )
        self._sleep = sleep

    def run[OutputT: BaseModel](
        self,
        route_name: str,
        parts: Sequence[Part],
        output_model: type[OutputT],
        *,
        user_id: UUID | None = None,
        subject: Subject | None = None,
    ) -> OutputT:
        attempt = self._start(route_name, time.perf_counter())
        self._check_budget(attempt, self._ledger.spent_today())
        base = self._request(attempt, parts, output_model)

        def finish(outcome: LLMCallOutcome, error: str | None = None) -> None:
            record = self._record(
                attempt, outcome, time.perf_counter(), error_type=error, user_id=user_id,
                subject=subject,
            )  # fmt: skip
            self._ledger.record(record)

        failure = _Failure()
        for index, model in enumerate(attempt.chain):
            if index:  # never switch models to get around the budget
                try:
                    self._check_budget(attempt, self._ledger.spent_today() + attempt.cost)
                except LLMUnavailableError:
                    finish(LLMCallOutcome.ERROR, "budget")
                    raise
            request = base.for_model(model)
            try:
                response = self._try(request, attempt)
                parsed, problems = _validate(output_model, response.text)
                if problems is not None:
                    response = self._try(request.with_repair(response.text, problems), attempt)
                    parsed, problems = _validate(output_model, response.text)
            except ProviderError as exc:
                failure.error = exc
                self._switch(attempt, model, exc.error_type, exc.kind.value)
                continue
            except _CircuitOpen:
                self._switch(attempt, model, "circuit_open", "circuit_open")
                continue
            if parsed is None:
                finish(LLMCallOutcome.INVALID_OUTPUT, "ValidationError")
                raise LLMCallFailedError("invalid_output")
            finish(LLMCallOutcome.OK)
            return parsed
        outcome, reason = self._exhausted(attempt, failure)
        finish(outcome, failure.error.error_type if failure.error else None)
        raise LLMCallFailedError(reason)

    def _try(self, request: ProviderRequest, attempt: _Attempt) -> ProviderResponse:
        """One model: retries transient errors with jittered backoff, raises the rest."""
        retry = 0
        while True:
            try:
                return self._once(request, attempt)
            except ProviderError as exc:
                if not self._retry_or_raise(attempt, request.model, exc, retry):
                    raise
                self._sleep(backoff(retry))
                retry += 1

    def _once(self, request: ProviderRequest, attempt: _Attempt) -> ProviderResponse:
        outcome: ProviderResponse | ProviderError
        try:
            with self._breakers.get_breaker(f"llm:{request.model}"):
                try:
                    outcome = self._provider.complete(request)
                except ProviderError as exc:
                    if exc.kind is not FailureKind.BLOCKED:
                        raise
                    outcome = exc  # a refusal is not an outage: the breaker stays closed
        except OpenedState:
            raise _CircuitOpen from None
        if isinstance(outcome, ProviderError):
            raise outcome
        attempt.add(request.model, outcome.usage, self._settings)
        return outcome


class AsyncLLMGateway(_GatewayCore):
    def __init__(
        self,
        settings: LLMSettings,
        provider: AsyncProvider,
        ledger: AsyncSpendLedger,
        *,
        breakers: AsyncCircuitBreakerFactory | None = None,
        cache: AnswerCache | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(settings)
        self._provider = provider
        self._ledger = ledger
        self._breakers = breakers or AsyncCircuitBreakerFactory(
            default_threshold=BREAKER_THRESHOLD, default_ttl=BREAKER_RESET_SECONDS
        )
        self._cache = cache if cache is not None and cache.enabled else None
        self._sleep = sleep

    async def run[OutputT: BaseModel](
        self,
        route_name: str,
        parts: Sequence[Part],
        output_model: type[OutputT],
        *,
        user_id: UUID | None = None,
        subject: Subject | None = None,
    ) -> OutputT:
        attempt = self._start(route_name, time.perf_counter())  # the kill switch first
        base = self._request(attempt, parts, output_model)
        cacheable = self._cache is not None and attempt.route.cacheable
        if cacheable:
            cached = await self._cached(attempt, base, output_model)
            if cached is not None:
                return cached
        self._check_budget(attempt, await self._ledger.spent_today())

        async def finish(outcome: LLMCallOutcome, error: str | None = None) -> None:
            record = self._record(
                attempt, outcome, time.perf_counter(), error_type=error, user_id=user_id,
                subject=subject,
            )  # fmt: skip
            await self._ledger.record(record)

        failure = _Failure()
        for index, model in enumerate(attempt.chain):
            if index:  # never switch models to get around the budget
                try:
                    self._check_budget(attempt, await self._ledger.spent_today() + attempt.cost)
                except LLMUnavailableError:
                    await finish(LLMCallOutcome.ERROR, "budget")
                    raise
            request = base.for_model(model)
            try:
                response = await self._try(request, attempt)
                parsed, problems = _validate(output_model, response.text)
                if problems is not None:
                    response = await self._try(
                        request.with_repair(response.text, problems), attempt
                    )
                    parsed, problems = _validate(output_model, response.text)
            except ProviderError as exc:
                failure.error = exc
                self._switch(attempt, model, exc.error_type, exc.kind.value)
                continue
            except _CircuitOpen:
                self._switch(attempt, model, "circuit_open", "circuit_open")
                continue
            if parsed is None:
                await finish(LLMCallOutcome.INVALID_OUTPUT, "ValidationError")
                raise LLMCallFailedError("invalid_output")
            await finish(LLMCallOutcome.OK)
            if cacheable and self._cache is not None:
                # Keyed by the model that answered: a fallback's answer is never served as the
                # primary's.
                await self._cache.put(
                    answer_key(request, attempt.route.prompt_version), response.text
                )
            return parsed
        outcome, reason = self._exhausted(attempt, failure)
        await finish(outcome, failure.error.error_type if failure.error else None)
        raise LLMCallFailedError(reason)

    async def _cached[OutputT: BaseModel](
        self, attempt: _Attempt, base: ProviderRequest, output_model: type[OutputT]
    ) -> OutputT | None:
        """A validated answer any model of the chain gave to this exact request, the primary's
        first."""
        if self._cache is None:
            return None
        for model in attempt.chain:
            cached = await self._cache.get(
                answer_key(base.for_model(model), attempt.route.prompt_version)
            )
            if cached is None:
                continue
            parsed, _ = _validate(output_model, cached)
            if parsed is not None:
                LLM_CACHE_HITS.labels(route=attempt.route.name).inc()
                log.info("llm.cache_hit", route=attempt.route.name, model=model)
                return parsed
        return None

    async def _try(self, request: ProviderRequest, attempt: _Attempt) -> ProviderResponse:
        retry = 0
        while True:
            try:
                return await self._once(request, attempt)
            except ProviderError as exc:
                if not self._retry_or_raise(attempt, request.model, exc, retry):
                    raise
                await self._sleep(backoff(retry))
                retry += 1

    async def _once(self, request: ProviderRequest, attempt: _Attempt) -> ProviderResponse:
        outcome: ProviderResponse | ProviderError
        try:
            async with await self._breakers.get_breaker(f"llm:{request.model}"):
                try:
                    outcome = await self._provider.complete(request)
                except ProviderError as exc:
                    if exc.kind is not FailureKind.BLOCKED:
                        raise
                    outcome = exc  # a refusal is not an outage: the breaker stays closed
        except OpenedState:
            raise _CircuitOpen from None
        if isinstance(outcome, ProviderError):
            raise outcome
        attempt.add(request.model, outcome.usage, self._settings)
        return outcome

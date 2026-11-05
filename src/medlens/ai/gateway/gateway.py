"""LLMGateway: the only way MedLens calls a model (CLAUDE.md §3, §5).

For every call, in order:

1. kill switch (MEDLENS_LLM__ENABLED) and today's spend against the daily budget, checked
   before the call: blocked calls raise ``LLMUnavailableError`` and are counted, not metered;
2. the route's tier picks the model; its versioned prompt is the cached system prompt, the
   variable content (images, untrusted text wrapped in tags) follows;
3. the provider call runs behind a circuit breaker; structured output is requested from the
   output model's JSON schema;
4. the output is validated against the output model, with one repair attempt;
5. one ``llm_calls`` row records model, prompt version, tokens, cost, latency and outcome.

``LLMGateway`` is synchronous (Celery workers); ``AsyncLLMGateway`` serves the API.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog
from anthropic import transform_schema
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
    LLM_LATENCY,
    LLM_SPENT_TODAY,
)
from medlens.ai.gateway.pricing import TokenUsage, cost_usd
from medlens.ai.gateway.providers import (
    AsyncProvider,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    SyncProvider,
    image_block,
)
from medlens.ai.gateway.registry import ROUTES, Route, Tier, model_for
from medlens.ai.guardrails import wrap_untrusted
from medlens.ai.prompts import load_prompt
from medlens.core.config import LLMSettings
from medlens.modules.control_plane.models import LLMCallOutcome
from medlens.ports.llm import LLMCallFailedError, LLMUnavailableError

log = structlog.stdlib.get_logger(__name__)

BREAKER_THRESHOLD = 5
BREAKER_RESET_SECONDS = 60


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
            blocks.append(image_block(part.data, part.media_type))
        elif isinstance(part, UntrustedPart):
            text = wrap_untrusted(part.text, kind=part.kind, attrs=part.attrs)
            blocks.append({"type": "text", "text": text})
        else:
            blocks.append({"type": "text", "text": part.text})
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


@dataclass
class _Attempt:
    """Bookkeeping for one gateway call (up to two provider requests)."""

    route: Route
    model: str
    started: float
    usage: TokenUsage = field(default_factory=TokenUsage)
    served_model: str | None = None
    attempts: int = 0

    def add(self, response: ProviderResponse) -> None:
        self.attempts += 1
        self.usage = self.usage + response.usage
        self.served_model = response.model


class _GatewayCore:
    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings

    def _start(self, route_name: str, clock: float) -> _Attempt:
        route = ROUTES[route_name]
        if not self._settings.enabled:
            raise self._blocked(route, "disabled")
        return _Attempt(route, model_for(route.tier, self._settings), clock)

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
            messages=({"role": "user", "content": _content(parts)},),
            schema=transform_schema(output_model),
            max_tokens=route.max_tokens,
            effort=route.effort,
            fallback=route.tier is Tier.BALANCED and self._settings.refusal_fallback,
        )

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
        prices = self._settings.prices
        price = prices.get(served) or prices[attempt.model]  # a fallback model may be unpriced
        cost = cost_usd(price, attempt.usage)
        latency = clock - attempt.started
        route = attempt.route.name
        LLM_CALLS.labels(route=route, outcome=outcome.value).inc()
        LLM_COST_USD.labels(route=route).inc(float(cost))
        LLM_LATENCY.labels(route=route).observe(latency)
        log.info(
            "llm.call",
            route=route,
            model=served,
            prompt_version=attempt.route.prompt_version,
            input_tokens=attempt.usage.input,
            output_tokens=attempt.usage.output,
            cache_read_tokens=attempt.usage.cache_read,
            cost_usd=str(cost),
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
            cost_usd=cost,
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
    ) -> None:
        super().__init__(settings)
        self._provider = provider
        self._ledger = ledger
        self._breakers = breakers or SyncCircuitBreakerFactory(
            default_threshold=BREAKER_THRESHOLD, default_ttl=BREAKER_RESET_SECONDS
        )

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
        request = self._request(attempt, parts, output_model)

        def finish(outcome: LLMCallOutcome, error: str | None = None) -> None:
            self._ledger.record(
                self._record(
                    attempt,
                    outcome,
                    time.perf_counter(),
                    error_type=error,
                    user_id=user_id,
                    subject=subject,
                )
            )

        try:
            response = self._call(request, attempt)
            if response.stop_reason == "refusal":
                finish(LLMCallOutcome.REFUSED)
                raise LLMCallFailedError("refused")
            parsed, problems = _validate(output_model, response.text)
            if problems is not None:
                response = self._call(request.with_repair(response.text, problems), attempt)
                parsed, problems = _validate(output_model, response.text)
        except ProviderError as exc:
            finish(LLMCallOutcome.ERROR, exc.error_type)
            raise LLMCallFailedError(exc.error_type) from None
        if parsed is None:
            finish(LLMCallOutcome.INVALID_OUTPUT, "ValidationError")
            raise LLMCallFailedError("invalid_output")
        finish(LLMCallOutcome.OK)
        return parsed

    def _call(self, request: ProviderRequest, attempt: _Attempt) -> ProviderResponse:
        try:
            with self._breakers.get_breaker(f"llm:{request.model}"):
                response = self._provider.complete(request)
        except OpenedState:
            raise self._blocked(attempt.route, "circuit_open") from None
        attempt.add(response)
        return response


class AsyncLLMGateway(_GatewayCore):
    def __init__(
        self,
        settings: LLMSettings,
        provider: AsyncProvider,
        ledger: AsyncSpendLedger,
        *,
        breakers: AsyncCircuitBreakerFactory | None = None,
        cache: AnswerCache | None = None,
    ) -> None:
        super().__init__(settings)
        self._provider = provider
        self._ledger = ledger
        self._breakers = breakers or AsyncCircuitBreakerFactory(
            default_threshold=BREAKER_THRESHOLD, default_ttl=BREAKER_RESET_SECONDS
        )
        self._cache = cache if cache is not None and cache.enabled else None

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
        request = self._request(attempt, parts, output_model)
        key = None
        if self._cache is not None and attempt.route.cacheable:
            key = answer_key(request, attempt.route.prompt_version)
            cached = await self._cache.get(key)
            if cached is not None:
                parsed, problems = _validate(output_model, cached)
                if parsed is not None:
                    LLM_CACHE_HITS.labels(route=attempt.route.name).inc()
                    log.info("llm.cache_hit", route=attempt.route.name)
                    return parsed
        self._check_budget(attempt, await self._ledger.spent_today())

        async def finish(outcome: LLMCallOutcome, error: str | None = None) -> None:
            await self._ledger.record(
                self._record(
                    attempt,
                    outcome,
                    time.perf_counter(),
                    error_type=error,
                    user_id=user_id,
                    subject=subject,
                )
            )

        try:
            response = await self._call(request, attempt)
            if response.stop_reason == "refusal":
                await finish(LLMCallOutcome.REFUSED)
                raise LLMCallFailedError("refused")
            parsed, problems = _validate(output_model, response.text)
            if problems is not None:
                response = await self._call(request.with_repair(response.text, problems), attempt)
                parsed, problems = _validate(output_model, response.text)
        except ProviderError as exc:
            await finish(LLMCallOutcome.ERROR, exc.error_type)
            raise LLMCallFailedError(exc.error_type) from None
        if parsed is None:
            await finish(LLMCallOutcome.INVALID_OUTPUT, "ValidationError")
            raise LLMCallFailedError("invalid_output")
        await finish(LLMCallOutcome.OK)
        if key is not None and self._cache is not None:
            await self._cache.put(key, response.text)
        return parsed

    async def _call(self, request: ProviderRequest, attempt: _Attempt) -> ProviderResponse:
        try:
            async with await self._breakers.get_breaker(f"llm:{request.model}"):
                response = await self._provider.complete(request)
        except OpenedState:
            raise self._blocked(attempt.route, "circuit_open") from None
        attempt.add(response)
        return response

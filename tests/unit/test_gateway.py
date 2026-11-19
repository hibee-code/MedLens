"""LLMGateway on Gemini: kill switch, budget, the per-tier fallback chain (retries, rate limits,
safety blocks), structured output with repair, metering, the answer cache and the provider's
mapping of a mocked Gemini client (no network)."""

from decimal import Decimal
from typing import Any

import httpx2
import pytest
from google.genai import errors, types
from prometheus_client import REGISTRY
from pydantic import BaseModel, SecretStr

from medlens.adapters.fakes import (
    AsyncFakeLLMProvider,
    AsyncMemoryLedger,
    FakeLLM,
    FakeLLMProvider,
    MemoryLedger,
)
from medlens.adapters.llm import GatewayPackReader
from medlens.ai.gateway import providers
from medlens.ai.gateway.cache import AnswerCache
from medlens.ai.gateway.gateway import (
    AsyncLLMGateway,
    ImagePart,
    LLMGateway,
    Subject,
    TextPart,
    UntrustedPart,
)
from medlens.ai.gateway.pricing import TokenUsage, cost_usd
from medlens.ai.gateway.providers import (
    AsyncGeminiProvider,
    FailureKind,
    GeminiProvider,
    ProviderError,
    ProviderRequest,
    image_part,
)
from medlens.ai.prompts import load_prompt
from medlens.core.config import LLMSettings, ModelPrice
from medlens.modules.control_plane.models import LLMCallOutcome
from medlens.modules.verification.pipeline import PackExtraction
from medlens.ports.llm import LLMCallFailedError, LLMUnavailableError

VALID = FakeLLM.reading(nafdac_number="A4-1234", product_name="Panadol Extra")
LITE, FLASH, HYDROGEN = "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-4-hydrogen"
# Real-looking prices, so metering is visible (the defaults are the free tier's 0).
PRICED = {
    LITE: ModelPrice(input=0.10, output=0.50, cache_write=0, cache_read=0.01),
    FLASH: ModelPrice(input=2.0, output=10.0, cache_write=0, cache_read=0.20),
    HYDROGEN: ModelPrice(input=1.5, output=7.5, cache_write=0, cache_read=0.15),
}


def transient(kind: str = "timeout") -> ProviderError:
    return ProviderError(kind, FailureKind.TRANSIENT)


RATE_LIMITED = ProviderError("http_429", FailureKind.RATE_LIMITED)


def blocked(usage: TokenUsage | None = None) -> ProviderError:
    return ProviderError("finish:SAFETY", FailureKind.BLOCKED, usage)


class Answer(BaseModel):
    answer: str


def settings(**overrides: Any) -> LLMSettings:
    return LLMSettings(**{"enabled": True, "prices": PRICED, **overrides})


def gateway(
    *script: object, ledger: MemoryLedger | None = None, **overrides: Any
) -> tuple[LLMGateway, FakeLLMProvider, MemoryLedger]:
    provider = FakeLLMProvider(script=list(script))
    ledger = ledger or MemoryLedger()
    sleeps: list[float] = []
    llm = LLMGateway(settings(**overrides), provider, ledger, sleep=sleeps.append)
    llm.sleeps = sleeps  # type: ignore[attr-defined]
    return llm, provider, ledger


def extract(llm: LLMGateway, text: str = "Read the pack.") -> PackExtraction:
    return llm.run(
        "pack_extraction",
        [ImagePart(b"jpeg"), TextPart(text)],
        PackExtraction,
        subject=Subject("scan", "s-1"),
    )


def models(provider: FakeLLMProvider) -> list[str]:
    return [request.model for request in provider.requests]


def fallbacks(route: str, reason: str) -> float:
    labels = {"route": route, "reason": reason}
    return REGISTRY.get_sample_value("medlens_llm_fallbacks_total", labels) or 0.0


# ------------------------------------------------------------------------- the basics
def test_a_valid_answer_is_parsed_metered_and_recorded_once() -> None:
    llm, provider, ledger = gateway(VALID)

    result = extract(llm)

    assert result.nafdac_number.value == "A4-1234"
    assert models(provider) == [LITE]
    [call] = ledger.calls
    assert call.outcome is LLMCallOutcome.OK
    assert (call.route, call.tier, call.requested_model, call.model) == (
        "pack_extraction", "fast", LITE, LITE,
    )  # fmt: skip
    assert call.prompt_version == "pack_extraction@v2"
    assert (call.subject_type, call.subject_id) == ("scan", "s-1")
    # 1000 input tokens at $0.10/M + 200 output tokens at $0.50/M
    assert call.cost_usd == Decimal("0.000200")
    assert call.attempts == 1


def test_the_kill_switch_blocks_before_any_call() -> None:
    llm, provider, ledger = gateway(VALID, enabled=False)

    with pytest.raises(LLMUnavailableError) as raised:
        extract(llm)

    assert raised.value.reason == "disabled"
    assert provider.requests == []
    assert ledger.calls == []


def test_an_exhausted_budget_stops_calls() -> None:
    llm, provider, _ = gateway(VALID, ledger=MemoryLedger(spent=Decimal("5.00")))

    with pytest.raises(LLMUnavailableError) as raised:
        extract(llm)

    assert raised.value.reason == "budget"
    assert provider.requests == []


def test_the_budget_is_metered_even_at_price_zero() -> None:
    llm, provider, _ = gateway(VALID, VALID, prices=LLMSettings().prices, daily_budget_usd=0)

    with pytest.raises(LLMUnavailableError):  # a zero budget blocks every call
        extract(llm)
    free, _, free_ledger = gateway(VALID, prices=LLMSettings().prices)
    extract(free)

    assert provider.requests == []
    assert free_ledger.calls[0].cost_usd == Decimal(0)
    assert free_ledger.calls[0].usage == TokenUsage(1000, 200)  # tokens are still recorded


def test_spend_from_earlier_calls_counts_against_the_budget() -> None:
    llm, provider, _ = gateway(VALID, VALID, daily_budget_usd=0.0002)

    extract(llm)  # costs exactly the budget
    with pytest.raises(LLMUnavailableError):
        extract(llm)

    assert len(provider.requests) == 1


def test_invalid_output_is_repaired_once_on_the_same_model() -> None:
    llm, provider, ledger = gateway('{"nafdac_number": "A4-1234"}', VALID)

    result = extract(llm)

    assert result.product_name.value == "Panadol Extra"
    first, repair = provider.requests
    assert repair.model == first.model == LITE
    assert repair.messages[: len(first.messages)] == first.messages
    assert repair.messages[-2]["role"] == "model"
    assert repair.messages[-1]["role"] == "user"
    assert "product_name" in repair.messages[-1]["parts"][0]["text"]
    [call] = ledger.calls
    assert call.attempts == 2
    assert call.cost_usd == Decimal("0.000400")  # both attempts are paid for


def test_output_still_invalid_after_the_repair_fails_the_call() -> None:
    llm, provider, ledger = gateway("not json", "still not json")

    with pytest.raises(LLMCallFailedError) as raised:
        extract(llm)

    assert raised.value.reason == "invalid_output"
    assert len(provider.requests) == 2
    assert ledger.calls[0].outcome is LLMCallOutcome.INVALID_OUTPUT


# ------------------------------------------------------------------------- the fallback chain
def test_a_transient_error_is_retried_on_the_same_model_then_succeeds() -> None:
    llm, provider, ledger = gateway(transient(), VALID)

    extract(llm)

    assert models(provider) == [LITE, LITE]
    assert len(llm.sleeps) == 1  # type: ignore[attr-defined]
    assert 0 <= llm.sleeps[0] <= 0.5  # type: ignore[attr-defined]  # jittered backoff
    assert ledger.calls[0].attempts == 2
    assert ledger.calls[0].model == LITE


def test_transient_errors_beyond_max_retries_move_to_the_next_model() -> None:
    before = fallbacks("pack_extraction", "transient")
    llm, provider, ledger = gateway(transient(), transient(), transient("http_503"), VALID)

    extract(llm)

    assert models(provider) == [LITE, LITE, LITE, FLASH]  # MAX_RETRIES=2, then the fallback
    assert ledger.calls[0].model == FLASH
    assert ledger.calls[0].requested_model == LITE
    assert fallbacks("pack_extraction", "transient") == before + 1


def test_a_rate_limit_moves_to_the_next_model_at_once() -> None:
    before = fallbacks("pack_extraction", "rate_limited")
    llm, provider, ledger = gateway(RATE_LIMITED, VALID)

    extract(llm)

    assert models(provider) == [LITE, FLASH]
    assert llm.sleeps == []  # type: ignore[attr-defined]
    assert ledger.calls[0].model == FLASH
    assert fallbacks("pack_extraction", "rate_limited") == before + 1


def test_an_overloaded_model_moves_to_the_next_at_once() -> None:
    """HTTP 503 "experiencing high demand": retrying would only add waiting."""
    before = fallbacks("pack_extraction", "overloaded")
    overloaded = ProviderError("http_503", FailureKind.OVERLOADED)
    llm, provider, ledger = gateway(overloaded, VALID)

    extract(llm)

    assert models(provider) == [LITE, FLASH]
    assert llm.sleeps == []  # type: ignore[attr-defined]
    assert ledger.calls[0].model == FLASH
    assert fallbacks("pack_extraction", "overloaded") == before + 1


def test_a_safety_block_moves_to_the_next_model_without_a_retry() -> None:
    llm, provider, ledger = gateway(blocked(), VALID)

    extract(llm)

    assert models(provider) == [LITE, FLASH]
    assert ledger.calls[0].outcome is LLMCallOutcome.OK
    assert ledger.calls[0].attempts == 2


def test_when_every_model_fails_the_call_fails_for_the_non_ai_fallback() -> None:
    llm, provider, ledger = gateway(RATE_LIMITED, RATE_LIMITED)

    with pytest.raises(LLMCallFailedError) as raised:
        extract(llm)

    assert raised.value.reason == "http_429"
    assert models(provider) == [LITE, FLASH]
    [call] = ledger.calls
    assert (call.outcome, call.error_type) == (LLMCallOutcome.ERROR, "http_429")


def test_when_every_model_refuses_the_call_is_recorded_as_refused() -> None:
    llm, _, ledger = gateway(blocked(), blocked())

    with pytest.raises(LLMCallFailedError) as raised:
        extract(llm)

    assert raised.value.reason == "refused"
    assert ledger.calls[0].outcome is LLMCallOutcome.REFUSED


def test_the_chain_never_switches_models_to_get_around_the_budget() -> None:
    """The primary's blocked answer still costs money; with the budget reached, no fallback."""
    usage = TokenUsage(1000, 200)  # $0.0002 on the fast tier's primary
    llm, provider, ledger = gateway(blocked(usage), VALID, daily_budget_usd=0.0002)

    with pytest.raises(LLMUnavailableError) as raised:
        extract(llm)

    assert raised.value.reason == "budget"
    assert models(provider) == [LITE]  # the fallback was never called
    [call] = ledger.calls  # what was spent is recorded
    assert (call.outcome, call.error_type, call.cost_usd) == (
        LLMCallOutcome.ERROR, "budget", Decimal("0.000200"),
    )  # fmt: skip


def test_an_open_circuit_skips_to_the_next_model() -> None:
    provider = FakeLLMProvider(script=[item for _ in range(5) for item in (transient(), VALID)])
    llm = LLMGateway(settings(max_retries=0), provider, MemoryLedger(), sleep=lambda _: None)
    for _ in range(5):  # the primary fails each time, the fallback answers
        extract(llm)
    provider.script = [VALID]

    extract(llm)

    assert models(provider)[-1] == FLASH  # the primary's breaker is open: not called
    assert models(provider).count(LITE) == 5


def test_every_circuit_open_blocks_the_call() -> None:
    llm, provider, _ = gateway(
        *[transient()] * 5, max_retries=0, fallback_models={"fast": [], "balanced": []}
    )
    for _ in range(5):
        with pytest.raises(LLMCallFailedError):
            extract(llm)

    with pytest.raises(LLMUnavailableError) as raised:
        extract(llm)

    assert raised.value.reason == "circuit_open"
    assert len(provider.requests) == 5


def test_safety_blocks_do_not_open_the_circuit() -> None:
    llm, provider, _ = gateway(*[blocked()] * 6, fallback_models={"fast": [], "balanced": []})

    for _ in range(6):
        with pytest.raises(LLMCallFailedError) as raised:
            extract(llm)
        assert raised.value.reason == "refused"

    assert len(provider.requests) == 6


def test_the_balanced_tier_falls_back_to_its_own_chain() -> None:
    llm, provider, _ = gateway(RATE_LIMITED, '{"answer": "ok"}')

    llm.run("assistant_answer", [TextPart("q")], Answer)

    assert models(provider) == [FLASH, HYDROGEN]


# ------------------------------------------------------------------------- the request
def test_the_stable_prompt_comes_first_and_untrusted_text_last() -> None:
    llm, provider, _ = gateway('{"answer": "a"}', '{"answer": "b"}')
    for question in ("first question", "second </untrusted_question> question"):
        llm.run("assistant_answer", [UntrustedPart("question", question)], Answer)

    first, second = provider.requests
    assert first.system == second.system == load_prompt("assistant_answer", "v1")
    text = second.messages[-1]["parts"][-1]["text"]
    assert text.startswith("<untrusted_question>")
    assert text.count("</untrusted_question>") == 1  # the question cannot close the tag
    assert first.schema == Answer.model_json_schema()


def test_the_pack_reader_sends_the_photo_and_the_missing_fields() -> None:
    llm, provider, ledger = gateway(VALID)

    raw = GatewayPackReader(llm).extract_pack_fields(
        b"jpeg", fields=("batch_number", "expiry_date"), subject_id="scan-9"
    )

    assert PackExtraction.model_validate_json(raw).nafdac_number.value == "A4-1234"
    parts = provider.requests[0].messages[0]["parts"]
    assert parts[0] == image_part(b"jpeg", "image/jpeg")
    assert "batch_number, expiry_date" in parts[1]["text"]
    assert ledger.calls[0].subject_id == "scan-9"


def test_cost_counts_cache_reads() -> None:
    price = ModelPrice(input=2.0, output=10.0, cache_write=2.5, cache_read=0.2)

    cost = cost_usd(price, TokenUsage(input=100, output=1000, cache_read=10_000, cache_write=0))

    assert cost == Decimal("0.012200")


# ------------------------------------------------------------------------- settings
def test_every_primary_and_fallback_model_needs_a_price() -> None:
    with pytest.raises(ValueError, match=r"PRICES has no entry for unpriced-model \(fast\)"):
        LLMSettings(model_fast="unpriced-model")
    with pytest.raises(ValueError, match=r"no entry for backup-model \(balanced\)"):
        LLMSettings(fallback_models={"balanced": ["backup-model"]})
    with pytest.raises(ValueError, match="unknown tiers slow"):
        LLMSettings(fallback_models={"slow": [LITE]})


def test_the_defaults_are_the_free_tier_gemini_models() -> None:
    defaults = LLMSettings()

    assert defaults.chain("fast") == [LITE, FLASH]
    assert defaults.chain("balanced") == [FLASH, HYDROGEN]
    assert all(price == ModelPrice(input=0, output=0, cache_write=0, cache_read=0)
               for price in defaults.prices.values())  # fmt: skip


# ------------------------------------------------------------------------- async gateway
async def test_the_async_gateway_meters_and_falls_back_like_the_sync_one() -> None:
    provider = AsyncFakeLLMProvider(script=[RATE_LIMITED, "nope", '{"answer": "ok"}'])
    ledger = AsyncMemoryLedger()
    llm = AsyncLLMGateway(settings(), provider, ledger, sleep=_no_sleep)

    result = await llm.run("assistant_answer", [TextPart("q")], Answer)

    assert result.answer == "ok"
    assert [request.model for request in provider.requests] == [FLASH, HYDROGEN, HYDROGEN]
    [call] = ledger.calls
    assert (call.attempts, call.tier, call.model) == (3, "balanced", HYDROGEN)


async def _no_sleep(_: float) -> None:
    return None


async def test_the_async_gateway_retries_blocks_and_records_failures() -> None:
    with pytest.raises(LLMUnavailableError):
        await AsyncLLMGateway(
            settings(), AsyncFakeLLMProvider(), AsyncMemoryLedger(spent=Decimal(100))
        ).run("assistant_answer", [], Answer)
    retried = AsyncFakeLLMProvider(script=[transient(), '{"answer": "ok"}'])
    await AsyncLLMGateway(settings(), retried, AsyncMemoryLedger(), sleep=_no_sleep).run(
        "assistant_answer", [], Answer
    )
    refusing = AsyncFakeLLMProvider(script=[blocked(), blocked()])
    with pytest.raises(LLMCallFailedError) as raised:
        await AsyncLLMGateway(settings(), refusing, AsyncMemoryLedger()).run(
            "assistant_answer", [], Answer
        )
    over_budget = AsyncFakeLLMProvider(script=[blocked(TokenUsage(1_000_000, 0)), "x"])
    with pytest.raises(LLMUnavailableError):
        await AsyncLLMGateway(settings(daily_budget_usd=1.0), over_budget, AsyncMemoryLedger()).run(
            "assistant_answer", [], Answer
        )

    assert [r.model for r in retried.requests] == [FLASH, FLASH]
    assert raised.value.reason == "refused"
    assert len(over_budget.requests) == 1


async def test_cached_answers_are_keyed_by_the_model_that_answered() -> None:
    from tests.unit.test_models_and_caches import AsyncDictRedis

    redis = AsyncDictRedis()
    cache = AnswerCache(redis, ttl_seconds=60)  # type: ignore[arg-type]
    provider = AsyncFakeLLMProvider(script=[RATE_LIMITED, '{"answer": "from hydrogen"}'])
    llm = AsyncLLMGateway(settings(), provider, AsyncMemoryLedger(), cache=cache)

    first = await llm.run("assistant_answer", [TextPart("q")], Answer)
    again = await llm.run("assistant_answer", [TextPart("q")], Answer)

    assert first.answer == again.answer == "from hydrogen"
    assert len(provider.requests) == 2  # the second answer came from the cache
    assert len(redis.data) == 1  # stored once, under the fallback's key


async def test_when_every_model_fails_the_assistant_degrades() -> None:
    """The feature's non-AI fallback: search results instead of an answer."""
    from medlens.core.config import KnowledgeSettings
    from medlens.modules.knowledge.service import SearchResult
    from medlens.modules.rag.service import AssistantService, Outcome
    from tests.unit.test_assistant_evidence import Knowledge, hit

    found = SearchResult([hit(None)], vector_search=True, reranked=False)
    provider = AsyncFakeLLMProvider(script=[RATE_LIMITED, blocked()])
    llm = AsyncLLMGateway(settings(), provider, AsyncMemoryLedger())
    assistant = AssistantService(Knowledge(found), llm, KnowledgeSettings())  # type: ignore[arg-type]

    actor = type("Actor", (), {"user_id": None})()
    reply = await assistant.ask(actor, "Any alert for Coartem?")

    assert reply.outcome is Outcome.DEGRADED
    assert reply.reason == "model_refused"
    assert reply.results == found.hits


# ------------------------------------------------------------------------- the Gemini provider
def response(**overrides: Any) -> types.GenerateContentResponse:
    candidate = types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text='{"answer": "ok"}')]),
        finish_reason=types.FinishReason.STOP,
    )
    values: dict[str, Any] = {
        "candidates": [candidate],
        "usage_metadata": types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1020,
            candidates_token_count=30,
            cached_content_token_count=900,
            thoughts_token_count=12,
        ),
        "model_version": FLASH,
    }
    return types.GenerateContentResponse(**(values | overrides))


class Models:
    """A mocked ``client.models`` (and ``client.aio.models``)."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **arguments: Any) -> Any:
        self.calls.append(arguments)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class AsyncModels(Models):
    async def generate_content(self, **arguments: Any) -> Any:  # type: ignore[override]
        return Models.generate_content(self, **arguments)


class Client:
    def __init__(self, models: Models) -> None:
        self.models = models
        self.aio = self
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


def request() -> ProviderRequest:
    return ProviderRequest(
        model=FLASH,
        system="system",
        messages=({"role": "user", "parts": [image_part(b"jpeg", "image/jpeg"), {"text": "q"}]},),
        schema=Answer.model_json_schema(),
        max_tokens=100,
        effort="medium",
    )


def test_the_gemini_provider_sends_the_request_and_maps_the_answer() -> None:
    models = Models(response())
    provider = GeminiProvider(settings(), Client(models))

    answer = provider.complete(request())

    assert answer.text == '{"answer": "ok"}'
    assert answer.model == FLASH
    # 1020 prompt tokens of which 900 cached; thinking tokens are billed as output
    assert answer.usage == TokenUsage(input=120, output=42, cache_read=900, cache_write=0)
    [call] = models.calls
    config = call["config"]
    assert call["model"] == FLASH
    assert config.system_instruction == "system"
    # The answer's 100 tokens plus the medium level's thinking headroom (thinking counts here)
    assert config.max_output_tokens == 100 + providers.THINKING_HEADROOM["medium"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == Answer.model_json_schema()
    assert config.thinking_config.thinking_level is types.ThinkingLevel.MEDIUM
    assert config.automatic_continuation is False
    [content] = call["contents"]
    assert content.role == "user"
    assert content.parts[0].inline_data.data == b"jpeg"  # the photo as a Gemini image input
    assert content.parts[0].inline_data.mime_type == "image/jpeg"
    assert content.parts[1].text == "q"


@pytest.mark.parametrize(
    ("outcome", "kind", "error_type"),
    [
        (errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}),
         FailureKind.RATE_LIMITED, "http_429"),
        (errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}}),
         FailureKind.OVERLOADED, "http_503"),  # "experiencing high demand": move on
        (errors.ServerError(504, {"error": {"code": 504, "status": "DEADLINE_EXCEEDED"}}),
         FailureKind.TRANSIENT, "http_504"),
        (errors.ServerError(500, {"error": {"code": 500}}), FailureKind.TRANSIENT, "http_500"),
        (errors.ClientError(408, {"error": {"code": 408}}), FailureKind.TRANSIENT, "http_408"),
        (errors.ClientError(400, {"error": {"code": 400}}), FailureKind.FAILED, "http_400"),
        (httpx2.ReadTimeout("slow"), FailureKind.TRANSIENT, "timeout"),
        (httpx2.ConnectError("down"), FailureKind.TRANSIENT, "ConnectError"),
        (RuntimeError("odd"), FailureKind.FAILED, "RuntimeError"),
        (response(prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.SAFETY)), FailureKind.BLOCKED,
         "prompt_blocked:SAFETY"),
        (response(candidates=[types.Candidate(finish_reason=types.FinishReason.SAFETY)]),
         FailureKind.BLOCKED, "finish:SAFETY"),
        (response(candidates=[]), FailureKind.BLOCKED, "empty_answer"),
        (response(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text="  ")]),
            finish_reason=types.FinishReason.STOP)]), FailureKind.BLOCKED, "empty_answer"),
    ],
)  # fmt: skip
def test_the_gemini_provider_classifies_failures(
    outcome: object, kind: FailureKind, error_type: str
) -> None:
    provider = GeminiProvider(settings(), Client(Models(outcome)))

    with pytest.raises(ProviderError) as raised:
        provider.complete(request())

    assert (raised.value.kind, raised.value.error_type) == (kind, error_type)


def test_a_blocked_answer_still_reports_its_usage() -> None:
    blocked_response = response(
        candidates=[types.Candidate(finish_reason=types.FinishReason.PROHIBITED_CONTENT)]
    )
    provider = GeminiProvider(settings(), Client(Models(blocked_response)))

    with pytest.raises(ProviderError) as raised:
        provider.complete(request())

    assert raised.value.usage.input == 120  # billed even though nothing was returned


def test_the_client_is_made_on_first_use_with_the_key_timeout_and_no_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[dict[str, Any]] = []

    def client(**arguments: Any) -> Client:
        made.append(arguments)
        return Client(Models(response()))

    monkeypatch.setattr(providers.genai, "Client", client)
    provider = GeminiProvider(
        settings(gemini_api_key=SecretStr("test-key"), request_timeout_seconds=12.5)
    )
    assert made == []  # nothing at start-up: AI may be off and the key unset

    provider.complete(request())
    provider.close()

    [arguments] = made
    assert arguments["api_key"] == "test-key"  # pragma: allowlist secret
    assert arguments["http_options"].timeout == 12_500  # milliseconds
    assert arguments["http_options"].retry_options.attempts == 1  # the gateway retries


async def test_the_async_gemini_provider_maps_answers_and_errors() -> None:
    client = Client(AsyncModels(response(), errors.ClientError(429, {"error": {"code": 429}})))
    provider = AsyncGeminiProvider(settings(), client)

    answer = await provider.complete(request())
    with pytest.raises(ProviderError) as raised:
        await provider.complete(request())
    await provider.aclose()

    assert answer.text == '{"answer": "ok"}'
    assert raised.value.kind is FailureKind.RATE_LIMITED
    assert client.closed


# ------------------------------------------------------------------------- AI switched on
def test_with_ai_on_a_key_is_required_at_startup() -> None:
    from medlens.core.config import SettingsError

    with pytest.raises(SettingsError, match="MEDLENS_LLM__ENABLED is true but no Gemini API key"):
        providers.require_key(LLMSettings(enabled=True), environ={})


@pytest.mark.parametrize(
    ("llm_settings", "environ"),
    [
        (LLMSettings(enabled=True, gemini_api_key=SecretStr("test-key")), {}),
        # The SDK's own variables:
        (LLMSettings(enabled=True), {"GEMINI_API_KEY": "test-key"}),  # pragma: allowlist secret
        (LLMSettings(enabled=True), {"GOOGLE_API_KEY": "test-key"}),  # pragma: allowlist secret
        (LLMSettings(enabled=False), {}),  # AI off: nothing is required
    ],
)
def test_a_key_in_settings_or_the_sdk_variables_is_enough(
    llm_settings: LLMSettings, environ: dict[str, str]
) -> None:
    providers.require_key(llm_settings, environ=environ)

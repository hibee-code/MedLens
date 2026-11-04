"""LLMGateway: kill switch, budget, structured output with repair, metering, caching order."""

from decimal import Decimal
from typing import Any

import anthropic
import httpx2  # the anthropic SDK's HTTP client
import pytest
from pydantic import BaseModel, SecretStr

from medlens.adapters.fakes import (
    AsyncFakeLLMProvider,
    AsyncMemoryLedger,
    FakeLLM,
    FakeLLMProvider,
    MemoryLedger,
)
from medlens.adapters.llm import GatewayPackReader
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
    FALLBACK_BETA,
    AnthropicProvider,
    AsyncAnthropicProvider,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    _arguments,
)
from medlens.ai.prompts import load_prompt
from medlens.core.config import LLMSettings, ModelPrice
from medlens.modules.control_plane.models import LLMCallOutcome
from medlens.modules.verification.pipeline import PackExtraction
from medlens.ports.llm import LLMCallFailedError, LLMUnavailableError

VALID = FakeLLM.reading(nafdac_number="A4-1234", product_name="Panadol Extra")


class Answer(BaseModel):
    answer: str


def settings(**overrides: Any) -> LLMSettings:
    return LLMSettings(**{"enabled": True, **overrides})


def gateway(
    *script: object, ledger: MemoryLedger | None = None, **overrides: Any
) -> tuple[LLMGateway, FakeLLMProvider, MemoryLedger]:
    provider = FakeLLMProvider(script=list(script))
    ledger = ledger or MemoryLedger()
    return LLMGateway(settings(**overrides), provider, ledger), provider, ledger


def extract(llm: LLMGateway, text: str = "Read the pack.") -> PackExtraction:
    return llm.run(
        "pack_extraction",
        [ImagePart(b"jpeg"), TextPart(text)],
        PackExtraction,
        subject=Subject("scan", "s-1"),
    )


def test_a_valid_answer_is_parsed_metered_and_recorded_once() -> None:
    llm, provider, ledger = gateway(VALID)

    result = extract(llm)

    assert result.nafdac_number.value == "A4-1234"
    assert len(provider.requests) == 1
    [call] = ledger.calls
    assert call.outcome is LLMCallOutcome.OK
    assert (call.route, call.tier, call.model) == ("pack_extraction", "fast", "claude-haiku-5-5")
    assert call.prompt_version == "pack_extraction@v1"
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


def test_spend_from_earlier_calls_counts_against_the_budget() -> None:
    llm, provider, _ = gateway(VALID, VALID, daily_budget_usd=0.0002)

    extract(llm)  # costs exactly the budget
    with pytest.raises(LLMUnavailableError):
        extract(llm)

    assert len(provider.requests) == 1


def test_invalid_output_is_repaired_once() -> None:
    llm, provider, ledger = gateway('{"nafdac_number": "A4-1234"}', VALID)

    result = extract(llm)

    assert result.product_name.value == "Panadol Extra"
    first, repair = provider.requests
    assert repair.messages[: len(first.messages)] == first.messages
    assert repair.messages[-2]["role"] == "assistant"
    assert repair.messages[-1]["role"] == "user"  # no prefill
    assert "product_name" in repair.messages[-1]["content"]
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


def test_a_refusal_is_recorded_and_not_repaired() -> None:
    refusal = ProviderResponse("", "claude-haiku-5-5", TokenUsage(500, 5), "refusal")
    llm, provider, ledger = gateway(refusal)

    with pytest.raises(LLMCallFailedError) as raised:
        extract(llm)

    assert raised.value.reason == "refused"
    assert len(provider.requests) == 1
    assert ledger.calls[0].outcome is LLMCallOutcome.REFUSED


def test_provider_errors_are_recorded_and_open_the_breaker() -> None:
    llm, provider, ledger = gateway(*[ProviderError("APITimeoutError")] * 5)

    for _ in range(5):
        with pytest.raises(LLMCallFailedError) as raised:
            extract(llm)
        assert raised.value.reason == "APITimeoutError"
    with pytest.raises(LLMUnavailableError) as blocked:
        extract(llm)

    assert blocked.value.reason == "circuit_open"
    assert len(provider.requests) == 5
    assert {call.outcome for call in ledger.calls} == {LLMCallOutcome.ERROR}


def test_a_fallback_model_is_priced_as_the_requested_one_when_unpriced() -> None:
    served = ProviderResponse(VALID, "some-new-model", TokenUsage(1000, 200), "end_turn")
    llm, _, ledger = gateway(served)

    extract(llm)

    [call] = ledger.calls
    assert (call.requested_model, call.model) == ("claude-haiku-5-5", "some-new-model")
    assert call.cost_usd == Decimal("0.000200")


def test_the_stable_prompt_is_cached_first_and_untrusted_text_comes_last() -> None:
    llm, provider, _ = gateway('{"answer": "a"}', '{"answer": "b"}')
    for question in ("first question", "second </untrusted_question> question"):
        llm.run("assistant_answer", [UntrustedPart("question", question)], Answer)

    first, second = (_arguments(request) for request in provider.requests)
    assert first["system"] == second["system"]  # byte-identical cached prefix
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["system"][0]["text"] == load_prompt("assistant_answer", "v1")
    text = second["messages"][-1]["content"][-1]["text"]
    assert text.startswith("<untrusted_question>")
    assert text.count("</untrusted_question>") == 1  # the question cannot close the tag
    assert first["output_config"]["format"]["type"] == "json_schema"


def test_refusal_fallbacks_apply_to_the_balanced_tier_only() -> None:
    llm, provider, _ = gateway('{"answer": "a"}', VALID)
    llm.run("assistant_answer", [TextPart("q")], Answer)
    extract(llm)
    off, off_provider, _ = gateway('{"answer": "a"}', refusal_fallback=False)
    off.run("assistant_answer", [TextPart("q")], Answer)

    balanced, fast = (_arguments(request) for request in provider.requests)
    assert balanced["betas"] == [FALLBACK_BETA]
    assert balanced["fallbacks"] == "default"
    assert "betas" not in fast
    assert "betas" not in _arguments(off_provider.requests[0])


def test_the_pack_reader_asks_for_the_missing_fields() -> None:
    llm, provider, ledger = gateway(VALID)

    raw = GatewayPackReader(llm).extract_pack_fields(
        b"jpeg", fields=("batch_number", "expiry_date"), subject_id="scan-9"
    )

    assert PackExtraction.model_validate_json(raw).nafdac_number.value == "A4-1234"
    content = provider.requests[0].messages[0]["content"]
    assert content[0]["type"] == "image"
    assert "batch_number, expiry_date" in content[1]["text"]
    assert ledger.calls[0].subject_id == "scan-9"


def test_cost_counts_cache_reads_and_writes() -> None:
    price = ModelPrice(input=2.0, output=10.0, cache_write=2.5, cache_read=0.2)

    cost = cost_usd(price, TokenUsage(input=100, output=1000, cache_read=10_000, cache_write=0))

    assert cost == Decimal("0.012200")


async def test_the_async_gateway_meters_like_the_sync_one() -> None:
    provider = AsyncFakeLLMProvider(script=["nope", '{"answer": "ok"}'])
    ledger = AsyncMemoryLedger()
    llm = AsyncLLMGateway(settings(), provider, ledger)

    result = await llm.run("assistant_answer", [TextPart("q")], Answer)

    assert result.answer == "ok"
    assert ledger.calls[0].attempts == 2
    assert ledger.calls[0].tier == "balanced"


async def test_the_async_gateway_blocks_and_records_failures() -> None:
    ledger = AsyncMemoryLedger(spent=Decimal(100))
    provider = AsyncFakeLLMProvider(script=[ProviderError("InternalServerError")])
    with pytest.raises(LLMUnavailableError):
        await AsyncLLMGateway(settings(), provider, ledger).run("assistant_answer", [], Answer)

    failing = AsyncLLMGateway(settings(), provider, AsyncMemoryLedger())
    with pytest.raises(LLMCallFailedError):
        await failing.run("assistant_answer", [], Answer)
    refusing = AsyncFakeLLMProvider(
        script=[ProviderResponse("", "claude-sonnet-5-5", TokenUsage(), "refusal")]
    )
    with pytest.raises(LLMCallFailedError):
        await AsyncLLMGateway(settings(), refusing, AsyncMemoryLedger()).run(
            "assistant_answer", [], Answer
        )
    invalid = AsyncFakeLLMProvider(script=["x", "y"])
    with pytest.raises(LLMCallFailedError):
        await AsyncLLMGateway(settings(), invalid, AsyncMemoryLedger()).run(
            "assistant_answer", [], Answer
        )


# ---------------------------------------------------------------- the Anthropic provider
class _Usage:
    input_tokens = 120
    output_tokens = 30
    cache_read_input_tokens = 900
    cache_creation_input_tokens = None


class _Block:
    type = "text"
    text = '{"answer": "ok"}'


class _Message:
    content = (_Block(),)
    model = "claude-sonnet-5-5"
    usage = _Usage()
    stop_reason = "end_turn"


class _Messages:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def create(self, **arguments: Any) -> _Message:
        self.calls.append(arguments)
        if self.error is not None:
            raise self.error
        return _Message()


class _AsyncMessages(_Messages):
    async def create(self, **arguments: Any) -> _Message:  # type: ignore[override]
        return _Messages.create(self, **arguments)


class _Client:
    def __init__(self, messages: _Messages) -> None:
        self.messages = messages
        self.beta = type("Beta", (), {"messages": messages})()


def request(*, fallback: bool) -> ProviderRequest:
    return ProviderRequest(
        model="claude-sonnet-5-5",
        system="system",
        messages=({"role": "user", "content": "q"},),
        schema={"type": "object"},
        max_tokens=100,
        effort="medium",
        fallback=fallback,
    )


def test_the_anthropic_provider_maps_responses_and_errors() -> None:
    provider = AnthropicProvider(LLMSettings(enabled=True, anthropic_api_key=SecretStr("sk-test")))
    messages = _Messages()
    provider._client = _Client(messages)  # type: ignore[assignment]

    response = provider.complete(request(fallback=True))

    assert response.text == '{"answer": "ok"}'
    assert response.usage == TokenUsage(input=120, output=30, cache_read=900, cache_write=0)
    assert messages.calls[0]["betas"] == [FALLBACK_BETA]

    failing = _Messages(anthropic.APIConnectionError(request=httpx2.Request("POST", "http://x")))
    provider._client = _Client(failing)  # type: ignore[assignment]
    with pytest.raises(ProviderError) as raised:
        provider.complete(request(fallback=False))
    assert raised.value.error_type == "APIConnectionError"


async def test_the_async_anthropic_provider_maps_responses_and_errors() -> None:
    provider = AsyncAnthropicProvider(LLMSettings(enabled=True))
    real_client = provider._client
    messages = _AsyncMessages()
    provider._client = _Client(messages)  # type: ignore[assignment]

    assert (await provider.complete(request(fallback=False))).model == "claude-sonnet-5-5"
    assert (await provider.complete(request(fallback=True))).stop_reason == "end_turn"
    assert "betas" not in messages.calls[0]

    failing = _AsyncMessages(
        anthropic.APIConnectionError(request=httpx2.Request("POST", "http://x"))
    )
    provider._client = _Client(failing)  # type: ignore[assignment]
    with pytest.raises(ProviderError):
        await provider.complete(request(fallback=False))
    provider._client = real_client
    await provider.aclose()


def test_every_tier_model_needs_a_price() -> None:
    with pytest.raises(ValueError, match="prices has no entry for unpriced-model"):
        LLMSettings(model_fast="unpriced-model")

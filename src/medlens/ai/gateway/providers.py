"""Model providers behind the gateway. Only this module imports the ``anthropic`` SDK
(CLAUDE.md §3). The SDK retries transient failures itself (connection errors, 408, 409, 429,
5xx, honouring Retry-After) up to MEDLENS_LLM__MAX_RETRIES within the request timeout; whatever
still fails surfaces as ``ProviderError`` and counts against the gateway's circuit breaker."""

import base64
from dataclasses import dataclass, replace
from typing import Any, Protocol

import anthropic

from medlens.ai.gateway.pricing import TokenUsage
from medlens.core.config import LLMSettings

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    model: str
    system: str
    messages: tuple[dict[str, Any], ...]
    schema: dict[str, Any]
    max_tokens: int
    effort: str
    fallback: bool = False

    def with_repair(self, previous: str, problems: str) -> "ProviderRequest":
        """The conversation so far plus a request to fix the output (no prefill: the last
        message stays a user turn)."""
        return replace(
            self,
            messages=(
                *self.messages,
                {"role": "assistant", "content": previous or "(no output)"},
                {
                    "role": "user",
                    "content": (
                        f"That output did not match the schema: {problems}\n"
                        "Answer again with the corrected JSON object only."
                    ),
                },
            ),
        )


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    text: str
    model: str
    usage: TokenUsage
    stop_reason: str | None


class ProviderError(Exception):
    def __init__(self, error_type: str) -> None:
        super().__init__(error_type)
        self.error_type = error_type


class SyncProvider(Protocol):
    def complete(self, request: ProviderRequest) -> ProviderResponse: ...


class AsyncProvider(Protocol):
    async def complete(self, request: ProviderRequest) -> ProviderResponse: ...


def image_block(data: bytes, media_type: str) -> dict[str, Any]:
    encoded = base64.standard_b64encode(data).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": encoded},
    }


def _arguments(request: ProviderRequest) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        # The stable instructions come first and are cached; the variable content follows.
        "system": [
            {"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": list(request.messages),
        "output_config": {
            "effort": request.effort,
            "format": {"type": "json_schema", "schema": request.schema},
        },
    }
    if request.fallback:
        arguments |= {"betas": [FALLBACK_BETA], "fallbacks": "default"}
    return arguments


def _response(message: Any) -> ProviderResponse:
    usage = message.usage
    return ProviderResponse(
        text="".join(block.text for block in message.content if block.type == "text"),
        model=str(message.model),
        usage=TokenUsage(
            input=int(usage.input_tokens or 0),
            output=int(usage.output_tokens or 0),
            cache_read=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        ),
        stop_reason=message.stop_reason,
    )


def _client_options(settings: LLMSettings) -> dict[str, Any]:
    key = settings.anthropic_api_key
    options: dict[str, Any] = {
        "timeout": settings.request_timeout_seconds,
        "max_retries": settings.max_retries,
    }
    if key is not None:  # otherwise the SDK's own credential lookup (env, profile)
        options["api_key"] = key.get_secret_value()
    return options


class AnthropicProvider:
    def __init__(self, settings: LLMSettings) -> None:
        self._client = anthropic.Anthropic(**_client_options(settings))

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        arguments = _arguments(request)
        try:
            if request.fallback:
                return _response(self._client.beta.messages.create(**arguments))
            return _response(self._client.messages.create(**arguments))
        except anthropic.APIError as exc:
            raise ProviderError(type(exc).__name__) from None


class AsyncAnthropicProvider:
    def __init__(self, settings: LLMSettings) -> None:
        self._client = anthropic.AsyncAnthropic(**_client_options(settings))

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        arguments = _arguments(request)
        try:
            if request.fallback:
                return _response(await self._client.beta.messages.create(**arguments))
            return _response(await self._client.messages.create(**arguments))
        except anthropic.APIError as exc:
            raise ProviderError(type(exc).__name__) from None

    async def aclose(self) -> None:
        await self._client.close()

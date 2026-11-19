"""The model provider behind the gateway: Google's Gemini API through the official Gen AI SDK
(``google-genai``). Only this module imports it (CLAUDE.md §3; decision log, "Switched LLM
provider from Anthropic to Gemini").

The SDK's own retries are off (``HttpRetryOptions(attempts=1)``): the gateway decides, per
failure, whether to retry the same model or move to the next one in the tier's fallback chain.
Every failure therefore surfaces as a ``ProviderError`` with one of four kinds:

    transient     timeouts, connection errors, HTTP 408, 500, 502, 504: retry the same model
    rate_limited  HTTP 429 (quota exhausted): the next model at once
    overloaded    HTTP 503 ("experiencing high demand"): the next model at once, since
                  retrying an overloaded model only adds waiting
    blocked       a blocked prompt, a safety-type finish reason or an empty answer: the next
                  model at once, no retry
    failed        anything else (bad request, authentication, unknown model): the next model

The client is created on first use, so the API and workers start without a key while AI is
switched off (MEDLENS_LLM__ENABLED=false). With AI on, ``require_key`` stops the API and the
workers at startup when no key is set, rather than letting every call fail.

Thinking tokens count against Gemini's ``max_output_tokens``: each request asks for the route's
answer limit plus a headroom for the thinking level (THINKING_HEADROOM), so thinking cannot
crowd out the JSON answer (seen live: high thinking used 246 of 256 tokens and cut the answer).
"""

import base64
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

import httpx2
import structlog
from google import genai
from google.genai import errors, types

from medlens.ai.gateway.pricing import TokenUsage
from medlens.core.config import LLMSettings, SettingsError

log = structlog.stdlib.get_logger(__name__)

# Route effort (registry.Route.effort) to Gemini's thinking level.
THINKING_LEVELS = {
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}
# Output tokens added to a route's answer limit for the model's thinking, by thinking level.
THINKING_HEADROOM = {"low": 1024, "medium": 4096, "high": 8192}
# Finish reasons that mean the model's output was withheld or refused.
BLOCKING_FINISH_REASONS = frozenset(
    {
        types.FinishReason.SAFETY,
        types.FinishReason.RECITATION,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.SPII,
        types.FinishReason.IMAGE_SAFETY,
        types.FinishReason.IMAGE_PROHIBITED_CONTENT,
        types.FinishReason.MODEL_ARMOR,
    }
)


class FailureKind(StrEnum):
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    OVERLOADED = "overloaded"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """A provider-neutral request. ``messages`` are turns ``{"role": "user" | "model",
    "parts": [{"text": ...} | {"image": base64, "mime_type": ...}]}``."""

    model: str
    system: str
    messages: tuple[dict[str, Any], ...]
    schema: dict[str, Any]
    max_tokens: int
    effort: str

    def with_repair(self, previous: str, problems: str) -> "ProviderRequest":
        """The conversation so far plus a request to fix the output."""
        return replace(
            self,
            messages=(
                *self.messages,
                {"role": "model", "parts": [{"text": previous or "(no output)"}]},
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"That output did not match the schema: {problems}\n"
                                "Answer again with the corrected JSON object only."
                            )
                        }
                    ],
                },
            ),
        )

    def for_model(self, model: str) -> "ProviderRequest":
        return replace(self, model=model)


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    text: str
    model: str
    usage: TokenUsage
    finish_reason: str | None = None


class ProviderError(Exception):
    """A failed provider call. ``usage`` is what the call consumed anyway (a blocked answer is
    still billed)."""

    def __init__(self, error_type: str, kind: FailureKind, usage: TokenUsage | None = None) -> None:
        super().__init__(error_type)
        self.error_type = error_type
        self.kind = kind
        self.usage = usage or TokenUsage()


class SyncProvider(Protocol):
    def complete(self, request: ProviderRequest) -> ProviderResponse: ...


class AsyncProvider(Protocol):
    async def complete(self, request: ProviderRequest) -> ProviderResponse: ...


def image_part(data: bytes, media_type: str) -> dict[str, Any]:
    return {"image": base64.standard_b64encode(data).decode("ascii"), "mime_type": media_type}


def _part(part: dict[str, Any]) -> types.Part:
    if "image" in part:
        return types.Part.from_bytes(
            data=base64.standard_b64decode(part["image"]), mime_type=part["mime_type"]
        )
    return types.Part.from_text(text=part["text"])


def _contents(request: ProviderRequest) -> list[types.Content]:
    return [
        types.Content(role=turn["role"], parts=[_part(part) for part in turn["parts"]])
        for turn in request.messages
    ]


def _config(request: ProviderRequest) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        # The stable instructions come first (Gemini caches repeated prefixes implicitly);
        # the variable content follows in the user turn.
        system_instruction=request.system,
        # The answer's limit plus room to think: thinking tokens count against this limit.
        max_output_tokens=request.max_tokens + THINKING_HEADROOM.get(request.effort, 1024),
        response_mime_type="application/json",
        response_json_schema=request.schema,
        thinking_config=types.ThinkingConfig(
            thinking_level=THINKING_LEVELS.get(request.effort, types.ThinkingLevel.LOW)
        ),
        # One response per call: the gateway's token limits and metering assume it.
        automatic_continuation=False,
    )


def _usage(response: types.GenerateContentResponse) -> TokenUsage:
    meta = response.usage_metadata
    if meta is None:
        return TokenUsage()
    cached = int(meta.cached_content_token_count or 0)
    return TokenUsage(
        input=max(0, int(meta.prompt_token_count or 0) - cached),
        # Thinking tokens are billed as output.
        output=int(meta.candidates_token_count or 0) + int(meta.thoughts_token_count or 0),
        cache_read=cached,
        cache_write=0,  # implicit caching: no separate write charge
    )


def _response(
    request: ProviderRequest, response: types.GenerateContentResponse
) -> ProviderResponse:
    usage = _usage(response)
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason is not None:
        raise ProviderError(
            f"prompt_blocked:{feedback.block_reason.name}", FailureKind.BLOCKED, usage
        )
    candidate = response.candidates[0] if response.candidates else None
    finish = candidate.finish_reason if candidate is not None else None
    if finish is not None and finish in BLOCKING_FINISH_REASONS:
        raise ProviderError(f"finish:{finish.name}", FailureKind.BLOCKED, usage)
    text = "".join(
        part.text
        for part in (candidate.content.parts or [] if candidate and candidate.content else [])
        if part.text and not part.thought
    )
    if not text.strip():
        raise ProviderError("empty_answer", FailureKind.BLOCKED, usage)
    return ProviderResponse(text, request.model, usage, finish.name if finish else None)


def _failure(exc: Exception) -> ProviderError:
    if isinstance(exc, errors.APIError):
        code = int(exc.code or 0)
        if code == 429:
            return ProviderError("http_429", FailureKind.RATE_LIMITED)
        if code == 503:  # "This model is currently experiencing high demand"
            return ProviderError("http_503", FailureKind.OVERLOADED)
        if code == 408 or code >= 500:
            return ProviderError(f"http_{code}", FailureKind.TRANSIENT)
        return ProviderError(f"http_{code}", FailureKind.FAILED)
    if isinstance(exc, httpx2.TimeoutException):
        return ProviderError("timeout", FailureKind.TRANSIENT)
    if isinstance(exc, httpx2.TransportError):
        return ProviderError(type(exc).__name__, FailureKind.TRANSIENT)
    return ProviderError(type(exc).__name__, FailureKind.FAILED)


def require_key(settings: LLMSettings, environ: Mapping[str, str] = os.environ) -> None:
    """With AI on (MEDLENS_LLM__ENABLED=true), a Gemini key must be set, either as
    MEDLENS_LLM__GEMINI_API_KEY or as the SDK's own GEMINI_API_KEY / GOOGLE_API_KEY: called by the
    API and the workers at startup, so a missing key stops them with a clear message instead of
    failing every call. With AI off, nothing is required."""
    if not settings.enabled:
        log.info("llm.disabled")
        return
    key = settings.gemini_api_key
    if (key is None or not key.get_secret_value().strip()) and not (
        environ.get("GEMINI_API_KEY") or environ.get("GOOGLE_API_KEY")
    ):
        raise SettingsError(
            "MedLens configuration is invalid:\n"
            "  - MEDLENS_LLM__ENABLED is true but no Gemini API key is set: set "
            "MEDLENS_LLM__GEMINI_API_KEY (Google AI Studio), or switch AI off with "
            "MEDLENS_LLM__ENABLED=false."
        )
    log.info(
        "llm.enabled",
        fast=settings.chain("fast"),
        balanced=settings.chain("balanced"),
        max_retries=settings.max_retries,
        daily_budget_usd=settings.daily_budget_usd,
    )


def _client(settings: LLMSettings) -> genai.Client:
    key = settings.gemini_api_key
    return genai.Client(
        # Without a key here, the SDK reads GEMINI_API_KEY / GOOGLE_API_KEY itself.
        api_key=key.get_secret_value() if key is not None else None,
        http_options=types.HttpOptions(
            timeout=int(settings.request_timeout_seconds * 1000),  # milliseconds
            retry_options=types.HttpRetryOptions(attempts=1),  # the gateway retries
        ),
    )


@dataclass
class GeminiProvider:
    settings: LLMSettings
    client: Any = field(default=None, repr=False)  # tests pass a mocked client

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        if self.client is None:
            self.client = _client(self.settings)
        try:
            response = self.client.models.generate_content(
                model=request.model, contents=_contents(request), config=_config(request)
            )
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - every SDK or transport failure is classified
            raise _failure(exc) from None
        return _response(request, response)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


@dataclass
class AsyncGeminiProvider:
    settings: LLMSettings
    client: Any = field(default=None, repr=False)  # tests pass a mocked client

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        if self.client is None:
            self.client = _client(self.settings)
        try:
            response = await self.client.aio.models.generate_content(
                model=request.model, contents=_contents(request), config=_config(request)
            )
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - every SDK or transport failure is classified
            raise _failure(exc) from None
        return _response(request, response)

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aio.aclose()

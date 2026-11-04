from collections.abc import Sequence
from typing import Protocol


class LLMUnavailableError(Exception):
    """No model call was made: the AI kill switch is off, today's budget is spent, or the
    provider's circuit breaker is open. Callers fall back to their non-AI behaviour."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class LLMCallFailedError(Exception):
    """A model call was made (and metered) but produced nothing usable: a provider error,
    a refusal, or output that stayed invalid after one repair attempt."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class LLMPort(Protocol):
    """Vision extraction for hard images, implemented by the LLM gateway (adapters/llm.py),
    never by a provider SDK directly. Returns the extraction as JSON text: the pipeline
    validates it again and normalises every value like user input."""

    def extract_pack_fields(
        self, image: bytes, *, fields: Sequence[str], subject_id: str | None = None
    ) -> str: ...

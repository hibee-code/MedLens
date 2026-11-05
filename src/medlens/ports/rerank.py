from collections.abc import Sequence
from typing import Protocol


class RerankPort(Protocol):
    """A cross-encoder in the worker: relevance in 0..1 of each passage to the query."""

    @property
    def model_version(self) -> str: ...

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]: ...


class RerankingPort(Protocol):
    """The API's view: reranking done elsewhere (a worker) within a time limit. None means
    unavailable; the fused order then stands."""

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float] | None: ...

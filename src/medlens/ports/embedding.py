from collections.abc import Sequence
from typing import Protocol


class EmbeddingPort(Protocol):
    """Image embeddings for visual matching. The model and its dimension are configuration."""

    @property
    def model_version(self) -> str: ...

    def embed_image(self, image: bytes) -> list[float]: ...


class TextEmbeddingPort(Protocol):
    """Text embeddings for document search. Passages and queries may be encoded differently
    (e5 models need a "passage: " / "query: " prefix)."""

    @property
    def model_version(self) -> str: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class QueryEmbeddingPort(Protocol):
    """Question embeddings for the API, which has no model of its own: computed elsewhere (the
    worker) within a time limit. None means unavailable; search then runs without vectors."""

    async def embed_query(self, text: str) -> list[float] | None: ...

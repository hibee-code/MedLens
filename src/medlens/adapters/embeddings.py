"""Embeddings with sentence-transformers (the `ml` dependency group, worker only).

Models load once per worker process (CLAUDE.md §7) and the first load downloads them from the
Hugging Face hub into its cache. ``load`` takes the model name; tests pass a stand-in model.
"""

import io
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from PIL import Image


class _Model(Protocol):
    def encode(self, inputs: Any, **kwargs: Any) -> Any: ...


def load_sentence_transformer(name: str) -> _Model:
    from sentence_transformers import SentenceTransformer  # heavy: imported on use

    model: _Model = SentenceTransformer(name, device="cpu")
    return model


def _rows(vectors: Any) -> list[list[float]]:
    return [[float(value) for value in vector] for vector in vectors]


class SentenceTransformerImageEmbedder:
    """clip-ViT-B-32 by default: unit-length vectors, compared by cosine similarity."""

    def __init__(
        self,
        model_name: str,
        dimension: int,
        load: Callable[[str], _Model] = load_sentence_transformer,
    ) -> None:
        self._model = load(model_name)
        self._dimension = dimension
        self.model_version = model_name

    def embed_image(self, image: bytes) -> list[float]:
        with Image.open(io.BytesIO(image)) as opened:
            picture = opened.convert("RGB")
        (vector,) = _rows(self._model.encode([picture], normalize_embeddings=True))
        if len(vector) != self._dimension:
            raise ValueError(f"{self.model_version} returned {len(vector)} dimensions")
        return vector


class SentenceTransformerTextEmbedder:
    """intfloat/multilingual-e5-small by default, which needs the e5 prefixes."""

    def __init__(
        self,
        model_name: str,
        dimension: int,
        load: Callable[[str], _Model] = load_sentence_transformer,
    ) -> None:
        self._model = load(model_name)
        self._dimension = dimension
        self.model_version = model_name

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}"])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = _rows(self._model.encode(texts, normalize_embeddings=True, batch_size=32))
        if any(len(vector) != self._dimension for vector in vectors):
            raise ValueError(f"{self.model_version} returned the wrong dimension")
        return vectors

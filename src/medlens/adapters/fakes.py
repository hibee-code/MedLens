"""Fake adapters for every pipeline port, for tests and for running the pipeline without the ml
dependency group. Each is scripted per image (keyed by the SHA-256 of its bytes) with a default,
so an end-to-end test decides what each photo "contains"."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal

from medlens.ai.gateway.ledger import CallRecord
from medlens.ai.gateway.pricing import TokenUsage
from medlens.ai.gateway.providers import ProviderRequest, ProviderResponse
from medlens.ports.barcode import BarcodeRead
from medlens.ports.image_analysis import ImageAssessment
from medlens.ports.llm import LLMUnavailableError
from medlens.ports.ocr import OcrResult
from medlens.ports.storage import ObjectNotFoundError

FAKE_USAGE = TokenUsage(input=1000, output=200)
GOOD_PHOTO = ImageAssessment(width=1200, height=1600, blur=250.0, brightness=128.0, phash=0)
NO_TEXT = OcrResult(text="", confidence=0.0)


def image_key(image: bytes) -> str:
    return hashlib.sha256(image).hexdigest()


@dataclass
class FakeImageAnalysis:
    """Scripted assessments; otherwise a good photo whose perceptual hash comes from its bytes,
    so different photos never look like near-duplicates."""

    by_image: dict[str, ImageAssessment] = field(default_factory=dict)
    default: ImageAssessment | None = None

    def assess(self, image: bytes) -> ImageAssessment:
        found = self.by_image.get(image_key(image), self.default)
        if found is not None:
            return found
        phash = int.from_bytes(hashlib.sha256(image).digest()[:8], "big", signed=True)
        return replace(GOOD_PHOTO, phash=phash)


@dataclass
class FakeBarcode:
    by_image: dict[str, list[BarcodeRead]] = field(default_factory=dict)
    default: list[BarcodeRead] = field(default_factory=list)

    def decode(self, image: bytes) -> list[BarcodeRead]:
        return list(self.by_image.get(image_key(image), self.default))


@dataclass
class FakeOcr:
    by_image: dict[str, OcrResult] = field(default_factory=dict)
    default: OcrResult = NO_TEXT

    def read(self, image: bytes) -> OcrResult:
        return self.by_image.get(image_key(image), self.default)


@dataclass
class FakeLLM:
    """Pack reading (LLMPort): scripted raw model output (valid JSON or not) per image, or
    unavailable when unscripted."""

    by_image: dict[str, str] = field(default_factory=dict)
    default: str | None = None
    calls: list[Sequence[str]] = field(default_factory=list)
    subjects: list[str | None] = field(default_factory=list)

    def extract_pack_fields(
        self, image: bytes, *, fields: Sequence[str], subject_id: str | None = None
    ) -> str:
        self.calls.append(tuple(fields))
        self.subjects.append(subject_id)
        output = self.by_image.get(image_key(image), self.default)
        if output is None:
            raise LLMUnavailableError("no scripted output")
        return output

    @staticmethod
    def reading(confidence: float = 0.9, **fields: str | None) -> str:
        """PackExtraction JSON: each of the four fields as printed, or not visible."""
        names = ("nafdac_number", "product_name", "batch_number", "expiry_date")
        return json.dumps(
            {
                name: (
                    {"value": fields[name], "confidence": confidence}
                    if fields.get(name)
                    else {"value": "not_visible", "confidence": 0.0}
                )
                for name in names
            }
        )


@dataclass
class FakeLLMProvider:
    """Gateway provider (sync and async): scripted responses in order. Each script entry is the
    output text, a ``ProviderResponse``, or an exception to raise."""

    script: list[object] = field(default_factory=list)
    model: str = "claude-haiku-5-5"
    usage: TokenUsage = FAKE_USAGE
    requests: list[ProviderRequest] = field(default_factory=list)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        entry = self.script.pop(0) if self.script else "{}"
        if isinstance(entry, BaseException):
            raise entry
        if isinstance(entry, ProviderResponse):
            return entry
        return ProviderResponse(str(entry), request.model, self.usage, "end_turn")


@dataclass
class AsyncFakeLLMProvider(FakeLLMProvider):
    async def complete(self, request: ProviderRequest) -> ProviderResponse:  # type: ignore[override]
        return FakeLLMProvider.complete(self, request)


@dataclass
class MemoryLedger:
    """Gateway ledger (sync and async) in memory: ``spent`` is today's spend before any call."""

    spent: Decimal = Decimal(0)
    calls: list[CallRecord] = field(default_factory=list)

    def spent_today(self) -> Decimal:
        return self.spent + sum((call.cost_usd for call in self.calls), Decimal(0))

    def record(self, call: CallRecord) -> None:
        self.calls.append(call)


@dataclass
class AsyncMemoryLedger(MemoryLedger):
    async def spent_today(self) -> Decimal:  # type: ignore[override]
        return MemoryLedger.spent_today(self)

    async def record(self, call: CallRecord) -> None:  # type: ignore[override]
        MemoryLedger.record(self, call)


@dataclass
class FakeQueryEmbedder:
    """QueryEmbeddingPort: the fake text model in-process, or unavailable."""

    model: "FakeTextEmbedder | None" = None
    calls: list[str] = field(default_factory=list)

    async def embed_query(self, text: str) -> list[float] | None:
        self.calls.append(text)
        return self.model.embed_query(text) if self.model else None


@dataclass
class FakeReranker:
    """RerankPort and RerankingPort: the share of the query's words found in each passage."""

    model_version: str = "fake-reranker-1"
    calls: list[tuple[str, int]] = field(default_factory=list)
    available: bool = True

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, len(passages)))
        words = set(query.casefold().split())
        return [
            round(len(words & set(passage.casefold().split())) / max(len(words), 1), 6)
            for passage in passages
        ]


@dataclass
class AsyncFakeReranker(FakeReranker):
    async def rerank(self, query: str, passages: Sequence[str]) -> list[float] | None:  # type: ignore[override]
        if not self.available:
            return None
        return FakeReranker.rerank(self, query, passages)


@dataclass
class FakeTextEmbedder:
    """Deterministic text vectors: texts sharing words get similar vectors."""

    dimension: int = 384
    model_version: str = "fake-text-embedder-1"

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def probe_dimension(self) -> int:
        return self.dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in text.casefold().split():
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


@dataclass
class FakeEmbedder:
    by_image: dict[str, list[float]] = field(default_factory=dict)
    dimension: int = 4
    model_version: str = "fake-embedder-1"

    def embed_image(self, image: bytes) -> list[float]:
        found = self.by_image.get(image_key(image))
        if found is not None:
            return found
        return [1.0] + [0.0] * (self.dimension - 1)

    def probe_dimension(self) -> int:
        return self.dimension


@dataclass
class InMemoryStorage:
    """Both the async (API) and sync (worker) storage ports."""

    objects: dict[str, tuple[bytes, str]] = field(default_factory=dict)

    async def put(self, key: str, data: bytes, *, content_type: str) -> None:
        self.objects[key] = (data, content_type)

    async def get(self, key: str) -> bytes:
        return self.get_sync(key)

    async def delete(self, key: str) -> None:
        self.objects.pop(key, None)

    def get_sync(self, key: str) -> bytes:
        try:
            return self.objects[key][0]
        except KeyError:
            raise ObjectNotFoundError(key) from None

    def put_sync(self, key: str, data: bytes, *, content_type: str) -> None:
        self.objects[key] = (data, content_type)


class SyncView:
    """Adapts InMemoryStorage to SyncStoragePort (same objects)."""

    def __init__(self, storage: InMemoryStorage) -> None:
        self._storage = storage

    def get(self, key: str) -> bytes:
        return self._storage.get_sync(key)

    def put(self, key: str, data: bytes, *, content_type: str) -> None:
        self._storage.put_sync(key, data, content_type=content_type)


def scripted(entries: Mapping[bytes, object]) -> dict[str, object]:
    """{image bytes: value} -> {image key: value}, for building the fakes above."""
    return {image_key(image): value for image, value in entries.items()}

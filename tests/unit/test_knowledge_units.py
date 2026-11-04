"""Knowledge, RAG and hardening building blocks without a database: chunking, rank fusion,
citation grounding, document validation, the query-embedding adapter, cache-aside, worker
metrics, and response schemas that must never expose secrets."""

import io
import json
from datetime import date
from typing import Any
from uuid import uuid4

import pytest
from fastapi import UploadFile
from redis.exceptions import ConnectionError as RedisConnectionError
from starlette.datastructures import Headers

from medlens.adapters.query_embedding import CeleryQueryEmbedder
from medlens.cache.aside import cached_json
from medlens.core.config import Settings
from medlens.core.errors import PayloadTooLarge, UnsupportedMediaType, ValidationFailed
from medlens.core.uploads import read_document_upload
from medlens.main import create_app
from medlens.modules.knowledge.models import DocumentKind
from medlens.modules.knowledge.repository import ChunkDetail, ChunkHit
from medlens.modules.knowledge.service import SearchHit, fuse
from medlens.modules.knowledge.text import (
    Page,
    UnreadableDocument,
    chunk,
    count_tokens,
    extract_pages,
    normalise,
)
from medlens.modules.rag.service import Citation, ModelAnswer, check_grounding
from medlens.modules.verification.extraction import find_batch_numbers, find_nafdac_numbers
from medlens.workers import metrics as worker_metrics
from tests.fixtures.documents import simple_pdf

# ------------------------------------------------------------------------- text
NOTICE = """# Public Alert 041/2026

Counterfeit Coartem 20/120 in circulation

NAFDAC alerts the public. NAFDAC Reg. No: A4-1234, Batch No: F2034.

## Advice

Report suspicious packs to the nearest office."""


def test_normalise_collapses_space_and_control_characters() -> None:
    assert normalise("  a\x00b \t c\n\n\n\nd ") == "ab c\n\nd"


def test_chunks_keep_their_heading_and_stay_within_the_limit() -> None:
    long = NOTICE + "\n\n" + "The tablets contain no artemether. " * 200
    chunks = chunk([Page(1, normalise(long))], max_tokens=120, overlap=20)

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.token_count <= 120 for c in chunks)
    assert chunks[0].section == "Public Alert 041/2026"
    assert any(c.section == "Advice" for c in chunks)
    assert all(c.page == 1 for c in chunks)
    # Overlap: each long-paragraph chunk starts with the end of the previous one.
    assert chunks[3].text.split()[0] in chunks[2].text


def test_a_single_huge_word_still_becomes_a_chunk() -> None:
    chunks = chunk([Page(None, "x" * 5000)], max_tokens=50, overlap=5)

    assert len(chunks) == 1
    assert count_tokens(chunks[0].text) == 1


def test_text_files_must_be_utf8_and_pdfs_need_text() -> None:
    with pytest.raises(UnreadableDocument):
        extract_pages(b"\xff\xfe\x00bad", "text/plain")
    with pytest.raises(UnreadableDocument):
        extract_pages(b"%PDF-1.4 not really", "application/pdf")
    with pytest.raises(UnreadableDocument):
        extract_pages(simple_pdf([[""]]), "application/pdf")
    assert extract_pages(simple_pdf([["Hello"], ["World"]]), "application/pdf") == [
        Page(1, "Hello"),
        Page(2, "World"),
    ]


def test_documents_need_labels_questions_do_not() -> None:
    assert find_nafdac_numbers(NOTICE) == ["A4-1234"]
    assert find_nafdac_numbers("any alert for 04-8655?") == []
    assert find_nafdac_numbers("any alert for 04-8655?", bare=True) == ["04-8655"]
    assert find_batch_numbers("check the batch the holder made; Batch No: F2034") == ["F2034"]


# ------------------------------------------------------------------------- fusion and grounding
def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    a, b, c = uuid4(), uuid4(), uuid4()

    fused = fuse(
        {
            "keyword": [ChunkHit(a, 0.9), ChunkHit(b, 0.5)],
            "vector": [ChunkHit(b, 0.95), ChunkHit(c, 0.9)],
        },
        k=60,
    )

    assert [chunk_id for chunk_id, _ in fused] == [b, a, c]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def hit(text: str) -> SearchHit:
    detail = ChunkDetail(
        uuid4(), uuid4(), "Alert", DocumentKind.ALERT, date(2026, 10, 1), None, 1, None, text
    )
    return SearchHit(detail, 0.03, ("keyword",), None)


def test_grounding_checks_ids_quotes_and_missing_citations() -> None:
    source = hit("Batch No: F2034 was found in Kano.")
    hits = {str(source.chunk.chunk_id): source}
    chunk_id = str(source.chunk.chunk_id)

    def answer(*citations: tuple[str, str], insufficient: bool = False) -> ModelAnswer:
        return ModelAnswer(
            answer="x",
            citations=[Citation(chunk_id=i, quote=q) for i, q in citations],
            insufficient_evidence=insufficient,
        )

    assert check_grounding(answer((chunk_id, "batch no: f2034")), hits) == []
    assert check_grounding(answer(insufficient=True), hits) == []
    assert check_grounding(answer(), hits) == ["the answer cites no document"]
    assert "not one of the documents" in check_grounding(answer(("other", "x")), hits)[0]
    assert "not in that document" in check_grounding(answer((chunk_id, "Lagos")), hits)[0]


# ------------------------------------------------------------------------- document uploads
def upload_file(data: bytes, content_type: str = "application/octet-stream") -> UploadFile:
    return UploadFile(
        io.BytesIO(data), filename="doc", headers=Headers({"content-type": content_type})
    )


async def test_document_uploads_are_checked_by_content(settings: Settings) -> None:
    pdf = await read_document_upload(upload_file(simple_pdf([["Notice"]])), settings)
    text = await read_document_upload(upload_file(b"# Note\n\nText"), settings)

    assert pdf.media_type == "application/pdf"
    assert text.media_type == "text/plain"
    with pytest.raises(UnsupportedMediaType):
        await read_document_upload(upload_file(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64), settings)
    with pytest.raises(ValidationFailed):
        await read_document_upload(upload_file(b""), settings)
    with pytest.raises(ValidationFailed):
        await read_document_upload(upload_file(simple_pdf([["/Launch evil"]])), settings)
    with pytest.raises(ValidationFailed):
        await read_document_upload(upload_file("café".encode("latin-1") * 40), settings)


async def test_oversized_documents_are_refused(settings_factory: Any) -> None:
    small = settings_factory(knowledge={"max_document_bytes": 100})

    with pytest.raises(PayloadTooLarge):
        await read_document_upload(upload_file(b"a" * 200), small)


# ------------------------------------------------------------------------- query embeddings
class FakeResult:
    def __init__(self, value: object = None, error: Exception | None = None) -> None:
        self.value, self.error = value, error

    def get(self, timeout: float) -> object:
        if self.error:
            raise self.error
        return self.value


class FakeCelery:
    def __init__(self, result: FakeResult) -> None:
        self.result = result
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def send_task(self, name: str, **options: Any) -> FakeResult:
        self.sent.append((name, options))
        return self.result


async def test_query_embeddings_come_from_a_worker_or_not_at_all() -> None:
    worker = FakeCelery(FakeResult([1, 0.5]))
    embedder = CeleryQueryEmbedder(worker, timeout_seconds=2)  # type: ignore[arg-type]

    assert await embedder.embed_query("q") == [1.0, 0.5]
    name, options = worker.sent[0]
    assert name == "medlens.knowledge.embed_query"
    assert options["expires"] == 2
    assert options["priority"] == 0
    timeout = CeleryQueryEmbedder(FakeCelery(FakeResult(error=TimeoutError())), timeout_seconds=1)  # type: ignore[arg-type]
    assert await timeout.embed_query("q") is None
    no_model = CeleryQueryEmbedder(FakeCelery(FakeResult(None)), timeout_seconds=1)  # type: ignore[arg-type]
    assert await no_model.embed_query("q") is None


# ------------------------------------------------------------------------- cache-aside
class DictRedis:
    def __init__(self, broken: bool = False) -> None:
        self.data: dict[str, str] = {}
        self.broken = broken

    async def get(self, key: str) -> str | None:
        if self.broken:
            raise RedisConnectionError
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int) -> None:
        if self.broken:
            raise RedisConnectionError
        self.data[key] = value


async def test_cache_aside_computes_once_and_survives_redis_outages() -> None:
    calls = []

    async def produce() -> dict[str, int]:
        calls.append(1)
        return {"n": len(calls)}

    redis = DictRedis()
    first = await cached_json(redis, "k", 60, produce)  # type: ignore[arg-type]
    second = await cached_json(redis, "k", 60, produce)  # type: ignore[arg-type]
    down = await cached_json(DictRedis(broken=True), "k", 60, produce)  # type: ignore[arg-type]

    assert (first, second) == (({"n": 1}, False), ({"n": 1}, True))
    assert down == ({"n": 2}, False)
    assert json.loads(redis.data["k"]) == {"n": 1}


# ------------------------------------------------------------------------- worker metrics
def test_worker_metrics_use_the_multiprocess_directory(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(worker_metrics.MULTIPROC_ENV, raising=False)
    assert worker_metrics.registry() is worker_metrics.REGISTRY

    directory = tmp_path / "prom"
    monkeypatch.setenv(worker_metrics.MULTIPROC_ENV, str(directory))
    (tmp_path / "prom").mkdir()
    (directory / "stale.db").write_text("old")
    worker_metrics.reset_multiproc_dir()

    assert list(directory.iterdir()) == []
    assert worker_metrics.registry() is not worker_metrics.REGISTRY
    worker_metrics.mark_process_dead()


def test_the_worker_metrics_server_starts_only_with_a_port(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    started: list[int] = []
    monkeypatch.delenv(worker_metrics.MULTIPROC_ENV, raising=False)
    monkeypatch.setattr(
        worker_metrics, "start_http_server", lambda port, registry: started.append(port)
    )

    for port in (0, 9200):
        configured = settings_factory(observability={"worker_metrics_port": port})
        monkeypatch.setattr(
            worker_metrics, "get_settings", lambda configured=configured: configured
        )
        worker_metrics.start_metrics_server()

    assert started == [9200]


# ------------------------------------------------------------------------- response schemas
SECRET_FIELDS = {
    "password_hash",
    "phone_encrypted",
    "phone_hash",
    "totp_secret",
    "totp_secret_encrypted",
    "token_hash",
    "key_hash",
    "device_fingerprint_hash",
    "image_s3_key",
    "s3_key",
}


def test_no_response_schema_exposes_secrets_or_storage_keys(settings: Settings) -> None:
    schemas = create_app(settings).openapi()["components"]["schemas"]

    exposed = {
        f"{name}.{field}"
        for name, schema in schemas.items()
        for field in schema.get("properties", {})
        if field in SECRET_FIELDS
    }

    assert exposed == set()

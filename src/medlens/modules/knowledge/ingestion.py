"""Document ingestion, in the worker (the text embedding model is worker-only).

For an uploaded document: fetch the stored file, extract text per page, skip it if the same
text is already indexed, scrub personal data, chunk, find NAFDAC and batch numbers in each chunk,
embed the chunks ("passage: " prefix) when the model is available, and record the NAFDAC
numbers named in alerts and recalls as published alerts (a scan signal). Without the model the
chunks are stored unembedded (keyword search still finds them) and the hourly backfill embeds
them later.
"""

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

import structlog
from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from medlens.ai.guardrails import scrub_pii
from medlens.core.config import KnowledgeSettings
from medlens.modules.knowledge.models import (
    ALERT_KINDS,
    DocumentStatus,
    KnowledgeChunk,
    KnowledgeDocument,
    PublishedAlert,
)
from medlens.modules.knowledge.text import Page, UnreadableDocument, chunk, extract_pages
from medlens.modules.verification.extraction import find_batch_numbers, find_nafdac_numbers
from medlens.ports.embedding import TextEmbeddingPort
from medlens.ports.storage import ObjectNotFoundError, SyncStoragePort

log = structlog.stdlib.get_logger(__name__)
EMBED_BATCH = 32


def ingest_document(
    session: Session,
    storage: SyncStoragePort,
    embedder: TextEmbeddingPort | None,
    document_id: UUID,
    settings: KnowledgeSettings,
) -> DocumentStatus | None:
    """Idempotent: only a pending document is processed. Returns its new status."""
    document = session.get(KnowledgeDocument, document_id, with_for_update=True)
    if document is None or document.status is not DocumentStatus.PENDING:
        return None
    try:
        if document.s3_key is None:
            raise ObjectNotFoundError(str(document_id))
        pages = extract_pages(storage.get(document.s3_key), document.media_type)
    except (UnreadableDocument, ObjectNotFoundError) as exc:
        document.status = DocumentStatus.FAILED
        document.error_code = "unreadable" if isinstance(exc, UnreadableDocument) else "missing"
        log.warning("knowledge.ingest_failed", document_id=str(document_id), reason=str(exc))
        return document.status
    return index_pages(session, document, pages, embedder, settings)


def index_pages(
    session: Session,
    document: KnowledgeDocument,
    pages: list[Page],
    embedder: TextEmbeddingPort | None,
    settings: KnowledgeSettings,
) -> DocumentStatus:
    content_hash = hashlib.sha256("\n\f".join(page.text for page in pages).encode()).hexdigest()
    document.content_sha256 = content_hash
    document.page_count = len(pages) if pages[0].number is not None else None
    earlier = session.scalar(
        select(KnowledgeDocument.id)
        .where(
            KnowledgeDocument.content_sha256 == content_hash,
            KnowledgeDocument.status == DocumentStatus.INDEXED,
            KnowledgeDocument.id != document.id,
        )
        .limit(1)
    )
    if earlier is not None:
        document.status = DocumentStatus.DUPLICATE
        document.duplicate_of_id = earlier
        log.info("knowledge.duplicate", document_id=str(document.id), duplicate_of=str(earlier))
        return document.status

    scrubbed = [Page(page.number, scrub_pii(page.text)) for page in pages]
    pieces = chunk(
        scrubbed, max_tokens=settings.chunk_tokens, overlap=settings.chunk_overlap_tokens
    )
    vectors = _embed([piece.text for piece in pieces], embedder)
    rows = [
        KnowledgeChunk(
            document_id=document.id,
            ordinal=piece.ordinal,
            page=piece.page,
            section=piece.section,
            text=piece.text,
            token_count=piece.token_count,
            nafdac_numbers=find_nafdac_numbers(piece.text),
            batch_numbers=find_batch_numbers(piece.text),
            embedding=vectors[index] if vectors else None,
            embedding_model_version=embedder.model_version if vectors and embedder else None,
        )
        for index, piece in enumerate(pieces)
    ]
    session.add_all(rows)
    session.flush()
    if document.kind in ALERT_KINDS:
        _record_alerts(session, document, rows)
    document.status = DocumentStatus.INDEXED
    document.chunk_count = len(rows)
    document.indexed_at = datetime.now(UTC)
    log.info(
        "knowledge.indexed",
        document_id=str(document.id),
        chunks=len(rows),
        embedded=bool(vectors),
    )
    return document.status


def _embed(texts: Sequence[str], embedder: TextEmbeddingPort | None) -> list[list[float]] | None:
    if embedder is None or not texts:
        return None
    try:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            vectors.extend(embedder.embed_passages(texts[start : start + EMBED_BATCH]))
    except Exception as exc:  # noqa: BLE001 - index without vectors; the backfill retries
        log.warning("knowledge.embedding_failed", error=type(exc).__name__)
        return None
    return vectors


def _record_alerts(
    session: Session, document: KnowledgeDocument, chunks: list[KnowledgeChunk]
) -> None:
    """Pairs of NAFDAC number and batch named together in a chunk. A chunk naming batches
    flags those batches only; one naming none flags every batch of the product."""
    rows: list[dict[str, object]] = []
    for item in chunks:
        batches = item.batch_numbers or [""]
        for number in item.nafdac_numbers:
            rows.extend(
                {
                    "document_id": document.id,
                    "chunk_id": item.id,
                    "kind": document.kind.value,
                    "nafdac_number": number,
                    "batch_number": batch,
                    "published_on": document.published_on,
                }
                for batch in batches
            )
    if rows:
        session.execute(
            insert(PublishedAlert)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["document_id", "nafdac_number", "batch_number"])
        )


def backfill_chunk_embeddings(
    session: Session, embedder: TextEmbeddingPort, *, limit: int = 512
) -> int:
    """Embed chunks with no vector, or one from another model (a model change re-embeds
    everything, in batches, never mixing models in one index)."""
    stale = session.scalars(
        select(KnowledgeChunk)
        .where(
            or_(
                KnowledgeChunk.embedding.is_(None),
                KnowledgeChunk.embedding_model_version.is_distinct_from(embedder.model_version),
            )
        )
        .order_by(KnowledgeChunk.created_at)
        .limit(limit)
        .with_for_update(skip_locked=True)
    ).all()
    vectors = _embed([item.text for item in stale], embedder)
    if vectors is None:
        return 0
    for item, vector in zip(stale, vectors, strict=True):
        item.embedding = vector
        item.embedding_model_version = embedder.model_version
    return len(stale)

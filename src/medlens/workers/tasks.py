"""Celery tasks. Every task is idempotent (delivery is at least once), retries only transient
errors with exponential backoff and jitter, and lands in ``failed_tasks`` when it finally fails
(``workers/failures.py``)."""

from datetime import UTC, date, datetime, timedelta
from functools import partial
from typing import Any
from uuid import UUID

import structlog
from celery import Task
from celery.signals import worker_init, worker_process_init, worker_process_shutdown
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy import delete, or_, select, text, update
from sqlalchemy.engine import Result
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from medlens.cache.keys import redis_key
from medlens.db.models import IdempotencyKey, OutboxEvent
from medlens.db.partitions import ensure_monthly_partitions
from medlens.modules.auth.models import RefreshToken
from medlens.modules.cases.service import open_case_for_scan
from medlens.modules.hotspots.service import record_snapshots
from medlens.modules.knowledge.ingestion import backfill_chunk_embeddings
from medlens.modules.knowledge.ingestion import ingest_document as ingest
from medlens.modules.organizations.models import Invitation
from medlens.modules.products.models import ReferenceImage
from medlens.modules.scans.models import Scan, ScanStatus
from medlens.ports.storage import ObjectNotFoundError, StorageUnavailableError
from medlens.queue import outbox
from medlens.queue.celery_app import celery_app, task
from medlens.queue.relay import relay_once
from medlens.workers.context import context_headers
from medlens.workers.metrics import mark_process_dead, start_metrics_server
from medlens.workers.runtime import get_runtime, init_runtime, shutdown_runtime

log = structlog.stdlib.get_logger(__name__)

# Connections and models once per worker process, after the fork (CLAUDE.md §7).
worker_process_init.connect(init_runtime, weak=False)
worker_process_shutdown.connect(shutdown_runtime, weak=False)
worker_init.connect(start_metrics_server, weak=False)
worker_process_shutdown.connect(mark_process_dead, weak=False)

# Worth retrying: the dependency may be back in a moment.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    StorageUnavailableError,
    OperationalError,
    InterfaceError,
    PoolTimeoutError,
    RedisConnectionError,
    RedisTimeoutError,
    TimeoutError,
    ConnectionError,
)
_RETRY: dict[str, Any] = {
    "autoretry_for": TRANSIENT_ERRORS,
    "retry_backoff": 2,
    "retry_backoff_max": 300,
    "retry_jitter": True,
    "max_retries": 5,
}
PURGE_GRACE = timedelta(days=30)
BACKFILL_BATCH = 200  # reference images queued per backfill run
SCAN_PRIORITY = 0  # Redis transport: lower is served first
_UNFINISHED = (ScanStatus.RECEIVED, ScanStatus.PROCESSING)


# --------------------------------------------------------------------------- scans
class _ScanTask(Task):  # type: ignore[misc]
    def on_failure(
        self, _exc: BaseException, _task_id: str, args: Any, kwargs: Any, _einfo: Any
    ) -> None:
        """Retries exhausted (or a bug): the scan must not stay ``processing`` forever."""
        scan_id = UUID(str(args[0] if args else kwargs["scan_id"]))
        with get_runtime().sessions() as session, session.begin():
            session.execute(
                update(Scan)
                .where(
                    Scan.id_window(scan_id),
                    Scan.status.in_([ScanStatus.RECEIVED, ScanStatus.PROCESSING]),
                )
                .values(status=ScanStatus.FAILED, error_code="retries_exhausted")
            )


@task(name="medlens.scans.process", base=_ScanTask, **_RETRY)
def process_scan(scan_id: str) -> str:
    runtime = get_runtime()
    result = runtime.verification.process_scan(UUID(scan_id))
    if result.outcome != "skipped":
        runtime.announce_scan_done(scan_id)
    return result.outcome


# --------------------------------------------------------------------------- outbox
def dispatch(event: OutboxEvent) -> None:
    """Map an outbox event to its task. Unknown events raise, so they end up dead-lettered."""
    payload = event.payload or {}
    headers = context_headers(payload.get("context") or {})
    # By name, Celery cannot see the task's ignore_result and would subscribe to a result
    # channel per message (results are ignored: business results live in Postgres).
    send = partial(celery_app.send_task, ignore_result=True)
    match event.event_type:
        case outbox.SCAN_RECEIVED:
            send(
                "medlens.scans.process",
                args=[event.aggregate_id],
                headers=headers,
                priority=SCAN_PRIORITY,
            )
        case outbox.SCAN_COMPLETED:
            if payload.get("needs_case"):
                send("medlens.cases.open", args=[event.aggregate_id], headers=headers)
        case outbox.REFERENCE_IMAGE_UPLOADED:
            send("medlens.reference_images.embed", args=[event.aggregate_id], headers=headers)
        case outbox.KNOWLEDGE_DOCUMENT_UPLOADED:
            send("medlens.knowledge.ingest", args=[event.aggregate_id], headers=headers)
        case outbox.SESSIONS_REVOKED:
            send(
                "medlens.auth.publish_revocations",
                args=[payload.get("session_ids", []), payload.get("ttl_seconds", 960)],
                headers=headers,
            )
        case _:
            raise ValueError(f"no task for outbox event {event.event_type!r}")


@task(name="medlens.outbox.relay", **_RETRY)
def relay_outbox() -> int:
    runtime = get_runtime()
    queue = runtime.settings.queue
    with runtime.sessions() as session, session.begin():
        result = relay_once(
            session,
            dispatch,
            batch_size=queue.relay_batch_size,
            max_attempts=queue.relay_max_attempts,
        )
    return result.published


# --------------------------------------------------------------------------- follow-ups
@task(name="medlens.cases.open", **_RETRY)
def open_case(scan_id: str) -> str | None:
    with get_runtime().sessions() as session, session.begin():
        case = open_case_for_scan(session, UUID(scan_id))
        return str(case.id) if case else None


@task(name="medlens.reference_images.embed", **_RETRY)
def embed_reference_image(reference_image_id: str) -> bool:
    """Perceptual hash and embedding of a reference image; computed on upload, used once a
    reviewer approves the image."""
    runtime = get_runtime()
    embedder = runtime.embedder
    if embedder is None:
        log.warning("reference_image.not_embedded", reason="no image embedding model")
        return False
    with runtime.sessions() as session, session.begin():
        image = session.get(ReferenceImage, UUID(reference_image_id), with_for_update=True)
        if image is None or image.deleted_at is not None:
            return False
        model = embedder.model_version
        if image.embedding is not None and image.embedding_model_version == model:
            return False  # already done
        try:
            data = runtime.storage.get(image.s3_key)
        except ObjectNotFoundError:
            log.warning("reference_image.missing", reference_image_id=reference_image_id)
            return False
        image.phash = runtime.analyzer.assess(data).phash
        image.embedding = embedder.embed_image(data)
        image.embedding_model_version = model
    return True


@task(name="medlens.reference_images.backfill", **_RETRY)
def backfill_reference_embeddings(limit: int = BACKFILL_BATCH) -> int:
    """Queue embeddings for reference images that have none, or one from another model (seeded
    images, uploads made while the model was unavailable, a model change). Hourly in Beat;
    ``make embeddings`` runs it now. Never mixes models: each image is re-embedded whole
    by ``embed_reference_image``."""
    embedder = get_runtime().embedder
    if embedder is None:
        log.warning("reference_image.backfill_skipped", reason="no image embedding model")
        return 0
    stale = or_(
        ReferenceImage.embedding.is_(None),
        ReferenceImage.embedding_model_version.is_distinct_from(embedder.model_version),
    )
    with get_runtime().sessions() as session:
        ids = session.scalars(
            select(ReferenceImage.id)
            .where(ReferenceImage.deleted_at.is_(None), stale)
            .order_by(ReferenceImage.created_at)
            .limit(limit)
        ).all()
    for image_id in ids:
        embed_reference_image.apply_async(args=[str(image_id)])
    log.info("reference_image.backfill_queued", count=len(ids))
    return len(ids)


# --------------------------------------------------------------------------- knowledge
@task(name="medlens.knowledge.ingest", **_RETRY)
def ingest_document(document_id: str) -> str | None:
    """Extract, chunk, embed and index an uploaded document (modules/knowledge/ingestion.py)."""
    runtime = get_runtime()
    with runtime.sessions() as session, session.begin():
        status = ingest(
            session,
            runtime.storage,
            runtime.text_embedder,
            UUID(document_id),
            runtime.settings.knowledge,
        )
    return status.value if status else None


@task(name="medlens.knowledge.embed_query", ignore_result=False, **_RETRY)
def embed_query(text: str) -> dict[str, Any] | None:
    """For the API's search: the question's embedding and the model version that made it,
    returned through the result backend. The API stops waiting after
    MEDLENS_KNOWLEDGE__QUERY_EMBEDDING_TIMEOUT_SECONDS."""
    embedder = get_runtime().text_embedder
    if embedder is None:
        return None
    return {"model_version": embedder.model_version, "vector": embedder.embed_query(text)}


@task(name="medlens.knowledge.rerank", ignore_result=False, **_RETRY)
def rerank(query: str, passages: list[str]) -> list[float] | None:
    """For the API's search: the cross-encoder's relevance of each passage, returned through
    the result backend. The API stops waiting after MEDLENS_KNOWLEDGE__RERANK_TIMEOUT_SECONDS."""
    reranker = get_runtime().reranker
    return reranker.rerank(query, passages) if reranker is not None else None


@task(name="medlens.knowledge.backfill_embeddings", **_RETRY)
def backfill_knowledge_embeddings() -> int:
    """Hourly: embed chunks indexed while the text model was unavailable, or after a model
    change."""
    runtime = get_runtime()
    if runtime.text_embedder is None:
        log.warning("knowledge.backfill_skipped", reason="no text embedding model")
        return 0
    with runtime.sessions() as session, session.begin():
        count = backfill_chunk_embeddings(session, runtime.text_embedder)
    log.info("knowledge.backfilled", chunks=count)
    return count


@task(name="medlens.auth.publish_revocations", **_RETRY)
def publish_revocations(session_ids: list[str], ttl_seconds: int) -> int:
    """Durable copy of the API's direct publish: revocations made while Redis was down reach
    Redis once it is back."""
    redis = get_runtime().redis
    with redis.pipeline(transaction=False) as pipe:
        for session_id in session_ids:
            pipe.set(redis_key("revoked_session", session_id), 1, ex=ttl_seconds)
        pipe.execute()
    return len(session_ids)


# --------------------------------------------------------------------------- maintenance
@task(name="medlens.hotspots.recompute", **_RETRY)
def recompute_hotspots() -> int:
    """Every 15 minutes: the last week's clusters as HotspotSnapshot rows (trends)."""
    with get_runtime().sessions() as session, session.begin():
        count = record_snapshots(session)
    log.info("hotspots.recomputed", clusters=count)
    return count


@task(name="medlens.maintenance.create_partitions", **_RETRY)
def create_partitions() -> list[str]:
    with get_runtime().sessions() as session, session.begin():
        return ensure_monthly_partitions(
            lambda ddl: session.execute(text(ddl)),
            "scans",
            today=date.today(),  # noqa: DTZ011
        )


@task(name="medlens.maintenance.purge_expired", **_RETRY)
def purge_expired() -> dict[str, int]:
    """Expired idempotency keys, long-dead refresh tokens and invitations, and the photos of
    scans past the retention period (the bucket's lifecycle rule deletes the objects; clearing
    the key stops anything from fetching them)."""
    runtime = get_runtime()
    now = datetime.now(UTC)
    retention = timedelta(days=runtime.settings.queue.scan_image_retention_days)
    statements = {
        "idempotency_keys": delete(IdempotencyKey).where(IdempotencyKey.expires_at < now),
        "refresh_tokens": delete(RefreshToken).where(RefreshToken.expires_at < now - PURGE_GRACE),
        "invitations": delete(Invitation).where(
            Invitation.accepted_at.is_(None), Invitation.expires_at < now - PURGE_GRACE
        ),
        "scan_images": update(Scan)
        .where(Scan.created_at < now - retention, Scan.image_s3_key.is_not(None))
        .values(image_s3_key=None),
    }
    with runtime.sessions() as session, session.begin():
        counts = {name: _rows(session.execute(stmt)) for name, stmt in statements.items()}
    log.info("maintenance.purged", **counts)
    return counts


def _rows(result: Result[Any]) -> int:
    return int(getattr(result, "rowcount", 0) or 0)

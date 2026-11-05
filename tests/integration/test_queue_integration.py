"""The queue against real Postgres, Redis and a real Celery worker."""

import logging
import threading
import time
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from sqlalchemy import func, select, update

from medlens.adapters.fakes import (
    FakeEmbedder,
    FakeImageAnalysis,
    FakeReranker,
    FakeTextEmbedder,
    SyncView,
)
from medlens.core.config import Settings
from medlens.db.models import FailedTask, OutboxEvent
from medlens.modules.rbac.models import RoleName
from medlens.modules.scans.models import Scan, ScanStatus
from medlens.modules.scans.service import scan_done_channel
from medlens.queue.backpressure import QueueDepth, queue_keys
from medlens.queue.celery_app import build_celery_config, celery_app
from medlens.queue.relay import relay_once
from medlens.workers import failures, tasks
from medlens.workers.runtime import WorkerRuntime, set_runtime
from tests.integration.api_support import Api, photo
from tests.support import reset_celery_clients

pytestmark = pytest.mark.integration


def add_events(api: Api, count: int, event_type: str) -> set[str]:
    ids: set[str] = set()
    with api.sync_sessions() as session, session.begin():
        for _ in range(count):
            event = OutboxEvent(
                event_type=event_type, aggregate_type="test", aggregate_id=str(uuid4())
            )
            session.add(event)
            session.flush()
            ids.add(str(event.id))
    return ids


# ------------------------------------------------------------------------- relay
async def test_two_relays_publish_each_event_exactly_once(api: Api) -> None:
    event_type = f"test.{uuid4().hex[:8]}"
    ours = add_events(api, 60, event_type)
    dispatched: Counter[str] = Counter()
    lock = threading.Lock()

    def dispatch(event: OutboxEvent) -> None:
        time.sleep(0.002)  # widen the window for overlap
        with lock:
            dispatched[str(event.id)] += 1

    def relay() -> None:
        for _ in range(10):
            with api.sync_sessions() as session, session.begin():
                relay_once(session, dispatch, batch_size=7)

    threads = [threading.Thread(target=relay) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with api.sync_sessions() as session:
        unpublished = session.scalars(
            select(OutboxEvent.id).where(
                OutboxEvent.event_type == event_type, OutboxEvent.published_at.is_(None)
            )
        ).all()
    assert {event_id for event_id in ours if dispatched[event_id] != 1} == set()
    assert unpublished == []


@pytest.fixture(autouse=True)
def empty_outbox(api: Api) -> None:
    """Earlier tests (in any order) leave unrelayed events in the shared database. The relay
    takes the oldest first, so a test's own events could miss its relay runs: mark the
    leftovers published before each test here."""
    with api.sync_sessions() as session, session.begin():
        session.execute(
            update(OutboxEvent)
            .where(OutboxEvent.published_at.is_(None), OutboxEvent.dead_lettered_at.is_(None))
            .values(published_at=func.now())
        )


async def test_failing_events_back_off_then_dead_letter(api: Api) -> None:
    event_type = f"test.{uuid4().hex[:8]}"
    (event_id,) = add_events(api, 1, event_type)

    def dispatch(event: OutboxEvent) -> None:
        if event.event_type == event_type:
            raise ConnectionError("broker down")

    now = datetime.now(UTC)
    for attempt in range(3):
        with api.sync_sessions() as session, session.begin():
            relay_once(session, dispatch, max_attempts=3, now=now + timedelta(hours=attempt))

    with api.sync_sessions() as session:
        event = session.get(OutboxEvent, UUID(event_id))
    assert event is not None
    assert event.attempts == 3
    assert event.dead_lettered_at is not None
    assert event.published_at is None
    assert event.last_error == "ConnectionError: broker down"


# ------------------------------------------------------------------------- dead letters
@pytest.fixture
def worker_runtime(api: Api, app_settings: Settings) -> Iterator[WorkerRuntime]:
    redis = Redis.from_url(app_settings.redis.url(app_settings.redis.cache_db))
    runtime = WorkerRuntime(
        settings=app_settings,
        sessions=api.sync_sessions,
        storage=SyncView(api.storage),
        redis=redis,
        analyzer=FakeImageAnalysis(),
        embedder=FakeEmbedder(dimension=app_settings.embeddings.image_dimension),
        verification=api.worker(),
        text_embedder=FakeTextEmbedder(dimension=app_settings.embeddings.text_dimension),
    )
    previous = set_runtime(runtime)
    yield runtime
    set_runtime(previous)
    redis.close()


async def test_failures_are_recorded_once_per_task_and_arguments(
    api: Api, worker_runtime: WorkerRuntime
) -> None:
    task_name = f"medlens.test.{uuid4().hex[:8]}"

    for _ in range(2):
        failures.record_failure(task_name, "t1", ["a"], {}, ValueError("boom"), None)
    failures.record_failure(task_name, "t2", ["b"], {}, ValueError("boom"), None)

    with api.sync_sessions() as session:
        rows = session.scalars(select(FailedTask).where(FailedTask.task_name == task_name)).all()
    assert sorted(row.failure_count for row in rows) == [1, 2]


async def test_admins_list_and_requeue_failed_tasks(
    api: Api, worker_runtime: WorkerRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_name = f"medlens.test.{uuid4().hex[:8]}"
    failures.record_failure(task_name, "t1", ["x"], {"y": 1}, ValueError("boom"), None)
    sent: list[tuple[str, Any, Any]] = []
    monkeypatch.setattr(
        "medlens.modules.operations.router._send",
        lambda name, args, kwargs: sent.append((name, args, kwargs)),
    )
    admin, _ = await api.user_with_role(RoleName.PLATFORM_ADMIN)
    headers = api.bearer(api.token(admin, auth_methods=("pwd", "mfa")))
    staff, pharmacy = await api.user_with_role(RoleName.PHARMACY_ADMIN)

    listed = await api.client.get(
        "/api/v1/admin/failed-tasks", params={"task_name": task_name}, headers=headers
    )
    failed_id = listed.json()["items"][0]["id"]
    requeued = await api.client.post(
        f"/api/v1/admin/failed-tasks/{failed_id}/requeue", headers=headers
    )
    pending = await api.client.get(
        "/api/v1/admin/failed-tasks",
        params={"task_name": task_name, "pending": True},
        headers=headers,
    )
    forbidden = await api.client.get(
        "/api/v1/admin/failed-tasks", headers=api.bearer(api.token(staff, organization=pharmacy))
    )

    assert listed.json()["items"][0]["error_type"] == "ValueError"
    assert requeued.status_code == 200, requeued.text
    assert requeued.json()["requeue_count"] == 1
    assert sent == [(task_name, ["x"], {"y": 1})]
    assert pending.json()["items"] == []
    assert forbidden.status_code == 403


# ------------------------------------------------------------------------- backpressure
async def test_queue_depth_reads_the_broker(app_settings: Settings) -> None:
    queue = f"scans.test.{uuid4().hex[:6]}"
    url = app_settings.redis.url(app_settings.redis.broker_db)
    sync = Redis.from_url(url)
    broker = AsyncRedis.from_url(url)
    try:
        keys = queue_keys(queue)
        sync.rpush(keys[0], "a", "b")
        sync.rpush(keys[5], "c")

        depth = await QueueDepth(broker, queue, cache_seconds=0).depth()
    finally:
        sync.delete(*queue_keys(queue))
        sync.close()
        await broker.aclose()

    assert depth == 3


# ------------------------------------------------------------------------- a real worker
@pytest.fixture
def broker(app_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the Celery app at the test Redis; restore afterwards."""
    monkeypatch.setattr("medlens.queue.celery_app.get_settings", lambda: app_settings)
    config = build_celery_config(app_settings)
    keys = ("broker_url", "result_backend", "task_always_eager")
    previous = {key: celery_app.conf.get(key) for key in keys}
    reset_celery_clients()
    with Redis.from_url(config["broker_url"]) as broker_redis:
        broker_redis.flushdb()  # messages left by earlier tests would be consumed first
    celery_app.conf.update(
        broker_url=config["broker_url"],
        result_backend=config["result_backend"],
        task_always_eager=False,
    )
    yield
    celery_app.conf.update(previous)
    reset_celery_clients()


class EventCollector(logging.Handler):
    """Collects structlog event dicts from every thread. Attached after the worker starts,
    because starting it reconfigures logging (which removes pytest's capture handler)."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if isinstance(record.msg, dict):
            self.events.append(record.msg)


async def test_a_real_worker_processes_a_scan_end_to_end(
    api: Api, worker_runtime: WorkerRuntime, broker: None
) -> None:
    """API -> outbox -> relay -> Redis -> worker -> result, with the request id in the worker's
    log lines."""
    from celery.contrib.testing.worker import start_worker

    product = await api.create_product()
    worker_runtime.verification = api.worker(
        text=f"{product.brand_name} NAFDAC REG NO: {product.nafdac_number}"
    )
    request_id = str(uuid4())
    response = await api.client.post(
        "/api/v1/scans",
        files={"image": ("pack.jpg", photo(), "image/jpeg")},
        headers={"X-Device-Id": f"device-{uuid4().hex}", "X-Request-ID": request_id},
    )
    scan_id = UUID(response.json()["id"])

    collector = EventCollector()
    with start_worker(
        celery_app,
        pool="solo",
        perform_ping_check=False,
        queues=["scans.pipeline", "events"],
        shutdown_timeout=10,
    ):
        logging.getLogger().addHandler(collector)
        try:
            with worker_runtime.sessions() as session, session.begin():
                relay_once(session, tasks.dispatch)
            status = wait_for_scan(worker_runtime, scan_id)
        finally:
            logging.getLogger().removeHandler(collector)

    scored = [event for event in collector.events if event.get("event") == "scan.scored"]
    assert status is ScanStatus.COMPLETED
    assert scored
    assert scored[0]["request_id"] == request_id


def wait_for_scan(runtime: WorkerRuntime, scan_id: UUID, seconds: float = 20) -> ScanStatus:
    """Poll the database (the worker runs in its own thread)."""
    deadline = time.monotonic() + seconds
    while True:
        with runtime.sessions() as session:
            status = session.scalar(select(Scan.status).where(Scan.id == scan_id))
        if status in {ScanStatus.COMPLETED, ScanStatus.FAILED} or time.monotonic() > deadline:
            assert status is not None
            return status
        time.sleep(0.2)


# ------------------------------------------------------------------------- task bodies
async def unregistered_scan(api: Api) -> UUID:
    response = await api.client.post(
        "/api/v1/scans",
        files={"image": ("pack.jpg", photo(), "image/jpeg")},
        data={"state": "NG-LA"},
        headers={"X-Device-Id": f"device-{uuid4().hex}"},
    )
    return UUID(response.json()["id"])


async def test_process_scan_announces_the_result(api: Api, worker_runtime: WorkerRuntime) -> None:
    scan_id = await unregistered_scan(api)
    worker_runtime.verification = api.worker(text="NAFDAC REG NO: Z9-777777")
    pubsub = worker_runtime.redis.pubsub()
    pubsub.subscribe(scan_done_channel(scan_id))
    pubsub.get_message(timeout=1)  # the subscription confirmation

    outcome = tasks.process_scan.run(str(scan_id))
    message = pubsub.get_message(timeout=2)
    pubsub.close()

    assert outcome == "completed"
    assert message is not None
    assert message["data"] == b"done"
    assert tasks.open_case.run(str(scan_id)) is not None  # UNREGISTERED opens a case


async def test_exhausted_retries_fail_the_scan(api: Api, worker_runtime: WorkerRuntime) -> None:
    scan_id = await unregistered_scan(api)

    tasks.process_scan.on_failure(TimeoutError(), "task-id", [str(scan_id)], {}, None)

    with worker_runtime.sessions() as session:
        scan = session.scalar(select(Scan).where(Scan.id == scan_id))
    assert scan is not None
    assert scan.status is ScanStatus.FAILED
    assert scan.error_code == "retries_exhausted"


async def test_reference_images_are_embedded_once(api: Api, worker_runtime: WorkerRuntime) -> None:
    from medlens.modules.products.models import ImageSide, ReferenceImage

    product = await api.create_product()
    key = f"reference-images/{uuid4()}.jpg"
    api.storage.put_sync(key, photo(), content_type="image/jpeg")
    with worker_runtime.sessions() as session, session.begin():
        image = ReferenceImage(product_id=product.id, side=ImageSide.FRONT, s3_key=key)
        session.add(image)
        session.flush()
        image_id = str(image.id)

    first = tasks.embed_reference_image.run(image_id)
    second = tasks.embed_reference_image.run(image_id)
    missing = tasks.embed_reference_image.run(str(uuid4()))

    with worker_runtime.sessions() as session:
        stored = session.get(ReferenceImage, UUID(image_id))
    assert (first, second, missing) == (True, False, False)
    assert stored is not None
    assert stored.embedding is not None
    assert len(stored.embedding) == worker_runtime.settings.embeddings.image_dimension
    assert stored.phash is not None


async def test_the_backfill_queues_images_without_a_current_embedding(
    api: Api, worker_runtime: WorkerRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    from medlens.modules.products.models import ImageSide, ReferenceImage

    product = await api.create_product()
    with worker_runtime.sessions() as session, session.begin():
        images = [
            ReferenceImage(product_id=product.id, side=ImageSide.FRONT, s3_key=f"r/{uuid4()}.jpg"),
            ReferenceImage(
                product_id=product.id,
                side=ImageSide.BACK,
                s3_key=f"r/{uuid4()}.jpg",
                embedding=[0.0] * worker_runtime.settings.embeddings.image_dimension,
                embedding_model_version="an-older-model",
            ),
        ]
        session.add_all(images)
        session.flush()
        wanted = {str(image.id) for image in images}
    queued: list[str] = []
    monkeypatch.setattr(
        tasks.embed_reference_image, "apply_async", lambda args: queued.append(args[0])
    )

    count = tasks.backfill_reference_embeddings.run(limit=10_000)

    assert count == len(queued)
    assert wanted <= set(queued)
    worker_runtime.embedder = None
    assert tasks.backfill_reference_embeddings.run() == 0


async def test_revocations_reach_redis_through_the_outbox(
    api: Api, worker_runtime: WorkerRuntime
) -> None:
    from medlens.queue.outbox import SESSIONS_REVOKED

    email = f"revoked.{uuid4().hex[:8]}@medlens.test"
    await api.create_user(email=email)
    login = (await api.password_login(email)).json()

    await api.client.post("/api/v1/auth/logout-all", headers=api.bearer(login["access_token"]))
    with worker_runtime.sessions() as session:
        event = session.scalar(
            select(OutboxEvent)
            .where(OutboxEvent.event_type == SESSIONS_REVOKED)
            .order_by(OutboxEvent.created_at.desc())
            .limit(1)
        )
    assert event is not None
    session_ids = event.payload["session_ids"]
    worker_runtime.redis.delete(*[f"medlens:v1:revoked_session:{sid}" for sid in session_ids])

    published = tasks.publish_revocations.run(session_ids, event.payload["ttl_seconds"])

    assert published == len(session_ids) >= 1
    assert worker_runtime.redis.exists(f"medlens:v1:revoked_session:{session_ids[0]}")


async def test_relay_task_dispatches_due_events(
    api: Api, worker_runtime: WorkerRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[str] = []
    monkeypatch.setattr(celery_app, "send_task", lambda name, **kwargs: sent.append(name))
    scan_id = await unregistered_scan(api)

    published = tasks.relay_outbox.run()

    assert published >= 1
    assert "medlens.scans.process" in sent
    with worker_runtime.sessions() as session:
        event = session.scalar(select(OutboxEvent).where(OutboxEvent.aggregate_id == str(scan_id)))
    assert event is not None
    assert event.published_at is not None


async def test_maintenance_tasks(api: Api, worker_runtime: WorkerRuntime) -> None:
    from medlens.db.models import IdempotencyKey

    with worker_runtime.sessions() as session, session.begin():
        session.add(
            IdempotencyKey(
                scope=f"user:{uuid4()}",
                key="expired-key-1",
                request_hash="0" * 64,
                expires_at=datetime.now(UTC) - timedelta(hours=1),
            )
        )

    partitions = tasks.create_partitions.run()
    purged = tasks.purge_expired.run()
    tasks.recompute_hotspots.run()

    assert len(partitions) == 4
    assert purged["idempotency_keys"] >= 1
    assert set(purged) == {"idempotency_keys", "refresh_tokens", "invitations", "scan_images"}


async def test_knowledge_tasks(api: Api, worker_runtime: WorkerRuntime) -> None:
    from medlens.core.ids import uuid7
    from medlens.modules.knowledge.models import DocumentKind, KnowledgeChunk, KnowledgeDocument

    key = f"documents/{uuid4()}.txt"
    number = f"Z9-{uuid4().int % 900000 + 100000}"
    text = f"# Recall\n\nNAFDAC Reg. No: {number}. Batch No: RC{uuid4().hex[:5]}1."
    api.storage.put_sync(key, text.encode(), content_type="text/plain")
    document_id = uuid7()
    with worker_runtime.sessions() as session, session.begin():
        session.add(
            KnowledgeDocument(
                id=document_id,
                title="Recall",
                kind=DocumentKind.RECALL,
                media_type="text/plain",
                s3_key=key,
                size_bytes=len(text),
            )
        )

    status = tasks.ingest_document.run(str(document_id))
    vector = tasks.embed_query.run("is there a recall?")
    with worker_runtime.sessions() as session, session.begin():
        stored = session.scalars(
            select(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id)
        ).all()
        for item in stored:
            item.embedding_model_version = "an-older-model"  # a model change
    backfilled = tasks.backfill_knowledge_embeddings.run()
    worker_runtime.reranker = FakeReranker()
    scores = tasks.rerank.run("recall batch", ["a recall notice", "storage advice"])
    worker_runtime.text_embedder = None
    worker_runtime.reranker = None
    without_model = (
        tasks.embed_query.run("q"),
        tasks.backfill_knowledge_embeddings.run(),
        tasks.rerank.run("q", ["p"]),
    )

    assert status == "indexed"
    assert vector is not None
    assert len(vector) == worker_runtime.settings.embeddings.text_dimension
    assert backfilled >= len(stored) >= 1
    assert scores[0] > scores[1]
    assert without_model == (None, 0, None)

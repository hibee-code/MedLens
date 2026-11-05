"""Queue architecture without a broker: configuration, routing, retries, context, dispatch,
backpressure and dead letters (Celery eager mode)."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from celery.exceptions import Retry

from medlens.core.config import Settings
from medlens.core.context import org_id_var, request_id_var, request_scope, user_id_var
from medlens.core.errors import DependencyUnavailable
from medlens.db.models import OutboxEvent
from medlens.modules.scans.service import scan_done_channel
from medlens.ports.storage import StorageUnavailableError
from medlens.queue import outbox
from medlens.queue.backpressure import (
    BASE_WAIT_SECONDS,
    QUEUES,
    Backpressure,
    QueueDepth,
    queue_keys,
)
from medlens.queue.beat_schedule import beat_schedule
from medlens.queue.celery_app import build_celery_config, celery_app
from medlens.queue.relay import backoff
from medlens.workers import context, failures, runtime, tasks
from tests.support import reset_celery_clients


@pytest.fixture
def eager(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr("medlens.queue.celery_app.get_settings", lambda: settings)
    celery_app.conf.update(task_always_eager=True, task_eager_propagates=False)
    yield
    celery_app.conf.update(task_always_eager=False)
    reset_celery_clients()  # eager runs may have created a backend for the test settings


@dataclass
class FakeVerification:
    error: BaseException | None = None
    calls: int = 0

    def process_scan(self, scan_id: object) -> SimpleNamespace:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(outcome="completed")


@dataclass
class FakeRuntime:
    verification: FakeVerification = field(default_factory=FakeVerification)
    announced: list[str] = field(default_factory=list)

    def announce_scan_done(self, scan_id: str) -> None:
        self.announced.append(scan_id)


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> FakeRuntime:
    fake = FakeRuntime()
    monkeypatch.setattr(tasks, "get_runtime", lambda: fake)
    monkeypatch.setattr(tasks._ScanTask, "on_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(failures, "record_failure", lambda *args, **kwargs: None)
    return fake


# ------------------------------------------------------------------------- configuration
def test_queues_routing_and_limits(settings: Settings) -> None:
    config = build_celery_config(settings)
    routes = config["task_routes"]

    assert {route["queue"] for route in routes.values()} <= set(QUEUES)
    assert routes["medlens.scans.*"] == {"queue": "scans.pipeline"}
    assert routes["medlens.reference_images.*"] == {"queue": "ml.embed"}
    assert routes["medlens.outbox.*"] == {"queue": "events"}
    assert routes["medlens.maintenance.*"] == {"queue": "maintenance"}
    assert config["accept_content"] == ["json"]
    assert config["task_ignore_result"] is True
    assert config["task_acks_late"] is True
    assert config["task_reject_on_worker_lost"] is True
    assert config["worker_prefetch_multiplier"] == 1
    options = config["broker_transport_options"]
    assert (
        options["visibility_timeout"] > config["task_time_limit"] > config["task_soft_time_limit"]
    )


def test_beat_runs_the_relay_and_maintenance(settings: Settings) -> None:
    schedule = beat_schedule(settings)

    assert schedule["outbox-relay"]["task"] == "medlens.outbox.relay"
    assert schedule["outbox-relay"]["schedule"] == settings.queue.relay_interval_seconds
    assert {entry["task"] for entry in schedule.values()} == {
        "medlens.outbox.relay",
        "medlens.reference_images.backfill",
        "medlens.knowledge.backfill_embeddings",
        "medlens.hotspots.recompute",
        "medlens.maintenance.create_partitions",
        "medlens.maintenance.purge_expired",
    }


def test_every_routed_task_exists() -> None:
    registered = {name for name in celery_app.tasks if name.startswith("medlens.")}

    assert {
        "medlens.scans.process",
        "medlens.outbox.relay",
        "medlens.cases.open",
        "medlens.reference_images.embed",
        "medlens.reference_images.backfill",
        "medlens.knowledge.ingest",
        "medlens.knowledge.embed_query",
        "medlens.knowledge.backfill_embeddings",
        "medlens.auth.publish_revocations",
        "medlens.hotspots.recompute",
        "medlens.maintenance.create_partitions",
        "medlens.maintenance.purge_expired",
    } <= registered


# ------------------------------------------------------------------------- retries
def test_transient_errors_are_retried(eager: None, fake_runtime: FakeRuntime) -> None:
    fake_runtime.verification.error = StorageUnavailableError("down")

    result = tasks.process_scan.apply(args=[str(uuid4())])

    assert result.failed()
    assert fake_runtime.verification.calls == tasks.process_scan.max_retries + 1


def test_permanent_errors_are_not_retried(eager: None, fake_runtime: FakeRuntime) -> None:
    fake_runtime.verification.error = ValueError("bug")

    result = tasks.process_scan.apply(args=[str(uuid4())])

    assert result.failed()
    assert fake_runtime.verification.calls == 1


def test_a_processed_scan_is_announced(eager: None, fake_runtime: FakeRuntime) -> None:
    scan_id = str(uuid4())

    result = tasks.process_scan.apply(args=[scan_id])

    assert result.get() == "completed"
    assert fake_runtime.announced == [scan_id]


def test_retry_signals_are_not_transient() -> None:
    assert Retry not in tasks.TRANSIENT_ERRORS
    assert ValueError not in tasks.TRANSIENT_ERRORS


# ------------------------------------------------------------------------- context
def test_context_travels_in_headers_and_is_restored() -> None:
    headers: dict[str, Any] = {}
    user = uuid4()
    with request_scope("req-123", trace_id="trace-9"):
        user_id_var.set(user)
        context._add_context(headers=headers)
    task = SimpleNamespace(request=SimpleNamespace(headers=headers))

    context._restore_context(task_id="t1", task=task)  # type: ignore[arg-type]
    restored = (request_id_var.get(), user_id_var.get(), org_id_var.get())
    context._clear_context(task_id="t1")

    assert headers[context.CONTEXT_HEADER] == {
        "request_id": "req-123",
        "trace_id": "trace-9",
        "user_id": str(user),
    }
    assert restored == ("req-123", user, None)
    assert request_id_var.get() is None


def test_tasks_without_context_get_their_own_request_id() -> None:
    task = SimpleNamespace(request=SimpleNamespace(headers=None))

    context._restore_context(task_id="abc", task=task)  # type: ignore[arg-type]
    request_id = request_id_var.get()
    context._clear_context(task_id="abc")

    assert request_id == "task-abc"


def test_explicit_context_headers_are_kept() -> None:
    headers = context.context_headers({"request_id": "from-outbox"})

    context._add_context(headers=headers)

    assert headers == {context.CONTEXT_HEADER: {"request_id": "from-outbox"}}


def test_outbox_events_remember_the_request() -> None:
    with request_scope("req-outbox"):
        event = outbox._event(outbox.SCAN_RECEIVED, "scan", uuid4(), {"a": 1})

    assert event.payload == {"a": 1, "context": {"request_id": "req-outbox"}}


# ------------------------------------------------------------------------- dispatch
@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        celery_app, "send_task", lambda name, **kwargs: calls.append((name, kwargs))
    )
    return calls


def event(event_type: str, **payload: Any) -> OutboxEvent:
    return OutboxEvent(
        event_type=event_type, aggregate_type="x", aggregate_id=str(uuid4()), payload=payload
    )


def test_events_map_to_tasks(sent: list[tuple[str, dict[str, Any]]]) -> None:
    received = event(outbox.SCAN_RECEIVED, context={"request_id": "r1"})

    tasks.dispatch(received)
    tasks.dispatch(event(outbox.SCAN_COMPLETED, needs_case=True))
    tasks.dispatch(event(outbox.SCAN_COMPLETED, needs_case=False))
    tasks.dispatch(event(outbox.REFERENCE_IMAGE_UPLOADED))
    tasks.dispatch(event(outbox.SESSIONS_REVOKED, session_ids=["s1"], ttl_seconds=960))

    assert [name for name, _ in sent] == [
        "medlens.scans.process",
        "medlens.cases.open",
        "medlens.reference_images.embed",
        "medlens.auth.publish_revocations",
    ]
    assert sent[0][1]["args"] == [received.aggregate_id]
    assert sent[0][1]["priority"] == tasks.SCAN_PRIORITY
    assert all(options["ignore_result"] for _, options in sent)  # no result subscriptions
    assert sent[0][1]["headers"] == {context.CONTEXT_HEADER: {"request_id": "r1"}}
    assert sent[3][1]["args"] == [["s1"], 960]


def test_unknown_events_raise_so_they_are_dead_lettered(
    sent: list[tuple[str, dict[str, Any]]],
) -> None:
    with pytest.raises(ValueError, match="no task"):
        tasks.dispatch(event("thing.happened"))


def test_relay_backoff_grows_and_is_capped() -> None:
    assert backoff(1).total_seconds() == 2
    assert backoff(3).total_seconds() == 8
    assert backoff(30).total_seconds() == 600


# ------------------------------------------------------------------------- backpressure
class FakePipe:
    def __init__(self, lengths: list[int] | Exception) -> None:
        self._lengths = lengths

    async def __aenter__(self) -> "FakePipe":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def llen(self, key: str) -> None:
        return None

    async def execute(self) -> list[int]:
        if isinstance(self._lengths, Exception):
            raise self._lengths
        return self._lengths


class FakeBroker:
    def __init__(self, lengths: list[int] | Exception) -> None:
        self.lengths = lengths
        self.reads = 0

    def pipeline(self, transaction: bool = True) -> FakePipe:
        self.reads += 1
        return FakePipe(self.lengths)


async def test_queue_depth_sums_priorities_and_is_cached() -> None:
    clock = iter([0.0, 1.0, 5.0])
    broker = FakeBroker([3, 0, 2] + [0] * 7)
    depth = QueueDepth(broker, cache_seconds=3.0, clock=lambda: next(clock))  # type: ignore[arg-type]

    assert await depth.depth() == 5
    assert await depth.depth() == 5
    assert broker.reads == 1
    broker.lengths = [9] * 10
    assert await depth.depth() == 90
    assert queue_keys("q")[:3] == ["q", "q:1", "q:2"]


async def test_an_unreadable_queue_does_not_block() -> None:
    from redis.exceptions import ConnectionError as RedisConnectionError

    depth = QueueDepth(FakeBroker(RedisConnectionError("down")))  # type: ignore[arg-type]

    assert await depth.depth() is None
    assert Backpressure(10, 20, 1.0).estimated_wait(None) == BASE_WAIT_SECONDS


def test_backpressure_grows_the_wait_then_refuses() -> None:
    pressure = Backpressure(soft_limit=100, hard_limit=1000, seconds_per_queued_scan=0.5)

    assert pressure.estimated_wait(50) == BASE_WAIT_SECONDS
    assert pressure.estimated_wait(400) == BASE_WAIT_SECONDS + 200
    with pytest.raises(DependencyUnavailable) as raised:
        pressure.estimated_wait(5000)
    assert raised.value.headers == {"Retry-After": "300"}


# ------------------------------------------------------------------------- dead letters
def test_failure_hashes_are_stable_and_value_free() -> None:
    first = failures.args_hash("t", ["a", 1], {"k": "v"})

    assert first == failures.args_hash("t", ["a", 1], {"k": "v"})
    assert first != failures.args_hash("t", ["a", 2], {"k": "v"})
    assert failures.traceback_hash(None) is None


def test_failure_signal_records_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[str] = []

    def record(task_name: str, *args: object) -> None:
        recorded.append(task_name)
        raise RuntimeError("database down")

    monkeypatch.setattr(failures, "record_failure", record)
    sender = SimpleNamespace(name="medlens.cases.open")

    failures._on_task_failure(sender=sender, task_id="t", exception=ValueError("x"), args=["a"])
    failures._on_task_failure(sender=None, exception=ValueError("x"))

    assert recorded == ["medlens.cases.open"]


def test_runtime_can_be_swapped_for_tests() -> None:
    fake = object()
    previous = runtime.set_runtime(fake)  # type: ignore[arg-type]
    try:
        assert runtime.get_runtime() is fake
    finally:
        runtime.set_runtime(previous)
    assert scan_done_channel("abc") == "medlens:v1:scan_done:abc"

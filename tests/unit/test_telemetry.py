"""Tracing and error reporting: off unless configured; spans for requests, logs joined to them;
Sentry without personal data. Redis clients fail fast."""

import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import structlog
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr
from redis.exceptions import RedisError

from medlens.cache.redis import create_redis_client, create_sync_redis_client
from medlens.core import telemetry
from medlens.core.config import RedisSettings, Settings
from medlens.main import create_app


@pytest.fixture
def traced(settings_factory: Any) -> Settings:
    settings: Settings = settings_factory(observability={"traces_sample_rate": 1.0})
    return settings


@pytest.fixture
def spans(traced: Settings) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    assert telemetry.configure_tracing(traced, "api", exporter=exporter)
    yield exporter
    telemetry.reset()


def test_tracing_is_off_without_an_endpoint(settings: Settings) -> None:
    assert not telemetry.configure_tracing(settings, "api")
    assert not telemetry.tracing_enabled()
    telemetry.instrument_engine(object())  # no-ops while off
    telemetry.instrument_celery()
    telemetry.flush()


async def test_requests_are_traced_and_logs_carry_the_trace(
    traced: Settings, spans: InMemorySpanExporter
) -> None:
    app = create_app(traced)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.get("/api/v1/scans/not-a-uuid", headers={"X-Device-Id": "d-1"})
        await client.get("/health/live")  # not traced
    telemetry.flush()

    names = [span.name for span in spans.get_finished_spans()]
    assert any("/api/v1/scans/{scan_id}" in name for name in names)
    assert not any("health" in name for name in names)
    resource = spans.get_finished_spans()[0].resource.attributes
    assert resource["service.name"] == "medlens-api"


def test_log_lines_get_the_active_trace(spans: InMemorySpanExporter) -> None:
    from opentelemetry import trace

    tracer = trace.get_tracer("test", tracer_provider=telemetry._state["provider"])
    with tracer.start_as_current_span("work"):
        inside = telemetry.add_trace_context(None, "info", {})
    outside = telemetry.add_trace_context(None, "info", {})

    assert len(inside["trace_id"]) == 32
    assert len(inside["span_id"]) == 16
    assert "trace_id" not in outside


def test_the_otlp_exporter_is_built_from_the_endpoint(
    settings_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []

    class Exporter(InMemorySpanExporter):
        def __init__(self, endpoint: str, timeout: int) -> None:
            super().__init__()
            built.append(endpoint)

    import opentelemetry.exporter.otlp.proto.http.trace_exporter as otlp

    monkeypatch.setattr(otlp, "OTLPSpanExporter", Exporter)
    configured = settings_factory(observability={"otlp_endpoint": "http://collector:4318/"})
    try:
        assert telemetry.configure_tracing(configured, "worker")
        assert telemetry.configure_tracing(configured, "worker")  # once per process
        telemetry.instrument_engine(_engine())
        telemetry.instrument_celery()
    finally:
        telemetry.reset()
    assert built == ["http://collector:4318/v1/traces"]


def _engine() -> Any:
    from sqlalchemy import create_engine

    return create_engine("sqlite://")


def test_sentry_starts_only_with_a_dsn_and_sends_no_personal_data(
    settings_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sentry_sdk

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setitem(telemetry._state, "sentry", False)

    assert not telemetry.configure_sentry(settings_factory(), "api")
    dsn = SecretStr("https://key@sentry.example/1")  # pragma: allowlist secret
    configured = settings_factory(observability={"sentry_dsn": dsn})
    assert telemetry.configure_sentry(configured, "worker")
    assert telemetry.configure_sentry(configured, "worker")  # once per process

    [options] = calls
    assert options["dsn"] == dsn.get_secret_value()
    assert options["send_default_pii"] is False
    assert options["include_local_variables"] is False
    assert options["max_request_body_size"] == "never"
    assert options["server_name"] == "worker"


def test_the_logging_pipeline_includes_trace_ids(settings: Settings) -> None:
    from medlens.core.logging import configure_logging

    configure_logging(settings)
    processors = structlog.get_config()["processors"]

    assert telemetry.add_trace_context in processors


# ------------------------------------------------------------------------- Redis clients
async def test_redis_clients_fail_fast_when_redis_is_down() -> None:
    down = RedisSettings(host="127.0.0.1", port=1)
    client = create_redis_client(down, db=0)
    started = time.perf_counter()
    with pytest.raises(RedisError):
        await client.get("key")
    elapsed_async = time.perf_counter() - started
    await client.aclose()

    sync = create_sync_redis_client(down, db=0)
    started = time.perf_counter()
    with pytest.raises(RedisError):
        sync.get("key")
    elapsed_sync = time.perf_counter() - started
    sync.close()

    # redis-py's default policy (ten retries with backoff) took over three seconds here.
    assert elapsed_async < 1.0
    assert elapsed_sync < 1.0


def test_redis_retries_are_bounded_by_settings() -> None:
    settings = RedisSettings(retries=2)

    async_retry = create_redis_client(settings, db=0).connection_pool.connection_kwargs["retry"]
    sync_retry = create_sync_redis_client(settings, db=0).connection_pool.connection_kwargs["retry"]

    assert async_retry._retries == 2
    assert sync_retry._retries == 2

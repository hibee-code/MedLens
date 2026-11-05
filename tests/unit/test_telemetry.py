"""Tracing and error reporting: off unless configured; spans for requests, logs joined to them;
Sentry without personal data. Redis clients fail fast."""

import json
import logging
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
    assert logging.getLogger("httpx").level == logging.WARNING  # no URLs with query strings


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


# ------------------------------------------------------------------------- no personal data
PHONE, LOCAL_PHONE, EMAIL = "+2348031234567", "08031234567", "ada.obi@example.ng"
TOKEN = "eyJhbGciOiJFZERTQSJ9.eyJzdWIiOiJhZGEifQ.c2lnbmF0dXJlLXZhbHVl"  # pragma: allowlist secret
CLIENT_IP, AGENT = "203.0.113.7", "MedLensPiiProbe/9.9"
PERSONAL = (PHONE, LOCAL_PHONE, EMAIL, TOKEN, CLIENT_IP, AGENT, "Ada Obi")  # name: body only


def leaks(text: str) -> list[str]:
    return [value for value in PERSONAL if value in text]


def pii_app(settings: Settings) -> Any:
    """The real app plus one route that drags personal data through errors and Redis."""
    import sentry_sdk
    from fastapi import Request
    from opentelemetry import trace

    app = create_app(settings)

    @app.post("/api/v1/pii-probe")
    async def probe(request: Request) -> dict[str, str]:
        body = await request.json()
        tracer = trace.get_tracer("probe", tracer_provider=telemetry._state["provider"])
        with tracer.start_as_current_span("redis SET") as span:  # as the Redis instrumentation
            span.set_attribute("db.system", "redis")
            span.set_attribute("db.statement", f"SET otp:{body['phone']} {body['email']}")
        sentry_sdk.set_user({"email": body["email"], "ip_address": CLIENT_IP})
        sentry_sdk.add_breadcrumb(message=f"code sent to {body['phone']}")
        try:
            raise RuntimeError(f"sign-in failed for <{body['email']}> {PHONE}")
        except RuntimeError as exc:
            trace.get_current_span().record_exception(exc)
            sentry_sdk.capture_exception(exc)
        sentry_sdk.capture_message(f"Bearer {TOKEN} used from {LOCAL_PHONE}")
        return {"ok": "yes"}

    return app


async def send_personal_data(app: Any) -> None:
    transport = httpx.ASGITransport(app=app, client=(CLIENT_IP, 40000))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            f"/api/v1/pii-probe?phone={PHONE}&email={EMAIL}",
            json={"phone": PHONE, "email": EMAIL, "name": "Ada Obi"},
            headers={"Authorization": f"Bearer {TOKEN}", "User-Agent": AGENT},
        )


async def test_traces_carry_no_personal_data(traced: Settings, spans: InMemorySpanExporter) -> None:
    await send_personal_data(pii_app(traced))
    telemetry.flush()

    exported = [
        {
            "name": span.name,
            "attributes": dict(span.attributes or {}),
            "events": [dict(event.attributes or {}) for event in span.events],
        }
        for span in spans.get_finished_spans()
    ]
    text = json.dumps(exported, default=str)
    assert any("/api/v1/pii-probe" in item["name"] for item in exported)  # still useful
    assert any(item["attributes"].get("db.statement") == "SET" for item in exported)
    assert leaks(text) == []


async def test_error_reports_carry_no_personal_data(settings_factory: Any) -> None:
    import sentry_sdk
    from sentry_sdk.envelope import Envelope
    from sentry_sdk.transport import Transport

    sent: list[str] = []

    class Capture(Transport):
        def capture_envelope(self, envelope: Envelope) -> None:
            sent.append(envelope.serialize().decode("utf-8", "replace"))

    dsn = SecretStr("https://key@sentry.example/1")  # pragma: allowlist secret
    settings = settings_factory(observability={"sentry_dsn": dsn})
    sentry_sdk.init(**telemetry.sentry_options(settings, "api"), transport=Capture())
    try:
        await send_personal_data(pii_app(settings_factory()))  # the app must not re-init Sentry
        sentry_sdk.flush()
    finally:
        sentry_sdk.get_client().close()
        sentry_sdk.init()  # no DSN: off again

    text = "\n".join(sent)
    assert "sign-in failed for" in text  # the errors were reported...
    assert "/api/v1/pii-probe" in text
    assert leaks(text) == []  # ...without the personal data
    assert '"release":"medlens@' in text  # versions are not mistaken for addresses


def test_the_scrubbers_mask_contact_details_and_tokens() -> None:
    from medlens.core.pii import scrub_secrets_and_pii

    masked = scrub_secrets_and_pii(f"{EMAIL} {PHONE} 0803 123 4567 Bearer {TOKEN} {TOKEN}")

    assert masked == "[email] [phone] [phone] Bearer [token] [token]"
    assert scrub_secrets_and_pii("ApiKey mlk_live_abc.def") == "ApiKey [redacted]"
    assert telemetry.redact_attributes(None) == {}
    assert telemetry.redact_attributes({"http.target": "/a?b=1", "tags": ["x", EMAIL]}) == {
        "http.target": "/a",
        "tags": ("x", "[email]"),
    }

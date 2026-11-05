"""Tracing (OpenTelemetry) and error reporting (Sentry). Both are off unless configured.

* MEDLENS_OBSERVABILITY__OTLP_ENDPOINT (e.g. http://otel-collector:4318): spans for incoming
  HTTP requests, SQL, Redis, outgoing HTTP (SMS, model API) and Celery tasks, exported over
  OTLP/HTTP and sampled at MEDLENS_OBSERVABILITY__TRACES_SAMPLE_RATE (following the caller's
  decision when a request arrives with a trace). Log lines carry trace_id and span_id.
* MEDLENS_OBSERVABILITY__SENTRY_DSN: unhandled errors from the API and the workers, without
  request bodies, local variables or personal data.

Health probes and /metrics are not traced.

No personal data leaves in either (tests/unit/test_telemetry.py sends some and checks):
every span passes RedactingSpanExporter (no client IPs or user agents, no query strings, Redis
commands without arguments, exceptions by type only, e-mail addresses, phone numbers and tokens
masked in every text),
and every Sentry event and breadcrumb passes ``scrub_event`` (no request headers, cookies,
query strings, bodies or user, and the same masking). Masking finds contact details and
tokens, not names: code never formats personal fields into messages.
"""

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from structlog.typing import EventDict, WrappedLogger

from medlens import __version__
from medlens.core.config import Settings
from medlens.core.pii import scrub_secrets_and_pii

log = structlog.stdlib.get_logger(__name__)
_UNTRACED_URLS = "health/.*,metrics"
_state: dict[str, Any] = {"provider": None, "sentry": False}


# Span attributes that identify the person or device making a request.
_DROPPED_ATTRIBUTES = frozenset(
    {
        "client.address",
        "client.port",
        "http.client_ip",
        "net.peer.ip",
        "net.peer.port",
        "net.sock.peer.addr",
        "net.sock.peer.port",
        "network.peer.address",
        "network.peer.port",
        "user_agent.original",
        "http.user_agent",
        "url.query",
    }
)
_DROPPED_PREFIXES = ("http.request.header.", "http.response.header.", "enduser.")
_URL_ATTRIBUTES = frozenset({"http.url", "url.full", "http.target"})


def _without_query(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def redact_attributes(attributes: Any) -> dict[str, Any]:
    """A span's (or span event's) attributes without personal data."""
    clean: dict[str, Any] = {}
    redis = (attributes or {}).get("db.system") == "redis"
    for key, raw in (attributes or {}).items():
        if key in _DROPPED_ATTRIBUTES or key.startswith(_DROPPED_PREFIXES):
            continue
        value = raw
        if key in _URL_ATTRIBUTES and isinstance(value, str):
            value = _without_query(value)
        elif key == "db.statement" and redis and isinstance(value, str):
            value = value.split(" ", 1)[0]  # the command; keys and values can hold anything
        if isinstance(value, str):
            value = scrub_secrets_and_pii(value)
        elif isinstance(value, Sequence) and not isinstance(value, bytes):
            value = tuple(scrub_secrets_and_pii(v) if isinstance(v, str) else v for v in value)
        clean[key] = value
    return clean


def _event_attributes(event: Event) -> dict[str, Any]:
    """Exception events keep only the exception's type: messages and stack text can hold
    anything (names cannot be found by pattern). Error details go to Sentry, scrubbed."""
    attributes = redact_attributes(event.attributes)
    if event.name == "exception":
        return {key: attributes[key] for key in ("exception.type", "exception.escaped")
                if key in attributes}  # fmt: skip
    return attributes


def _redact(span: ReadableSpan) -> ReadableSpan:
    return ReadableSpan(
        name=scrub_secrets_and_pii(span.name),
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=redact_attributes(span.attributes),
        events=[
            Event(event.name, _event_attributes(event), event.timestamp) for event in span.events
        ],
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class RedactingSpanExporter(SpanExporter):
    """Wraps the real exporter: spans leave the process only after ``redact_attributes``."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([_redact(span) for span in spans])

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


def tracing_enabled() -> bool:
    return _state["provider"] is not None


def configure_tracing(
    settings: Settings, component: str, *, exporter: SpanExporter | None = None
) -> bool:
    """Once per process. ``exporter`` replaces the OTLP exporter (tests)."""
    observability = settings.observability
    if tracing_enabled():
        return True
    if exporter is None:
        if not observability.otlp_endpoint:
            return False
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = observability.otlp_endpoint.rstrip("/") + "/v1/traces"
        exporter = OTLPSpanExporter(endpoint=endpoint, timeout=5)
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": f"{observability.service_name}-{component}",
                "service.version": __version__,
                "deployment.environment": settings.app.environment.value,
            }
        ),
        sampler=ParentBased(TraceIdRatioBased(observability.traces_sample_rate)),
    )
    provider.add_span_processor(BatchSpanProcessor(RedactingSpanExporter(exporter)))
    trace.set_tracer_provider(provider)
    _state["provider"] = provider

    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    RedisInstrumentor().instrument(tracer_provider=provider)
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)
    log.info("telemetry.tracing_on", component=component)
    return True


def instrument_app(app: Any) -> None:
    if not tracing_enabled():
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=_state["provider"], excluded_urls=_UNTRACED_URLS
    )


def instrument_engine(engine: Any) -> None:
    """A SQLAlchemy engine (async engines: pass ``engine.sync_engine``)."""
    if not tracing_enabled():
        return
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    SQLAlchemyInstrumentor().instrument(engine=engine, tracer_provider=_state["provider"])


def instrument_celery() -> None:
    """``worker_process_init``: each pool process traces its own tasks."""
    if not tracing_enabled():
        return
    from opentelemetry.instrumentation.celery import CeleryInstrumentor

    CeleryInstrumentor().instrument(  # type: ignore[no-untyped-call]
        tracer_provider=_state["provider"]
    )


def add_trace_context(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    """structlog processor: the active span's ids, so logs and traces join up."""
    context = trace.get_current_span().get_span_context()
    if context.is_valid:
        event_dict.setdefault("trace_id", format(context.trace_id, "032x"))
        event_dict.setdefault("span_id", format(context.span_id, "016x"))
    return event_dict


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_secrets_and_pii(value)
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_scrub(item) for item in value]
    return value


# Event fields describing the software, not the request: left as they are.
_SENTRY_METADATA = frozenset(
    {"event_id", "timestamp", "release", "dist", "environment", "server_name", "platform",
     "sdk", "modules", "level", "type"}
)  # fmt: skip


def scrub_event(event: Any, _hint: Any = None) -> Any:
    """Sentry ``before_send``: the request reduced to its method and URL path, no user, and
    personal data masked everywhere else (messages, exception values, breadcrumbs, tags)."""
    request = event.get("request")
    if isinstance(request, dict):
        url = request.get("url")
        event["request"] = {
            "method": request.get("method"),
            "url": _without_query(url) if isinstance(url, str) else None,
        }
    event.pop("user", None)
    return {
        key: value if key in _SENTRY_METADATA else _scrub(value) for key, value in event.items()
    }


def scrub_breadcrumb(crumb: Any, _hint: Any = None) -> Any:
    """Sentry ``before_breadcrumb``."""
    return _scrub(crumb)


def sentry_options(settings: Settings, component: str) -> dict[str, Any]:
    dsn = settings.observability.sentry_dsn
    return {
        "dsn": dsn.get_secret_value() if dsn else None,
        "environment": settings.app.environment.value,
        "release": f"medlens@{__version__}",
        "server_name": component,
        "send_default_pii": False,  # no IPs, cookies, headers or user data
        "max_request_body_size": "never",
        "include_local_variables": False,  # locals can hold tokens or pack text
        "traces_sample_rate": 0.0,  # tracing goes to OpenTelemetry
        "before_send": scrub_event,
        "before_breadcrumb": scrub_breadcrumb,
    }


def configure_sentry(settings: Settings, component: str) -> bool:
    """Once per process; off without a DSN."""
    dsn = settings.observability.sentry_dsn
    if dsn is None or _state["sentry"]:
        return bool(_state["sentry"])
    import sentry_sdk

    sentry_sdk.init(**sentry_options(settings, component))
    _state["sentry"] = True
    log.info("telemetry.sentry_on", component=component)
    return True


def flush() -> None:
    """Export pending spans now (app shutdown)."""
    provider = _state["provider"]
    if provider is not None:
        provider.force_flush(timeout_millis=5000)


def reset() -> None:
    """Undo configure_tracing and the client instrumentation (tests)."""
    provider = _state["provider"]
    if provider is None:
        return
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    RedisInstrumentor().uninstrument()
    HTTPXClientInstrumentor().uninstrument()
    provider.shutdown()
    _state["provider"] = None

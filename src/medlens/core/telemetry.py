"""Tracing (OpenTelemetry) and error reporting (Sentry). Both are off unless configured.

* MEDLENS_OBSERVABILITY__OTLP_ENDPOINT (e.g. http://otel-collector:4318): spans for incoming
  HTTP requests, SQL, Redis, outgoing HTTP (SMS, model API) and Celery tasks, exported over
  OTLP/HTTP and sampled at MEDLENS_OBSERVABILITY__TRACES_SAMPLE_RATE (following the caller's
  decision when a request arrives with a trace). Log lines carry trace_id and span_id.
* MEDLENS_OBSERVABILITY__SENTRY_DSN: unhandled errors from the API and the workers, without
  request bodies, local variables or personal data.

Health probes and /metrics are not traced.
"""

from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from structlog.typing import EventDict, WrappedLogger

from medlens import __version__
from medlens.core.config import Settings

log = structlog.stdlib.get_logger(__name__)
_UNTRACED_URLS = "health/.*,metrics"
_state: dict[str, Any] = {"provider": None, "sentry": False}


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
    provider.add_span_processor(BatchSpanProcessor(exporter))
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


def configure_sentry(settings: Settings, component: str) -> bool:
    """Once per process; off without a DSN."""
    dsn = settings.observability.sentry_dsn
    if dsn is None or _state["sentry"]:
        return bool(_state["sentry"])
    import sentry_sdk

    sentry_sdk.init(
        dsn=dsn.get_secret_value(),
        environment=settings.app.environment.value,
        release=f"medlens@{__version__}",
        server_name=component,
        send_default_pii=False,  # no IPs, cookies, headers or user data
        max_request_body_size="never",
        include_local_variables=False,  # locals can hold tokens or pack text
        traces_sample_rate=0.0,  # tracing goes to OpenTelemetry
    )
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

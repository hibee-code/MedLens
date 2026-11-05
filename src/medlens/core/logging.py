"""structlog setup: JSON lines in staging/production, readable console output locally.

Standard-library loggers (uvicorn, celery, sqlalchemy, ...) are routed through the same
processors, so every line carries the request context from ``medlens.core.context``.
"""

import logging
import sys

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from medlens.core.config import Environment, Settings
from medlens.core.context import snapshot
from medlens.core.telemetry import add_trace_context

# Loggers that install their own handlers; we clear them and let records reach the root handler.
_PROPAGATED_LOGGERS = ("uvicorn", "uvicorn.error", "celery")
# AccessLogMiddleware writes one structured line per request; Uvicorn's access log duplicates it.
_SILENCED_LOGGERS = ("uvicorn.access",)
# HTTP clients log every request's full URL at INFO, query string included: warnings only.
_QUIET_LOGGERS = ("httpx", "httpcore", "urllib3")


def add_request_context(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    for key, value in snapshot().items():
        event_dict.setdefault(key, value)
    return event_dict


def configure_logging(settings: Settings) -> None:
    observability = settings.observability
    environment = settings.app.environment
    use_json = observability.log_format == "json" or (
        observability.log_format == "auto"
        and environment in {Environment.STAGING, Environment.PRODUCTION}
    )
    level = "DEBUG" if settings.app.debug else observability.log_level

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_request_context,
        add_trace_context,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    rendering: list[Processor] = [structlog.stdlib.ProcessorFormatter.remove_processors_meta]
    if use_json:
        rendering += [structlog.processors.dict_tracebacks, structlog.processors.JSONRenderer()]
    else:
        rendering.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # Off: module-level loggers cached on first use would ignore any later reconfiguration
        # (a second app instance, Celery's setup_logging, tests capturing logs).
        cache_logger_on_first_use=False,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(foreign_pre_chain=shared, processors=rendering)
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for name in _PROPAGATED_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    for name in _SILENCED_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = False
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

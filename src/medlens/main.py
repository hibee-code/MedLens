"""Application factory.

Run locally:  uv run uvicorn medlens.main:create_app --factory --reload
Containers:   uvicorn medlens.main:create_app --factory --host 0.0.0.0 --port 8000 --proxy-headers
              (one process per container; no Gunicorn)
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import cache

import structlog
from fastapi import APIRouter, FastAPI
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from medlens import __version__
from medlens.core import telemetry
from medlens.core.bootstrap import configure_process
from medlens.core.config import Settings, get_settings
from medlens.core.devices import DEVICE_HEADER
from medlens.core.errors import register_exception_handlers
from medlens.core.metrics import router as metrics_router
from medlens.core.middleware import (
    REQUEST_ID_HEADER,
    AccessLogMiddleware,
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
    UnhandledErrorMiddleware,
)
from medlens.core.openapi import (
    API_DESCRIPTION,
    API_TAGS,
    API_TITLE,
    MedLensAPI,
    generate_operation_id,
)
from medlens.modules.audit.router import router as audit_router
from medlens.modules.auth.router import api_keys_router, well_known_router
from medlens.modules.auth.router import router as auth_router
from medlens.modules.cases.router import router as cases_router
from medlens.modules.control_plane.router import router as ai_admin_router
from medlens.modules.health.deps import readiness_service
from medlens.modules.health.monitor import start_monitor, stop_monitor
from medlens.modules.health.router import router as health_router
from medlens.modules.hotspots.router import router as hotspots_router
from medlens.modules.knowledge.router import router as knowledge_router
from medlens.modules.operations.router import router as failed_tasks_router
from medlens.modules.organizations.router import invitations_router
from medlens.modules.organizations.router import router as organizations_router
from medlens.modules.products.router import claims_router as product_claims_router
from medlens.modules.products.router import reference_images_router
from medlens.modules.products.router import router as products_router
from medlens.modules.rag.router import router as assistant_router
from medlens.modules.rbac.router import router as admin_router
from medlens.modules.reports.router import manufacturer_router, stats_router
from medlens.modules.scans.router import router as scans_router
from medlens.modules.users.router import router as users_router
from medlens.queue.backpressure import QUEUES, SCANS_QUEUE, QueueDepth
from medlens.resources import open_resources

# Routers under /api/v1, each built with core.routing.api_v1_router() so it carries its full path.
API_V1_ROUTERS: tuple[APIRouter, ...] = (
    auth_router,
    api_keys_router,
    users_router,
    organizations_router,
    invitations_router,
    products_router,
    reference_images_router,
    product_claims_router,
    scans_router,
    cases_router,
    hotspots_router,
    stats_router,
    manufacturer_router,
    knowledge_router,
    assistant_router,
    admin_router,
    ai_admin_router,
    failed_tasks_router,
    audit_router,
)

log = structlog.stdlib.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    interval = settings.observability.dependency_probe_interval_seconds
    async with open_resources(settings) as resources:
        app.state.resources = resources
        monitor = (
            start_monitor(
                readiness_service(resources, settings),
                interval_seconds=interval,
                queues=[resources.scan_queue]
                + [
                    QueueDepth(resources.broker_redis, name)
                    for name in QUEUES
                    if name != SCANS_QUEUE
                ],
            )
            if interval
            else None
        )
        log.info("app.started", environment=settings.app.environment, version=__version__)
        try:
            yield
        finally:
            log.info("app.stopping")
            if monitor is not None:
                await stop_monitor(monitor)
            del app.state.resources
            telemetry.flush()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_process(settings, component="api")

    show_docs = settings.app.show_docs
    app = MedLensAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=__version__,
        openapi_tags=API_TAGS,
        generate_unique_id_function=generate_operation_id,
        docs_url="/docs" if show_docs else None,
        redoc_url="/redoc" if show_docs else None,
        openapi_url="/openapi.json" if show_docs else None,
        lifespan=lifespan,
        middleware=_middleware(settings),
    )
    app.state.settings = settings
    register_exception_handlers(app)
    telemetry.instrument_app(app)
    # HTTP request metrics (served by the /metrics route below), by route template.
    Instrumentator(
        excluded_handlers=["/metrics", "/health/.*"], should_group_status_codes=False
    ).add(_http_metrics()).instrument(app)

    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(well_known_router)
    for router in API_V1_ROUTERS:
        app.include_router(router)
    return app


@cache
def _http_metrics() -> Callable[[Info], None]:
    """Created once per process: the instrumentator skips metrics that already exist in the
    registry, so a second app in the same process (tests) would otherwise record nothing."""
    instrumentation = metrics.default()
    if instrumentation is None:  # only if something else registered the same metrics first
        raise RuntimeError("HTTP metrics are already registered in this process")
    return instrumentation


def _middleware(settings: Settings) -> list[Middleware]:
    """Outermost first. Errors raised inside are still answered with problem+json, a request id
    and (when configured) CORS headers, because those layers wrap the error handling."""
    stack = [Middleware(SecurityHeadersMiddleware), Middleware(RequestContextMiddleware)]
    if settings.app.cors_origins:
        stack.append(
            Middleware(
                CORSMiddleware,
                allow_origins=settings.app.cors_origins,
                allow_credentials=True,
                allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                allow_headers=[
                    "Authorization",
                    "Content-Type",
                    "Idempotency-Key",
                    "X-API-Key",
                    DEVICE_HEADER,
                    REQUEST_ID_HEADER,
                ],
                expose_headers=[REQUEST_ID_HEADER, "Retry-After"],
                max_age=600,
            )
        )
    stack += [
        Middleware(AccessLogMiddleware),
        Middleware(UnhandledErrorMiddleware),
        Middleware(BodySizeLimitMiddleware, max_body_bytes=settings.app.max_request_body_bytes),
    ]
    return stack

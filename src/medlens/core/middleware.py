"""Pure ASGI middleware.

Deliberately not ``BaseHTTPMiddleware``: it runs the downstream app in a separate task (so context
variables set there are invisible to the middleware) and re-wraps every response body as a stream.

Order, outermost first (see ``medlens.main``): SecurityHeaders → RequestContext → CORS → AccessLog
→ UnhandledError → BodySizeLimit → FastAPI exception handlers → router.
"""

import time
import uuid

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from medlens.core.context import request_scope
from medlens.core.errors import PayloadTooLarge, domain_error_response, problem_response
from medlens.core.ids import uuid7

REQUEST_ID_HEADER = "X-Request-ID"
_MAX_REQUEST_ID_LENGTH = 64

log = structlog.stdlib.get_logger(__name__)


def _parse_request_id(raw: str | None) -> str | None:
    """Accept a client-supplied request id only if it is a UUID; return it in canonical form."""
    if raw is None or len(raw) > _MAX_REQUEST_ID_LENGTH:
        return None
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return None


def route_template(scope: Scope) -> str:
    """Matched route path (``/scans/{scan_id}``), not the raw path: bounded cardinality, no IDs."""
    path = getattr(scope.get("route"), "path", None)
    return path if isinstance(path, str) else "<unmatched>"


# On every response, errors included. HSTS is ignored by browsers over plain HTTP, so it is safe
# in development; TLS ends at the load balancer in deployments.
SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), geolocation=(), microphone=()",
}
# API responses carry personal and tenant data: never stored by browsers or shared caches.
_NO_STORE_PREFIX = "/api/"


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        no_store = str(scope.get("path", "")).startswith(_NO_STORE_PREFIX)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers.setdefault(name, value)
                if no_store:
                    headers.setdefault("Cache-Control", "no-store")
            await send(message)

        await self.app(scope, receive, send_with_headers)


class RequestContextMiddleware:
    """Assigns the request id, binds the per-request context and echoes ``X-Request-ID``."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _parse_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER))
        request_id = request_id or str(uuid7())

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        with request_scope(request_id):
            await self.app(scope, receive, send_with_request_id)


class AccessLogMiddleware:
    """Times each request and writes one structured access-log line."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_code = 500  # reported if the app dies before sending a response

        async def send_and_capture(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_and_capture)
        finally:
            route = route_template(scope)
            # Probes and scrapes arrive every few seconds; keep them out of INFO logs.
            emit = log.debug if route.startswith("/health/") or route == "/metrics" else log.info
            emit(
                "http.request",
                method=scope["method"],
                route=route,
                status=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )


class UnhandledErrorMiddleware:
    """Last line of defence: log unexpected exceptions with a stack trace, answer a generic 500."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def send_and_track(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_and_track)
        except Exception:
            log.exception(
                "http.unhandled_exception", method=scope["method"], route=route_template(scope)
            )
            if response_started:
                raise  # too late for a clean response; let the server drop the connection
            response = problem_response(
                status=500,
                title="Internal Server Error",
                detail="An unexpected error occurred. Quote the request_id when reporting it.",
                instance=scope["path"],
            )
            await response(scope, receive, send)


class RequestBodyTooLarge(Exception):
    """Raised from ``receive`` when the body exceeds the limit."""


class BodySizeLimitMiddleware:
    """Rejects bodies above ``max_body_bytes`` with 413, whether or not Content-Length is sent."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            await self._reject(scope, receive, send)
            return

        received = 0
        exceeded = False
        response_started = False

        async def receive_limited() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    exceeded = True
                    raise RequestBodyTooLarge
            return message

        async def send_unless_exceeded(message: Message) -> None:
            nonlocal response_started
            if exceeded:
                # FastAPI turns body-read failures into a 400; we answer with 413 instead.
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive_limited, send_unless_exceeded)
        except Exception:
            if not exceeded or response_started:
                raise
        if exceeded and not response_started:
            await self._reject(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        error = PayloadTooLarge(
            f"The request body exceeds the limit of {self.max_body_bytes} bytes."
        )
        await domain_error_response(error, instance=scope["path"])(scope, receive, send)

"""Domain errors and their RFC 9457 ``application/problem+json`` responses.

Services raise ``DomainError`` subclasses; the handlers registered here turn them, Starlette HTTP
errors and request-validation errors into problem details. Unexpected exceptions are handled by
``UnhandledErrorMiddleware`` (core/middleware.py) so they never leak stack traces.
"""

from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import Any, ClassVar, cast

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from starlette.exceptions import HTTPException as StarletteHTTPException

from medlens.core.context import get_request_id

PROBLEM_CONTENT_TYPE = "application/problem+json"
PROBLEM_TYPE_PREFIX = "urn:medlens:problem:"

log = structlog.stdlib.get_logger(__name__)


class ProblemDetail(BaseModel):
    """RFC 9457 problem details. Extension members (such as ``errors``) are allowed."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(
        default="about:blank",
        description="URI identifying the problem type; stable, so clients can branch on it.",
    )
    title: str = Field(description="Short summary of the problem type.")
    status: int = Field(description="HTTP status code.")
    detail: str | None = Field(default=None, description="Explanation of this occurrence.")
    instance: str | None = Field(default=None, description="Request path that failed.")
    request_id: str | None = Field(default=None, description="Quote this when reporting issues.")


class FieldError(BaseModel):
    """One invalid field. Never echoes the rejected value."""

    model_config = ConfigDict(extra="allow")

    loc: list[str | int] = Field(description='Path to the field, e.g. ["query", "limit"].')
    msg: str = Field(description="What is wrong with it.")
    type: str = Field(description="Machine-readable error type, e.g. int_parsing.")


class ValidationProblemDetail(ProblemDetail):
    """Problem details for 422 responses, listing every invalid field."""

    errors: list[FieldError] = Field(default_factory=list)


def problem_response(
    *,
    status: int,
    title: str,
    type_: str = "about:blank",
    detail: str | None = None,
    instance: str | None = None,
    headers: Mapping[str, str] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> JSONResponse:
    problem = ProblemDetail.model_validate(
        {
            "type": type_,
            "title": title,
            "status": status,
            "detail": detail,
            "instance": instance,
            "request_id": get_request_id(),
            **(extensions or {}),
        }
    )
    return JSONResponse(
        problem.model_dump(mode="json", exclude_none=True),
        status_code=int(status),  # plain int, not HTTPStatus, so logs show 404 not an enum repr
        headers={"Cache-Control": "no-store", **(headers or {})},
        media_type=PROBLEM_CONTENT_TYPE,
    )


class DomainError(Exception):
    """Base class for expected errors. Subclasses set the status, title and type slug."""

    status_code: ClassVar[int] = HTTPStatus.BAD_REQUEST
    title: ClassVar[str] = "Bad request"
    slug: ClassVar[str] = "bad-request"

    def __init__(self, detail: str | None = None, *, headers: Mapping[str, str] | None = None):
        super().__init__(detail or self.title)
        self.detail = detail
        self.headers: dict[str, str] = dict(headers or {})

    @property
    def type_uri(self) -> str:
        return f"{PROBLEM_TYPE_PREFIX}{self.slug}"

    def extensions(self) -> dict[str, Any]:
        return {}


class NotFound(DomainError):
    status_code = HTTPStatus.NOT_FOUND
    title = "Resource not found"
    slug = "not-found"


class Conflict(DomainError):
    status_code = HTTPStatus.CONFLICT
    title = "Conflict with the current state of the resource"
    slug = "conflict"


class IdempotencyKeyReused(Conflict):
    title = "Idempotency key reused"
    slug = "idempotency-key-reused"


class Forbidden(DomainError):
    status_code = HTTPStatus.FORBIDDEN
    title = "Forbidden"
    slug = "forbidden"


class Unauthorized(DomainError):
    status_code = HTTPStatus.UNAUTHORIZED
    title = "Authentication required"
    slug = "unauthorized"

    def __init__(self, detail: str | None = None):
        super().__init__(detail, headers={"WWW-Authenticate": "Bearer"})


class ValidationFailed(DomainError):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    title = "Validation failed"
    slug = "validation-failed"

    def __init__(self, detail: str | None = None, *, errors: Sequence[Mapping[str, Any]] = ()):
        """``errors`` items should follow FieldError: {"loc": [...], "msg": ..., "type": ...}."""
        super().__init__(detail)
        self.errors = [dict(error) for error in errors]

    def extensions(self) -> dict[str, Any]:
        return {"errors": self.errors} if self.errors else {}


class PayloadTooLarge(DomainError):
    status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    title = "Content too large"
    slug = "payload-too-large"


class UnsupportedMediaType(DomainError):
    status_code = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    title = "Unsupported media type"
    slug = "unsupported-media-type"


class RateLimited(DomainError):
    status_code = HTTPStatus.TOO_MANY_REQUESTS
    title = "Too many requests"
    slug = "rate-limited"

    def __init__(self, detail: str | None = None, *, retry_after_seconds: int | None = None):
        headers = {"Retry-After": str(retry_after_seconds)} if retry_after_seconds else None
        super().__init__(detail, headers=headers)


class DependencyUnavailable(DomainError):
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    title = "Service temporarily unavailable"
    slug = "dependency-unavailable"

    def __init__(
        self,
        detail: str | None = None,
        *,
        dependency: str,
        retry_after_seconds: int | None = None,
    ):
        headers = {"Retry-After": str(retry_after_seconds)} if retry_after_seconds else None
        super().__init__(detail, headers=headers)
        # Logged, never returned: clients do not need to know our topology.
        self.dependency = dependency


def domain_error_response(error: DomainError, *, instance: str | None) -> JSONResponse:
    return problem_response(
        status=error.status_code,
        title=error.title,
        type_=error.type_uri,
        detail=error.detail,
        instance=instance,
        headers=error.headers,
        extensions=error.extensions(),
    )


async def _handle_domain_error(request: Request, exc: Exception) -> Response:
    error = cast(DomainError, exc)
    if isinstance(error, DependencyUnavailable):
        log.warning("http.dependency_unavailable", dependency=error.dependency)
    return domain_error_response(error, instance=request.url.path)


async def _handle_http_exception(request: Request, exc: Exception) -> Response:
    error = cast(StarletteHTTPException, exc)
    try:
        title = HTTPStatus(error.status_code).phrase
    except ValueError:
        title = "Error"
    # Starlette's default detail is the status phrase itself; 5xx details may be internal.
    detail = error.detail if error.detail != title and error.status_code < 500 else None
    return problem_response(
        status=error.status_code,
        title=title,
        detail=detail,
        instance=request.url.path,
        headers=error.headers,
    )


async def _handle_request_validation(request: Request, exc: Exception) -> Response:
    error = cast(RequestValidationError, exc)
    # Only location, message and type: echoing the rejected input could leak PII or secrets.
    errors = [
        {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]
    return problem_response(
        status=HTTPStatus.UNPROCESSABLE_ENTITY,
        title=ValidationFailed.title,
        type_=f"{PROBLEM_TYPE_PREFIX}{ValidationFailed.slug}",
        detail="One or more request fields are invalid.",
        instance=request.url.path,
        extensions={"errors": errors},
    )


# A request that waited MEDLENS_DATABASE__POOL_TIMEOUT_SECONDS for a free connection: the
# container is overloaded. 503 with Retry-After sheds load (clients and the load balancer back
# off) instead of a 500 that looks like a bug.
DB_POOL_EXHAUSTED = Counter(
    "medlens_db_pool_exhausted",
    "Requests refused with 503 because no database connection was free in time.",
)
POOL_RETRY_AFTER_SECONDS = 2


async def _handle_pool_timeout(request: Request, _exc: Exception) -> Response:
    DB_POOL_EXHAUSTED.inc()
    log.warning("http.db_pool_exhausted", path=request.url.path)
    error = DependencyUnavailable(
        "The service is busy. Try again shortly.",
        dependency="database",
        retry_after_seconds=POOL_RETRY_AFTER_SECONDS,
    )
    return domain_error_response(error, instance=request.url.path)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, _handle_domain_error)
    app.add_exception_handler(PoolTimeoutError, _handle_pool_timeout)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(RequestValidationError, _handle_request_validation)

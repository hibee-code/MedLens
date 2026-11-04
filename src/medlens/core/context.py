"""Per-request context carried in contextvars, never in globals.

The request middleware opens a fresh scope for every request; auth (Phase 3) fills in the user and
organization. Log processors and Celery task headers read the current values via ``snapshot()``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import UUID

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
user_id_var: ContextVar[UUID | None] = ContextVar("user_id", default=None)
org_id_var: ContextVar[UUID | None] = ContextVar("org_id", default=None)


def get_request_id() -> str | None:
    return request_id_var.get()


@contextmanager
def request_scope(request_id: str, *, trace_id: str | None = None) -> Iterator[None]:
    """Bind a fresh context for one request. User and organization start empty."""
    request_token = request_id_var.set(request_id)
    trace_token = trace_id_var.set(trace_id)
    user_token = user_id_var.set(None)
    org_token = org_id_var.set(None)
    try:
        yield
    finally:
        org_id_var.reset(org_token)
        user_id_var.reset(user_token)
        trace_id_var.reset(trace_token)
        request_id_var.reset(request_token)


def snapshot() -> dict[str, str]:
    """Current non-empty context values as strings, for log lines and task headers."""
    values = {
        "request_id": request_id_var.get(),
        "trace_id": trace_id_var.get(),
        "user_id": user_id_var.get(),
        "org_id": org_id_var.get(),
    }
    return {key: str(value) for key, value in values.items() if value is not None}

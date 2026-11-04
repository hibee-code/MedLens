"""Request context (request id, trace id, user, organization) travels into tasks.

Producers add ``core.context.snapshot()`` to every message as the ``medlens_context`` header
(``before_task_publish``). Events relayed from the outbox carry the context of the request that
created them, stored with the event. Workers restore it into the contextvars before the task
runs (``task_prerun``), so log lines in the worker carry the original request id, and clear it
afterwards (``task_postrun``).
"""

from contextvars import Token
from typing import Any
from uuid import UUID

from celery import Task
from celery.signals import before_task_publish, task_postrun, task_prerun

from medlens.core.context import org_id_var, request_id_var, snapshot, trace_id_var, user_id_var

CONTEXT_HEADER = "medlens_context"

_tokens: dict[str, list[Token[Any]]] = {}


def context_headers(context: dict[str, str] | None = None) -> dict[str, dict[str, str]]:
    """Message headers for ``apply_async(headers=...)``: the given or the current context."""
    return {CONTEXT_HEADER: dict(context if context is not None else snapshot())}


def _add_context(headers: dict[str, Any] | None = None, **_: Any) -> None:
    if headers is not None and CONTEXT_HEADER not in headers:
        current = snapshot()
        if current:
            headers[CONTEXT_HEADER] = current


def task_context(task: Task) -> dict[str, str]:
    request = task.request
    context = getattr(request, CONTEXT_HEADER, None)
    if context is None:
        context = (getattr(request, "headers", None) or {}).get(CONTEXT_HEADER)
    return dict(context) if isinstance(context, dict) else {}


def _uuid(value: str | None) -> UUID | None:
    try:
        return UUID(value) if value else None
    except ValueError:
        return None


def _restore_context(task_id: str | None = None, task: Task | None = None, **_: Any) -> None:
    if task is None or task_id is None:
        return
    context = task_context(task)
    _tokens[task_id] = [
        request_id_var.set(context.get("request_id") or f"task-{task_id}"),
        trace_id_var.set(context.get("trace_id")),
        user_id_var.set(_uuid(context.get("user_id"))),
        org_id_var.set(_uuid(context.get("org_id"))),
    ]


def _clear_context(task_id: str | None = None, **_: Any) -> None:
    tokens = _tokens.pop(task_id or "", None)
    if tokens is None:
        return
    request_token, trace_token, user_token, org_token = tokens
    org_id_var.reset(org_token)
    user_id_var.reset(user_token)
    trace_id_var.reset(trace_token)
    request_id_var.reset(request_token)


# Celery's signal decorators are untyped; connect explicitly (weak=False: module-level handlers).
before_task_publish.connect(_add_context, weak=False)
task_prerun.connect(_restore_context, weak=False)
task_postrun.connect(_clear_context, weak=False)

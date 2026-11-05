"""Dead letters: tasks that finally failed (after their retries) are recorded in
``failed_tasks``, one row per task and arguments, so a platform admin can inspect and requeue
them (``/api/v1/admin/failed-tasks``).

Task arguments are ids and settings, never personal data; error messages are truncated and the
traceback is kept only as a hash (it may contain values)."""

import hashlib
import json
import traceback as tb
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

import structlog
from celery.signals import task_failure
from prometheus_client import Counter
from sqlalchemy.dialects.postgresql import insert

from medlens.db.models import FailedTask
from medlens.workers.runtime import get_runtime

log = structlog.stdlib.get_logger(__name__)
TASKS_FAILED = Counter(
    "medlens_tasks_failed", "Tasks that failed for good (retries exhausted or a bug).", ["task"]
)

ERROR_MAX_LENGTH = 500


def args_hash(task_name: str, args: Any, kwargs: Any) -> str:
    canonical = json.dumps([task_name, args or [], kwargs or {}], sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def traceback_hash(trace: TracebackType | None) -> str | None:
    if trace is None:
        return None
    frames = "".join(tb.format_tb(trace))
    return hashlib.sha256(frames.encode()).hexdigest()


def record_failure(
    task_name: str,
    task_id: str | None,
    args: Any,
    kwargs: Any,
    exception: BaseException,
    trace: TracebackType | None,
) -> None:
    now = datetime.now(UTC)
    values = {
        "task_name": task_name,
        "task_id": task_id,
        "args": list(args or []),
        "kwargs": dict(kwargs or {}),
        "args_hash": args_hash(task_name, args, kwargs),
        "error_type": type(exception).__name__,
        "error_message": str(exception)[:ERROR_MAX_LENGTH],
        "traceback_hash": traceback_hash(trace),
        "first_failed_at": now,
        "last_failed_at": now,
    }
    statement = insert(FailedTask).values(**values)
    statement = statement.on_conflict_do_update(
        index_elements=[FailedTask.task_name, FailedTask.args_hash],
        set_={
            "task_id": statement.excluded.task_id,
            "error_type": statement.excluded.error_type,
            "error_message": statement.excluded.error_message,
            "traceback_hash": statement.excluded.traceback_hash,
            "last_failed_at": now,
            "failure_count": FailedTask.failure_count + 1,
            "updated_at": now,
        },
    )
    with get_runtime().sessions() as session, session.begin():
        session.execute(statement)
    log.error(
        "task.failed",
        task=task_name,
        task_id=task_id,
        error=type(exception).__name__,
    )


def _on_task_failure(
    sender: Any = None,
    task_id: str | None = None,
    exception: BaseException | None = None,
    args: Any = None,
    kwargs: Any = None,
    traceback: TracebackType | None = None,
    **_: Any,
) -> None:
    if sender is None or exception is None:
        return
    TASKS_FAILED.labels(task=sender.name).inc()
    try:
        record_failure(sender.name, task_id, args, kwargs, exception, traceback)
    except Exception as exc:  # recording must never mask the original failure
        log.exception("task.failure_not_recorded", task=sender.name, error=type(exc).__name__)


task_failure.connect(_on_task_failure, weak=False)

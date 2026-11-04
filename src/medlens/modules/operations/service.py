"""Dead-lettered tasks: listing them and sending them again (platform admins)."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import ColumnElement, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.concurrency import run_in_threadpool

from medlens.core.errors import NotFound
from medlens.core.pagination import PageParams
from medlens.core.request_meta import RequestMeta
from medlens.db.models import FailedTask
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.rbac.access import Actor

# (task name, args, kwargs) -> publishes the task; blocking (kombu), so run in a thread.
SendTask = Callable[[str, list[Any], dict[str, Any]], None]


class FailedTaskService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], send: SendTask) -> None:
        self._sessions = sessions
        self._send = send

    async def list_page(
        self, params: PageParams, *filters: ColumnElement[bool]
    ) -> list[FailedTask]:
        statement = select(FailedTask).where(*filters)
        if params.after is not None:
            position = tuple_(params.after.created_at, params.after.id)
            statement = statement.where(tuple_(FailedTask.created_at, FailedTask.id) < position)
        statement = statement.order_by(FailedTask.created_at.desc(), FailedTask.id.desc()).limit(
            params.limit + 1
        )
        async with UnitOfWork(self._sessions) as uow:
            return list(await uow.session.scalars(statement))

    async def requeue(self, actor: Actor, failed_id: UUID, meta: RequestMeta) -> FailedTask:
        async with UnitOfWork(self._sessions) as uow:
            failed = await uow.session.get(FailedTask, failed_id, with_for_update=True)
            if failed is None:
                raise NotFound("Failed task not found.")
            await run_in_threadpool(self._send, failed.task_name, failed.args, failed.kwargs)
            failed.requeued_at = datetime.now(UTC)
            failed.requeue_count += 1
            await AuditService(uow.session).record(
                "task.requeued",
                actor_user_id=actor.user_id,
                resource_type="failed_task",
                resource_id=failed.id,
                meta=meta,
                details={"task_name": failed.task_name},
            )
            await uow.commit()
            return failed

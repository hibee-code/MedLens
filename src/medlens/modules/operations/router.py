from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, Query
from sqlalchemy import ColumnElement

from medlens.core.openapi import problem_responses
from medlens.core.pagination import Cursor, Page, PageParams, build_page, page_params
from medlens.core.routing import api_v1_router
from medlens.db.models import FailedTask
from medlens.modules.auth.deps import Meta as RequestMetaDep
from medlens.modules.operations.schemas import FailedTaskRead
from medlens.modules.operations.service import FailedTaskService
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.queue.celery_app import celery_app
from medlens.resources import Resources, get_resources
from medlens.workers.context import context_headers

router = api_v1_router("/admin/failed-tasks", tags=["admin"])
Admin = Annotated[Actor, Depends(require_permissions(Permission.PLATFORM_ADMIN, fresh=True))]


def _send(task_name: str, args: list[Any], kwargs: dict[str, Any]) -> None:
    celery_app.send_task(task_name, args=args, kwargs=kwargs, headers=context_headers())


def get_failed_task_service(
    resources: Annotated[Resources, Depends(get_resources)],
) -> FailedTaskService:
    return FailedTaskService(resources.session_factory, _send)


ServiceDep = Annotated[FailedTaskService, Depends(get_failed_task_service)]


@router.get(
    "",
    response_model=Page[FailedTaskRead],
    summary="List dead-lettered tasks",
    description=(
        "Background tasks that failed after all their retries, newest first, one entry per "
        "task and arguments. Platform admins only."
    ),
    responses=problem_responses(401, 403, 422),
    dependencies=[Depends(require_permissions(Permission.PLATFORM_ADMIN))],
)
async def list_failed_tasks(
    service: ServiceDep,
    params: Annotated[PageParams, Depends(page_params)],
    task_name: Annotated[str | None, Query(max_length=200)] = None,
    pending: Annotated[bool, Query(description="Only tasks not requeued since failing.")] = False,
) -> Page[FailedTaskRead]:
    filters: list[ColumnElement[bool]] = []
    if task_name is not None:
        filters.append(FailedTask.task_name == task_name)
    if pending:
        filters.append(
            FailedTask.requeued_at.is_(None) | (FailedTask.requeued_at < FailedTask.last_failed_at)
        )
    rows = await service.list_page(params, *filters)
    return build_page(
        rows,
        limit=params.limit,
        cursor_of=lambda row: Cursor(created_at=row.created_at, id=row.id),
        to_item=FailedTaskRead.model_validate,
    )


@router.post(
    "/{failed_task_id}/requeue",
    response_model=FailedTaskRead,
    summary="Run a dead-lettered task again",
    description=(
        "Sends the task again with its original arguments. Tasks are idempotent, so requeueing "
        "one that later succeeded is harmless."
    ),
    responses=problem_responses(401, 403, 404, 503),
)
async def requeue_failed_task(
    failed_task_id: UUID, actor: Admin, service: ServiceDep, meta: RequestMetaDep
) -> FailedTaskRead:
    return FailedTaskRead.model_validate(await service.requeue(actor, failed_task_id, meta))

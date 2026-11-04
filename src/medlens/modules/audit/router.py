from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, Query
from pydantic import ConfigDict

from medlens.core.openapi import problem_responses
from medlens.core.pagination import Cursor, Page, PageParams, build_page, page_params
from medlens.core.routing import api_v1_router
from medlens.core.schemas import ReadModel
from medlens.db.session import UnitOfWork
from medlens.modules.audit.models import AuditLog
from medlens.modules.audit.service import AuditService
from medlens.modules.rbac.deps import require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.resources import Resources, get_resources

router = api_v1_router("/admin/audit-logs", tags=["admin"])


class AuditLogRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e40",
                    "created_at": "2026-11-03T09:00:00Z",
                    "action": "auth.login_failed",
                    "actor_user_id": None,
                    "actor_org_id": None,
                    "resource_type": None,
                    "resource_id": None,
                    "request_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e41",
                    "metadata": {"method": "password"},
                }
            ]
        }
    )

    id: UUID
    created_at: datetime
    action: str
    actor_user_id: UUID | None
    actor_org_id: UUID | None
    resource_type: str | None
    resource_id: str | None
    request_id: str | None
    metadata: dict[str, Any]


def _read(entry: AuditLog) -> AuditLogRead:
    return AuditLogRead(
        id=entry.id,
        created_at=entry.created_at,
        action=entry.action,
        actor_user_id=entry.actor_user_id,
        actor_org_id=entry.actor_org_id,
        resource_type=entry.resource_type,
        resource_id=entry.resource_id,
        request_id=entry.request_id,
        metadata=entry.metadata_,
    )


@router.get(
    "",
    response_model=Page[AuditLogRead],
    summary="Read the security audit log",
    description="Security events, newest first, optionally filtered by action.",
    responses=problem_responses(401, 403, 422),
    dependencies=[Depends(require_permissions(Permission.AUDIT_READ))],
)
async def list_audit_logs(
    resources: Annotated[Resources, Depends(get_resources)],
    params: Annotated[PageParams, Depends(page_params)],
    action: Annotated[
        str | None, Query(max_length=64, description="e.g. auth.login_failed")
    ] = None,
) -> Page[AuditLogRead]:
    async with UnitOfWork(resources.session_factory) as uow:
        rows = await AuditService(uow.session).list_page(params, action=action)
    return build_page(
        rows,
        limit=params.limit,
        cursor_of=lambda row: Cursor(created_at=row.created_at, id=row.id),
        to_item=_read,
    )

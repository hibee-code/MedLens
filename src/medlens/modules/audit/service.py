"""Security audit trail. Events are added to the caller's transaction, so an event is recorded
exactly when the change it describes is committed. Failed attempts (bad password, reused
refresh token) are committed by the auth service before it raises, so they are never lost."""

from typing import Any
from uuid import UUID

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from medlens.core.context import get_request_id
from medlens.core.crypto import ip_hash
from medlens.core.pagination import PageParams
from medlens.core.request_meta import RequestMeta
from medlens.modules.audit.models import AuditLog


class AuditService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        action: str,
        *,
        actor_user_id: UUID | None = None,
        actor_org_id: UUID | None = None,
        resource_type: str | None = None,
        resource_id: object = None,
        meta: RequestMeta | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._session.add(
            AuditLog(
                action=action,
                actor_user_id=actor_user_id,
                actor_org_id=actor_org_id,
                resource_type=resource_type,
                resource_id=str(resource_id) if resource_id is not None else None,
                ip_hash=ip_hash(meta.ip) if meta and meta.ip else None,
                user_agent=meta.user_agent if meta else None,
                request_id=get_request_id(),
                metadata_=details or {},
            )
        )
        await self._session.flush()

    async def list_page(self, params: PageParams, *, action: str | None = None) -> list[AuditLog]:
        """Newest first, keyset-paginated (audit logs have no updated_at, so no Repository)."""
        statement = select(AuditLog)
        if action:
            statement = statement.where(AuditLog.action == action)
        if params.after is not None:
            position = tuple_(params.after.created_at, params.after.id)
            statement = statement.where(tuple_(AuditLog.created_at, AuditLog.id) < position)
        statement = statement.order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
        return list(await self._session.scalars(statement.limit(params.limit + 1)))

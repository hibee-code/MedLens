"""Branches: an organization's premises, with an exact location (a business address, not
personal data). Members read them; administrators (``org:manage``) change them."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from geoalchemy2 import WKTElement
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.errors import NotFound
from medlens.core.pagination import PageParams
from medlens.core.request_meta import RequestMeta
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.organizations.models import Branch
from medlens.modules.organizations.repository import BranchRepository
from medlens.modules.organizations.schemas import BranchCreate
from medlens.modules.rbac import policies
from medlens.modules.rbac.access import Actor

_ORG_NOT_FOUND = "Organization not found."
_NOT_FOUND = "Branch not found."


def _point(latitude: float | None, longitude: float | None) -> WKTElement | None:
    """Geography columns are bound through ST_GeogFromText, so points are written as text."""
    if latitude is None or longitude is None:
        return None
    return WKTElement(f"POINT({longitude} {latitude})", srid=4326)


class BranchService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def list_page(
        self, actor: Actor, organization_id: UUID, params: PageParams
    ) -> list[Branch]:
        if not policies.can_read_organization(actor, organization_id):
            raise NotFound(_ORG_NOT_FOUND)
        async with UnitOfWork(self._sessions) as uow:
            repo = BranchRepository(uow.session)
            return await repo.list_page(
                actor.tenant, params, Branch.organization_id == organization_id
            )

    async def create(
        self, actor: Actor, organization_id: UUID, body: BranchCreate, meta: RequestMeta
    ) -> Branch:
        if not policies.can_manage_organization(actor, organization_id):
            raise NotFound(_ORG_NOT_FOUND)
        async with UnitOfWork(self._sessions) as uow:
            branch = Branch(
                organization_id=organization_id,
                name=body.name,
                address=body.address,
                state=body.state,
                location=_point(body.latitude, body.longitude),
            )
            uow.session.add(branch)
            await uow.session.flush()
            await self._audit(uow.session, "branch.created", actor, branch, meta)
            await uow.commit()
            return branch

    async def update(
        self,
        actor: Actor,
        organization_id: UUID,
        branch_id: UUID,
        changes: dict[str, Any],
        meta: RequestMeta,
    ) -> Branch:
        async with UnitOfWork(self._sessions) as uow:
            branch = await self._writable(uow.session, actor, organization_id, branch_id)
            if "latitude" in changes:
                branch.location = _point(changes.pop("latitude"), changes.pop("longitude"))
            for name, value in changes.items():
                setattr(branch, name, value)
            await self._audit(uow.session, "branch.updated", actor, branch, meta)
            await uow.commit()
            return branch

    async def delete(
        self, actor: Actor, organization_id: UUID, branch_id: UUID, meta: RequestMeta
    ) -> None:
        async with UnitOfWork(self._sessions) as uow:
            branch = await self._writable(uow.session, actor, organization_id, branch_id)
            branch.deleted_at = datetime.now(UTC)
            await self._audit(uow.session, "branch.deleted", actor, branch, meta)
            await uow.commit()

    @staticmethod
    async def _writable(
        session: AsyncSession, actor: Actor, organization_id: UUID, branch_id: UUID
    ) -> Branch:
        if not policies.can_manage_organization(actor, organization_id):
            raise NotFound(_ORG_NOT_FOUND)
        branch = await BranchRepository(session).get(actor.tenant, branch_id)
        if branch is None or branch.organization_id != organization_id:
            raise NotFound(_NOT_FOUND)
        return branch

    @staticmethod
    async def _audit(
        session: AsyncSession, action: str, actor: Actor, branch: Branch, meta: RequestMeta
    ) -> None:
        await AuditService(session).record(
            action,
            actor_user_id=actor.user_id,
            actor_org_id=branch.organization_id,
            resource_type="branch",
            resource_id=branch.id,
            meta=meta,
        )

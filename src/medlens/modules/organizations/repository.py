from typing import Any
from uuid import UUID

from sqlalchemy import ColumnElement, Select, delete, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from medlens.core.pagination import PageParams
from medlens.db.repository import OrganizationScopedRepository
from medlens.modules.organizations.models import (
    Branch,
    Invitation,
    Membership,
    MembershipStatus,
    Organization,
)
from medlens.modules.rbac.permissions import ORGANIZATION_ADMIN_ROLES
from medlens.modules.users.models import User

_ADMIN_ROLES = sorted(role.value for role in ORGANIZATION_ADMIN_ROLES)


class BranchRepository(OrganizationScopedRepository[Branch]):
    model = Branch


class MembershipRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def organization(self, organization_id: UUID) -> Organization | None:
        statement = select(Organization).where(
            Organization.id == organization_id, Organization.deleted_at.is_(None)
        )
        return (await self.session.scalars(statement)).first()

    async def membership(
        self, organization_id: UUID, user_id: UUID, *, lock: bool = True
    ) -> Membership | None:
        statement = select(Membership).where(
            Membership.organization_id == organization_id, Membership.user_id == user_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self.session.scalars(statement)).first()

    async def organizations_page(
        self, params: PageParams, *filters: ColumnElement[bool]
    ) -> list[Organization]:
        statement = select(Organization).where(*filters)
        return await _keyset(self.session, statement, Organization, params)

    async def invitations_page(self, organization_id: UUID, params: PageParams) -> list[Invitation]:
        statement = select(Invitation).where(Invitation.organization_id == organization_id)
        return await _keyset(self.session, statement, Invitation, params)

    async def members_page(
        self, organization_id: UUID, params: PageParams
    ) -> list[tuple[Membership, User]]:
        statement = (
            select(Membership, User)
            .join(User, User.id == Membership.user_id)
            .where(Membership.organization_id == organization_id)
        )
        if params.after is not None:
            position = tuple_(params.after.created_at, params.after.id)
            statement = statement.where(tuple_(Membership.created_at, Membership.id) < position)
        statement = statement.order_by(Membership.created_at.desc(), Membership.id.desc())
        return [(m, u) for m, u in await self.session.execute(statement.limit(params.limit + 1))]

    async def active_admin_count(self, organization_id: UUID) -> int:
        """Locks the organization's admin memberships so concurrent demotions serialize."""
        rows = await self.session.scalars(
            select(Membership.id)
            .where(
                Membership.organization_id == organization_id,
                Membership.status == MembershipStatus.ACTIVE,
                Membership.role.in_(_ADMIN_ROLES),
            )
            .with_for_update()
        )
        return len(list(rows))

    async def organizations_where_last_admin(self, user_id: UUID) -> list[UUID]:
        mine = (
            select(Membership.organization_id)
            .where(
                Membership.user_id == user_id,
                Membership.status == MembershipStatus.ACTIVE,
                Membership.role.in_(_ADMIN_ROLES),
            )
            .scalar_subquery()
        )
        statement = (
            select(Membership.organization_id)
            .where(
                Membership.organization_id.in_(mine),
                Membership.status == MembershipStatus.ACTIVE,
                Membership.role.in_(_ADMIN_ROLES),
            )
            .group_by(Membership.organization_id)
            .having(func.count() == 1)
        )
        return list(await self.session.scalars(statement))

    async def delete_all_for_user(self, user_id: UUID) -> None:
        await self.session.execute(delete(Membership).where(Membership.user_id == user_id))


async def _keyset[RowT](
    session: AsyncSession, statement: Select[RowT], model: Any, params: PageParams
) -> list[RowT]:
    """Newest first by (created_at, id); up to ``limit + 1`` rows for build_page."""
    if params.after is not None:
        position = tuple_(params.after.created_at, params.after.id)
        statement = statement.where(tuple_(model.created_at, model.id) < position)
    statement = statement.order_by(model.created_at.desc(), model.id.desc()).limit(params.limit + 1)
    return list(await session.scalars(statement))

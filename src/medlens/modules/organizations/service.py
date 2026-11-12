"""Organization membership: listing members, assigning roles, removing members.

Rules:
* organization admins act only on their own organization; any other organization answers 404;
  platform admins act anywhere;
* the role must be an organization role valid for the organization's type, and the actor must
  hold every permission it grants (no escalation);
* adding someone who is not yet a member is reserved to platform admins (organization admins
  add people through invitations, Phase 5);
* the last active admin can be neither demoted nor removed;
* every change bumps the member's perm_ver after commit, so it applies on their next request.
"""

from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.errors import Conflict, NotFound, ValidationFailed
from medlens.core.pagination import PageParams
from medlens.core.request_meta import RequestMeta
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.organizations.models import Membership, MembershipStatus, Organization
from medlens.modules.organizations.repository import MembershipRepository
from medlens.modules.rbac import policies
from medlens.modules.rbac.access import Actor, PermissionVersions, bump_perm_ver
from medlens.modules.rbac.models import RoleName
from medlens.modules.rbac.permissions import ORGANIZATION_ADMIN_ROLES, ROLE_ORGANIZATION_TYPES
from medlens.modules.rbac.service import ensure_no_escalation, role_permissions
from medlens.modules.users.models import User, UserStatus

_NOT_FOUND = "Organization not found."


class MembershipService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], redis: Redis) -> None:
        self._sessions = sessions
        self._versions = PermissionVersions(redis)

    async def list_members(
        self, actor: Actor, organization_id: UUID, params: PageParams
    ) -> list[tuple[Membership, User]]:
        if not policies.can_read_organization(actor, organization_id):
            raise NotFound(_NOT_FOUND)
        async with UnitOfWork(self._sessions) as uow:
            repo = MembershipRepository(uow.session)
            if await repo.organization(organization_id) is None:
                raise NotFound(_NOT_FOUND)
            return await repo.members_page(organization_id, params)

    async def assign_role(
        self,
        actor: Actor,
        organization_id: UUID,
        user_id: UUID,
        role: RoleName,
        meta: RequestMeta,
    ) -> tuple[Membership, User]:
        if not policies.can_manage_members(actor, organization_id):
            raise NotFound(_NOT_FOUND)
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            repo = MembershipRepository(session)
            organization = await repo.organization(organization_id)
            if organization is None:
                raise NotFound(_NOT_FOUND)
            validate_role_for(organization, role)
            ensure_no_escalation(actor, await role_permissions(session, role.value))

            user = await session.get(User, user_id)
            if user is None or user.status is UserStatus.DELETED:
                raise NotFound("User not found.")
            membership = await repo.membership(organization_id, user_id)
            previous = membership.role if membership else None
            if membership is None:
                if not actor.is_platform_admin:
                    raise NotFound("This user is not a member of the organization.")
                membership = Membership(
                    organization_id=organization_id,
                    user_id=user_id,
                    role=role.value,
                    invited_by=actor.user_id,
                )
                session.add(membership)
            else:
                if _is_admin(membership) and role not in ORGANIZATION_ADMIN_ROLES:
                    await _keep_an_admin(repo, organization_id)
                membership.role = role.value
                membership.status = MembershipStatus.ACTIVE
            version = await bump_perm_ver(session, user_id)
            await AuditService(session).record(
                "member.role_assigned",
                actor_user_id=actor.user_id,
                actor_org_id=actor.organization.id if actor.organization else None,
                resource_type="membership",
                resource_id=user_id,
                meta=meta,
                details={
                    "organization_id": str(organization_id),
                    "from": previous,
                    "to": role.value,
                },
            )
            await uow.commit()
        await self._versions.remember(user_id, version)
        return membership, user

    async def remove_member(
        self, actor: Actor, organization_id: UUID, user_id: UUID, meta: RequestMeta
    ) -> None:
        if not policies.can_manage_members(actor, organization_id):
            raise NotFound(_NOT_FOUND)
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            repo = MembershipRepository(session)
            if await repo.organization(organization_id) is None:
                raise NotFound(_NOT_FOUND)
            membership = await repo.membership(organization_id, user_id)
            if membership is None:
                raise NotFound("This user is not a member of the organization.")
            if _is_admin(membership):
                await _keep_an_admin(repo, organization_id)
            await session.delete(membership)
            version = await bump_perm_ver(session, user_id)
            await AuditService(session).record(
                "member.removed",
                actor_user_id=actor.user_id,
                actor_org_id=actor.organization.id if actor.organization else None,
                resource_type="membership",
                resource_id=user_id,
                meta=meta,
                details={"organization_id": str(organization_id), "role": membership.role},
            )
            await uow.commit()
        await self._versions.remember(user_id, version)


async def organizations_where_last_admin(session: AsyncSession, user_id: UUID) -> list[UUID]:
    """For account erasure: organizations that would be left without an admin."""
    return await MembershipRepository(session).organizations_where_last_admin(user_id)


async def delete_memberships(session: AsyncSession, user_id: UUID) -> None:
    await MembershipRepository(session).delete_all_for_user(user_id)


def validate_role_for(organization: Organization, role: RoleName) -> None:
    allowed = ROLE_ORGANIZATION_TYPES.get(role)
    if allowed is None or organization.type not in allowed:
        message = (
            f"The role {role.value!r} cannot be held in a {organization.type.value} organization."
        )
        raise ValidationFailed(
            message, errors=[{"loc": ["body", "role"], "msg": message, "type": "role"}]
        )


def _is_admin(membership: Membership) -> bool:
    return membership.status is MembershipStatus.ACTIVE and membership.role in {
        role.value for role in ORGANIZATION_ADMIN_ROLES
    }


async def _keep_an_admin(repo: MembershipRepository, organization_id: UUID) -> None:
    if await repo.active_admin_count(organization_id) <= 1:
        raise Conflict(
            "An organization must keep at least one administrator. Appoint another one first."
        )

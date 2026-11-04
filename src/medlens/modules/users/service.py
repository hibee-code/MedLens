"""The signed-in user's own account: profile, memberships and erasure (NDPA right to erasure)."""

from dataclasses import dataclass
from typing import cast
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.errors import Conflict, Unauthorized
from medlens.core.request_meta import RequestMeta
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.auth.service import AuthService
from medlens.modules.organizations.models import Membership, Organization
from medlens.modules.organizations.service import delete_memberships, organizations_where_last_admin
from medlens.modules.rbac.access import PermissionVersions, bump_perm_ver
from medlens.modules.scans.service import detach_user
from medlens.modules.users.models import User, UserStatus


@dataclass(frozen=True, slots=True)
class MembershipView:
    organization_id: UUID
    organization_name: str
    organization_type: str
    role: str
    status: str


class UserService:
    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], redis: Redis, auth: AuthService
    ) -> None:
        self._sessions = sessions
        self._versions = PermissionVersions(redis)
        self._auth = auth

    async def profile(self, user_id: UUID) -> tuple[User, list[MembershipView]]:
        async with UnitOfWork(self._sessions) as uow:
            user = await _active_user(uow.session, user_id)
            return user, await _memberships(uow.session, user_id)

    async def update_profile(
        self, user_id: UUID, changes: dict[str, object], meta: RequestMeta
    ) -> tuple[User, list[MembershipView]]:
        async with UnitOfWork(self._sessions) as uow:
            user = await _active_user(uow.session, user_id)
            if "full_name" in changes:
                user.full_name = cast("str | None", changes["full_name"])
            await AuditService(uow.session).record(
                "user.profile_updated",
                actor_user_id=user.id,
                meta=meta,
                details={"fields": sorted(changes)},
            )
            await uow.commit()
            return user, await _memberships(uow.session, user_id)

    async def erase(self, user_id: UUID, meta: RequestMeta) -> None:
        """Remove every piece of personal data. The row stays (status deleted) so scans and audit
        entries keep a stable, meaningless id; scans no longer reference the user at all."""
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            user = await _active_user(session, user_id, lock=True)
            if await organizations_where_last_admin(session, user_id):
                raise Conflict(
                    "You are the last administrator of an organization. Appoint another "
                    "administrator before deleting your account."
                )
            ended = await self._auth.end_sessions(session, user_id)
            await delete_memberships(session, user_id)
            await detach_user(session, user_id)
            user.set_phone(None)
            user.email = None
            user.password_hash = None
            user.full_name = None
            user.mfa_secret = None
            user.mfa_enabled = False
            user.last_login_at = None
            user.status = UserStatus.DELETED
            version = await bump_perm_ver(session, user_id)
            await AuditService(session).record("user.erased", actor_user_id=user_id, meta=meta)
            await uow.commit()
        await self._auth.publish_revocations(ended)
        await self._versions.remember(user_id, version)


async def _active_user(session: AsyncSession, user_id: UUID, *, lock: bool = False) -> User:
    user = await session.get(User, user_id, with_for_update=lock)
    if user is None or user.status is not UserStatus.ACTIVE:
        raise Unauthorized("This account is not active.")
    return user


async def _memberships(session: AsyncSession, user_id: UUID) -> list[MembershipView]:
    rows = await session.execute(
        select(Membership, Organization)
        .join(Organization, Organization.id == Membership.organization_id)
        .where(Membership.user_id == user_id, Organization.deleted_at.is_(None))
        .order_by(Organization.name)
    )
    return [
        MembershipView(
            organization_id=organization.id,
            organization_name=organization.name,
            organization_type=organization.type.value,
            role=membership.role,
            status=membership.status.value,
        )
        for membership, organization in rows
    ]

"""Platform admins managing accounts: find, suspend, reactivate, sign out everywhere, and reset
MFA for a lost authenticator.

Every action is audited with the admin as actor. Admins cannot act on their own account (they
use their own settings), and the last active platform admin cannot be suspended. Suspending ends
every session and bumps perm_ver, so open access tokens stop working on their next request.
"""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import ColumnElement, func, or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.crypto import phone_lookup_hash
from medlens.core.errors import Conflict, NotFound
from medlens.core.identifiers import normalize_ng_phone
from medlens.core.pagination import PageParams
from medlens.core.request_meta import RequestMeta
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.auth.service import AuthService
from medlens.modules.organizations.models import Membership
from medlens.modules.rbac.access import PermissionVersions, bump_perm_ver
from medlens.modules.rbac.models import RoleName
from medlens.modules.users.models import User, UserStatus
from medlens.modules.users.service import MembershipView, memberships_of

_NOT_FOUND = "User not found."
_PLATFORM_ADMIN_LOCK = 0x4D4C_5041  # "MLPA": changes to who is an active platform admin
SELF = "Use your own account settings for your own account."


class AdminAction(StrEnum):
    SUSPEND = "suspend"
    REACTIVATE = "reactivate"
    SIGN_OUT = "sign_out"
    RESET_MFA = "reset_mfa"


@dataclass(frozen=True, slots=True)
class UserFilter:
    query: str | None = None  # part of an e-mail address or name, or a phone number
    status: UserStatus | None = None
    global_role: str | None = None
    organization_id: UUID | None = None


class UserAdminService:
    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], redis: Redis, auth: AuthService
    ) -> None:
        self._sessions = sessions
        self._versions = PermissionVersions(redis)
        self._auth = auth

    async def list_page(
        self, where: UserFilter, params: PageParams
    ) -> list[tuple[User, list[MembershipView]]]:
        statement = select(User).where(User.status != UserStatus.DELETED)
        if where.query:
            statement = statement.where(_matches(where.query))
        if where.status is not None:
            statement = statement.where(User.status == where.status)
        if where.global_role is not None:
            statement = statement.where(User.global_role == where.global_role)
        if where.organization_id is not None:
            statement = statement.where(
                select(Membership.id)
                .where(
                    Membership.user_id == User.id,
                    Membership.organization_id == where.organization_id,
                )
                .exists()
            )
        if params.after is not None:
            position = tuple_(params.after.created_at, params.after.id)
            statement = statement.where(tuple_(User.created_at, User.id) < position)
        statement = statement.order_by(User.created_at.desc(), User.id.desc())
        async with UnitOfWork(self._sessions) as uow:
            users = list(await uow.session.scalars(statement.limit(params.limit + 1)))
            return [(user, await memberships_of(uow.session, user.id)) for user in users]

    async def get(self, user_id: UUID) -> tuple[User, list[MembershipView]]:
        async with UnitOfWork(self._sessions) as uow:
            user = await _target(uow.session, user_id)
            return user, await memberships_of(uow.session, user.id)

    async def act(
        self,
        actor_id: UUID,
        user_id: UUID,
        action: AdminAction,
        reason: str,
        meta: RequestMeta,
    ) -> tuple[User, list[MembershipView]]:
        if actor_id == user_id:
            raise Conflict(SELF)
        version: int | None = None
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            await session.execute(select(func.pg_advisory_xact_lock(_PLATFORM_ADMIN_LOCK)))
            user = await _target(session, user_id, lock=True)
            ended: list[UUID] = []
            if action is AdminAction.SUSPEND:
                if user.status is UserStatus.SUSPENDED:
                    raise Conflict("The account is already suspended.")
                if user.global_role == RoleName.PLATFORM_ADMIN.value:
                    await _keep_one_platform_admin(session, user.id)
                user.status = UserStatus.SUSPENDED
                ended = await self._auth.end_sessions(session, user.id)
                version = await bump_perm_ver(session, user.id)
            elif action is AdminAction.REACTIVATE:
                if user.status is UserStatus.ACTIVE:
                    raise Conflict("The account is already active.")
                user.status = UserStatus.ACTIVE
                version = await bump_perm_ver(session, user.id)
            elif action is AdminAction.SIGN_OUT:
                ended = await self._auth.end_sessions(session, user.id)
            else:
                if not user.mfa_enabled and user.mfa_secret is None:
                    raise Conflict("The account has no MFA to reset.")
                ended = await self._auth.reset_mfa(session, user)
            await AuditService(session).record(
                f"user.admin_{action.value}",
                actor_user_id=actor_id,
                resource_type="user",
                resource_id=user.id,
                meta=meta,
                details={"reason": reason, "sessions_ended": len(ended)},
            )
            await uow.commit()
            memberships = await memberships_of(session, user.id)
        if ended:
            await self._auth.publish_revocations(ended)
        if version is not None:
            await self._versions.remember(user.id, version)
        return user, memberships


def _matches(query: str) -> ColumnElement[bool]:
    text = query.strip()
    try:
        phone = normalize_ng_phone(text)
    except ValueError:
        phone = None
    if phone is not None:
        return User.phone_hash == phone_lookup_hash(phone)
    pattern = "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    return or_(User.email.ilike(pattern), User.full_name.ilike(pattern))


async def _target(session: AsyncSession, user_id: UUID, *, lock: bool = False) -> User:
    user = await session.get(User, user_id, with_for_update=lock)
    if user is None or user.status is UserStatus.DELETED:
        raise NotFound(_NOT_FOUND)
    return user


async def _keep_one_platform_admin(session: AsyncSession, leaving: UUID) -> None:
    others = await session.scalar(
        select(func.count())
        .select_from(User)
        .where(
            User.global_role == RoleName.PLATFORM_ADMIN.value,
            User.status == UserStatus.ACTIVE,
            User.id != leaving,
        )
    )
    if not others:
        raise Conflict("This is the last active platform admin.")

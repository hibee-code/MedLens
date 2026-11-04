"""Role administration: reading the role catalogue and changing users' global roles."""

from dataclasses import dataclass
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.errors import Forbidden, NotFound, ValidationFailed
from medlens.core.request_meta import RequestMeta
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.rbac.access import Actor, PermissionVersions, bump_perm_ver
from medlens.modules.rbac.models import Permission as PermissionRow
from medlens.modules.rbac.models import Role, RolePermission, RoleScope
from medlens.modules.rbac.permissions import Permission
from medlens.modules.users.models import User, UserStatus


@dataclass(frozen=True, slots=True)
class RoleView:
    name: str
    scope: RoleScope
    description: str
    permissions: list[str]


async def role_permissions(session: AsyncSession, role: str) -> frozenset[Permission]:
    rows = await session.scalars(
        select(RolePermission.permission_name).where(RolePermission.role_name == role)
    )
    return frozenset(Permission(name) for name in rows)


def ensure_no_escalation(actor: Actor, granted: frozenset[Permission]) -> None:
    """Nobody can hand out a permission they do not hold themselves."""
    missing = granted - actor.permissions
    if missing:
        raise Forbidden(
            "You cannot grant a role with permissions you do not have: "
            + ", ".join(sorted(missing))
        )


class RbacService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], redis: Redis) -> None:
        self._sessions = sessions
        self._versions = PermissionVersions(redis)

    async def roles(self) -> list[RoleView]:
        async with UnitOfWork(self._sessions) as uow:
            roles = list(await uow.session.scalars(select(Role).order_by(Role.name)))
            grants = await uow.session.execute(select(RolePermission))
            by_role: dict[str, list[str]] = {}
            for grant in grants.scalars():
                by_role.setdefault(grant.role_name, []).append(grant.permission_name)
        return [
            RoleView(role.name, role.scope, role.description, sorted(by_role.get(role.name, [])))
            for role in roles
        ]

    async def permissions(self) -> list[PermissionRow]:
        async with UnitOfWork(self._sessions) as uow:
            return list(
                await uow.session.scalars(select(PermissionRow).order_by(PermissionRow.name))
            )

    async def set_global_role(
        self, actor: Actor, user_id: UUID, role_name: str, meta: RequestMeta
    ) -> User:
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            role = await session.get(Role, role_name)
            if role is None or role.scope is not RoleScope.GLOBAL:
                raise ValidationFailed(
                    f"{role_name!r} is not a global role.",
                    errors=[{"loc": ["body", "role"], "msg": "Not a global role", "type": "role"}],
                )
            ensure_no_escalation(actor, await role_permissions(session, role_name))
            user = await session.get(User, user_id, with_for_update=True)
            if user is None or user.status is UserStatus.DELETED:
                raise NotFound("User not found.")
            previous = user.global_role
            user.global_role = role_name
            version = await bump_perm_ver(session, user.id)
            await AuditService(session).record(
                "user.global_role_changed",
                actor_user_id=actor.user_id,
                resource_type="user",
                resource_id=user.id,
                meta=meta,
                details={"from": previous, "to": role_name},
            )
            await uow.commit()
        await self._versions.remember(user_id, version)
        return user

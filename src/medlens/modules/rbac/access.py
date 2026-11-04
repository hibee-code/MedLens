"""What a principal may do: an ``Actor`` with permissions and organization context.

User permissions = permissions of their global role plus those of their role in the active
organization (the token's ``org``), read from ``role_permissions``. The result is cached in
Redis under a key that contains

* the user's current ``perm_ver``, bumped on every role or membership change, and
* a global RBAC version, bumped when the role -> permission matrix changes,

so a change applies on the very next request, even with an old access token. Redis problems
never fail a request: resolution falls back to the database.
"""

import json
from dataclasses import dataclass
from typing import TypedDict
from uuid import UUID

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.cache.keys import redis_key
from medlens.core.errors import Unauthorized
from medlens.core.tenancy import TenantContext
from medlens.db.session import UnitOfWork
from medlens.modules.auth.principal import Principal
from medlens.modules.organizations.models import (
    Membership,
    MembershipStatus,
    Organization,
    OrganizationStatus,
    OrganizationType,
)
from medlens.modules.rbac.models import RolePermission
from medlens.modules.rbac.permissions import NON_DELEGABLE_TO_API_KEYS, Permission
from medlens.modules.users.models import User, UserStatus

log = structlog.stdlib.get_logger(__name__)

PERMISSIONS_TTL_SECONDS = 900
PERM_VER_TTL_SECONDS = 3600
_RBAC_VERSION_KEY = redis_key("rbac_version")

# Sets the cached perm_ver only upwards, so a slow reader cannot overwrite a newer bump.
_MONOTONIC_SET = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local candidate = tonumber(ARGV[1])
if candidate > current then
    redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
    return candidate
end
redis.call('EXPIRE', KEYS[1], ARGV[2])
return current
"""


class _OrganizationData(TypedDict):
    id: str
    type: str
    role: str | None
    jurisdiction_states: list[str] | None


class _AccessData(TypedDict):
    """What is cached in Redis per (user, organization, perm_ver)."""

    permissions: list[str]
    global_role: str | None
    organization: _OrganizationData | None
    perm_ver: int


@dataclass(frozen=True, slots=True)
class OrganizationContext:
    id: UUID
    type: OrganizationType
    role: str | None  # the member's role; None for API keys
    jurisdiction_states: frozenset[str] | None  # regulators only


@dataclass(frozen=True, slots=True)
class Actor:
    principal: Principal
    permissions: frozenset[Permission]
    global_role: str | None
    organization: OrganizationContext | None
    perm_ver: int  # current version; the token may carry an older one

    def can(self, permission: Permission) -> bool:
        return permission in self.permissions

    @property
    def user_id(self) -> UUID | None:
        return self.principal.user_id

    @property
    def is_user(self) -> bool:
        return self.principal.is_user

    @property
    def is_platform_admin(self) -> bool:
        return Permission.PLATFORM_ADMIN in self.permissions

    @property
    def stale(self) -> bool:
        """The access token predates a change to this user's roles or memberships."""
        return self.principal.is_user and self.principal.perm_ver < self.perm_ver

    @property
    def tenant(self) -> TenantContext:
        return TenantContext(
            user_id=self.user_id,
            organization_id=self.organization.id if self.organization else None,
            unrestricted=self.is_platform_admin,
        )


class PermissionVersions:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._set = redis.register_script(_MONOTONIC_SET)

    @staticmethod
    def _key(user_id: UUID) -> str:
        return redis_key("perm_ver", user_id)

    async def cached(self, user_id: UUID) -> int | None:
        value = await self._redis.get(self._key(user_id))
        return int(value) if value is not None else None

    async def remember(self, user_id: UUID, version: int) -> None:
        try:
            await self._set(keys=[self._key(user_id)], args=[version, PERM_VER_TTL_SECONDS])
        except RedisError as exc:
            log.warning("rbac.perm_ver_cache_failed", error=type(exc).__name__)

    async def rbac_version(self) -> int:
        value = await self._redis.get(_RBAC_VERSION_KEY)
        return int(value) if value is not None else 0

    async def bump_rbac_version(self) -> None:
        """Call after changing role_permissions: invalidates every cached permission set."""
        await self._redis.incr(_RBAC_VERSION_KEY)


async def bump_perm_ver(session: AsyncSession, user_id: UUID) -> int:
    """Increment the user's permission version in the caller's transaction. After committing,
    pass the result to ``PermissionVersions.remember``."""
    statement = (
        update(User)
        .where(User.id == user_id)
        .values(perm_ver=User.perm_ver + 1)
        .returning(User.perm_ver)
    )
    return int((await session.execute(statement)).scalar_one())


class AccessResolver:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], redis: Redis) -> None:
        self._sessions = sessions
        self._redis = redis
        self.versions = PermissionVersions(redis)

    async def resolve(self, principal: Principal) -> Actor:
        if principal.kind == "api_key":
            return await self._resolve_api_key(principal)
        if principal.user_id is None:
            raise Unauthorized()
        user_id, org_id = principal.user_id, principal.organization_id

        cache_key: str | None = None
        try:
            perm_ver = await self.versions.cached(user_id)
            rbac_version = await self.versions.rbac_version()
            if perm_ver is not None:
                cache_key = redis_key("perms", rbac_version, user_id, org_id or "-", perm_ver)
                cached = await self._redis.get(cache_key)
                if cached is not None:
                    data: _AccessData = json.loads(cached)
                    return _actor_from_json(principal, data)
        except RedisError as exc:
            log.warning("rbac.cache_unavailable", error=type(exc).__name__)
            rbac_version = None

        loaded = await self._load_user(user_id, org_id)
        try:
            await self.versions.remember(user_id, loaded["perm_ver"])
            if rbac_version is not None:
                cache_key = redis_key(
                    "perms", rbac_version, user_id, org_id or "-", loaded["perm_ver"]
                )
                await self._redis.set(cache_key, json.dumps(loaded), ex=PERMISSIONS_TTL_SECONDS)
        except RedisError as exc:
            log.warning("rbac.cache_unavailable", error=type(exc).__name__)
        return _actor_from_json(principal, loaded)

    async def _load_user(self, user_id: UUID, org_id: UUID | None) -> _AccessData:
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            user = await session.get(User, user_id)
            if user is None or user.status is not UserStatus.ACTIVE:
                raise Unauthorized("This account is not active.")
            permissions = set(await _role_permissions(session, user.global_role))
            organization: _OrganizationData | None = None
            if org_id is not None:
                row = (
                    await session.execute(
                        select(Membership.role, Organization)
                        .join(Organization, Organization.id == Membership.organization_id)
                        .where(
                            Membership.user_id == user_id,
                            Membership.organization_id == org_id,
                            Membership.status == MembershipStatus.ACTIVE,
                            Organization.status == OrganizationStatus.ACTIVE,
                            Organization.deleted_at.is_(None),
                        )
                    )
                ).first()
                if row is not None:
                    role, org = row
                    permissions |= set(await _role_permissions(session, role))
                    organization = _organization_json(org, role)
            return {
                "permissions": sorted(permissions),
                "global_role": user.global_role,
                "organization": organization,
                "perm_ver": user.perm_ver,
            }

    async def _resolve_api_key(self, principal: Principal) -> Actor:
        """An API key may use its scopes, minus anything its organization's roles could never
        hold and anything administrative."""
        async with UnitOfWork(self._sessions) as uow:
            org = (
                await uow.session.get(Organization, principal.organization_id)
                if principal.organization_id
                else None
            )
        if org is None:
            raise Unauthorized("Invalid API key.")
        scopes = {
            Permission(scope)
            for scope in principal.scopes
            if scope in Permission._value2member_map_
        }
        permissions = frozenset(scopes - NON_DELEGABLE_TO_API_KEYS)
        return Actor(
            principal=principal,
            permissions=permissions,
            global_role=None,
            organization=OrganizationContext(
                id=org.id,
                type=org.type,
                role=None,
                jurisdiction_states=frozenset(org.jurisdiction_states or ()) or None,
            ),
            perm_ver=0,
        )


async def _role_permissions(session: AsyncSession, role: str) -> list[str]:
    statement = select(RolePermission.permission_name).where(RolePermission.role_name == role)
    return list(await session.scalars(statement))


def _organization_json(org: Organization, role: str) -> _OrganizationData:
    return {
        "id": str(org.id),
        "type": org.type.value,
        "role": role,
        "jurisdiction_states": sorted(org.jurisdiction_states or []) or None,
    }


def _actor_from_json(principal: Principal, data: _AccessData) -> Actor:
    org = data["organization"]
    organization = None
    if org is not None:
        states = org["jurisdiction_states"]
        organization = OrganizationContext(
            id=UUID(org["id"]),
            type=OrganizationType(org["type"]),
            role=org["role"],
            jurisdiction_states=frozenset(states) if states else None,
        )
    return Actor(
        principal=principal,
        permissions=frozenset(Permission(p) for p in data["permissions"]),
        global_role=data["global_role"],
        organization=organization,
        perm_ver=data["perm_ver"],
    )

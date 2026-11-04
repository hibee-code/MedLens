"""Route-level authorization.

    actor: Annotated[Actor, Depends(require_permissions(Permission.MEMBER_MANAGE, fresh=True))]

Row-level rules are not checked here: services apply ``policies.py``.
"""

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends

from medlens.core.errors import Forbidden, Unauthorized
from medlens.modules.auth.deps import AuthServiceDep, CurrentPrincipal, OptionalPrincipal
from medlens.modules.auth.principal import Principal
from medlens.modules.auth.service import AuthService
from medlens.modules.rbac.access import AccessResolver, Actor
from medlens.modules.rbac.permissions import CONSUMER_PERMISSIONS, Permission
from medlens.resources import Resources, get_resources


def get_access_resolver(resources: Annotated[Resources, Depends(get_resources)]) -> AccessResolver:
    return AccessResolver(resources.session_factory, resources.redis)


async def _resolve(principal: Principal, resolver: AccessResolver, auth: AuthService) -> Actor:
    actor = await resolver.resolve(principal)
    if actor.permissions - CONSUMER_PERMISSIONS:
        # A privileged session: if Redis could not confirm it is still live, ask Postgres.
        await auth.ensure_not_revoked(principal)
    return actor


async def get_actor(
    principal: CurrentPrincipal,
    resolver: Annotated[AccessResolver, Depends(get_access_resolver)],
    auth: AuthServiceDep,
) -> Actor:
    return await _resolve(principal, resolver, auth)


async def get_optional_actor(
    principal: OptionalPrincipal,
    resolver: Annotated[AccessResolver, Depends(get_access_resolver)],
    auth: AuthServiceDep,
) -> Actor | None:
    """None for guests, on routes open to them (submitting and reading scans)."""
    return None if principal is None else await _resolve(principal, resolver, auth)


CurrentActor = Annotated[Actor, Depends(get_actor)]
OptionalActor = Annotated[Actor | None, Depends(get_optional_actor)]


def require_permissions(
    *permissions: Permission, fresh: bool = False
) -> Callable[..., Awaitable[Actor]]:
    """Dependency requiring every listed permission.

    ``fresh=True`` also rejects access tokens issued before the user's roles last changed, so
    sensitive operations (changing roles, managing API keys) never run on outdated authority.
    Other routes simply use the current permissions.
    """

    async def dependency(actor: CurrentActor) -> Actor:
        if fresh and actor.stale:
            raise Unauthorized("Your permissions have changed. Refresh your access token.")
        if not all(actor.can(permission) for permission in permissions):
            raise Forbidden("You do not have permission to do this.")
        return actor

    return dependency

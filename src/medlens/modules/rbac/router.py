from typing import Annotated
from uuid import UUID

from fastapi import Depends

from medlens.core.openapi import problem_responses
from medlens.core.routing import api_v1_router
from medlens.modules.auth.deps import Meta as RequestMetaDep
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.modules.rbac.schemas import (
    GlobalRoleAssignment,
    PermissionRead,
    RoleRead,
    UserRoleRead,
)
from medlens.modules.rbac.service import RbacService
from medlens.resources import Resources, get_resources

router = api_v1_router("/admin", tags=["admin"])


def get_rbac_service(resources: Annotated[Resources, Depends(get_resources)]) -> RbacService:
    return RbacService(resources.session_factory, resources.redis)


ServiceDep = Annotated[RbacService, Depends(get_rbac_service)]
can_manage_members = Depends(require_permissions(Permission.MEMBER_MANAGE))
UserManager = Annotated[Actor, Depends(require_permissions(Permission.USER_MANAGE, fresh=True))]


@router.get(
    "/roles",
    response_model=list[RoleRead],
    summary="List roles and their permissions",
    description="The role catalogue as currently stored (it can change without a deploy).",
    responses=problem_responses(401, 403),
    dependencies=[can_manage_members],
)
async def list_roles(service: ServiceDep) -> list[RoleRead]:
    return [
        RoleRead(
            name=r.name, scope=r.scope.value, description=r.description, permissions=r.permissions
        )
        for r in await service.roles()
    ]


@router.get(
    "/permissions",
    response_model=list[PermissionRead],
    summary="List permissions",
    description="Every permission (`resource:action`) with what it allows.",
    responses=problem_responses(401, 403),
    dependencies=[can_manage_members],
)
async def list_permissions(service: ServiceDep) -> list[PermissionRead]:
    return [PermissionRead.model_validate(p) for p in await service.permissions()]


@router.put(
    "/users/{user_id}/global-role",
    response_model=UserRoleRead,
    summary="Set a user's global role",
    description=(
        "Platform administration: makes a user a consumer, health worker or platform admin. "
        "Takes effect on the user's next request."
    ),
    responses=problem_responses(401, 403, 404, 422),
)
async def set_global_role(
    user_id: UUID,
    body: GlobalRoleAssignment,
    actor: UserManager,
    service: ServiceDep,
    meta: RequestMetaDep,
) -> UserRoleRead:
    user = await service.set_global_role(actor, user_id, body.role.value, meta)
    return UserRoleRead(id=user.id, global_role=user.global_role)

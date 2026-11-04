from typing import Annotated

from fastapi import Depends, Response, status

from medlens.core.openapi import problem_responses
from medlens.core.routing import api_v1_router
from medlens.modules.auth.deps import AuthServiceDep, SessionCheckedUser, UserPrincipal
from medlens.modules.auth.deps import Meta as RequestMetaDep
from medlens.modules.users.models import User
from medlens.modules.users.schemas import MembershipSummary, PasswordChange, UserRead, UserUpdate
from medlens.modules.users.service import MembershipView, UserService
from medlens.resources import Resources, get_resources

router = api_v1_router("/users", tags=["users"])


def get_user_service(
    auth: AuthServiceDep, resources: Annotated[Resources, Depends(get_resources)]
) -> UserService:
    return UserService(resources.session_factory, resources.redis, auth)


UserServiceDep = Annotated[UserService, Depends(get_user_service)]


def _user_read(user: User, memberships: list[MembershipView]) -> UserRead:
    return UserRead(
        id=user.id,
        email=user.email,
        phone=user.phone_e164,
        full_name=user.full_name,
        status=user.status.value,
        global_role=user.global_role,
        mfa_enabled=user.mfa_enabled,
        created_at=user.created_at,
        memberships=[MembershipSummary.model_validate(m) for m in memberships],
    )


@router.get(
    "/me",
    response_model=UserRead,
    summary="Your account",
    description="Profile, global role and organization memberships of the signed-in user.",
    responses=problem_responses(401, 403),
)
async def read_me(principal: UserPrincipal, service: UserServiceDep) -> UserRead:
    return _user_read(*await service.profile(principal.require_user_id()))


@router.patch(
    "/me",
    response_model=UserRead,
    summary="Update your profile",
    description="Only the fields sent are changed. E-mail and phone changes need verification "
    "and are not available here.",
    responses=problem_responses(401, 403, 422),
)
async def update_me(
    body: UserUpdate, principal: UserPrincipal, service: UserServiceDep, meta: RequestMetaDep
) -> UserRead:
    user, memberships = await service.update_profile(
        principal.require_user_id(), body.changes(), meta
    )
    return _user_read(user, memberships)


@router.post(
    "/me/password",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Change your password",
    description=(
        "Needs the current password. The new one must have 12+ characters, not be a common "
        "password and not contain your e-mail address. Your other sessions are signed out."
    ),
    responses=problem_responses(401, 403, 409, 422),
)
async def change_password(
    body: PasswordChange,
    principal: SessionCheckedUser,
    auth: AuthServiceDep,
    meta: RequestMetaDep,
) -> None:
    await auth.change_password(
        principal,
        body.current_password.get_secret_value(),
        body.new_password.get_secret_value(),
        meta,
    )


@router.delete(
    "/me",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete your account (right to erasure)",
    description=(
        "Removes your personal data (phone, e-mail, name, password, MFA) and memberships, and "
        "signs you out everywhere. Scans you submitted are kept for public-health analytics "
        "but no longer linked to you. An organization's last administrator must appoint "
        "another one first."
    ),
    responses=problem_responses(401, 403, 409),
)
async def delete_me(
    principal: SessionCheckedUser, service: UserServiceDep, meta: RequestMetaDep
) -> None:
    await service.erase(principal.require_user_id(), meta)

from typing import Annotated
from uuid import UUID

from fastapi import Depends, Query, Response, status

from medlens.core.openapi import problem_responses
from medlens.core.pagination import Cursor, Page, PageParams, build_page, page_params
from medlens.core.routing import api_v1_router
from medlens.modules.auth.deps import (
    AuthServiceDep,
    SessionCheckedUser,
    UserPrincipal,
    require_mfa_verified,
)
from medlens.modules.auth.deps import Meta as RequestMetaDep
from medlens.modules.auth.schemas import EmailCodeSent, EmailOnly, EmailVerify
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import require_permissions
from medlens.modules.rbac.models import RoleName
from medlens.modules.rbac.permissions import Permission
from medlens.modules.users.admin import AdminAction, UserAdminService, UserFilter
from medlens.modules.users.models import User, UserStatus
from medlens.modules.users.schemas import (
    AdminAccountAction,
    AdminUserRead,
    MembershipSummary,
    PasswordChange,
    UserRead,
    UserUpdate,
)
from medlens.modules.users.service import MembershipView, UserService
from medlens.ratelimit.deps import limit_logins
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
        email_verified=user.email_verified_at is not None,
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


@router.post(
    "/me/email",
    response_model=EmailCodeSent,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Add an e-mail address to your account",
    description=(
        "E-mails a 6-digit code (valid 15 minutes) to the address; confirm it with "
        "`POST /users/me/email/verify`. An address another account uses gets no code (same "
        "answer). With a verified address you can set a password (`POST /auth/password/forgot`) "
        "and accept invitations sent to it."
    ),
    responses=problem_responses(401, 403, 422, 429, 503),
    dependencies=[Depends(limit_logins)],
)
async def add_email(
    body: EmailOnly, principal: SessionCheckedUser, auth: AuthServiceDep, meta: RequestMetaDep
) -> EmailCodeSent:
    expires_in = await auth.request_email_change(principal, body.email, meta)
    return EmailCodeSent(
        detail="If the address can receive e-mail, a code is on its way.", expires_in=expires_in
    )


@router.post(
    "/me/email/verify",
    response_model=UserRead,
    summary="Confirm the e-mail address with its code",
    description=(
        "Sets the verified address on your account. Five wrong attempts invalidate the code."
    ),
    responses=problem_responses(401, 403, 409, 422, 429, 503),
    dependencies=[Depends(limit_logins)],
)
async def verify_added_email(
    body: EmailVerify,
    principal: SessionCheckedUser,
    auth: AuthServiceDep,
    service: UserServiceDep,
    meta: RequestMetaDep,
) -> UserRead:
    await auth.confirm_email_change(principal, body.email, body.code, meta)
    return _user_read(*await service.profile(principal.require_user_id()))


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


# ------------------------------------------------------------------ platform admins
admin_users_router = api_v1_router("/admin/users", tags=["admin"])
PlatformAdmin = Annotated[
    Actor, Depends(require_permissions(Permission.PLATFORM_ADMIN, fresh=True))
]
# A person signed in with MFA, never an API key.
_ADMIN_ONLY = [
    Depends(require_mfa_verified),
    Depends(require_permissions(Permission.PLATFORM_ADMIN, fresh=True)),
]


def get_user_admin_service(
    auth: AuthServiceDep, resources: Annotated[Resources, Depends(get_resources)]
) -> UserAdminService:
    return UserAdminService(resources.session_factory, resources.redis, auth)


AdminServiceDep = Annotated[UserAdminService, Depends(get_user_admin_service)]


def _mask_phone(phone: str | None) -> str | None:
    return f"{phone[:7]}*****{phone[-2:]}" if phone and len(phone) > 9 else None


def _admin_read(row: tuple[User, list[MembershipView]]) -> AdminUserRead:
    user, memberships = row
    return AdminUserRead(
        **_user_read(user, memberships).model_dump(exclude={"phone"}),
        phone=_mask_phone(user.phone_e164),
        last_login_at=user.last_login_at,
    )


def _user_cursor(row: tuple[User, list[MembershipView]]) -> Cursor:
    return Cursor(created_at=row[0].created_at, id=row[0].id)


@admin_users_router.get(
    "",
    response_model=Page[AdminUserRead],
    summary="Find accounts",
    description=(
        "Newest first. `q` matches part of an e-mail address or name, or a whole Nigerian phone "
        "number in any format. Erased accounts are not listed. Platform admins, signed in with "
        "MFA."
    ),
    responses=problem_responses(401, 403, 422),
    dependencies=_ADMIN_ONLY,
)
async def list_users(
    service: AdminServiceDep,
    params: Annotated[PageParams, Depends(page_params)],
    q: Annotated[str | None, Query(min_length=2, max_length=100)] = None,
    user_status: Annotated[UserStatus | None, Query(alias="status")] = None,
    global_role: Annotated[RoleName | None, Query()] = None,
    organization_id: Annotated[UUID | None, Query()] = None,
) -> Page[AdminUserRead]:
    where = UserFilter(
        query=q,
        status=user_status,
        global_role=global_role.value if global_role else None,
        organization_id=organization_id,
    )
    rows = await service.list_page(where, params)
    return build_page(rows, limit=params.limit, cursor_of=_user_cursor, to_item=_admin_read)


@admin_users_router.get(
    "/{user_id}",
    response_model=AdminUserRead,
    summary="One account",
    description="The account with its memberships; erased accounts answer 404.",
    responses=problem_responses(401, 403, 404),
    dependencies=_ADMIN_ONLY,
)
async def read_user(user_id: UUID, service: AdminServiceDep) -> AdminUserRead:
    return _admin_read(await service.get(user_id))


def _action_route(action: AdminAction, path: str, summary: str, description: str) -> None:
    async def handler(
        user_id: UUID,
        body: AdminAccountAction,
        actor: PlatformAdmin,
        service: AdminServiceDep,
        meta: RequestMetaDep,
    ) -> AdminUserRead:
        acted = await service.act(
            actor.principal.require_user_id(), user_id, action, body.reason, meta
        )
        return _admin_read(acted)

    handler.__name__ = f"admin_{action.value}_user"
    admin_users_router.post(
        f"/{{user_id}}/{path}",
        response_model=AdminUserRead,
        summary=summary,
        description=description + " Audited with the reason. Not for your own account (409).",
        responses=problem_responses(401, 403, 404, 409, 422),
        dependencies=_ADMIN_ONLY,
    )(handler)


_action_route(
    AdminAction.SUSPEND,
    "suspend",
    "Suspend an account",
    "Ends every session and blocks sign-in until reactivated; open access tokens stop working "
    "on their next request. The last active platform admin cannot be suspended.",
)
_action_route(
    AdminAction.REACTIVATE,
    "reactivate",
    "Reactivate an account",
    "Allows sign-in again. Earlier sessions stay ended.",
)
_action_route(
    AdminAction.SIGN_OUT,
    "sign-out",
    "Sign an account out everywhere",
    "Ends every session (a lost or stolen device); the account signs in again as usual.",
)
_action_route(
    AdminAction.RESET_MFA,
    "mfa/reset",
    "Reset an account's MFA",
    "For a lost authenticator without recovery codes, after checking who is asking: removes "
    "the authenticator and the recovery codes and ends every session. Staff set up MFA again "
    "at their next sign-in.",
)

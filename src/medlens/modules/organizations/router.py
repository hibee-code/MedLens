from typing import Annotated
from uuid import UUID

from fastapi import Depends, Path, Query, Response, status

from medlens.core.geo import point_coordinates
from medlens.core.openapi import problem_responses
from medlens.core.pagination import Cursor, Page, PageParams, build_page, page_params
from medlens.core.routing import api_v1_router
from medlens.modules.auth.deps import AuthServiceDep, SessionCheckedUser
from medlens.modules.auth.deps import Meta as RequestMetaDep
from medlens.modules.auth.router import login_response
from medlens.modules.auth.schemas import (
    EmailCodeSent,
    EmailVerify,
    LoginResponse,
    MfaChallengeResponse,
    TokenResponse,
)
from medlens.modules.organizations.branches import BranchService
from medlens.modules.organizations.models import (
    Branch,
    Invitation,
    Membership,
    Organization,
    OrganizationStatus,
    OrganizationType,
)
from medlens.modules.organizations.onboarding import (
    InvitationService,
    OrganizationService,
    invitation_status,
    mask_recipient,
)
from medlens.modules.organizations.schemas import (
    BranchCreate,
    BranchRead,
    BranchUpdate,
    InvitationAccept,
    InvitationAccepted,
    InvitationAcceptNew,
    InvitationCreate,
    InvitationCreated,
    InvitationPreview,
    InvitationRead,
    MemberRead,
    OrganizationCreate,
    OrganizationRead,
    OrganizationRegistration,
    PlatformInvitationCreate,
    RoleAssignment,
)
from medlens.modules.organizations.service import MembershipService
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import get_actor, require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.modules.users.models import User
from medlens.ratelimit.deps import limit_logins
from medlens.resources import Resources, get_resources

EMAIL_SENT = "If the address can receive e-mail, a code is on its way."
REGISTRATION_CODE_SECONDS = 900  # the answer when no code is sent (as long as a real code)

router = api_v1_router("/organizations", tags=["organizations"])


def get_membership_service(
    resources: Annotated[Resources, Depends(get_resources)],
) -> MembershipService:
    return MembershipService(resources.session_factory, resources.redis)


ServiceDep = Annotated[MembershipService, Depends(get_membership_service)]
Reader = Annotated[Actor, Depends(require_permissions(Permission.ORG_READ))]
Manager = Annotated[Actor, Depends(require_permissions(Permission.MEMBER_MANAGE, fresh=True))]


def _member(membership: Membership, user: User) -> MemberRead:
    return MemberRead(
        organization_id=membership.organization_id,
        user_id=user.id,
        full_name=user.full_name,
        email=user.email,
        role=membership.role,
        status=membership.status.value,
        created_at=membership.created_at,
    )


@router.get(
    "/{organization_id}/members",
    response_model=Page[MemberRead],
    summary="List an organization's members",
    description=(
        "Members of your own organization (platform admins: any organization), newest first. "
        "Other organizations answer 404."
    ),
    responses=problem_responses(401, 403, 404, 422),
)
async def list_members(
    organization_id: UUID,
    actor: Reader,
    service: ServiceDep,
    params: Annotated[PageParams, Depends(page_params)],
) -> Page[MemberRead]:
    rows = await service.list_members(actor, organization_id, params)
    return build_page(
        rows,
        limit=params.limit,
        cursor_of=lambda row: Cursor(created_at=row[0].created_at, id=row[0].id),
        to_item=lambda row: _member(*row),
    )


@router.put(
    "/{organization_id}/members/{user_id}",
    response_model=MemberRead,
    summary="Set a member's role",
    description=(
        "Changes a member's role. The role must suit the organization type and carry no "
        "permission you lack. Only platform admins can add people who are not members yet. "
        "The last administrator cannot be demoted. Takes effect on the member's next request."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def assign_role(
    organization_id: UUID,
    user_id: UUID,
    body: RoleAssignment,
    actor: Manager,
    service: ServiceDep,
    meta: RequestMetaDep,
) -> MemberRead:
    membership, user = await service.assign_role(actor, organization_id, user_id, body.role, meta)
    return _member(membership, user)


@router.delete(
    "/{organization_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Remove a member",
    description="Removes the membership. The last administrator cannot be removed.",
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def remove_member(
    organization_id: UUID, user_id: UUID, actor: Manager, service: ServiceDep, meta: RequestMetaDep
) -> None:
    await service.remove_member(actor, organization_id, user_id, meta)


# ------------------------------------------------------------------ organizations
invitations_router = api_v1_router("/invitations", tags=["organizations"])
SignedIn = Annotated[Actor, Depends(get_actor)]  # any signed-in user or API key
PlatformAdmin = Annotated[
    Actor, Depends(require_permissions(Permission.PLATFORM_ADMIN, fresh=True))
]
Inviter = Annotated[Actor, Depends(require_permissions(Permission.MEMBER_INVITE, fresh=True))]
OrgManager = Annotated[Actor, Depends(require_permissions(Permission.ORG_MANAGE))]
Params = Annotated[PageParams, Depends(page_params)]
ResourcesDep = Annotated[Resources, Depends(get_resources)]
TokenPath = Annotated[str, Path(min_length=16, max_length=128)]


def _invitations(resources: Resources) -> InvitationService:
    """With the e-mail channel and the link base, for the routes that send invitations."""
    return InvitationService(
        resources.session_factory,
        resources.redis,
        resources.email,
        frontend_url=resources.frontend_url,
    )


def _organization(organization: Organization) -> OrganizationRead:
    return OrganizationRead.model_validate(organization)


def _cursor(row: Organization | Invitation | Branch) -> Cursor:
    return Cursor(created_at=row.created_at, id=row.id)


@router.post(
    "/register",
    response_model=EmailCodeSent,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Register your organization (public sign-up form)",
    description=(
        "For pharmacies, PPMVs, clinics, manufacturers and distributors: the organization, its "
        "licence number and its first administrator, who gets a 6-digit code by e-mail (valid "
        "15 minutes) to verify the address with `POST /organizations/register/verify`. The "
        "organization starts as `pending_verification` and grants nothing until a platform "
        "admin checks the licence and verifies it. An address that already has an account "
        "gets no new account (the same answer). Rate-limited. Regulators are created by "
        "platform admins."
    ),
    responses=problem_responses(409, 422, 429, 503),
    dependencies=[Depends(limit_logins)],
)
async def register_organization(
    body: OrganizationRegistration,
    resources: ResourcesDep,
    auth: AuthServiceDep,
    meta: RequestMetaDep,
) -> EmailCodeSent:
    service = OrganizationService(resources.session_factory, resources.redis)
    expires_in = REGISTRATION_CODE_SECONDS
    if await service.register(body, meta):
        expires_in = await auth.send_email_verification(body.admin.email)
    return EmailCodeSent(detail=EMAIL_SENT, expires_in=expires_in)


@router.post(
    "/register/verify",
    response_model=LoginResponse,
    summary="Verify the administrator's e-mail address",
    description=(
        "Proves the address with the e-mailed code. The answer is an MFA enrolment challenge "
        "(`mfa_enrolment_required`): every staff account sets up an authenticator app before "
        "its first session. Five wrong attempts invalidate the code."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "examples": {
                        "mfa_enrolment_required": {
                            "summary": "Verified: set up MFA next",
                            "value": MfaChallengeResponse.model_json_schema()["examples"][0],
                        }
                    }
                }
            }
        },
        **problem_responses(401, 403, 422, 429, 503),
    },
    dependencies=[Depends(limit_logins)],
)
async def verify_registration(
    body: EmailVerify, auth: AuthServiceDep, meta: RequestMetaDep
) -> TokenResponse | MfaChallengeResponse:
    return login_response(await auth.verify_email(body.email, body.code, meta))


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=OrganizationRead,
    summary="Create an organization (platform admins)",
    description=(
        "Platform admins create organizations directly (regulators only this way), then invite "
        "their administrators. Everyone else registers with `POST /organizations/register`. "
        "Access tokens issued before your roles last changed are refused."
    ),
    responses=problem_responses(401, 403, 409, 422),
)
async def create_organization(
    body: OrganizationCreate, actor: PlatformAdmin, resources: ResourcesDep, meta: RequestMetaDep
) -> OrganizationRead:
    service = OrganizationService(resources.session_factory, resources.redis)
    return _organization(await service.create(actor, body, meta))


@router.get(
    "",
    response_model=Page[OrganizationRead],
    summary="List organizations",
    description=(
        "Platform admins: every organization, optionally filtered. Everyone else: the "
        "organizations they belong to. Newest first."
    ),
    responses=problem_responses(401, 403, 422),
)
async def list_organizations(
    actor: SignedIn,
    resources: ResourcesDep,
    params: Params,
    org_status: Annotated[OrganizationStatus | None, Query(alias="status")] = None,
    org_type: Annotated[OrganizationType | None, Query(alias="type")] = None,
) -> Page[OrganizationRead]:
    service = OrganizationService(resources.session_factory, resources.redis)
    rows = await service.list_page(actor, params, status=org_status, kind=org_type)
    return build_page(rows, limit=params.limit, cursor_of=_cursor, to_item=_organization)


@router.get(
    "/{organization_id}",
    response_model=OrganizationRead,
    summary="Get an organization",
    description="Organizations you belong to (platform admins: any). Others answer 404.",
    responses=problem_responses(401, 403, 404),
)
async def get_organization(
    organization_id: UUID, actor: SignedIn, resources: ResourcesDep
) -> OrganizationRead:
    service = OrganizationService(resources.session_factory, resources.redis)
    return _organization(await service.get(actor, organization_id))


@router.post(
    "/{organization_id}/verify",
    response_model=OrganizationRead,
    summary="Verify an organization",
    description=(
        "Platform admins only, after checking its registration. Activates the organization and "
        "its memberships from the members' next request."
    ),
    responses=problem_responses(401, 403, 404, 409),
)
async def verify_organization(
    organization_id: UUID, actor: PlatformAdmin, resources: ResourcesDep, meta: RequestMetaDep
) -> OrganizationRead:
    service = OrganizationService(resources.session_factory, resources.redis)
    return _organization(
        await service.set_status(actor, organization_id, OrganizationStatus.ACTIVE, meta)
    )


@router.post(
    "/{organization_id}/suspend",
    response_model=OrganizationRead,
    summary="Suspend an organization",
    description=(
        "Platform admins only. Its memberships stop granting permissions from the members' next "
        "request; verifying it again restores them."
    ),
    responses=problem_responses(401, 403, 404, 409),
)
async def suspend_organization(
    organization_id: UUID, actor: PlatformAdmin, resources: ResourcesDep, meta: RequestMetaDep
) -> OrganizationRead:
    service = OrganizationService(resources.session_factory, resources.redis)
    return _organization(
        await service.set_status(actor, organization_id, OrganizationStatus.SUSPENDED, meta)
    )


# ------------------------------------------------------------------ invitations
def _invitation(invitation: Invitation) -> InvitationRead:
    return InvitationRead(
        id=invitation.id,
        organization_id=invitation.organization_id,
        role=invitation.role,
        recipient=mask_recipient(invitation),
        status=invitation_status(invitation),
        expires_at=invitation.expires_at,
        created_at=invitation.created_at,
    )


@router.post(
    "/{organization_id}/invitations",
    status_code=status.HTTP_201_CREATED,
    response_model=InvitationCreated,
    summary="Invite someone by e-mail",
    description=(
        "E-mails the invitation (a link when MEDLENS_APP__FRONTEND_URL is set, else the code) "
        "and returns its token once. A new person creates an account from it with "
        "`POST /api/v1/invitations/{token}/accept-new`; someone with an account accepts it "
        "with `POST /api/v1/invitations/accept`, signed in with that verified address. "
        "Invitations expire after 7 days; a new one replaces any open invitation for the same "
        "address. The role must suit the organization and carry no permission you lack. "
        "Platform admins are invited with `POST /api/v1/admin/invitations`."
    ),
    responses=problem_responses(401, 403, 404, 422),
)
async def create_invitation(
    organization_id: UUID,
    body: InvitationCreate,
    actor: Inviter,
    resources: ResourcesDep,
    meta: RequestMetaDep,
) -> InvitationCreated:
    created = await _invitations(resources).create(actor, organization_id, body, meta)
    return InvitationCreated(**_invitation(created.invitation).model_dump(), token=created.token)


@router.get(
    "/{organization_id}/invitations",
    response_model=Page[InvitationRead],
    summary="List an organization's invitations",
    description="Every invitation with its status, newest first. Recipients are masked.",
    responses=problem_responses(401, 403, 404, 422),
)
async def list_invitations(
    organization_id: UUID, actor: Inviter, resources: ResourcesDep, params: Params
) -> Page[InvitationRead]:
    service = InvitationService(resources.session_factory, resources.redis)
    rows = await service.list_page(actor, organization_id, params)
    return build_page(rows, limit=params.limit, cursor_of=_cursor, to_item=_invitation)


@router.delete(
    "/{organization_id}/invitations/{invitation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Revoke an invitation",
    description="The token stops working immediately.",
    responses=problem_responses(401, 403, 404, 409),
)
async def revoke_invitation(
    organization_id: UUID,
    invitation_id: UUID,
    actor: Inviter,
    resources: ResourcesDep,
    meta: RequestMetaDep,
) -> None:
    service = InvitationService(resources.session_factory, resources.redis)
    await service.revoke(actor, organization_id, invitation_id, meta)


@invitations_router.get(
    "/{token}",
    response_model=InvitationPreview,
    summary="Look at an invitation",
    description=(
        "For the invitation's landing page (the e-mailed link): the organization, the role, the "
        "masked address and whether that address already has an account. With an account, sign "
        "in and use `POST /invitations/accept`; without one, `POST /invitations/{token}/"
        "accept-new`. Unknown, used, revoked and expired invitations answer 404. Rate-limited."
    ),
    responses=problem_responses(404, 429, 503),
    dependencies=[Depends(limit_logins)],
)
async def preview_invitation(token: TokenPath, resources: ResourcesDep) -> InvitationPreview:
    service = InvitationService(resources.session_factory, resources.redis)
    found = await service.preview(token)
    invitation, organization = found.invitation, found.organization
    return InvitationPreview(
        organization_name=organization.name if organization else None,
        organization_type=organization.type.value if organization else None,
        role=invitation.role,
        recipient=mask_recipient(invitation),
        expires_at=invitation.expires_at,
        account_exists=found.account_exists,
    )


@invitations_router.post(
    "/{token}/accept-new",
    response_model=LoginResponse,
    summary="Create your account from an invitation",
    description=(
        "For a person without a MedLens account: creates it with the invited address (receiving "
        "the invitation proves it), your name and a password, and joins the organization with "
        "the invited role (or becomes a platform admin). The answer is an MFA enrolment "
        "challenge (`mfa_enrolment_required`): every staff account sets up an authenticator "
        "app before its first session. An address that already has an account answers 409: "
        "sign in and use `POST /invitations/accept`. Rate-limited."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "examples": {
                        "mfa_enrolment_required": {
                            "summary": "Account created: set up MFA next",
                            "value": MfaChallengeResponse.model_json_schema()["examples"][0],
                        }
                    }
                }
            }
        },
        **problem_responses(404, 409, 422, 429, 503),
    },
    dependencies=[Depends(limit_logins)],
)
async def accept_invitation_new_account(
    token: TokenPath,
    body: InvitationAcceptNew,
    resources: ResourcesDep,
    auth: AuthServiceDep,
    meta: RequestMetaDep,
) -> TokenResponse | MfaChallengeResponse:
    service = InvitationService(resources.session_factory, resources.redis)
    accepted = await service.accept_new(
        token, body.full_name, body.password.get_secret_value(), meta
    )
    return login_response(await auth.sign_in_invited(accepted.user.id, meta))


@invitations_router.post(
    "/accept",
    response_model=InvitationAccepted,
    summary="Accept an invitation with your account",
    description=(
        "Joins the organization with the invited role (or makes you a platform admin). You must "
        "be signed in with the verified e-mail address the invitation was sent to; unknown, "
        "expired, used and other people's invitations all answer 404. An account without a "
        "password sets one here (`password`). An account that becomes a staff account here is "
        "signed out everywhere (`signed_out: true`): staff sign in with e-mail, password and an "
        "authenticator app, which it sets up on its next sign-in."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def accept_invitation(
    body: InvitationAccept,
    principal: SessionCheckedUser,
    resources: ResourcesDep,
    auth: AuthServiceDep,
    meta: RequestMetaDep,
) -> InvitationAccepted:
    service = InvitationService(resources.session_factory, resources.redis)
    password = body.password.get_secret_value() if body.password is not None else None
    accepted = await service.accept(
        principal.require_user_id(), body.token, password, meta, end_sessions=auth.end_sessions
    )
    if accepted.ended_sessions:
        await auth.publish_revocations(list(accepted.ended_sessions))
    return InvitationAccepted(
        organization_id=accepted.organization_id,
        role=accepted.role,
        signed_out=accepted.became_staff,
    )


# ------------------------------------------------------------------ platform-admin invitations
platform_invitations_router = api_v1_router("/admin/invitations", tags=["admin"])


@platform_invitations_router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=InvitationCreated,
    summary="Invite a platform admin",
    description=(
        "E-mails an invitation to become a platform admin and returns its token once. The "
        "invitee creates an account from it (`POST /invitations/{token}/accept-new`) or, with "
        "an account, accepts it signed in. A new invitation replaces an open one for the same "
        "address. Access tokens issued before your roles last changed are refused."
    ),
    responses=problem_responses(401, 403, 422),
)
async def create_platform_invitation(
    body: PlatformInvitationCreate,
    actor: PlatformAdmin,
    resources: ResourcesDep,
    meta: RequestMetaDep,
) -> InvitationCreated:
    service = _invitations(resources)
    created = await service.create_platform(actor.user_id, body.email, meta)
    return InvitationCreated(**_invitation(created.invitation).model_dump(), token=created.token)


@platform_invitations_router.get(
    "",
    response_model=Page[InvitationRead],
    summary="List platform-admin invitations",
    description="Every platform-admin invitation with its status, newest first; masked.",
    responses=problem_responses(401, 403, 422),
    dependencies=[Depends(require_permissions(Permission.PLATFORM_ADMIN, fresh=True))],
)
async def list_platform_invitations(
    resources: ResourcesDep, params: Params
) -> Page[InvitationRead]:
    rows = await _invitations(resources).list_platform_page(params)
    return build_page(rows, limit=params.limit, cursor_of=_cursor, to_item=_invitation)


@platform_invitations_router.delete(
    "/{invitation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Revoke a platform-admin invitation",
    description="The token stops working immediately.",
    responses=problem_responses(401, 403, 404, 409),
)
async def revoke_platform_invitation(
    invitation_id: UUID, actor: PlatformAdmin, resources: ResourcesDep, meta: RequestMetaDep
) -> None:
    await _invitations(resources).revoke(actor, None, invitation_id, meta)


# ------------------------------------------------------------------ branches
def _branch(branch: Branch) -> BranchRead:
    coordinates = point_coordinates(branch.location)
    return BranchRead(
        id=branch.id,
        organization_id=branch.organization_id,
        name=branch.name,
        address=branch.address,
        state=branch.state,
        latitude=coordinates[0] if coordinates else None,
        longitude=coordinates[1] if coordinates else None,
        created_at=branch.created_at,
    )


@router.get(
    "/{organization_id}/branches",
    response_model=Page[BranchRead],
    summary="List an organization's branches",
    description="Premises of your own organization (platform admins: any), newest first.",
    responses=problem_responses(401, 403, 404, 422),
)
async def list_branches(
    organization_id: UUID, actor: Reader, resources: ResourcesDep, params: Params
) -> Page[BranchRead]:
    rows = await BranchService(resources.session_factory).list_page(actor, organization_id, params)
    return build_page(rows, limit=params.limit, cursor_of=_cursor, to_item=_branch)


@router.post(
    "/{organization_id}/branches",
    status_code=status.HTTP_201_CREATED,
    response_model=BranchRead,
    summary="Add a branch",
    description="A premises of the organization, optionally with its location.",
    responses=problem_responses(401, 403, 404, 422),
)
async def create_branch(
    organization_id: UUID,
    body: BranchCreate,
    actor: OrgManager,
    resources: ResourcesDep,
    meta: RequestMetaDep,
) -> BranchRead:
    service = BranchService(resources.session_factory)
    return _branch(await service.create(actor, organization_id, body, meta))


@router.patch(
    "/{organization_id}/branches/{branch_id}",
    response_model=BranchRead,
    summary="Update a branch",
    description="Only the fields sent change; latitude and longitude change together.",
    responses=problem_responses(401, 403, 404, 422),
)
async def update_branch(
    organization_id: UUID,
    branch_id: UUID,
    body: BranchUpdate,
    actor: OrgManager,
    resources: ResourcesDep,
    meta: RequestMetaDep,
) -> BranchRead:
    service = BranchService(resources.session_factory)
    return _branch(await service.update(actor, organization_id, branch_id, body.changes(), meta))


@router.delete(
    "/{organization_id}/branches/{branch_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Remove a branch",
    description="Soft delete: the branch disappears from lists; its history stays.",
    responses=problem_responses(401, 403, 404),
)
async def delete_branch(
    organization_id: UUID,
    branch_id: UUID,
    actor: OrgManager,
    resources: ResourcesDep,
    meta: RequestMetaDep,
) -> None:
    await BranchService(resources.session_factory).delete(actor, organization_id, branch_id, meta)

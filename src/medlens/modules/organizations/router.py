from typing import Annotated
from uuid import UUID

from fastapi import Depends, Query, Response, status

from medlens.core.geo import point_coordinates
from medlens.core.openapi import problem_responses
from medlens.core.pagination import Cursor, Page, PageParams, build_page, page_params
from medlens.core.routing import api_v1_router
from medlens.modules.auth.deps import Meta as RequestMetaDep
from medlens.modules.auth.deps import SessionCheckedUser
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
    InvitationCreate,
    InvitationCreated,
    InvitationRead,
    MemberRead,
    OrganizationCreate,
    OrganizationRead,
    RoleAssignment,
)
from medlens.modules.organizations.service import MembershipService
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import get_actor, require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.modules.users.models import User
from medlens.resources import Resources, get_resources

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
OrgCreator = Annotated[Actor, Depends(get_actor)]
PlatformAdmin = Annotated[
    Actor, Depends(require_permissions(Permission.PLATFORM_ADMIN, fresh=True))
]
Inviter = Annotated[Actor, Depends(require_permissions(Permission.MEMBER_INVITE, fresh=True))]
OrgManager = Annotated[Actor, Depends(require_permissions(Permission.ORG_MANAGE))]
Params = Annotated[PageParams, Depends(page_params)]
ResourcesDep = Annotated[Resources, Depends(get_resources)]


def _organization(organization: Organization) -> OrganizationRead:
    return OrganizationRead.model_validate(organization)


def _cursor(row: Organization | Invitation | Branch) -> Cursor:
    return Cursor(created_at=row.created_at, id=row.id)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=OrganizationRead,
    summary="Register an organization",
    description=(
        "Registers a pharmacy, PPMV, clinic, distributor or manufacturer; you become its "
        "administrator. It starts as `pending_verification`: memberships grant nothing until a "
        "platform admin verifies it. Regulators are created by platform admins only."
    ),
    responses=problem_responses(401, 403, 409, 422),
)
async def create_organization(
    body: OrganizationCreate, actor: OrgCreator, resources: ResourcesDep, meta: RequestMetaDep
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
    actor: OrgCreator,
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
    organization_id: UUID, actor: OrgCreator, resources: ResourcesDep
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
    summary="Invite someone by phone number or e-mail address",
    description=(
        "Returns a token once; MedLens does not send it. The invitee accepts it with "
        "`POST /api/v1/invitations/accept` while signed in with that phone number or e-mail "
        "address. Invitations expire after 7 days; a new one replaces any open invitation for "
        "the same person. The role must suit the organization and carry no permission you lack."
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
    service = InvitationService(resources.session_factory, resources.redis)
    created = await service.create(actor, organization_id, body, meta)
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


@invitations_router.post(
    "/accept",
    response_model=MemberRead,
    summary="Accept an invitation",
    description=(
        "Joins the organization with the invited role. You must be signed in with the phone "
        "number or e-mail address the invitation was sent to; unknown, expired, used and "
        "other people's invitations all answer 404."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def accept_invitation(
    body: InvitationAccept,
    principal: SessionCheckedUser,
    resources: ResourcesDep,
    meta: RequestMetaDep,
) -> MemberRead:
    service = InvitationService(resources.session_factory, resources.redis)
    membership, user = await service.accept(principal.require_user_id(), body.token, meta)
    return _member(membership, user)


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

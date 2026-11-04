"""Creating and verifying organizations, and inviting people into them.

* Anyone signed in may register a pharmacy, PPMV, clinic, distributor or manufacturer; it starts
  as ``pending_verification`` and its creator becomes its administrator. Memberships of an
  unverified organization grant nothing until a platform admin verifies it.
* Regulators are created by platform admins only: a verified regulator sees every scan in its
  jurisdiction.
* Invitations go to a phone number or e-mail address. MedLens sends no messages besides login
  codes, so the inviter gets the token once and passes it on; the invitee accepts it while
  signed in with that phone number or e-mail address. Tokens expire after INVITATION_TTL.
* Suspending or re-activating an organization bumps every member's perm_ver, so the change
  applies on their next request rather than when cached permissions expire.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import ColumnElement, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.crypto import phone_lookup_hash
from medlens.core.errors import Conflict, Forbidden, NotFound
from medlens.core.pagination import PageParams
from medlens.core.request_meta import RequestMeta
from medlens.core.security import new_opaque_token, token_digest
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.organizations.models import (
    Invitation,
    Membership,
    MembershipStatus,
    Organization,
    OrganizationStatus,
    OrganizationType,
)
from medlens.modules.organizations.repository import MembershipRepository
from medlens.modules.organizations.schemas import (
    InvitationCreate,
    InvitationStatus,
    OrganizationCreate,
)
from medlens.modules.organizations.service import validate_role_for
from medlens.modules.rbac import policies
from medlens.modules.rbac.access import Actor, PermissionVersions, bump_perm_ver
from medlens.modules.rbac.models import RoleName
from medlens.modules.rbac.service import ensure_no_escalation, role_permissions
from medlens.modules.users.models import User, UserStatus

INVITATION_TTL = timedelta(days=7)
_NOT_FOUND = "Organization not found."
# The role an organization's creator receives.
_FOUNDER_ROLES = {
    OrganizationType.PHARMACY: RoleName.PHARMACY_ADMIN,
    OrganizationType.PPMV: RoleName.PHARMACY_ADMIN,
    OrganizationType.CLINIC: RoleName.PHARMACY_ADMIN,
    OrganizationType.DISTRIBUTOR: RoleName.PHARMACY_ADMIN,
    OrganizationType.MANUFACTURER: RoleName.MANUFACTURER_ADMIN,
    OrganizationType.REGULATOR: RoleName.REGULATOR_ADMIN,
}


def _now() -> datetime:
    return datetime.now(UTC)


def _org_id(actor: Actor) -> UUID | None:
    return actor.organization.id if actor.organization else None


class OrganizationService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], redis: Redis) -> None:
        self._sessions = sessions
        self._versions = PermissionVersions(redis)

    async def create(
        self, actor: Actor, body: OrganizationCreate, meta: RequestMeta
    ) -> Organization:
        user_id = actor.user_id
        if user_id is None:
            raise Forbidden("Organizations are registered by people, not API keys.")
        if body.type is OrganizationType.REGULATOR and not actor.is_platform_admin:
            raise Forbidden("Regulator organizations are created by platform administrators.")
        version: int | None = None
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            organization = Organization(
                name=body.name,
                type=body.type,
                registration_number=body.registration_number,
                state=body.state,
                jurisdiction_states=(
                    sorted({state.value for state in body.jurisdiction_states})
                    if body.jurisdiction_states
                    else None
                ),
            )
            session.add(organization)
            try:
                await session.flush()
            except IntegrityError:
                raise Conflict(
                    "An organization of this type with this registration number already exists."
                ) from None
            if not actor.is_platform_admin:
                session.add(
                    Membership(
                        organization_id=organization.id,
                        user_id=user_id,
                        role=_FOUNDER_ROLES[body.type].value,
                    )
                )
                version = await bump_perm_ver(session, user_id)
            await AuditService(session).record(
                "organization.created",
                actor_user_id=user_id,
                resource_type="organization",
                resource_id=organization.id,
                meta=meta,
                details={"type": body.type.value},
            )
            await uow.commit()
        if version is not None:
            await self._versions.remember(user_id, version)
        return organization

    async def list_page(
        self,
        actor: Actor,
        params: PageParams,
        *,
        status: OrganizationStatus | None = None,
        kind: OrganizationType | None = None,
    ) -> list[Organization]:
        """Platform admins: every organization. Everyone else: those they belong to."""
        filters: list[ColumnElement[bool]] = [Organization.deleted_at.is_(None)]
        if not actor.is_platform_admin:
            if actor.user_id is None:
                raise Forbidden("This endpoint is for signed-in users, not API keys.")
            member_of = select(Membership.organization_id).where(
                Membership.user_id == actor.user_id
            )
            filters.append(Organization.id.in_(member_of))
        if status is not None:
            filters.append(Organization.status == status)
        if kind is not None:
            filters.append(Organization.type == kind)
        async with UnitOfWork(self._sessions) as uow:
            return await MembershipRepository(uow.session).organizations_page(params, *filters)

    async def get(self, actor: Actor, organization_id: UUID) -> Organization:
        async with UnitOfWork(self._sessions) as uow:
            repo = MembershipRepository(uow.session)
            organization = await repo.organization(organization_id)
            visible = actor.is_platform_admin or (
                actor.user_id is not None
                and await repo.membership(organization_id, actor.user_id, lock=False) is not None
            )
            if organization is None or not visible:
                raise NotFound(_NOT_FOUND)
            return organization

    async def set_status(
        self,
        actor: Actor,
        organization_id: UUID,
        status: OrganizationStatus,
        meta: RequestMeta,
    ) -> Organization:
        if not actor.is_platform_admin:
            raise Forbidden("Only platform administrators verify or suspend organizations.")
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            organization = await session.get(Organization, organization_id, with_for_update=True)
            if organization is None or organization.deleted_at is not None:
                raise NotFound(_NOT_FOUND)
            previous = organization.status
            if previous is status:
                raise Conflict(f"The organization is already {status.value}.")
            organization.status = status
            if status is OrganizationStatus.ACTIVE and organization.verified_at is None:
                organization.verified_at = _now()
            bumped = await _bump_members(session, organization_id)
            await AuditService(session).record(
                f"organization.{_STATUS_ACTIONS[status]}",
                actor_user_id=actor.user_id,
                resource_type="organization",
                resource_id=organization_id,
                meta=meta,
                details={"from": previous.value, "to": status.value},
            )
            await uow.commit()
        for user_id, version in bumped:
            await self._versions.remember(user_id, version)
        return organization


_STATUS_ACTIONS = {
    OrganizationStatus.ACTIVE: "verified",
    OrganizationStatus.SUSPENDED: "suspended",
    OrganizationStatus.PENDING_VERIFICATION: "unverified",
}


async def _bump_members(session: AsyncSession, organization_id: UUID) -> list[tuple[UUID, int]]:
    members = select(Membership.user_id).where(Membership.organization_id == organization_id)
    result = await session.execute(
        update(User)
        .where(User.id.in_(members))
        .values(perm_ver=User.perm_ver + 1)
        .returning(User.id, User.perm_ver)
    )
    return [(user_id, version) for user_id, version in result]


# --------------------------------------------------------------------------- invitations
@dataclass(frozen=True, slots=True)
class CreatedInvitation:
    invitation: Invitation
    token: str
    recipient: str


def invitation_status(invitation: Invitation, now: datetime | None = None) -> InvitationStatus:
    if invitation.accepted_at is not None:
        return "accepted"
    if invitation.revoked_at is not None:
        return "revoked"
    return "expired" if invitation.expires_at <= (now or _now()) else "pending"


def mask_recipient(invitation: Invitation) -> str:
    if invitation.email:
        local, _, domain = invitation.email.partition("@")
        return f"{local[:2]}***@{domain}"
    phone = invitation.phone_e164 or ""
    return f"{phone[:7]}*****{phone[-2:]}" if len(phone) > 9 else "***"


class InvitationService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], redis: Redis) -> None:
        self._sessions = sessions
        self._versions = PermissionVersions(redis)

    async def create(
        self, actor: Actor, organization_id: UUID, body: InvitationCreate, meta: RequestMeta
    ) -> CreatedInvitation:
        if not policies.can_invite_members(actor, organization_id):
            raise NotFound(_NOT_FOUND)
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            organization = await MembershipRepository(session).organization(organization_id)
            if organization is None:
                raise NotFound(_NOT_FOUND)
            validate_role_for(organization, body.role)
            ensure_no_escalation(actor, await role_permissions(session, body.role.value))
            # A new invitation replaces any open one for the same person.
            same_recipient = (
                Invitation.email == body.email
                if body.email
                else Invitation.phone_hash == phone_lookup_hash(body.phone or "")
            )
            await session.execute(
                update(Invitation)
                .where(
                    Invitation.organization_id == organization_id,
                    same_recipient,
                    Invitation.accepted_at.is_(None),
                    Invitation.revoked_at.is_(None),
                )
                .values(revoked_at=_now())
            )
            token = new_opaque_token()
            invitation = Invitation(
                organization_id=organization_id,
                role=body.role.value,
                email=body.email,
                token_hash=token_digest(token),
                expires_at=_now() + INVITATION_TTL,
                invited_by=actor.user_id,
            )
            if body.phone:
                invitation.set_phone(body.phone)
            session.add(invitation)
            await session.flush()
            await AuditService(session).record(
                "invitation.created",
                actor_user_id=actor.user_id,
                actor_org_id=_org_id(actor),
                resource_type="invitation",
                resource_id=invitation.id,
                meta=meta,
                details={"organization_id": str(organization_id), "role": body.role.value},
            )
            await uow.commit()
        return CreatedInvitation(invitation, token, mask_recipient(invitation))

    async def list_page(
        self, actor: Actor, organization_id: UUID, params: PageParams
    ) -> list[Invitation]:
        if not policies.can_invite_members(actor, organization_id):
            raise NotFound(_NOT_FOUND)
        async with UnitOfWork(self._sessions) as uow:
            return await MembershipRepository(uow.session).invitations_page(organization_id, params)

    async def revoke(
        self, actor: Actor, organization_id: UUID, invitation_id: UUID, meta: RequestMeta
    ) -> None:
        if not policies.can_invite_members(actor, organization_id):
            raise NotFound(_NOT_FOUND)
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            invitation = await session.get(Invitation, invitation_id, with_for_update=True)
            if invitation is None or invitation.organization_id != organization_id:
                raise NotFound("Invitation not found.")
            if invitation_status(invitation) != "pending":
                raise Conflict(f"The invitation is already {invitation_status(invitation)}.")
            invitation.revoked_at = _now()
            await AuditService(session).record(
                "invitation.revoked",
                actor_user_id=actor.user_id,
                actor_org_id=_org_id(actor),
                resource_type="invitation",
                resource_id=invitation_id,
                meta=meta,
            )
            await uow.commit()

    async def accept(self, user_id: UUID, token: str, meta: RequestMeta) -> tuple[Membership, User]:
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            invitation = await session.scalar(
                select(Invitation)
                .where(Invitation.token_hash == token_digest(token))
                .with_for_update()
            )
            user = await session.get(User, user_id)
            if user is None or user.status is not UserStatus.ACTIVE:
                raise Forbidden("This account cannot accept invitations.")
            # One answer for unknown, used, expired and someone else's invitations.
            addressed_to_user = invitation is not None and (
                (
                    invitation.email is not None
                    and user.email is not None
                    and invitation.email.lower() == user.email.lower()  # CITEXT semantics
                )
                or (invitation.phone_hash is not None and invitation.phone_hash == user.phone_hash)
            )
            if (
                invitation is None
                or not addressed_to_user
                or invitation_status(invitation) != "pending"
            ):
                raise NotFound("This invitation is not valid for your account.")
            repo = MembershipRepository(session)
            if await repo.membership(invitation.organization_id, user_id) is not None:
                raise Conflict("You are already a member of this organization.")
            membership = Membership(
                organization_id=invitation.organization_id,
                user_id=user_id,
                role=invitation.role,
                status=MembershipStatus.ACTIVE,
                invited_by=invitation.invited_by,
            )
            session.add(membership)
            invitation.accepted_at = _now()
            invitation.accepted_by = user_id
            version = await bump_perm_ver(session, user_id)
            await AuditService(session).record(
                "invitation.accepted",
                actor_user_id=user_id,
                actor_org_id=invitation.organization_id,
                resource_type="invitation",
                resource_id=invitation.id,
                meta=meta,
                details={"role": invitation.role},
            )
            await uow.commit()
        await self._versions.remember(user_id, version)
        return membership, user

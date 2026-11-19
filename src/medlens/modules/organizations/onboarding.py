"""Creating and verifying organizations, and inviting people into them.

* Pharmacies, PPMVs, clinics, distributors and manufacturers register themselves through the
  public form (``register``): the organization starts as ``pending_verification`` and its first
  administrator's memberships grant nothing until a platform admin checks the licence.
* Regulators and other organizations set up on someone's behalf are created by platform
  admins only.
* Staff join by invitation, which goes to an e-mail address (with a link when the frontend URL
  is configured; the inviter also gets the token once). A new person creates the account from
  the invitation, which verifies the address; an existing account accepts while signed in with
  that verified address. Platform admins are invited the same way (organization_id NULL); the
  first one by the bootstrap command. Tokens expire after INVITATION_TTL.
* Suspending or re-activating an organization bumps every member's perm_ver, so the change
  applies on their next request rather than when cached permissions expire.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import structlog
from redis.asyncio import Redis
from sqlalchemy import ColumnElement, exists, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.errors import Conflict, Forbidden, NotFound, ValidationFailed
from medlens.core.pagination import PageParams
from medlens.core.request_meta import RequestMeta
from medlens.core.security import hash_password, new_opaque_token, token_digest
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.auth.passwords import password_problems
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
    OrganizationRegistration,
)
from medlens.modules.organizations.service import validate_role_for
from medlens.modules.rbac import policies
from medlens.modules.rbac.access import Actor, PermissionVersions, bump_perm_ver
from medlens.modules.rbac.models import PUBLIC_ROLES, RoleName
from medlens.modules.rbac.service import ensure_no_escalation, role_permissions
from medlens.modules.users.models import User, UserStatus
from medlens.ports.email import EmailDeliveryError, EmailPort, EmailUnavailableError

log = structlog.stdlib.get_logger(__name__)

INVITATION_TTL = timedelta(days=7)
_NOT_FOUND = "Organization not found."
_BOOTSTRAP_LOCK = 0x4D4C_4241  # "MLBA": one bootstrap at a time
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
        """Platform admins create organizations directly (regulators only this way); the
        organization's administrators are then invited. Everyone else uses ``register``."""
        if actor.user_id is None or not actor.is_platform_admin:
            raise Forbidden("Organizations are created by platform administrators.")
        async with UnitOfWork(self._sessions) as uow:
            organization = await _new_organization(uow.session, body)
            await AuditService(uow.session).record(
                "organization.created",
                actor_user_id=actor.user_id,
                resource_type="organization",
                resource_id=organization.id,
                meta=meta,
                details={"type": body.type.value},
            )
            await uow.commit()
        return organization

    async def register(self, form: OrganizationRegistration, meta: RequestMeta) -> bool:
        """The public sign-up form: the organization (pending verification: it grants nothing
        until a platform admin checks its licence) and its first administrator, whose address
        must then be verified. Returns whether a verification code should be sent: an address
        that already has an account gets no new account and no new organization (the same
        answer to the caller); an unverified one from an unfinished registration gets a new
        code."""
        admin = form.admin
        problems = password_problems(admin.password.get_secret_value(), email=admin.email)
        if problems:
            raise ValidationFailed(
                "The password is not acceptable.",
                errors=[
                    {"loc": ["body", "admin", "password"], "msg": problem, "type": "password"}
                    for problem in problems
                ],
            )
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            existing = await session.scalar(select(User).where(User.email == admin.email))
            audit = AuditService(session)
            if existing is not None:
                await audit.record(
                    "organization.registration_existing_account",
                    actor_user_id=existing.id,
                    meta=meta,
                )
                await uow.commit()
                return existing.email_verified_at is None
            organization = await _new_organization(session, form.organization)
            founder = User(
                email=admin.email,
                full_name=admin.full_name,
                password_hash=hash_password(admin.password.get_secret_value()),
            )
            session.add(founder)
            await session.flush()
            session.add(
                Membership(
                    organization_id=organization.id,
                    user_id=founder.id,
                    role=_FOUNDER_ROLES[form.organization.type].value,
                )
            )
            await audit.record(
                "organization.registered",
                actor_user_id=founder.id,
                resource_type="organization",
                resource_id=organization.id,
                meta=meta,
                details={"type": form.organization.type.value},
            )
            await uow.commit()
        return True

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
    emailed: bool = False


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


INVITATION_SUBJECT = "You are invited to {organization} on MedLens"
INVITATION_TEXT = (
    "You have been invited to join {organization} on MedLens as {role}.\n\n"
    "{how}\n\n"
    "The invitation expires in 7 days. If you were not expecting it, ignore this message."
)
HOW_LINK = "Open this link to accept: {url}"
HOW_CODE = (
    "To accept, open MedLens and enter this invitation code: {token}\n"
    "New to MedLens? The code lets you create your account with this address."
)
PLATFORM = "the MedLens platform team"
PLATFORM_SUBJECT = "You are invited to administer MedLens"


@dataclass(frozen=True, slots=True)
class InvitationPreviewData:
    organization: Organization | None
    invitation: Invitation
    account_exists: bool


@dataclass(frozen=True, slots=True)
class AcceptedInvitation:
    user: User
    organization_id: UUID | None
    role: str
    became_staff: bool  # its passwordless sessions must end: staff sign in with MFA
    perm_ver: int
    ended_sessions: tuple[UUID, ...] = ()


# Ends a user's sessions inside the caller's transaction (AuthService.end_sessions).
SessionEnder = Callable[[AsyncSession, UUID], Awaitable[list[UUID]]]


async def _is_staff(session: AsyncSession, user: User) -> bool:
    if user.global_role not in PUBLIC_ROLES:
        return True
    return bool(await session.scalar(select(exists().where(Membership.user_id == user.id))))


async def _new_organization(session: AsyncSession, body: OrganizationCreate) -> Organization:
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
    return organization


class InvitationService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        redis: Redis,
        email: EmailPort | None = None,
        frontend_url: str | None = None,
    ) -> None:
        self._sessions = sessions
        self._versions = PermissionVersions(redis)
        self._email = email
        self._frontend_url = frontend_url.rstrip("/") if frontend_url else None

    # -------------------------------------------------------------- creating
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
            created = await self._new(
                session, actor, organization_id, body.role.value, body.email, meta=meta
            )
            await uow.commit()
        return replace(created, emailed=await self._send(created, organization.name))

    async def create_platform(
        self, actor_user_id: UUID | None, email: str, meta: RequestMeta
    ) -> CreatedInvitation:
        """An invitation to become a platform admin, from a platform admin (the route checks
        that, with fresh MFA)."""
        async with UnitOfWork(self._sessions) as uow:
            created = await self._new(
                uow.session, None, None, RoleName.PLATFORM_ADMIN.value, email, actor_user_id, meta
            )
            await uow.commit()
        return replace(created, emailed=await self._send(created, PLATFORM))

    async def bootstrap_platform(self, email: str, meta: RequestMeta) -> CreatedInvitation:
        """The first platform admin's invitation (the bootstrap command, run by an operator).
        Refused once an active platform admin exists: from then on, admins invite admins."""
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            # Serialises concurrent bootstraps on the transaction-scoped advisory lock.
            await session.execute(select(func.pg_advisory_xact_lock(_BOOTSTRAP_LOCK)))
            admin = await session.scalar(
                select(User.id).where(
                    User.global_role == RoleName.PLATFORM_ADMIN.value,
                    User.status == UserStatus.ACTIVE,
                )
            )
            if admin is not None:
                raise Conflict(
                    "A platform admin already exists: platform admins invite further admins."
                )
            created = await self._new(
                session, None, None, RoleName.PLATFORM_ADMIN.value, email, None, meta
            )
            await AuditService(session).record(
                "platform.bootstrap_invitation_created",
                resource_type="invitation",
                resource_id=created.invitation.id,
                meta=meta,
            )
            await uow.commit()
        return replace(created, emailed=await self._send(created, PLATFORM))

    async def _new(
        self,
        session: AsyncSession,
        actor: Actor | None,
        organization_id: UUID | None,
        role: str,
        email: str,
        actor_user_id: UUID | None = None,
        meta: RequestMeta | None = None,
    ) -> CreatedInvitation:
        # A new invitation replaces any open one for the same person and scope.
        scope = (
            Invitation.organization_id == organization_id
            if organization_id is not None
            else Invitation.organization_id.is_(None)
        )
        await session.execute(
            update(Invitation)
            .where(
                scope,
                Invitation.email == email,
                Invitation.accepted_at.is_(None),
                Invitation.revoked_at.is_(None),
            )
            .values(revoked_at=_now())
        )
        token = new_opaque_token()
        inviter = actor.user_id if actor is not None else actor_user_id
        invitation = Invitation(
            organization_id=organization_id,
            role=role,
            email=email,
            token_hash=token_digest(token),
            expires_at=_now() + INVITATION_TTL,
            invited_by=inviter,
        )
        session.add(invitation)
        await session.flush()
        await AuditService(session).record(
            "invitation.created",
            actor_user_id=inviter,
            actor_org_id=_org_id(actor) if actor is not None else None,
            resource_type="invitation",
            resource_id=invitation.id,
            meta=meta,
            details={
                "organization_id": str(organization_id) if organization_id else None,
                "role": role,
            },
        )
        return CreatedInvitation(invitation, token, mask_recipient(invitation))

    async def _send(self, created: CreatedInvitation, organization: str) -> bool:
        """E-mails the invitation (the inviter also gets the token); True when it was sent. A
        failure to send does not undo the invitation: the inviter can pass the token on."""
        invitation = created.invitation
        if self._email is None or invitation.email is None:
            return False
        how = (
            HOW_LINK.format(url=f"{self._frontend_url}/invitations/{created.token}")
            if self._frontend_url
            else HOW_CODE.format(token=created.token)
        )
        try:
            await self._email.send(
                invitation.email,
                PLATFORM_SUBJECT
                if invitation.organization_id is None
                else INVITATION_SUBJECT.format(organization=organization),
                INVITATION_TEXT.format(
                    organization=organization, role=invitation.role.replace("_", " "), how=how
                ),
            )
        except (EmailUnavailableError, EmailDeliveryError) as exc:
            log.warning("invitation.email_not_sent", error=type(exc).__name__)
            return False
        return True

    # -------------------------------------------------------------- listing and revoking
    async def list_page(
        self, actor: Actor, organization_id: UUID, params: PageParams
    ) -> list[Invitation]:
        if not policies.can_invite_members(actor, organization_id):
            raise NotFound(_NOT_FOUND)
        async with UnitOfWork(self._sessions) as uow:
            return await MembershipRepository(uow.session).invitations_page(organization_id, params)

    async def list_platform_page(self, params: PageParams) -> list[Invitation]:
        async with UnitOfWork(self._sessions) as uow:
            return await MembershipRepository(uow.session).invitations_page(None, params)

    async def revoke(
        self,
        actor: Actor,
        organization_id: UUID | None,
        invitation_id: UUID,
        meta: RequestMeta,
    ) -> None:
        if organization_id is not None and not policies.can_invite_members(actor, organization_id):
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

    # -------------------------------------------------------------- accepting
    async def _pending(self, session: AsyncSession, token: str) -> Invitation | None:
        invitation = await session.scalar(
            select(Invitation).where(Invitation.token_hash == token_digest(token)).with_for_update()
        )
        if invitation is None or invitation_status(invitation) != "pending":
            return None
        return invitation

    async def preview(self, token: str) -> InvitationPreviewData:
        """What a pending invitation offers; unknown, used, revoked and expired ones are 404."""
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            invitation = await session.scalar(
                select(Invitation).where(Invitation.token_hash == token_digest(token))
            )
            if invitation is None or invitation_status(invitation) != "pending":
                raise NotFound(_INVALID_INVITATION)
            organization = (
                await session.get(Organization, invitation.organization_id)
                if invitation.organization_id
                else None
            )
            account = (
                await session.scalar(select(User.id).where(User.email == invitation.email))
                if invitation.email
                else None
            )
            return InvitationPreviewData(organization, invitation, account is not None)

    async def accept_new(
        self, token: str, full_name: str, password: str, meta: RequestMeta
    ) -> AcceptedInvitation:
        """A new person creates their account from an e-mail invitation: receiving the token
        proves the address, so it is verified at once. An address with an account must sign
        in and accept instead (409)."""
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            invitation = await self._pending(session, token)
            if invitation is None or invitation.email is None:
                raise NotFound(_INVALID_INVITATION)
            if await session.scalar(select(User.id).where(User.email == invitation.email)):
                raise Conflict(
                    "This address already has an account: sign in and accept the invitation."
                )
            _password_ok(password, invitation.email)
            user = User(
                email=invitation.email,
                email_verified_at=_now(),
                full_name=full_name,
                password_hash=hash_password(password),
            )
            session.add(user)
            await session.flush()
            accepted = await self._apply(session, invitation, user, meta)
            await uow.commit()
        await self._versions.remember(accepted.user.id, accepted.perm_ver)
        return accepted

    async def accept(
        self,
        user_id: UUID,
        token: str,
        password: str | None,
        meta: RequestMeta,
        end_sessions: SessionEnder | None = None,
    ) -> AcceptedInvitation:
        """A signed-in account with the invited, verified address accepts. An account without a
        password (it signed in with codes) sets one here: staff sign in with a password and
        MFA. An account that becomes staff here has its sessions ended in the same transaction
        (``end_sessions``): they were opened without MFA."""
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            invitation = await self._pending(session, token)
            user = await session.get(User, user_id)
            if user is None or user.status is not UserStatus.ACTIVE:
                raise Forbidden("This account cannot accept invitations.")
            # One answer for unknown, used, expired and someone else's invitations.
            addressed_to_user = invitation is not None and (
                (
                    invitation.email is not None
                    and user.email is not None
                    and user.email_verified_at is not None  # a proven address only
                    and invitation.email.lower() == user.email.lower()  # CITEXT semantics
                )
                or (invitation.phone_hash is not None and invitation.phone_hash == user.phone_hash)
            )
            if invitation is None or not addressed_to_user:
                raise NotFound("This invitation is not valid for your account.")
            if user.password_hash is None:
                if password is None:
                    raise ValidationFailed(
                        "Set a password to accept: staff sign in with e-mail, password and an "
                        "authenticator app.",
                        errors=[
                            {"loc": ["body", "password"], "msg": "required", "type": "missing"}
                        ],
                    )
                _password_ok(password, user.email or "")
                user.password_hash = hash_password(password)
            accepted = await self._apply(session, invitation, user, meta)
            if accepted.became_staff and end_sessions is not None:
                ended = await end_sessions(session, user.id)
                accepted = replace(accepted, ended_sessions=tuple(ended))
            await uow.commit()
        await self._versions.remember(accepted.user.id, accepted.perm_ver)
        return accepted

    async def _apply(
        self, session: AsyncSession, invitation: Invitation, user: User, meta: RequestMeta
    ) -> AcceptedInvitation:
        was_staff = await _is_staff(session, user)
        if invitation.organization_id is None:  # a platform-admin invitation
            if user.global_role == RoleName.PLATFORM_ADMIN.value:
                raise Conflict("You are already a platform admin.")
            user.global_role = RoleName.PLATFORM_ADMIN.value
        else:
            repo = MembershipRepository(session)
            if await repo.membership(invitation.organization_id, user.id) is not None:
                raise Conflict("You are already a member of this organization.")
            session.add(
                Membership(
                    organization_id=invitation.organization_id,
                    user_id=user.id,
                    role=invitation.role,
                    status=MembershipStatus.ACTIVE,
                    invited_by=invitation.invited_by,
                )
            )
        invitation.accepted_at = _now()
        invitation.accepted_by = user.id
        version = await bump_perm_ver(session, user.id)
        await AuditService(session).record(
            "invitation.accepted",
            actor_user_id=user.id,
            actor_org_id=invitation.organization_id,
            resource_type="invitation",
            resource_id=invitation.id,
            meta=meta,
            details={"role": invitation.role},
        )
        await session.flush()
        return AcceptedInvitation(
            user,
            invitation.organization_id,
            invitation.role,
            became_staff=not was_staff,
            perm_ver=version,
        )


_INVALID_INVITATION = "This invitation is invalid, used or expired."


def _password_ok(password: str, email: str) -> None:
    problems = password_problems(password, email=email)
    if problems:
        raise ValidationFailed(
            "The password is not acceptable.",
            errors=[
                {"loc": ["body", "password"], "msg": problem, "type": "password"}
                for problem in problems
            ],
        )

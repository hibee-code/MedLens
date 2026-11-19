"""One-link invitations, platform-admin invitations and the first admin's bootstrap
(decision log, "Accounts: invitations and the first platform admin")."""

from typing import Any

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.adapters.email import ConsoleEmail
from medlens.core.errors import Conflict
from medlens.core.request_meta import RequestMeta
from medlens.modules.organizations.models import Invitation, Membership
from medlens.modules.organizations.onboarding import InvitationService
from medlens.modules.rbac.models import RoleName
from medlens.modules.users.models import User, UserStatus
from tests.integration.api_support import Api, new_email, new_phone

pytestmark = pytest.mark.integration
NEW_PASSWORD = "Tafawa-Balewa-Square-1960"  # pragma: allowlist secret
META = RequestMeta(ip=None, user_agent="tests")


def fresh_mfa(api: Api, user: User, organization: Any = None) -> dict[str, str]:
    return api.bearer(api.token(user, organization=organization, auth_methods=("pwd", "mfa")))


async def invite(api: Api, email: str, role: str = "pharmacy_staff") -> tuple[str, Any]:
    admin, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    response = await api.client.post(
        f"/api/v1/organizations/{organization.id}/invitations",
        json={"email": email, "role": role},
        headers=fresh_mfa(api, admin, organization),
    )
    assert response.status_code == 201, response.text
    return response.json()["token"], organization


def stored_user(api: Api, email: str) -> User | None:
    with api.sync_sessions() as session:
        return session.scalar(select(User).where(User.email == email))


# ------------------------------------------------------------------------- one link
async def test_a_new_person_creates_the_account_from_the_invitation(api: Api) -> None:
    email = new_email("newstaff")
    token, organization = await invite(api, email)

    preview = await api.client.get(f"/api/v1/invitations/{token}")
    created = await api.client.post(
        f"/api/v1/invitations/{token}/accept-new",
        json={"full_name": "Amina Bello", "password": NEW_PASSWORD},
    )
    used_preview = await api.client.get(f"/api/v1/invitations/{token}")
    used_again = await api.client.post(
        f"/api/v1/invitations/{token}/accept-new",
        json={"full_name": "Someone Else", "password": NEW_PASSWORD},
    )
    signed_in = await api.password_login(email, NEW_PASSWORD)

    assert preview.status_code == 200, preview.text
    assert preview.json() | {"expires_at": None} == {
        "organization_name": organization.name,
        "organization_type": "pharmacy",
        "role": "pharmacy_staff",
        "recipient": f"{email[:2]}***@{email.partition('@')[2]}",
        "expires_at": None,
        "account_exists": False,
    }
    assert created.status_code == 200, created.text
    assert created.json()["status"] == "mfa_enrolment_required"  # staff set up MFA first
    assert used_preview.status_code == 404
    assert used_again.status_code == 404
    assert signed_in.json()["status"] == "mfa_enrolment_required"
    user = stored_user(api, email)
    assert user is not None
    assert user.email_verified_at is not None  # the invitation proved the address
    assert user.full_name == "Amina Bello"
    with api.sync_sessions() as session:
        membership = session.scalar(select(Membership).where(Membership.user_id == user.id))
    assert membership is not None
    assert (membership.organization_id, membership.role) == (organization.id, "pharmacy_staff")


async def test_accept_new_refuses_existing_accounts_and_weak_passwords(api: Api) -> None:
    taken = new_email("taken")
    await api.create_user(email=taken)
    token_taken, _ = await invite(api, taken)
    fresh = new_email("weak")
    token_fresh, _ = await invite(api, fresh)

    preview = await api.client.get(f"/api/v1/invitations/{token_taken}")
    existing = await api.client.post(
        f"/api/v1/invitations/{token_taken}/accept-new",
        json={"full_name": "Impostor", "password": NEW_PASSWORD},
    )
    weak = await api.client.post(
        f"/api/v1/invitations/{token_fresh}/accept-new",
        json={"full_name": "Weak", "password": "password"},  # pragma: allowlist secret
    )
    unknown = await api.client.get("/api/v1/invitations/" + "x" * 43)

    assert preview.json()["account_exists"] is True
    assert existing.status_code == 409  # sign in and accept instead
    assert weak.status_code == 422
    assert stored_user(api, fresh) is None  # nothing was created
    assert unknown.status_code == 404


async def test_invitation_emails_carry_a_link_when_the_frontend_is_known(api: Api) -> None:
    mailer = ConsoleEmail()
    service = InvitationService(
        api.resources.session_factory,
        api.resources.redis,
        mailer,
        frontend_url="https://app.medlens.test/",
    )

    created = await service.create_platform(None, new_email("linked"), META)

    assert created.emailed is True
    ((_, subject, text),) = mailer.sent
    assert subject == "You are invited to administer MedLens"
    assert f"https://app.medlens.test/invitations/{created.token}" in text


# ------------------------------------------------------------------------- existing accounts
async def test_a_consumer_who_becomes_staff_sets_a_password_and_is_signed_out(api: Api) -> None:
    phone, email = new_phone(), new_email("shopper")
    consumer = await api.create_user(email=email, phone=phone, password=None)
    session = await api.otp_login(phone)  # a real, passwordless consumer session
    headers = api.bearer(session["access_token"])
    token, _ = await invite(api, email)

    without_password = await api.client.post(
        "/api/v1/invitations/accept", json={"token": token}, headers=headers
    )
    accepted = await api.client.post(
        "/api/v1/invitations/accept",
        json={"token": token, "password": NEW_PASSWORD},
        headers=headers,
    )
    refreshed = await api.client.post(
        "/api/v1/auth/refresh", json={"refresh_token": session["refresh_token"]}
    )
    code_sign_in = await api.client.post("/api/v1/auth/otp/request", json={"phone": phone})
    password_sign_in = await api.password_login(email, NEW_PASSWORD)

    assert without_password.status_code == 422, without_password.text
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["signed_out"] is True
    assert refreshed.status_code == 401  # the passwordless session ended
    assert code_sign_in.status_code == 202  # same answer, but staff get no code
    assert password_sign_in.json()["status"] == "mfa_enrolment_required"
    user = stored_user(api, email)
    assert user is not None
    assert user.id == consumer.id
    assert user.password_hash is not None


async def test_existing_staff_join_another_organization_without_signing_out(api: Api) -> None:
    staff, _ = await api.user_with_role(RoleName.PHARMACY_STAFF)
    assert staff.email is not None
    token, organization = await invite(api, staff.email)

    accepted = await api.client.post(
        "/api/v1/invitations/accept",
        json={"token": token},
        headers=api.bearer(api.token(staff)),
    )

    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["signed_out"] is False
    assert accepted.json()["organization_id"] == str(organization.id)


# ------------------------------------------------------------------------- platform admins
async def test_platform_admins_invite_platform_admins(api: Api) -> None:
    admin = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)
    pharmacist, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    email = new_email("ops")

    by_pharmacist = await api.client.post(
        "/api/v1/admin/invitations",
        json={"email": email},
        headers=fresh_mfa(api, pharmacist, organization),
    )
    created = await api.client.post(
        "/api/v1/admin/invitations", json={"email": email}, headers=fresh_mfa(api, admin)
    )
    token = created.json()["token"]
    listed = await api.client.get("/api/v1/admin/invitations", headers=fresh_mfa(api, admin))
    preview = await api.client.get(f"/api/v1/invitations/{token}")
    wrong_scope = await api.client.delete(
        f"/api/v1/organizations/{organization.id}/invitations/{created.json()['id']}",
        headers=fresh_mfa(api, pharmacist, organization),
    )
    joined = await api.client.post(
        f"/api/v1/invitations/{token}/accept-new",
        json={"full_name": "Ops Admin", "password": NEW_PASSWORD},
    )

    assert by_pharmacist.status_code == 403
    assert created.status_code == 201, created.text
    assert created.json()["organization_id"] is None
    assert created.json()["role"] == "platform_admin"
    assert created.json()["id"] in [item["id"] for item in listed.json()["items"]]
    assert preview.json()["organization_name"] is None
    assert wrong_scope.status_code == 404
    assert joined.json()["status"] == "mfa_enrolment_required"
    user = stored_user(api, email)
    assert user is not None
    assert user.global_role == "platform_admin"
    assert any("administer MedLens" in subject for subject, _ in api.emails_to(email))


async def test_platform_invitations_can_be_revoked(api: Api) -> None:
    admin = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)
    created = await api.client.post(
        "/api/v1/admin/invitations",
        json={"email": new_email("revoked")},
        headers=fresh_mfa(api, admin),
    )
    url = f"/api/v1/admin/invitations/{created.json()['id']}"

    revoked = await api.client.delete(url, headers=fresh_mfa(api, admin))
    again = await api.client.delete(url, headers=fresh_mfa(api, admin))
    preview = await api.client.get(f"/api/v1/invitations/{created.json()['token']}")

    assert revoked.status_code == 204
    assert again.status_code == 409
    assert preview.status_code == 404


async def test_an_existing_account_accepts_a_platform_invitation(api: Api) -> None:
    admin = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)
    staff, _ = await api.user_with_role(RoleName.REGULATOR_ADMIN)
    assert staff.email is not None
    created = await api.client.post(
        "/api/v1/admin/invitations", json={"email": staff.email}, headers=fresh_mfa(api, admin)
    )
    accepted = await api.client.post(
        "/api/v1/invitations/accept",
        json={"token": created.json()["token"]},
        headers=api.bearer(api.token(staff)),
    )

    assert accepted.status_code == 200, accepted.text
    assert accepted.json() == {
        "organization_id": None,
        "role": "platform_admin",
        "signed_out": False,
    }
    assert (await api.reload(User, staff.id)).global_role == "platform_admin"


# ------------------------------------------------------------------------- bootstrap
async def test_bootstrap_invites_the_first_admin_only(api: Api) -> None:
    """Runs in a rolled-back transaction where no active platform admin exists."""
    email = new_email("first-admin")
    mailer = ConsoleEmail()
    async with api.resources.db_engine.connect() as connection:
        outer = await connection.begin()
        try:
            sessions = async_sessionmaker(
                bind=connection,
                class_=AsyncSession,
                expire_on_commit=False,
                join_transaction_mode="create_savepoint",
            )
            async with sessions() as session:
                await session.execute(
                    update(User)
                    .where(User.global_role == RoleName.PLATFORM_ADMIN.value)
                    .values(status=UserStatus.SUSPENDED)
                )
                await session.commit()
            service = InvitationService(sessions, api.resources.redis, mailer)

            created = await service.bootstrap_platform(email, META)
            replaced = await service.bootstrap_platform(email, META)  # a rerun replaces it
            async with sessions() as session:
                rows = list(
                    await session.scalars(select(Invitation).where(Invitation.email == email))
                )
                session.add(User(email=new_email(), global_role=RoleName.PLATFORM_ADMIN.value))
                await session.commit()
            with pytest.raises(Conflict):
                await service.bootstrap_platform(email, META)
        finally:
            await outer.rollback()

    assert created.emailed
    assert replaced.emailed
    assert created.invitation.organization_id is None
    assert sorted(row.revoked_at is None for row in rows) == [False, True]
    assert f"code: {replaced.token}" in mailer.sent[-1][2]

"""Accounts (decision log, "Accounts: consumers sign themselves up, staff are invited"):
consumers sign in with a code by SMS or e-mail; staff never sign in with a code, need MFA and
join through an organization (the public registration form, verified by a platform admin, or
an invitation); password resets, adding an address, and the e-mail channel's failure modes."""

import dataclasses
from uuid import uuid4

import pytest
from sqlalchemy import select

from medlens.modules.organizations.models import Organization, OrganizationStatus, OrganizationType
from medlens.modules.rbac.models import RoleName
from medlens.modules.users.models import User
from medlens.ports.email import EmailUnavailableError
from tests.integration.api_support import PASSWORD, Api, new_phone

pytestmark = pytest.mark.integration
STRONG = "Ojuelegba-Bridge-2026"  # pragma: allowlist secret
NEWER = "Third-Mainland-Bridge-77"  # pragma: allowlist secret


def address() -> str:
    return f"person-{uuid4().hex[:10]}@example.com"


async def login(api: Api, email: str, password: str = STRONG) -> tuple[int, dict]:  # type: ignore[type-arg]
    response = await api.client.post(
        "/api/v1/auth/login", json={"email": email, "password": password}
    )
    return response.status_code, response.json()


def registration(email: str, *, number: str | None = None, **organization: object) -> dict:  # type: ignore[type-arg]
    return {
        "organization": {
            "name": f"Ikeja Pharmacy {uuid4().hex[:4]}",
            "type": "pharmacy",
            "registration_number": number or f"PCN/LA/{uuid4().hex[:8]}",
            "state": "NG-LA",
            **organization,
        },
        "admin": {"full_name": "Amina Bello", "email": email, "password": STRONG},
    }


# ------------------------------------------------------------------------- consumers
async def test_consumers_sign_in_with_an_emailed_code(api: Api) -> None:
    email = address()

    asked = await api.client.post("/api/v1/auth/email-otp/request", json={"email": email})
    signed_in = await api.client.post(
        "/api/v1/auth/email-otp/verify", json={"email": email, "code": api.last_email_code(email)}
    )
    await api.client.post("/api/v1/auth/email-otp/request", json={"email": email})
    again = await api.client.post(
        "/api/v1/auth/email-otp/verify", json={"email": email, "code": api.last_email_code(email)}
    )
    wrong = await api.client.post(
        "/api/v1/auth/email-otp/verify", json={"email": email, "code": "000000"}
    )
    me = await api.client.get(
        "/api/v1/users/me", headers=api.bearer(signed_in.json()["access_token"])
    )
    me_again = await api.client.get(
        "/api/v1/users/me", headers=api.bearer(again.json()["access_token"])
    )

    assert asked.status_code == 202
    assert signed_in.status_code == 200, signed_in.text
    assert (me.json()["global_role"], me.json()["email_verified"]) == ("consumer", True)
    assert me.json()["memberships"] == []
    assert me_again.json()["id"] == me.json()["id"]  # the same account, not a new one
    assert wrong.status_code == 401
    assert [subject for subject, _ in api.emails_to(email)] == ["Your MedLens sign-in code"] * 2


async def test_staff_never_sign_in_with_a_code(api: Api) -> None:
    """A staff account has e-mail, password and MFA: neither an SMS nor an e-mail code opens it."""
    phone = new_phone()
    staff = await api.create_user(phone=phone)
    organization = await api.create_org(OrganizationType.PHARMACY)
    await api.add_member(staff, organization, RoleName.PHARMACY_STAFF)
    assert staff.email is not None

    by_email = await api.client.post("/api/v1/auth/email-otp/request", json={"email": staff.email})
    by_sms = await api.client.post("/api/v1/auth/otp/request", json={"phone": phone})

    assert (by_email.status_code, by_sms.status_code) == (202, 202)  # the same answers
    assert api.emails_to(staff.email) == []  # but no codes
    assert all(to != phone for to, _ in api.resources.sms.sent)  # type: ignore[attr-defined]


async def test_a_consumer_who_became_staff_cannot_use_an_earlier_code(api: Api) -> None:
    phone = new_phone()
    consumer = await api.create_user(phone=phone)
    await api.client.post("/api/v1/auth/otp/request", json={"phone": phone})
    code = api.last_code(phone)
    organization = await api.create_org(OrganizationType.PHARMACY)
    await api.add_member(consumer, organization, RoleName.PHARMACY_STAFF)

    refused = await api.client.post("/api/v1/auth/otp/verify", json={"phone": phone, "code": code})

    assert refused.status_code == 403
    assert "authenticator app" in refused.json()["detail"]


async def test_every_staff_account_needs_mfa(api: Api) -> None:
    for role in (RoleName.PHARMACY_STAFF, RoleName.MANUFACTURER_STAFF, RoleName.REGULATOR_ANALYST):
        staff, _ = await api.user_with_role(role)
        assert staff.email is not None

        status, body = await login(api, staff.email, PASSWORD)

        assert (status, body["status"]) == (200, "mfa_enrolment_required"), role


# ------------------------------------------------------------------------- organizations
async def test_an_organization_registers_itself_and_waits_for_verification(api: Api) -> None:
    email = address()
    form = registration(email)

    registered = await api.client.post("/api/v1/organizations/register", json=form)
    early, problem = await login(api, email)  # the address is not verified yet
    verified = await api.client.post(
        "/api/v1/organizations/register/verify",
        json={"email": email, "code": api.last_email_code(email)},
    )
    with api.sync_sessions() as session:
        organization = session.scalars(
            select(Organization).where(
                Organization.registration_number == form["organization"]["registration_number"]
            )
        ).one()

    assert registered.status_code == 202, registered.text
    assert (early, problem["detail"]) == (
        403, "Verify your e-mail address first: enter the code we sent to it.",
    )  # fmt: skip
    assert verified.status_code == 200, verified.text
    assert verified.json()["status"] == "mfa_enrolment_required"  # MFA before any session
    assert organization.status is OrganizationStatus.PENDING_VERIFICATION  # grants nothing yet


async def test_registration_checks_its_form(api: Api) -> None:
    number = f"PCN/LA/{uuid4().hex[:8]}"
    first = await api.client.post(
        "/api/v1/organizations/register", json=registration(address(), number=number)
    )
    duplicate = await api.client.post(
        "/api/v1/organizations/register", json=registration(address(), number=number)
    )
    regulator = await api.client.post(
        "/api/v1/organizations/register",
        json=registration(address(), type="regulator", jurisdiction_states=["NG-LA"]),
    )
    no_licence = registration(address())
    del no_licence["organization"]["registration_number"]
    unlicensed = await api.client.post("/api/v1/organizations/register", json=no_licence)
    weak = registration(address())
    weak["admin"]["password"] = "password123"  # pragma: allowlist secret
    refused = await api.client.post("/api/v1/organizations/register", json=weak)

    assert first.status_code == 202
    assert duplicate.status_code == 409
    assert regulator.status_code == 422
    assert unlicensed.status_code == 422
    assert refused.status_code == 422


async def test_registration_never_reveals_or_takes_over_an_account(api: Api) -> None:
    taken = await api.create_user()
    assert taken.email is not None
    form = registration(taken.email)

    response = await api.client.post("/api/v1/organizations/register", json=form)
    with api.sync_sessions() as session:
        created = session.scalars(
            select(Organization).where(
                Organization.registration_number == form["organization"]["registration_number"]
            )
        ).first()
    status, _ = await login(api, taken.email, PASSWORD)

    assert response.status_code == 202  # the same answer as a new address
    assert created is None  # but no organization
    assert api.emails_to(taken.email) == []
    assert status == 200  # the account is untouched


async def test_only_platform_admins_create_organizations_directly(api: Api) -> None:
    consumer = await api.create_user()

    response = await api.client.post(
        "/api/v1/organizations",
        json={"name": "My Pharmacy", "type": "pharmacy"},
        headers=api.bearer(api.token(consumer)),
    )

    assert response.status_code == 403


# ------------------------------------------------------------------------- passwords
async def test_a_password_reset_sets_a_new_password_and_ends_every_session(api: Api) -> None:
    user = await api.create_user()
    assert user.email is not None
    _, session = await login(api, user.email, PASSWORD)

    asked = await api.client.post("/api/v1/auth/password/forgot", json={"email": user.email})
    unknown = await api.client.post("/api/v1/auth/password/forgot", json={"email": address()})
    code = api.last_email_code(user.email)
    reset = await api.client.post(
        "/api/v1/auth/password/reset",
        json={"email": user.email, "code": code, "new_password": NEWER},
    )
    replay = await api.client.post(
        "/api/v1/auth/password/reset",
        json={"email": user.email, "code": code, "new_password": STRONG},
    )
    old, _ = await login(api, user.email, PASSWORD)
    new, _ = await login(api, user.email, NEWER)
    refreshed = await api.client.post(
        "/api/v1/auth/refresh", json={"refresh_token": session["refresh_token"]}
    )

    assert (asked.status_code, unknown.status_code) == (202, 202)  # same for any address
    assert reset.status_code == 204, reset.text
    assert replay.status_code == 401  # a code works once
    assert (old, new) == (401, 200)
    assert refreshed.status_code == 401  # signed out everywhere


async def test_a_phone_account_adds_a_verified_address_and_sets_a_password(api: Api) -> None:
    tokens = await api.otp_login(new_phone())
    headers = api.bearer(tokens["access_token"])
    email = address()

    asked = await api.client.post("/api/v1/users/me/email", json={"email": email}, headers=headers)
    code = api.last_email_code(email)
    added = await api.client.post(
        "/api/v1/users/me/email/verify", json={"email": email, "code": code}, headers=headers
    )
    await api.client.post("/api/v1/auth/password/forgot", json={"email": email})
    await api.client.post(
        "/api/v1/auth/password/reset",
        json={"email": email, "code": api.last_email_code(email), "new_password": STRONG},
    )
    status, _ = await login(api, email)

    assert asked.status_code == 202
    assert added.status_code == 200, added.text
    assert (added.json()["email"], added.json()["email_verified"]) == (email, True)
    assert status == 200


async def test_an_address_of_another_account_cannot_be_added(api: Api) -> None:
    other = await api.create_user()
    assert other.email is not None
    tokens = await api.otp_login(new_phone())
    headers = api.bearer(tokens["access_token"])

    asked = await api.client.post(
        "/api/v1/users/me/email", json={"email": other.email}, headers=headers
    )
    guessed = await api.client.post(
        "/api/v1/users/me/email/verify",
        json={"email": other.email, "code": "123456"},
        headers=headers,
    )

    assert asked.status_code == 202  # the same answer
    assert api.emails_to(other.email) == []  # but no code
    assert guessed.status_code == 401


# ------------------------------------------------------------------------- invitations
async def test_invitations_are_emailed_and_need_a_verified_address(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    assert organization is not None
    email = address()
    invited = await api.client.post(
        f"/api/v1/organizations/{organization.id}/invitations",
        json={"email": email, "role": "pharmacy_staff"},
        headers=api.bearer(
            api.token(admin, organization=organization, auth_methods=("pwd", "mfa"))
        ),
    )
    token = invited.json()["token"]
    invitee = await api.create_user(email=email, email_verified_at=None)
    early = await api.client.post(
        "/api/v1/invitations/accept", json={"token": token}, headers=api.bearer(api.token(invitee))
    )
    with api.sync_sessions() as session, session.begin():
        stored = session.get(User, invitee.id)
        assert stored is not None
        stored.email_verified_at = stored.created_at  # proven, e.g. with an e-mailed code
    accepted = await api.client.post(
        "/api/v1/invitations/accept", json={"token": token}, headers=api.bearer(api.token(invitee))
    )

    assert invited.status_code == 201, invited.text
    invitation_mail = [text for subject, text in api.emails_to(email) if "invited" in subject]
    assert len(invitation_mail) == 1
    assert token in invitation_mail[0]
    assert organization.name in invitation_mail[0]
    assert early.status_code == 404  # an unproven address cannot accept
    assert accepted.status_code == 200, accepted.text


# ------------------------------------------------------------------------- failure modes
async def test_mail_trouble_answers_503_with_retry_after(api: Api) -> None:
    class Down:
        async def send(self, to: str, subject: str, text: str) -> None:
            raise EmailUnavailableError("smtp: down")

        async def aclose(self) -> None:
            return None

    api.app.state.resources = dataclasses.replace(api.resources, email=Down())

    response = await api.client.post("/api/v1/auth/email-otp/request", json={"email": address()})

    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"

"""Registering and verifying organizations, invitations, and branches."""

from uuid import uuid4

import pytest
from sqlalchemy import select

from medlens.db.session import UnitOfWork
from medlens.modules.organizations.models import (
    Invitation,
    Membership,
    Organization,
    OrganizationType,
)
from medlens.modules.rbac.models import RoleName
from medlens.modules.users.models import User
from tests.integration.api_support import Api, new_email, new_phone

pytestmark = pytest.mark.integration
FOUNDER_PASSWORD = "Ojuelegba-Bridge-2026"  # pragma: allowlist secret


def auth(api: Api, user: User, organization: Organization | None = None) -> dict[str, str]:
    return api.bearer(api.token(user, organization=organization))


def admin_session(api: Api, user: User, organization: Organization | None = None) -> dict[str, str]:
    """A fresh MFA session: sensitive routes reject tokens older than the user's last change."""
    return api.bearer(api.token(user, organization=organization, auth_methods=("pwd", "mfa")))


async def fresh(api: Api, user: User) -> User:
    reloaded: User = await api.reload(User, user.id)
    return reloaded


# ------------------------------------------------------------------------- registration
async def register(api: Api, **organization: object) -> tuple[User, Organization]:
    """The public registration form, then the founder verifies the address."""
    email = f"founder-{uuid4().hex[:8]}@example.com"
    number = f"PCN/{uuid4().hex[:8]}"
    form = {
        "organization": {"name": "Ikeja Pharmacy", "type": "pharmacy", "state": "Lagos",
                         "registration_number": number, **organization},
        "admin": {"full_name": "Founder", "email": email, "password": FOUNDER_PASSWORD},
    }  # fmt: skip
    registered = await api.client.post("/api/v1/organizations/register", json=form)
    assert registered.status_code == 202, registered.text
    await api.client.post(
        "/api/v1/organizations/register/verify",
        json={"email": email, "code": api.last_email_code(email)},
    )
    async with UnitOfWork(api.resources.session_factory) as uow:
        founder = await uow.session.scalar(select(User).where(User.email == email))
        organization_row = await uow.session.scalar(
            select(Organization).where(Organization.registration_number == number)
        )
    assert founder is not None
    assert organization_row is not None
    return founder, organization_row


async def test_a_registered_organization_waits_for_verification(api: Api) -> None:
    founder, organization = await register(api)
    platform, _ = await api.user_with_role(RoleName.PLATFORM_ADMIN)
    org_id = str(organization.id)
    before = await api.client.get(
        f"/api/v1/organizations/{org_id}/members",
        headers=auth(api, await fresh(api, founder), organization),
    )
    verified = await api.client.post(
        f"/api/v1/organizations/{org_id}/verify", headers=admin_session(api, platform)
    )
    after = await api.client.get(
        f"/api/v1/organizations/{org_id}/members",
        headers=auth(api, await fresh(api, founder), organization),
    )
    again = await api.client.post(
        f"/api/v1/organizations/{org_id}/verify", headers=admin_session(api, platform)
    )

    assert organization.status.value == "pending_verification"
    assert organization.state == "NG-LA"
    assert before.status_code == 403  # an unverified organization grants nothing
    assert verified.json()["status"] == "active"
    assert verified.json()["verified_at"]
    assert after.status_code == 200
    assert after.json()["items"][0]["role"] == "pharmacy_admin"
    assert again.status_code == 409


async def test_only_platform_admins_create_regulators_or_verify(api: Api) -> None:
    user = await api.create_user()
    platform, _ = await api.user_with_role(RoleName.PLATFORM_ADMIN)
    regulator = {"name": "Kano Drug Agency", "type": "regulator", "jurisdiction_states": ["NG-KN"]}

    by_user = await api.client.post(
        "/api/v1/organizations", json=regulator, headers=auth(api, user)
    )
    by_admin = await api.client.post(
        "/api/v1/organizations", json=regulator, headers=auth(api, platform)
    )
    verify_by_user = await api.client.post(
        f"/api/v1/organizations/{by_admin.json()['id']}/verify", headers=admin_session(api, user)
    )
    no_jurisdiction = await api.client.post(
        "/api/v1/organizations",
        json={"name": "X", "type": "regulator"},
        headers=auth(api, platform),
    )

    assert by_user.status_code == 403
    assert by_admin.status_code == 201
    assert by_admin.json()["jurisdiction_states"] == ["NG-KN"]
    assert verify_by_user.status_code == 403
    assert no_jurisdiction.status_code == 422


async def test_registration_numbers_are_unique_per_type(api: Api) -> None:
    platform, _ = await api.user_with_role(RoleName.PLATFORM_ADMIN)
    number = f"PCN/{new_phone()}"
    body = {"name": "A", "type": "pharmacy", "registration_number": number}

    first = await api.client.post(
        "/api/v1/organizations", json=body, headers=admin_session(api, platform)
    )
    second = await api.client.post(
        "/api/v1/organizations", json=body, headers=admin_session(api, platform)
    )

    assert first.status_code == 201
    assert second.status_code == 409


async def test_people_see_only_their_organizations(api: Api) -> None:
    member, organization = await api.user_with_role(RoleName.PHARMACY_STAFF)
    assert organization is not None
    other = await api.create_org(OrganizationType.PHARMACY)
    platform, _ = await api.user_with_role(RoleName.PLATFORM_ADMIN)

    mine = await api.client.get("/api/v1/organizations", headers=auth(api, member, organization))
    foreign = await api.client.get(
        f"/api/v1/organizations/{other.id}", headers=auth(api, member, organization)
    )
    as_admin = await api.client.get(
        f"/api/v1/organizations/{other.id}", headers=auth(api, platform)
    )

    assert [item["id"] for item in mine.json()["items"]] == [str(organization.id)]
    assert foreign.status_code == 404
    assert as_admin.status_code == 200


# ------------------------------------------------------------------------- invitations
async def test_an_invitation_is_accepted_by_the_invited_account(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    email = new_email("invitee")
    invitee = await api.create_user(email=email)
    stranger = await api.create_user()

    invited = await api.client.post(
        f"/api/v1/organizations/{organization.id}/invitations",
        json={"email": email.upper(), "role": "pharmacy_staff"},
        headers=admin_session(api, admin, organization),
    )
    token = invited.json()["token"]
    wrong_account = await api.client.post(
        "/api/v1/invitations/accept", json={"token": token}, headers=auth(api, stranger)
    )
    accepted = await api.client.post(
        "/api/v1/invitations/accept", json={"token": token}, headers=auth(api, invitee)
    )
    reused = await api.client.post(
        "/api/v1/invitations/accept", json={"token": token}, headers=auth(api, invitee)
    )
    listed = await api.client.get(
        f"/api/v1/organizations/{organization.id}/invitations",
        headers=admin_session(api, admin, organization),
    )

    assert invited.status_code == 201, invited.text
    assert invited.json()["recipient"].startswith("in***@")
    assert wrong_account.status_code == 404
    assert accepted.status_code == 200, accepted.text
    assert accepted.json() == {
        "organization_id": str(organization.id),
        "role": "pharmacy_staff",
        "signed_out": True,  # a consumer became staff: it now signs in with MFA
    }
    assert reused.status_code in {401, 404}
    assert listed.json()["items"][0]["status"] == "accepted"
    assert "token" not in listed.json()["items"][0]
    assert (await fresh(api, invitee)).perm_ver > invitee.perm_ver


async def test_invitations_are_replaced_and_revocable(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.MANUFACTURER_ADMIN)
    email = new_email("maker")
    invitee = await api.create_user(email=email)
    url = f"/api/v1/organizations/{organization.id}/invitations"
    headers = admin_session(api, admin, organization)

    first = await api.client.post(
        url, json={"email": email, "role": "manufacturer_staff"}, headers=headers
    )
    second = await api.client.post(
        url, json={"email": email, "role": "manufacturer_staff"}, headers=headers
    )
    replaced = await api.client.post(
        "/api/v1/invitations/accept",
        json={"token": first.json()["token"]},
        headers=auth(api, invitee),
    )
    revoked = await api.client.delete(f"{url}/{second.json()['id']}", headers=headers)
    after_revoke = await api.client.post(
        "/api/v1/invitations/accept",
        json={"token": second.json()["token"]},
        headers=auth(api, invitee),
    )

    assert first.json()["recipient"].startswith("ma***@")
    assert email not in first.text
    assert replaced.status_code == 404  # a new invitation replaces the open one
    assert revoked.status_code == 204
    assert after_revoke.status_code == 404


async def test_invitations_cannot_escalate_or_mismatch_the_organization(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    url = f"/api/v1/organizations/{organization.id}/invitations"
    headers = admin_session(api, admin, organization)

    regulator_role = await api.client.post(
        url, json={"email": new_email(), "role": "regulator_admin"}, headers=headers
    )
    by_phone = await api.client.post(  # invitations go to e-mail addresses only
        url, json={"phone": new_phone(), "role": "pharmacy_staff"}, headers=headers
    )
    other_org = await api.create_org(OrganizationType.PHARMACY)
    foreign = await api.client.post(
        f"/api/v1/organizations/{other_org.id}/invitations",
        json={"email": new_email(), "role": "pharmacy_staff"},
        headers=headers,
    )

    assert regulator_role.status_code in {403, 422}
    assert by_phone.status_code == 422
    assert foreign.status_code == 404
    async with UnitOfWork(api.resources.session_factory) as uow:
        count = len(
            list(
                await uow.session.scalars(
                    select(Invitation).where(Invitation.organization_id == other_org.id)
                )
            )
        )
    assert count == 0


# ------------------------------------------------------------------------- branches
async def test_branches_have_locations_and_stay_in_their_organization(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    staff = await api.create_user()
    await api.add_member(staff, organization, RoleName.PHARMACY_STAFF)
    rival, rival_org = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    url = f"/api/v1/organizations/{organization.id}/branches"
    body = {
        "name": "Ikeja",
        "address": "12 Allen Avenue",
        "state": "NG-LA",
        "latitude": 6.6018,
        "longitude": 3.3515,
    }

    created = await api.client.post(url, json=body, headers=auth(api, admin, organization))
    branch_url = f"{url}/{created.json()['id']}"
    moved = await api.client.patch(
        branch_url,
        json={"latitude": 6.45, "longitude": 3.39},
        headers=auth(api, admin, organization),
    )
    half_moved = await api.client.patch(
        branch_url, json={"latitude": 6.0}, headers=auth(api, admin, organization)
    )
    by_staff = await api.client.patch(
        branch_url, json={"name": "X"}, headers=auth(api, staff, organization)
    )
    listed = await api.client.get(url, headers=auth(api, staff, organization))
    by_rival = await api.client.get(url, headers=auth(api, rival, rival_org))
    deleted = await api.client.delete(branch_url, headers=auth(api, admin, organization))
    after = await api.client.get(url, headers=auth(api, staff, organization))

    assert created.status_code == 201, created.text
    assert created.json()["latitude"] == pytest.approx(6.6018)
    assert moved.json()["longitude"] == pytest.approx(3.39)
    assert half_moved.status_code == 422
    assert by_staff.status_code == 403
    assert [item["name"] for item in listed.json()["items"]] == ["Ikeja"]
    assert by_rival.status_code == 404
    assert deleted.status_code == 204
    assert after.json()["items"] == []


async def test_founders_become_admins_of_their_type(api: Api) -> None:
    _, organization = await register(api, name="Lagos Distributors", type="distributor")

    async with UnitOfWork(api.resources.session_factory) as uow:
        membership = await uow.session.scalar(
            select(Membership).where(Membership.organization_id == organization.id)
        )
    assert membership is not None
    assert membership.role == "pharmacy_admin"  # distributors use the pharmacy roles


async def test_invitations_match_e_mail_addresses_case_insensitively(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    local = f"Amina.{uuid4().hex[:6]}"
    invitee = await api.create_user(email=f"{local}@Example.NG")

    invited = await api.client.post(
        f"/api/v1/organizations/{organization.id}/invitations",
        json={"email": f"{local.lower()}@example.ng", "role": "pharmacy_staff"},
        headers=admin_session(api, admin, organization),
    )
    accepted = await api.client.post(
        "/api/v1/invitations/accept",
        json={"token": invited.json()["token"]},
        headers=auth(api, invitee),
    )

    assert accepted.status_code == 200, accepted.text

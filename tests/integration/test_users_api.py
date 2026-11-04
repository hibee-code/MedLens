"""The signed-in user's account: profile and NDPA erasure."""

import pytest
from sqlalchemy import select

from medlens.db.session import UnitOfWork
from medlens.modules.organizations.models import Membership, OrganizationType
from medlens.modules.rbac.models import RoleName
from medlens.modules.scans.models import Scan, Verdict
from medlens.modules.users.models import User, UserStatus
from tests.integration.api_support import Api, new_phone

pytestmark = pytest.mark.integration


async def test_read_and_update_profile(api: Api) -> None:
    user, organization = await api.user_with_role(RoleName.MANUFACTURER_STAFF)
    headers = api.bearer(api.token(user, organization=organization))

    me = (await api.client.get("/api/v1/users/me", headers=headers)).json()
    updated = await api.client.patch(
        "/api/v1/users/me", headers=headers, json={"full_name": "  Chidi <b>Okeke</b> "}
    )
    unknown_field = await api.client.patch(
        "/api/v1/users/me", headers=headers, json={"global_role": "platform_admin"}
    )

    assert me["memberships"][0]["role"] == "manufacturer_staff"
    assert me["memberships"][0]["organization_type"] == "manufacturer"
    assert updated.status_code == 200
    assert updated.json()["full_name"] == "Chidi Okeke"  # free text: HTML stripped
    assert unknown_field.status_code == 422  # no privilege escalation through PATCH


async def test_erasure_removes_personal_data_and_keeps_scans(api: Api) -> None:
    phone = new_phone()
    session = await api.otp_login(phone)
    headers = api.bearer(session["access_token"])
    user_id = (await api.client.get("/api/v1/users/me", headers=headers)).json()["id"]
    pharmacy = await api.create_org(OrganizationType.PHARMACY)
    user: User = await api.reload(User, user_id)
    await api.add_member(user, pharmacy, RoleName.PHARMACY_STAFF)
    scan = await api._save(Scan(user_id=user.id, verdict=Verdict.SUSPICIOUS))

    response = await api.client.delete("/api/v1/users/me", headers=headers)

    assert response.status_code == 204
    erased: User = await api.reload(User, user.id)
    assert erased.status is UserStatus.DELETED
    assert (erased.phone_e164, erased.phone_hash, erased.email, erased.password_hash) == (
        None,
        None,
        None,
        None,
    )
    async with UnitOfWork(api.resources.session_factory) as uow:
        kept = (await uow.session.scalars(select(Scan).where(Scan.id == scan.id))).one()
        memberships = list(
            await uow.session.scalars(select(Membership).where(Membership.user_id == user.id))
        )
    assert kept.user_id is None
    assert memberships == []
    assert (await api.client.get("/api/v1/users/me", headers=headers)).status_code == 401
    refresh = await api.client.post(
        "/api/v1/auth/refresh", json={"refresh_token": session["refresh_token"]}
    )
    assert refresh.status_code == 401
    # The phone number is free again: signing in creates a brand-new account.
    fresh = await api.otp_login(phone)
    fresh_me = (
        await api.client.get("/api/v1/users/me", headers=api.bearer(fresh["access_token"]))
    ).json()
    assert fresh_me["id"] != user_id


async def test_last_admin_cannot_erase_their_account(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)

    response = await api.client.delete(
        "/api/v1/users/me", headers=api.bearer(api.token(admin, organization=organization))
    )

    assert response.status_code == 409
    assert (await api.reload(User, admin.id)).status is UserStatus.ACTIVE

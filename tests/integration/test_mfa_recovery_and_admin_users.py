"""MFA recovery codes and platform admins managing accounts (decision log, "Accounts: recovery
codes and account administration")."""

import re
import time
from typing import Any
from uuid import uuid4

import pyotp
import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.errors import Conflict
from medlens.core.request_meta import RequestMeta
from medlens.modules.audit.models import AuditLog
from medlens.modules.auth.models import MfaRecoveryCode
from medlens.modules.auth.service import AuthService
from medlens.modules.rbac.models import RoleName
from medlens.modules.users.admin import AdminAction, UserAdminService
from medlens.modules.users.models import User, UserStatus
from tests.integration.api_support import Api, new_email, new_phone

pytestmark = pytest.mark.integration
CODE_FORMAT = re.compile(r"^[a-z2-9]{5}-[a-z2-9]{5}$")


async def enrol(api: Api, user: User) -> tuple[str, dict[str, Any]]:
    """Signs in with the password and turns MFA on; returns the secret and the confirm answer."""
    challenge = (await api.password_login(user.email or "")).json()
    assert challenge["status"] == "mfa_enrolment_required", challenge
    bearer = api.bearer(challenge["mfa_token"])
    secret = (await api.client.post("/api/v1/auth/mfa/enrol", headers=bearer)).json()["secret"]
    confirmed = await api.client.post(
        "/api/v1/auth/mfa/confirm", json={"code": pyotp.TOTP(secret).now()}, headers=bearer
    )
    assert confirmed.status_code == 200, confirmed.text
    body: dict[str, Any] = confirmed.json()
    return secret, body


async def mfa_challenge(api: Api, user: User) -> str:
    body = (await api.password_login(user.email or "")).json()
    assert body["status"] == "mfa_required", body
    token: str = body["mfa_token"]
    return token


def next_code(secret: str) -> str:
    """The next 30-second step's code: the current one was used (codes work once)."""
    return pyotp.TOTP(secret).at(time.time() + 30)


# ------------------------------------------------------------------------- recovery codes
async def test_recovery_codes_sign_in_once_each(api: Api) -> None:
    staff, _ = await api.user_with_role(RoleName.PHARMACY_STAFF)
    _, enabled = await enrol(api, staff)
    codes: list[str] = enabled["recovery_codes"]
    session = api.bearer(enabled["access_token"])

    left_before = await api.client.get("/api/v1/auth/mfa/recovery-codes", headers=session)
    typed = codes[0].replace("-", "").upper()  # typed back loosely
    first = await api.client.post(
        "/api/v1/auth/mfa/verify",
        json={"mfa_token": await mfa_challenge(api, staff), "recovery_code": typed},
    )
    reused = await api.client.post(
        "/api/v1/auth/mfa/verify",
        json={"mfa_token": await mfa_challenge(api, staff), "recovery_code": codes[0]},
    )
    both = await api.client.post(
        "/api/v1/auth/mfa/verify",
        json={
            "mfa_token": await mfa_challenge(api, staff),
            "code": "123456",
            "recovery_code": codes[1],
        },
    )
    left_after = await api.client.get("/api/v1/auth/mfa/recovery-codes", headers=session)

    assert len(codes) == len(set(codes)) == 10
    assert all(CODE_FORMAT.fullmatch(code) for code in codes)
    assert left_before.json() == {"remaining": 10}
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "authenticated"
    assert reused.status_code == 401
    assert both.status_code == 422
    assert left_after.json() == {"remaining": 9}
    with api.sync_sessions() as db:
        stored = list(db.scalars(select(MfaRecoveryCode.code_hash)))
        audited = db.scalar(
            select(AuditLog.metadata_)
            .where(AuditLog.actor_user_id == staff.id, AuditLog.action == "auth.login_succeeded")
            .order_by(AuditLog.created_at.desc())
            .limit(1)
        )
    assert not set(codes) & set(stored)  # only hashes are stored
    assert audited == {"method": "password+recovery_code", "recovery_codes_left": 9}


async def test_recovery_codes_are_regenerated_with_the_authenticator(api: Api) -> None:
    staff, _ = await api.user_with_role(RoleName.MANUFACTURER_STAFF)
    secret, enabled = await enrol(api, staff)
    session = api.bearer(enabled["access_token"])
    without_mfa = api.bearer(api.token(staff, auth_methods=("pwd",)))

    no_mfa = await api.client.post(
        "/api/v1/auth/mfa/recovery-codes", json={"code": next_code(secret)}, headers=without_mfa
    )
    wrong = await api.client.post(
        "/api/v1/auth/mfa/recovery-codes", json={"code": "000000"}, headers=session
    )
    renewed = await api.client.post(
        "/api/v1/auth/mfa/recovery-codes", json={"code": next_code(secret)}, headers=session
    )
    old = await api.client.post(
        "/api/v1/auth/mfa/verify",
        json={
            "mfa_token": await mfa_challenge(api, staff),
            "recovery_code": enabled["recovery_codes"][0],
        },
    )
    new = await api.client.post(
        "/api/v1/auth/mfa/verify",
        json={
            "mfa_token": await mfa_challenge(api, staff),
            "recovery_code": renewed.json()["recovery_codes"][0],
        },
    )

    assert no_mfa.status_code == 403
    assert wrong.status_code == 422
    assert renewed.status_code == 200, renewed.text
    assert set(renewed.json()["recovery_codes"]).isdisjoint(enabled["recovery_codes"])
    assert old.status_code == 401
    assert new.status_code == 200, new.text


# ------------------------------------------------------------------------- account administration
def admin_headers(api: Api, admin: User) -> dict[str, str]:
    return api.bearer(api.token(admin, auth_methods=("pwd", "mfa")))


async def test_platform_admins_find_accounts(api: Api) -> None:
    admin = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)
    marker = uuid4().hex[:8]
    phone = new_phone()
    target = await api.create_user(
        email=f"findme-{marker}@example.ng", phone=phone, full_name=f"Kemi {marker}"
    )
    member, organization = await api.user_with_role(RoleName.PHARMACY_STAFF)
    assert organization is not None
    headers = admin_headers(api, admin)

    by_email = await api.client.get(f"/api/v1/admin/users?q=findme-{marker}", headers=headers)
    by_name = await api.client.get(f"/api/v1/admin/users?q=kemi {marker}", headers=headers)
    by_phone = await api.client.get(
        "/api/v1/admin/users", params={"q": "0" + phone[4:]}, headers=headers
    )
    by_org = await api.client.get(
        f"/api/v1/admin/users?organization_id={organization.id}", headers=headers
    )
    one = await api.client.get(f"/api/v1/admin/users/{target.id}", headers=headers)
    paged = await api.client.get("/api/v1/admin/users?limit=1", headers=headers)
    by_pharmacist = await api.client.get(
        "/api/v1/admin/users", headers=api.bearer(api.token(member, organization=organization))
    )
    without_mfa = await api.client.get(
        "/api/v1/admin/users", headers=api.bearer(api.token(admin, auth_methods=("pwd",)))
    )
    missing = await api.client.get(f"/api/v1/admin/users/{uuid4()}", headers=headers)

    assert [item["id"] for item in by_email.json()["items"]] == [str(target.id)]
    assert [item["id"] for item in by_name.json()["items"]] == [str(target.id)]
    assert [item["id"] for item in by_phone.json()["items"]] == [str(target.id)]
    assert [item["id"] for item in by_org.json()["items"]] == [str(member.id)]
    assert one.json()["phone"] == f"{phone[:7]}*****{phone[-2:]}"  # masked
    assert phone not in one.text
    assert paged.json()["has_more"] is True
    assert paged.json()["next_cursor"]
    assert by_pharmacist.status_code == 403
    assert without_mfa.status_code == 403
    assert missing.status_code == 404


async def test_suspend_and_reactivate(api: Api) -> None:
    admin = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)
    staff, _ = await api.user_with_role(RoleName.PHARMACY_STAFF)
    session = await api.staff_login(staff)
    headers = admin_headers(api, admin)
    url = f"/api/v1/admin/users/{staff.id}"
    reason = {"reason": "Left the pharmacy; access removed on the owner's request."}

    no_reason = await api.client.post(f"{url}/suspend", json={}, headers=headers)
    suspended = await api.client.post(f"{url}/suspend", json=reason, headers=headers)
    again = await api.client.post(f"{url}/suspend", json=reason, headers=headers)
    refreshed = await api.client.post(
        "/api/v1/auth/refresh", json={"refresh_token": session["refresh_token"]}
    )
    sign_in = await api.password_login(staff.email or "")
    own = await api.client.post(
        f"/api/v1/admin/users/{admin.id}/suspend", json=reason, headers=headers
    )
    reactivated = await api.client.post(f"{url}/reactivate", json=reason, headers=headers)
    sign_in_again = await api.password_login(staff.email or "")

    assert no_reason.status_code == 422
    assert suspended.status_code == 200, suspended.text
    assert suspended.json()["status"] == "suspended"
    assert again.status_code == 409
    assert refreshed.status_code == 401  # every session ended
    assert sign_in.status_code == 403
    assert own.status_code == 409
    assert reactivated.json()["status"] == "active"
    assert sign_in_again.json()["status"] == "mfa_required"
    with api.sync_sessions() as db:
        actions = set(
            db.scalars(
                select(AuditLog.action).where(
                    AuditLog.resource_id == str(staff.id), AuditLog.actor_user_id == admin.id
                )
            )
        )
    assert {"user.admin_suspend", "user.admin_reactivate"} <= actions


async def test_sign_out_everywhere_and_reset_mfa(api: Api) -> None:
    admin = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)
    staff, _ = await api.user_with_role(RoleName.REGULATOR_ANALYST)
    _, enabled = await enrol(api, staff)
    headers = admin_headers(api, admin)
    url = f"/api/v1/admin/users/{staff.id}"
    reason = {"reason": "Phone stolen; identity checked by video call."}

    signed_out = await api.client.post(f"{url}/sign-out", json=reason, headers=headers)
    refreshed = await api.client.post(
        "/api/v1/auth/refresh", json={"refresh_token": enabled["refresh_token"]}
    )
    reset = await api.client.post(f"{url}/mfa/reset", json=reason, headers=headers)
    reset_again = await api.client.post(f"{url}/mfa/reset", json=reason, headers=headers)
    sign_in = await api.password_login(staff.email or "")

    assert signed_out.status_code == 200, signed_out.text
    assert refreshed.status_code == 401
    assert reset.json()["mfa_enabled"] is False
    assert reset_again.status_code == 409
    assert sign_in.json()["status"] == "mfa_enrolment_required"  # set it up again
    with api.sync_sessions() as db:
        left = db.scalar(
            select(func.count())
            .select_from(MfaRecoveryCode)
            .where(MfaRecoveryCode.user_id == staff.id)
        )
    assert left == 0


async def test_the_last_platform_admin_cannot_be_suspended(api: Api) -> None:
    """In a rolled-back transaction where only one active platform admin remains."""
    last = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)
    other = await api.create_user(global_role=RoleName.PLATFORM_ADMIN, email=new_email())
    meta = RequestMeta(ip=None, user_agent="tests")
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
                    .where(User.global_role == RoleName.PLATFORM_ADMIN.value, User.id != last.id)
                    .values(status=UserStatus.SUSPENDED)
                )
                await session.commit()
            resources = api.resources
            auth = AuthService(
                sessions=sessions,
                keyring=resources.keyring,
                sms=resources.sms,
                redis=resources.redis,
                settings=api.app.state.settings.auth,
            )
            service = UserAdminService(sessions, resources.redis, auth)
            with pytest.raises(Conflict, match="last active platform admin"):
                await service.act(other.id, last.id, AdminAction.SUSPEND, "test", meta)
        finally:
            await outer.rollback()

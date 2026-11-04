"""Staff sign-in: e-mail + password, lockouts, rehashing and MFA."""

import statistics
import time

import pyotp
import pytest
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from medlens.core.config import Settings
from medlens.modules.organizations.models import OrganizationType
from medlens.modules.rbac.models import RoleName
from medlens.modules.users.models import User, UserStatus
from tests.integration.api_support import PASSWORD, Api, new_email, open_api
from tests.integration.support import make_settings

pytestmark = pytest.mark.integration


def claims(api: Api, token: str) -> tuple[str, ...]:
    return api.resources.keyring.verify(token, "access").auth_methods


async def test_staff_login_returns_a_session_with_the_single_organization(api: Api) -> None:
    user, organization = await api.user_with_role(RoleName.PHARMACY_STAFF)

    response = await api.password_login(user.email or "")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "authenticated"
    assert body["organization_id"] == str(organization.id if organization else None)
    assert claims(api, body["access_token"]) == ("pwd",)
    assert "auth.login_succeeded" in await api.audit_actions(user.id)


async def test_unknown_email_and_wrong_password_look_identical(
    app_settings: Settings, database_groups: dict, redis_groups: dict
) -> None:
    settings = make_settings(
        database=database_groups, redis=redis_groups, auth={"login_max_failures": 100}
    )
    async with open_api(settings) as api:
        user = await api.create_user()

        async def attempt(email: str) -> tuple[float, int, str]:
            started = time.perf_counter()
            response = await api.password_login(email, "wrong-password-123456")
            return time.perf_counter() - started, response.status_code, response.json()["detail"]

        wrong_password = [await attempt(user.email or "") for _ in range(4)]
        unknown_email = [await attempt(new_email()) for _ in range(4)]

    assert {(status, detail) for _, status, detail in wrong_password} == {
        (401, "Invalid e-mail address or password.")
    }
    assert {(s, d) for _, s, d in unknown_email} == {(s, d) for _, s, d in wrong_password}
    ratio = statistics.median(t for t, _, _ in wrong_password) / statistics.median(
        t for t, _, _ in unknown_email
    )
    assert 0.5 < ratio < 2.0  # both verify an argon2 hash


async def test_repeated_failures_lock_the_account(api: Api) -> None:
    user = await api.create_user()

    for _ in range(5):
        assert (
            await api.password_login(user.email or "", "wrong-password-123456")
        ).status_code == 401
    locked = await api.password_login(user.email or "")  # even the right password

    assert locked.status_code == 429
    assert 0 < int(locked.headers["retry-after"]) <= 60
    assert locked.headers["content-type"] == "application/problem+json"


async def test_failures_from_one_ip_lock_other_accounts_too(api: Api) -> None:
    for _ in range(5):
        await api.password_login(new_email(), "wrong-password-123456")  # unknown accounts count

    other = await api.create_user()
    response = await api.password_login(other.email or "")

    assert response.status_code == 429


async def test_outdated_hash_parameters_are_upgraded_on_login(api: Api) -> None:
    weak = PasswordHash((Argon2Hasher(time_cost=1, memory_cost=8192, parallelism=1),))
    user = await api.create_user(password=None, password_hash=weak.hash(PASSWORD))
    old_hash = user.password_hash

    assert (await api.password_login(user.email or "")).status_code == 200

    stored: User = await api.reload(User, user.id)
    assert stored.password_hash != old_hash
    assert "m=8192" not in (stored.password_hash or "")


async def test_suspended_accounts_cannot_sign_in(api: Api) -> None:
    user = await api.create_user(status=UserStatus.SUSPENDED)

    response = await api.password_login(user.email or "")

    assert response.status_code == 403
    assert "auth.login_blocked" in await api.audit_actions(user.id)


# ------------------------------------------------------------------ MFA
@pytest.mark.parametrize("role", [RoleName.REGULATOR_ANALYST, RoleName.PLATFORM_ADMIN])
async def test_regulators_and_platform_admins_must_enrol_in_mfa(api: Api, role: RoleName) -> None:
    user, _ = await api.user_with_role(role)

    first = (await api.password_login(user.email or "")).json()
    assert first["status"] == "mfa_enrolment_required"
    enrol_token = api.bearer(first["mfa_token"])

    # The enrolment token works for MFA set-up only.
    assert (await api.client.get("/api/v1/users/me", headers=enrol_token)).status_code == 401
    setup = (await api.client.post("/api/v1/auth/mfa/enrol", headers=enrol_token)).json()
    totp = pyotp.TOTP(setup["secret"])
    assert setup["otpauth_uri"].startswith("otpauth://totp/MedLens:")
    confirmed = await api.client.post(
        "/api/v1/auth/mfa/confirm", headers=enrol_token, json={"code": totp.now()}
    )
    assert confirmed.status_code == 200, confirmed.text
    assert claims(api, confirmed.json()["access_token"]) == ("pwd", "mfa")

    # From now on, every login needs the second factor.
    second = (await api.password_login(user.email or "")).json()
    assert second["status"] == "mfa_required"
    code = totp.at(time.time() + 30)  # the current step was used to confirm
    verified = await api.client.post(
        "/api/v1/auth/mfa/verify", json={"mfa_token": second["mfa_token"], "code": code}
    )
    assert verified.status_code == 200, verified.text
    assert claims(api, verified.json()["access_token"]) == ("pwd", "mfa")

    replay = await api.client.post(
        "/api/v1/auth/mfa/verify", json={"mfa_token": second["mfa_token"], "code": code}
    )
    assert replay.status_code == 401  # a code works once
    actions = await api.audit_actions(user.id)
    assert {"auth.mfa_enrolment_started", "auth.mfa_enabled", "auth.mfa_challenged"} <= set(actions)


async def test_other_staff_may_opt_in_to_mfa(api: Api) -> None:
    user, _ = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    session = (await api.password_login(user.email or "")).json()
    headers = api.bearer(session["access_token"])

    setup = (await api.client.post("/api/v1/auth/mfa/enrol", headers=headers)).json()
    wrong = await api.client.post(
        "/api/v1/auth/mfa/confirm", headers=headers, json={"code": "000000"}
    )
    confirmed = await api.client.post(
        "/api/v1/auth/mfa/confirm",
        headers=headers,
        json={"code": pyotp.TOTP(setup["secret"]).now()},
    )
    again = await api.client.post("/api/v1/auth/mfa/enrol", headers=headers)

    assert wrong.status_code == 422
    assert confirmed.status_code == 200
    assert again.status_code == 409
    assert (await api.password_login(user.email or "")).json()["status"] == "mfa_required"


async def test_mfa_challenge_allows_five_attempts(api: Api) -> None:
    user, _ = await api.user_with_role(RoleName.REGULATOR_ADMIN)
    secret = pyotp.random_base32()
    async with api.resources.session_factory() as session:
        stored = await session.get(User, user.id)
        assert stored is not None
        stored.mfa_secret, stored.mfa_enabled = secret, True
        await session.commit()
    challenge = (await api.password_login(user.email or "")).json()

    for _ in range(5):
        response = await api.client.post(
            "/api/v1/auth/mfa/verify", json={"mfa_token": challenge["mfa_token"], "code": "000000"}
        )
        assert response.status_code == 401
    exhausted = await api.client.post(
        "/api/v1/auth/mfa/verify",
        json={"mfa_token": challenge["mfa_token"], "code": pyotp.TOTP(secret).now()},
    )

    assert exhausted.status_code == 401
    assert exhausted.json()["detail"] == "Too many wrong codes. Sign in again."


async def test_mfa_token_is_not_an_access_token(api: Api) -> None:
    user, _ = await api.user_with_role(RoleName.REGULATOR_ANALYST)
    challenge = (await api.password_login(user.email or "")).json()

    response = await api.client.get("/api/v1/users/me", headers=api.bearer(challenge["mfa_token"]))

    assert response.status_code == 401


async def test_regulator_membership_in_any_role_requires_mfa(api: Api) -> None:
    regulator = await api.create_org(OrganizationType.REGULATOR)
    user = await api.create_user()
    await api.add_member(user, regulator, RoleName.REGULATOR_ANALYST)

    assert (await api.password_login(user.email or "")).json()["status"] == "mfa_enrolment_required"

"""Sessions: refresh rotation and theft detection, logout, organization switching, JWT checks."""

import base64
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from medlens.core.config import AuthSettings
from medlens.core.security import JwtKeyRing
from medlens.modules.organizations.models import OrganizationType
from medlens.modules.rbac.models import RoleName
from tests.integration.api_support import PASSWORD, Api, new_phone

pytestmark = pytest.mark.integration

NEW_PASSWORD = "Kola-nut tea at 7 o'clock"  # pragma: allowlist secret
COMMON_PASSWORD = "passwordpassword"  # pragma: allowlist secret


def password_change(current: str, new: str) -> dict[str, str]:
    return {"current_password": current, "new_password": new}


async def me_status(api: Api, access_token: str) -> int:
    return (await api.client.get("/api/v1/users/me", headers=api.bearer(access_token))).status_code


async def test_refresh_rotates_and_reuse_ends_the_session(api: Api) -> None:
    first = await api.otp_login(new_phone())

    rotated = await api.client.post(
        "/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
    )
    assert rotated.status_code == 200
    second = rotated.json()
    assert second["refresh_token"] != first["refresh_token"]
    assert await me_status(api, second["access_token"]) == 200

    # The first refresh token is replayed (as by a thief who copied it).
    replay = await api.client.post(
        "/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
    )
    assert replay.status_code == 401

    # The whole session is over: the legitimate client's tokens stop working too.
    after = await api.client.post(
        "/api/v1/auth/refresh", json={"refresh_token": second["refresh_token"]}
    )
    assert after.status_code == 401
    assert await me_status(api, second["access_token"]) == 401
    assert "auth.refresh_token_reuse" in await api.audit_actions()


async def test_unknown_refresh_token_is_rejected(api: Api) -> None:
    response = await api.client.post("/api/v1/auth/refresh", json={"refresh_token": "x" * 43})

    assert response.status_code == 401


async def test_logout_ends_this_session_only(api: Api) -> None:
    phone = new_phone()
    laptop, phone_app = await api.otp_login(phone), await api.otp_login(phone)

    response = await api.client.post(
        "/api/v1/auth/logout", headers=api.bearer(laptop["access_token"])
    )

    assert response.status_code == 204
    assert await me_status(api, laptop["access_token"]) == 401
    assert (
        await api.client.post(
            "/api/v1/auth/refresh", json={"refresh_token": laptop["refresh_token"]}
        )
    ).status_code == 401
    assert await me_status(api, phone_app["access_token"]) == 200


async def test_logout_all_ends_every_session(api: Api) -> None:
    phone = new_phone()
    sessions = [await api.otp_login(phone) for _ in range(3)]

    response = await api.client.post(
        "/api/v1/auth/logout-all", headers=api.bearer(sessions[0]["access_token"])
    )

    assert response.status_code == 204
    for session in sessions:
        assert await me_status(api, session["access_token"]) == 401


async def test_switch_organization_and_keep_it_across_refresh(api: Api) -> None:
    pharmacy = await api.create_org(OrganizationType.PHARMACY)
    clinic = await api.create_org(OrganizationType.CLINIC)
    stranger = await api.create_org(OrganizationType.PHARMACY)
    user = await api.create_user()
    await api.add_member(user, pharmacy, RoleName.PHARMACY_STAFF)
    await api.add_member(user, clinic, RoleName.PHARMACY_ADMIN)
    session = (await api.password_login(user.email or "")).json()
    assert session["organization_id"] is None  # two memberships: no default

    switched = await api.client.post(
        "/api/v1/auth/switch-organization",
        headers=api.bearer(session["access_token"]),
        json={"organization_id": str(clinic.id)},
    )
    refreshed = await api.client.post(
        "/api/v1/auth/refresh", json={"refresh_token": session["refresh_token"]}
    )
    not_member = await api.client.post(
        "/api/v1/auth/switch-organization",
        headers=api.bearer(session["access_token"]),
        json={"organization_id": str(stranger.id)},
    )

    assert switched.status_code == 200
    assert switched.json()["organization_id"] == str(clinic.id)
    token = switched.json()["access_token"]
    assert api.resources.keyring.verify(token, "access").organization_id == clinic.id
    assert refreshed.json()["organization_id"] == str(clinic.id)
    assert not_member.status_code == 404


async def test_unknown_api_key_is_rejected(api: Api) -> None:
    response = await api.client.post(
        "/api/v1/auth/logout", headers={"X-API-Key": "ml_live_000000000000_" + "x" * 43}
    )

    assert response.status_code == 401  # unknown key


# ------------------------------------------------------------------ JWT verification
def _ring(seed: bytes, *, audience: str = "medlens-api", kid: str = "local-dev") -> JwtKeyRing:
    key = base64.b64encode(seed).decode()
    return JwtKeyRing(
        AuthSettings(jwt_keys={kid: key}, jwt_active_key_id=kid, jwt_audience=audience)
    )


async def test_expired_forged_and_misdirected_tokens_are_rejected(api: Api) -> None:
    user = await api.create_user()
    keyring = api.resources.keyring
    expired = keyring.issue("access", subject=user.id, now=int(time.time()) - 3600)
    active_kid = jwt.get_unverified_header(api.token(user))["kid"]
    forged = _ring(b"f" * 32, kid=active_kid).issue("access", subject=user.id)
    wrong_audience = jwt.encode(
        {
            "sub": str(user.id),
            "aud": "another-api",
            "iss": "medlens",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
            "jti": "x",
        },
        Ed25519PrivateKey.from_private_bytes(b"f" * 32),
        algorithm="EdDSA",
        headers={"kid": active_kid, "typ": "at+jwt"},
    )
    unsigned = jwt.encode({"sub": str(user.id)}, key=None, algorithm="none")  # type: ignore[arg-type]
    mfa_token = keyring.issue("mfa", subject=user.id, purpose="verify")

    for token in (expired, forged, wrong_audience, unsigned, mfa_token, "not-a-jwt"):
        assert await me_status(api, token) == 401, token[:30]
    assert await me_status(api, api.token(user)) == 200


async def test_jwks_publishes_the_verification_keys(api: Api) -> None:
    response = await api.client.get("/.well-known/jwks.json")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=300"
    [key] = response.json()["keys"]
    assert key["kty"] == "OKP"
    assert key["crv"] == "Ed25519"
    assert key["alg"] == "EdDSA"
    assert len(base64.urlsafe_b64decode(key["x"] + "=")) == 32  # raw Ed25519 public key
    token = api.token(await api.create_user())
    assert jwt.get_unverified_header(token)["kid"] == key["kid"]


# ------------------------------------------------------------------ password change
async def test_password_change_rules_and_session_handling(api: Api) -> None:
    user = await api.create_user(email=f"ada.lovelace.{new_phone()[-6:]}@medlens.test")
    current = (await api.password_login(user.email or "")).json()
    other = (await api.password_login(user.email or "")).json()
    headers = api.bearer(current["access_token"])
    url = "/api/v1/users/me/password"

    wrong_current = await api.client.post(
        url,
        headers=headers,
        json=password_change("nope", "x" * 20),
    )
    weak = await api.client.post(
        url,
        headers=headers,
        json=password_change(PASSWORD, COMMON_PASSWORD),
    )
    changed = await api.client.post(
        url,
        headers=headers,
        json=password_change(PASSWORD, NEW_PASSWORD),
    )

    assert wrong_current.status_code == 422
    assert wrong_current.json()["errors"][0]["loc"] == ["body", "current_password"]
    assert weak.status_code == 422
    assert "too common" in weak.json()["errors"][0]["msg"]
    assert changed.status_code == 204
    assert await me_status(api, current["access_token"]) == 200  # this session continues
    assert await me_status(api, other["access_token"]) == 401  # others are signed out
    assert (await api.password_login(user.email or "", NEW_PASSWORD)).status_code == 200
    assert "auth.password_changed" in await api.audit_actions(user.id)


async def test_phone_accounts_have_no_password_to_change(api: Api) -> None:
    session = await api.otp_login(new_phone())

    response = await api.client.post(
        "/api/v1/users/me/password",
        headers=api.bearer(session["access_token"]),
        json=password_change("anything", NEW_PASSWORD),
    )

    assert response.status_code == 409

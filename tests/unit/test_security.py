"""JWT key ring, password hashing, API keys and TOTP."""

import base64
import time
from uuid import uuid4

import pyotp
import pytest

from medlens.core.config import AuthSettings
from medlens.core.security import (
    InvalidTokenError,
    JwtKeyRing,
    hash_password,
    new_api_key,
    parse_api_key_prefix,
    verify_password,
)
from medlens.modules.auth.totp import matching_timestep

OLD = base64.b64encode(b"o" * 32).decode()
NEW = base64.b64encode(b"n" * 32).decode()


def ring(keys: dict[str, str], active: str, **settings: object) -> JwtKeyRing:
    return JwtKeyRing(AuthSettings(jwt_keys=keys, jwt_active_key_id=active, **settings))  # type: ignore[arg-type]


def test_round_trip_carries_session_claims() -> None:
    keyring = ring({"k1": OLD}, "k1")
    user, session, organization = uuid4(), uuid4(), uuid4()

    token = keyring.issue(
        "access",
        subject=user,
        session_id=session,
        organization_id=organization,
        perm_ver=7,
        auth_methods=("pwd", "mfa"),
    )
    claims = keyring.verify(token, "access")

    assert (claims.subject, claims.session_id, claims.organization_id) == (
        user,
        session,
        organization,
    )
    assert claims.perm_ver == 7
    assert claims.auth_methods == ("pwd", "mfa")
    assert claims.expires_at - claims.issued_at == 900


def test_rotation_old_tokens_verify_until_the_key_is_removed() -> None:
    before = ring({"2026-10": OLD}, "2026-10")
    token = before.issue("access", subject=uuid4())

    during = ring({"2026-10": OLD, "2026-11": NEW}, "2026-11")
    after = ring({"2026-11": NEW}, "2026-11")

    assert during.verify(token, "access").subject  # old kid still verifies
    assert during.verify(during.issue("access", subject=uuid4()), "access")
    with pytest.raises(InvalidTokenError):
        after.verify(token, "access")
    assert [key["kid"] for key in during.jwks()["keys"]] == ["2026-10", "2026-11"]


@pytest.mark.parametrize(
    "make",
    [
        lambda k: k.issue("access", subject=uuid4(), now=int(time.time()) - 3600),  # expired
        lambda k: ring({"k1": NEW}, "k1").issue("access", subject=uuid4()),  # forged, same kid
        lambda k: ring({"k1": OLD}, "k1", jwt_audience="other").issue("access", subject=uuid4()),
        lambda k: ring({"k1": OLD}, "k1", jwt_issuer="other").issue("access", subject=uuid4()),
        lambda k: k.issue("mfa", subject=uuid4()),  # wrong token type
        lambda k: "garbage",
    ],
    ids=["expired", "forged", "audience", "issuer", "type", "garbage"],
)
def test_bad_tokens_are_rejected(make) -> None:  # type: ignore[no-untyped-def]
    keyring = ring({"k1": OLD}, "k1")

    with pytest.raises(InvalidTokenError):
        keyring.verify(make(keyring), "access")


def test_mfa_tokens_carry_their_purpose_and_short_lifetime() -> None:
    keyring = ring({"k1": OLD}, "k1")

    claims = keyring.verify(keyring.issue("mfa", subject=uuid4(), purpose="enrol"), "mfa")

    assert claims.purpose == "enrol"
    assert claims.session_id is None
    assert claims.expires_at - claims.issued_at == 300


def test_password_hashing() -> None:
    stored = hash_password("correct horse battery staple")

    assert stored.startswith("$argon2id$")
    assert verify_password("correct horse battery staple", stored) == (True, None)
    assert verify_password("wrong", stored)[0] is False
    assert verify_password("anything", None) == (False, None)  # still does the work


def test_api_key_format() -> None:
    created = new_api_key()

    assert created.key.startswith(f"ml_live_{created.prefix}_")
    assert len(created.prefix) == 12
    assert parse_api_key_prefix(created.key) == created.prefix
    assert created.digest not in created.key


@pytest.mark.parametrize(
    "key",
    ["", "ml_live_", "ml_test_abcdef123456_" + "x" * 40, "ml_live_ABCDEF123456_" + "x" * 40,
     "ml_live_abcdef123456-" + "x" * 40, "ml_live_abcdef123456_short"],
)  # fmt: skip
def test_malformed_api_keys(key: str) -> None:
    assert parse_api_key_prefix(key) is None


def test_totp_accepts_one_step_of_drift_and_reports_it() -> None:
    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    now = time.time()

    current = matching_timestep(secret, totp.at(now), now=now)
    previous = matching_timestep(secret, totp.at(now - 30), now=now)

    assert current is not None
    assert previous == current - 1
    assert matching_timestep(secret, totp.at(now - 90), now=now) is None
    assert (
        matching_timestep(secret, "000000" if totp.at(now) != "000000" else "111111", now=now)
        is None
    )

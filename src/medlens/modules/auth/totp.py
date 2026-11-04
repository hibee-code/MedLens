"""TOTP (RFC 6238) for MFA, via pyotp."""

import time
from datetime import UTC, datetime

import pyotp

from medlens.core.security import constant_time_equal


def new_secret() -> str:
    return pyotp.random_base32()


def provisioning_uri(secret: str, *, account: str, issuer: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=account, issuer_name=issuer)


def matching_timestep(secret: str, code: str, *, now: float | None = None) -> int | None:
    """The 30-second step whose code matches (current, previous or next step for clock drift),
    or None. Returning the step lets callers refuse a code that was already used."""
    totp = pyotp.TOTP(secret)
    instant = time.time() if now is None else now
    for offset in (0, -1, 1):
        moment = datetime.fromtimestamp(instant + offset * totp.interval, UTC)
        if constant_time_equal(totp.at(moment), code):
            return totp.timecode(moment)
    return None

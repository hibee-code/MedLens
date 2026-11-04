"""Credentials: JWT signing and verification, password hashing, opaque tokens and API keys.

Access tokens are EdDSA (Ed25519) JWTs. The ``kid`` header names the signing key; every key in
``MEDLENS_AUTH__JWT_KEYS`` verifies, so a key can be rotated without logging anyone out. The
``typ`` header separates access tokens from short-lived MFA tokens, so one can never be used as
the other.
"""

import base64
import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from pwdlib import PasswordHash

from medlens.core.config import AuthSettings
from medlens.core.ids import uuid7

TokenType = Literal["access", "mfa"]
_HEADER_TYPES: dict[TokenType, str] = {"access": "at+jwt", "mfa": "mfa+jwt"}
_ALGORITHM = "EdDSA"
_LEEWAY_SECONDS = 10


class InvalidTokenError(Exception):
    """Expired, forged, wrong audience or issuer, wrong type, or signed by an unknown key."""


@dataclass(frozen=True, slots=True)
class TokenClaims:
    subject: UUID  # user id
    session_id: UUID | None  # refresh-token family; None on MFA tokens
    organization_id: UUID | None
    perm_ver: int
    auth_methods: tuple[str, ...]  # RFC 8176 "amr": pwd, otp, mfa
    token_id: str
    issued_at: int
    expires_at: int
    purpose: str | None = None  # MFA tokens: "verify" or "enrol"


class JwtKeyRing:
    def __init__(self, settings: AuthSettings) -> None:
        self._issuer = settings.jwt_issuer
        self._audience = settings.jwt_audience
        self._access_ttl = settings.access_token_ttl_seconds
        self._mfa_ttl = settings.mfa_token_ttl_seconds
        self._private = {
            key_id: Ed25519PrivateKey.from_private_bytes(base64.b64decode(seed.get_secret_value()))
            for key_id, seed in settings.jwt_keys.items()
        }
        self._public: dict[str, Ed25519PublicKey] = {
            key_id: key.public_key() for key_id, key in self._private.items()
        }
        self._active_key_id = settings.jwt_active_key_id

    @property
    def access_ttl_seconds(self) -> int:
        return self._access_ttl

    @property
    def mfa_ttl_seconds(self) -> int:
        return self._mfa_ttl

    def issue(
        self,
        token_type: TokenType,
        *,
        subject: UUID,
        session_id: UUID | None = None,
        organization_id: UUID | None = None,
        perm_ver: int = 0,
        auth_methods: tuple[str, ...] = (),
        purpose: str | None = None,
        now: int | None = None,
    ) -> str:
        issued_at = int(time.time()) if now is None else now
        ttl = self._access_ttl if token_type == "access" else self._mfa_ttl  # noqa: S105
        claims: dict[str, Any] = {
            "iss": self._issuer,
            "aud": self._audience,
            "sub": str(subject),
            "sid": str(session_id) if session_id else None,
            "org": str(organization_id) if organization_id else None,
            "perm_ver": perm_ver,
            "amr": list(auth_methods),
            "iat": issued_at,
            "exp": issued_at + ttl,
            "jti": str(uuid7()),
        }
        if purpose is not None:
            claims["purpose"] = purpose
        return jwt.encode(
            claims,
            self._private[self._active_key_id],
            algorithm=_ALGORITHM,
            headers={"kid": self._active_key_id, "typ": _HEADER_TYPES[token_type]},
        )

    def verify(self, token: str, token_type: TokenType) -> TokenClaims:
        try:
            header = jwt.get_unverified_header(token)
            key = self._public.get(str(header.get("kid")))
            if key is None or header.get("typ") != _HEADER_TYPES[token_type]:
                raise InvalidTokenError("unknown key or wrong token type")
            claims = jwt.decode(
                token,
                key,
                algorithms=[_ALGORITHM],
                audience=self._audience,
                issuer=self._issuer,
                leeway=_LEEWAY_SECONDS,
                options={"require": ["exp", "iat", "sub", "jti", "iss", "aud"]},
            )
            return TokenClaims(
                subject=UUID(claims["sub"]),
                session_id=UUID(claims["sid"]) if claims.get("sid") else None,
                organization_id=UUID(claims["org"]) if claims.get("org") else None,
                perm_ver=int(claims.get("perm_ver", 0)),
                auth_methods=tuple(claims.get("amr", ())),
                token_id=str(claims["jti"]),
                issued_at=int(claims["iat"]),
                expires_at=int(claims["exp"]),
                purpose=claims.get("purpose"),
            )
        except (jwt.PyJWTError, KeyError, ValueError, TypeError) as exc:
            raise InvalidTokenError(str(exc)) from None

    def jwks(self) -> dict[str, list[dict[str, str]]]:
        """Public keys in JWK Set form (RFC 7517 / RFC 8037), for /.well-known/jwks.json."""
        keys = []
        for key_id, key in self._public.items():
            raw = key.public_bytes(Encoding.Raw, PublicFormat.Raw)
            keys.append(
                {
                    "kty": "OKP",
                    "crv": "Ed25519",
                    "x": base64.urlsafe_b64encode(raw).rstrip(b"=").decode(),
                    "kid": key_id,
                    "use": "sig",
                    "alg": _ALGORITHM,
                }
            )
        return {"keys": keys}


# ------------------------------------------------------------------------------ passwords
_password_hash = PasswordHash.recommended()  # argon2id with the library's current parameters
# Verified against when the account does not exist, so both paths take the same time.
_DUMMY_HASH = _password_hash.hash(secrets.token_urlsafe(16))


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str | None) -> tuple[bool, str | None]:
    """(matches, new hash if the stored one uses outdated parameters)."""
    if password_hash is None:
        _password_hash.verify(password, _DUMMY_HASH)
        return False, None
    return _password_hash.verify_and_update(password, password_hash)


# ------------------------------------------------------------------------------ opaque tokens
def new_opaque_token() -> str:
    """256 random bits, URL-safe (refresh tokens)."""
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    """Lookup hash of a high-entropy token; such tokens need no salt or slow hash."""
    return hashlib.sha256(token.encode()).hexdigest()


def constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode(), right.encode())


# ------------------------------------------------------------------------------ API keys
API_KEY_PREFIX = "ml_live_"  # pragma: allowlist secret
_KEY_ID_LENGTH = 12  # 48 bits: prefix collisions stay negligible
_KEY_ID = re.compile(r"^[0-9a-f]{12}$")
_MIN_SECRET_LENGTH = 32


@dataclass(frozen=True, slots=True)
class NewApiKey:
    key: str  # shown once, never stored
    prefix: str  # stored, identifies the key
    digest: str  # stored


def new_api_key() -> NewApiKey:
    prefix = secrets.token_hex(_KEY_ID_LENGTH // 2)
    key = f"{API_KEY_PREFIX}{prefix}_{secrets.token_urlsafe(32)}"
    return NewApiKey(key=key, prefix=prefix, digest=token_digest(key))


def parse_api_key_prefix(key: str) -> str | None:
    """The prefix of a well-formed key, else None. The secret part may itself contain '_'."""
    if not key.startswith(API_KEY_PREFIX):
        return None
    rest = key.removeprefix(API_KEY_PREFIX)
    prefix, separator, secret = (
        rest[:_KEY_ID_LENGTH],
        rest[_KEY_ID_LENGTH : _KEY_ID_LENGTH + 1],
        rest[_KEY_ID_LENGTH + 1 :],
    )
    well_formed = (
        separator == "_" and len(secret) >= _MIN_SECRET_LENGTH and _KEY_ID.fullmatch(prefix)
    )
    return prefix if well_formed else None

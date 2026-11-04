from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, SecretStr, StringConstraints, field_validator

from medlens.core.schemas import EmailAddress, InputModel, NonEmptyStr, PhoneNumberNG, ReadModel
from medlens.modules.rbac.permissions import Permission

OtpCode = Annotated[str, StringConstraints(pattern=r"^\d{6}$")]
Email = EmailAddress
Password = Annotated[SecretStr, Field(min_length=1, max_length=256)]

_EXAMPLE_ACCESS = "<access token (JWT)>"
_EXAMPLE_ORG = "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e12"


class OtpRequest(InputModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"phone": "0803 123 4567"}]})

    phone: PhoneNumberNG


class OtpRequested(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "detail": "If the number can receive SMS, a code is on its way.",
                    "expires_in": 300,
                }
            ]
        }
    )

    detail: str
    expires_in: int = Field(description="Seconds the code stays valid.")


class OtpVerify(InputModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"phone": "+2348031234567", "code": "482913"}]}
    )

    phone: PhoneNumberNG
    code: OtpCode


class LoginRequest(InputModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"email": "analyst@nafdac.gov.ng", "password": "••••••••••••"}]
        }
    )

    email: Email
    password: Password


class TokenResponse(ReadModel):
    """A signed-in session: a short-lived access token and a rotating refresh token."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "authenticated",
                    "access_token": _EXAMPLE_ACCESS,
                    "token_type": "Bearer",
                    "expires_in": 900,
                    "refresh_token": "<refresh token>",
                    "refresh_expires_in": 2592000,
                    "organization_id": _EXAMPLE_ORG,
                }
            ]
        }
    )

    status: Literal["authenticated"] = "authenticated"
    access_token: str
    token_type: Literal["Bearer"] = "Bearer"  # noqa: S105 - the scheme name, not a secret
    expires_in: int = Field(description="Access-token lifetime in seconds.")
    refresh_token: str = Field(description="Single use: every refresh returns a new one.")
    refresh_expires_in: int
    organization_id: UUID | None = Field(description="Active organization of the session.")


class MfaChallengeResponse(ReadModel):
    """Password accepted; a second factor is needed. ``mfa_required``: call /auth/mfa/verify.
    ``mfa_enrolment_required``: this role must use MFA; set it up with /auth/mfa/enrol and
    /auth/mfa/confirm, sending the mfa_token as the bearer token."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"status": "mfa_required", "mfa_token": _EXAMPLE_ACCESS, "expires_in": 300}
            ]
        }
    )

    status: Literal["mfa_required", "mfa_enrolment_required"]
    mfa_token: str
    expires_in: int


LoginResponse = Annotated[TokenResponse | MfaChallengeResponse, Field(discriminator="status")]


class MfaVerifyRequest(InputModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"mfa_token": _EXAMPLE_ACCESS, "code": "123456"}]}
    )

    mfa_token: str = Field(max_length=4096)
    code: OtpCode


class MfaEnrolment(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "secret": "<base32 secret>",
                    "otpauth_uri": (
                        "otpauth://totp/MedLens:analyst%40nafdac.gov.ng"
                        "?secret=<base32 secret>&issuer=MedLens"
                    ),
                }
            ]
        }
    )

    secret: str = Field(description="Base32 secret, for manual entry in an authenticator app.")
    otpauth_uri: str = Field(description="Encode as a QR code for authenticator apps.")


class MfaConfirmRequest(InputModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"code": "123456"}]})

    code: OtpCode


class RefreshRequest(InputModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"refresh_token": "<refresh token>"}]}
    )

    refresh_token: str = Field(min_length=20, max_length=128)


class SwitchOrganizationRequest(InputModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"organization_id": _EXAMPLE_ORG}]})

    organization_id: UUID | None = Field(description="null clears the active organization.")


class AccessTokenResponse(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "access_token": _EXAMPLE_ACCESS,
                    "token_type": "Bearer",
                    "expires_in": 900,
                    "organization_id": _EXAMPLE_ORG,
                }
            ]
        }
    )

    access_token: str
    token_type: Literal["Bearer"] = "Bearer"  # noqa: S105 - the scheme name, not a secret
    expires_in: int
    organization_id: UUID | None


class Jwk(ReadModel):
    kty: str
    crv: str
    x: str
    kid: str
    use: str
    alg: str


class JwksResponse(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "keys": [
                        {
                            "kty": "OKP",
                            "crv": "Ed25519",
                            "x": "<base64url Ed25519 public key>",
                            "kid": "2026-11",
                            "use": "sig",
                            "alg": "EdDSA",
                        }
                    ]
                }
            ]
        }
    )

    keys: list[Jwk]


class ApiKeyCreate(InputModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "ERP batch sync",
                    "scopes": ["product:read", "batch:write"],
                    "expires_at": "2027-11-03T00:00:00Z",
                }
            ]
        }
    )

    name: Annotated[NonEmptyStr, StringConstraints(max_length=128)]
    scopes: list[Permission] = Field(min_length=1)
    expires_at: datetime | None = Field(default=None, description="Never expires when null.")

    @field_validator("expires_at")
    @classmethod
    def _aware_and_future(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value <= datetime.now(value.tzinfo)):
            raise ValueError("must be a future time with a timezone")
        return value


class ApiKeyRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e20",
                    "name": "ERP batch sync",
                    "prefix": "3f9a1c2e0b7d",  # pragma: allowlist secret
                    "scopes": ["batch:write", "product:read"],
                    "created_at": "2026-11-03T09:00:00Z",
                    "last_used_at": None,
                    "expires_at": "2027-11-03T00:00:00Z",
                    "revoked_at": None,
                }
            ]
        }
    )

    id: UUID
    name: str
    prefix: str = Field(description="Public part of the key, to tell keys apart.")
    scopes: list[str]
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None


class ApiKeyCreated(ApiKeyRead):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e20",
                    "name": "ERP batch sync",
                    "prefix": "3f9a1c2e0b7d",  # pragma: allowlist secret
                    "scopes": ["batch:write", "product:read"],
                    "created_at": "2026-11-03T09:00:00Z",
                    "last_used_at": None,
                    "expires_at": "2027-11-03T00:00:00Z",
                    "revoked_at": None,
                    "key": "ml_live_3f9a1c2e0b7d_<secret>",
                }
            ]
        }
    )

    key: str = Field(description="The full key. Shown only once: store it securely now.")

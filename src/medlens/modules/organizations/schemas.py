from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, JsonValue, SecretStr, model_validator

from medlens.core.identifiers import NigerianState
from medlens.core.schemas import (
    EmailAddress,
    InputModel,
    Latitude,
    Longitude,
    NonEmptyStr,
    ReadModel,
    SafeText,
    UpdateModel,
)
from medlens.modules.organizations.models import OrganizationStatus, OrganizationType
from medlens.modules.rbac.models import RoleName


class MemberRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "organization_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e12",
                    "user_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e30",
                    "full_name": "Amina Bello",
                    "email": "pharmacist@example.ng",
                    "role": "pharmacy_staff",
                    "status": "active",
                    "created_at": "2026-11-03T09:00:00Z",
                }
            ]
        }
    )

    organization_id: UUID
    user_id: UUID
    full_name: str | None
    email: str | None
    role: str
    status: str
    created_at: datetime


class RoleAssignment(InputModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"role": "pharmacy_staff"}]})

    role: RoleName


# ------------------------------------------------------------------ organizations
class OrganizationCreate(InputModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "Ikeja Community Pharmacy",
                    "type": "pharmacy",
                    "registration_number": "PCN/LA/2024/0193",
                    "state": "NG-LA",
                }
            ]
        }
    )

    name: NonEmptyStr = Field(max_length=200)
    type: OrganizationType
    registration_number: str | None = Field(
        default=None,
        max_length=64,
        description="PCN premises licence, NAFDAC or CAC number, depending on the type.",
    )
    state: NigerianState | None = None
    jurisdiction_states: list[NigerianState] | None = Field(
        default=None,
        max_length=37,
        description="Regulators only: the states they oversee (all 37 for a national agency).",
    )

    @model_validator(mode="after")
    def _jurisdiction(self) -> "OrganizationCreate":
        is_regulator = self.type is OrganizationType.REGULATOR
        if is_regulator != bool(self.jurisdiction_states):
            raise ValueError("jurisdiction_states is required for regulators and only for them")
        return self


class FounderAccount(InputModel):
    """The organization's first administrator, who registers it."""

    full_name: NonEmptyStr = Field(max_length=200)
    email: EmailAddress
    password: Annotated[SecretStr, Field(min_length=1, max_length=256)] = Field(
        description="12+ characters, not a common password, not containing the address."
    )


class OrganizationRegistration(InputModel):
    """The public sign-up form for pharmacies, PPMVs, clinics, manufacturers and distributors:
    the organization and its first administrator. Regulators are created by platform admins."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "organization": {
                        "name": "Ikeja Community Pharmacy",
                        "type": "pharmacy",
                        "registration_number": "PCN/LA/2024/0193",
                        "state": "NG-LA",
                    },
                    "admin": {
                        "full_name": "Amina Bello",
                        "email": "amina@ikejapharmacy.ng",
                        "password": "••••••••••••••",
                    },
                }
            ]
        }
    )

    organization: OrganizationCreate
    admin: FounderAccount

    @model_validator(mode="after")
    def _not_a_regulator(self) -> "OrganizationRegistration":
        if self.organization.type is OrganizationType.REGULATOR:
            raise ValueError("regulator organizations are created by platform administrators")
        if not self.organization.registration_number:
            raise ValueError(
                "organization.registration_number is required: the licence an administrator "
                "checks before verifying the organization (PCN premises licence, NAFDAC or CAC)"
            )
        return self


class OrganizationRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e12",
                    "name": "Ikeja Community Pharmacy",
                    "type": "pharmacy",
                    "registration_number": "PCN/LA/2024/0193",
                    "state": "NG-LA",
                    "jurisdiction_states": None,
                    "status": "pending_verification",
                    "verified_at": None,
                    "created_at": "2026-11-03T09:00:00Z",
                }
            ]
        }
    )

    id: UUID
    name: str
    type: OrganizationType
    registration_number: str | None
    state: NigerianState | None
    jurisdiction_states: list[str] | None
    status: OrganizationStatus
    verified_at: datetime | None
    created_at: datetime


# ------------------------------------------------------------------ invitations
class InvitationCreate(InputModel):
    """Staff sign in with e-mail, password and MFA, so invitations go to an e-mail address."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"email": "pharmacist@example.ng", "role": "pharmacy_staff"}]
        }
    )

    email: EmailAddress
    role: RoleName


class PlatformInvitationCreate(InputModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"email": "ops@medlens.co"}]})

    email: EmailAddress


class InvitationPreview(ReadModel):
    """What an invitation offers, for its landing page (the token is the only key)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "organization_name": "Lekki Community Pharmacy",
                    "organization_type": "pharmacy",
                    "role": "pharmacy_staff",
                    "recipient": "am***@example.ng",
                    "expires_at": "2026-11-26T09:00:00Z",
                    "account_exists": False,
                }
            ]
        }
    )

    organization_name: str | None = Field(description="None: a platform-admin invitation.")
    organization_type: str | None
    role: str
    recipient: str = Field(description="The invited address, masked.")
    expires_at: datetime
    account_exists: bool = Field(
        description=(
            "Whether the invited address already has an account: sign in and accept "
            "(POST /invitations/accept); otherwise create it "
            "(POST /invitations/{token}/accept-new)."
        )
    )


class InvitationAcceptNew(InputModel):
    """A new person: the account is created from the invitation (receiving it proves the
    address)."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"full_name": "Amina Bello", "password": "••••••••••••••"}]}
    )

    full_name: NonEmptyStr = Field(max_length=200)
    password: Annotated[SecretStr, Field(min_length=1, max_length=256)] = Field(
        description="12+ characters, not a common password, not containing the address."
    )


class InvitationAccepted(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "organization_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e12",
                    "role": "pharmacy_staff",
                    "signed_out": True,
                }
            ]
        }
    )

    organization_id: UUID | None = Field(description="None: you are now a platform admin.")
    role: str
    signed_out: bool = Field(
        description=(
            "True when the account just became a staff account: its sessions ended, and it "
            "now signs in with e-mail, password and an authenticator app."
        )
    )


InvitationStatus = Literal["pending", "accepted", "revoked", "expired"]


class InvitationRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e60",
                    "organization_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e12",
                    "role": "pharmacy_staff",
                    "recipient": "ph***@example.ng",
                    "status": "pending",
                    "expires_at": "2026-11-10T09:00:00Z",
                    "created_at": "2026-11-03T09:00:00Z",
                }
            ]
        }
    )

    id: UUID
    organization_id: UUID | None = Field(description="None: a platform-admin invitation.")
    role: str
    recipient: str = Field(description="Masked e-mail address or phone number.")
    status: InvitationStatus
    expires_at: datetime
    created_at: datetime


class InvitationCreated(InvitationRead):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e60",
                    "organization_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e12",
                    "role": "pharmacy_staff",
                    "recipient": "ph***@example.ng",
                    "status": "pending",
                    "expires_at": "2026-11-10T09:00:00Z",
                    "created_at": "2026-11-03T09:00:00Z",
                    "token": "<invitation token>",
                }
            ]
        }
    )

    token: str = Field(
        description=(
            "Shown once; also e-mailed to the invitee (with a link when "
            "MEDLENS_APP__FRONTEND_URL is set)."
        )
    )


class InvitationAccept(InputModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"token": "<invitation token>"}]})

    token: str = Field(min_length=16, max_length=128)
    password: Annotated[SecretStr, Field(min_length=1, max_length=256)] | None = Field(
        default=None,
        description=(
            "Required when your account has no password yet (it signed in with codes): staff "
            "sign in with e-mail, password and an authenticator app."
        ),
    )


# ------------------------------------------------------------------ branches
_BRANCH_EXAMPLE: dict[str, JsonValue] = {
    "name": "Ikeja branch",
    "address": "12 Allen Avenue, Ikeja",
    "state": "NG-LA",
    "latitude": 6.6018,
    "longitude": 3.3515,
}


class BranchCreate(InputModel):
    model_config = ConfigDict(json_schema_extra={"examples": [_BRANCH_EXAMPLE]})

    name: NonEmptyStr = Field(max_length=200)
    address: SafeText | None = None
    state: NigerianState | None = None
    latitude: Latitude | None = None
    longitude: Longitude | None = None

    @model_validator(mode="after")
    def _both_coordinates(self) -> "BranchCreate":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("give both latitude and longitude, or neither")
        return self


class BranchUpdate(UpdateModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"address": "14 Allen Avenue"}]})

    name: NonEmptyStr | None = Field(default=None, max_length=200)
    address: SafeText | None = None
    state: NigerianState | None = None
    latitude: Latitude | None = None
    longitude: Longitude | None = None

    @model_validator(mode="after")
    def _both_coordinates(self) -> "BranchUpdate":
        sent = self.model_fields_set & {"latitude", "longitude"}
        if len(sent) == 1:
            raise ValueError("change latitude and longitude together")
        return self


class BranchRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e70",
                    "organization_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e12",
                    **_BRANCH_EXAMPLE,
                    "created_at": "2026-11-03T09:00:00Z",
                }
            ]
        }
    )

    id: UUID
    organization_id: UUID
    name: str
    address: str | None
    state: NigerianState | None
    latitude: float | None
    longitude: float | None
    created_at: datetime

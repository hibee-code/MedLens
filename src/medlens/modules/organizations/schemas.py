from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import ConfigDict, Field, JsonValue, model_validator

from medlens.core.identifiers import NigerianState
from medlens.core.schemas import (
    EmailAddress,
    InputModel,
    Latitude,
    Longitude,
    NonEmptyStr,
    PhoneNumberNG,
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
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"email": "pharmacist@example.ng", "role": "pharmacy_staff"},
                {"phone": "+2348031234567", "role": "pharmacy_staff"},
            ]
        }
    )

    email: EmailAddress | None = None
    phone: PhoneNumberNG | None = None
    role: RoleName

    @model_validator(mode="after")
    def _one_recipient(self) -> "InvitationCreate":
        if (self.email is None) == (self.phone is None):
            raise ValueError("give either email or phone")
        return self


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
    organization_id: UUID
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
            "Shown once. Pass it to the invitee, who accepts it while signed in with the "
            "invited e-mail address or phone number. MedLens does not send it."
        )
    )


class InvitationAccept(InputModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"token": "<invitation token>"}]})

    token: str = Field(min_length=16, max_length=128)


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

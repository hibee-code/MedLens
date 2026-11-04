from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import ConfigDict, Field, SecretStr, StringConstraints

from medlens.core.schemas import InputModel, ReadModel, SafeText, UpdateModel


class MembershipSummary(ReadModel):
    organization_id: UUID
    organization_name: str
    organization_type: str
    role: str
    status: str


class UserRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e30",
                    "email": "pharmacist@example.ng",
                    "phone": None,
                    "full_name": "Amina Bello",
                    "status": "active",
                    "global_role": "consumer",
                    "mfa_enabled": False,
                    "created_at": "2026-11-03T09:00:00Z",
                    "memberships": [
                        {
                            "organization_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e12",
                            "organization_name": "Lekki Community Pharmacy",
                            "organization_type": "pharmacy",
                            "role": "pharmacy_admin",
                            "status": "active",
                        }
                    ],
                }
            ]
        }
    )

    id: UUID
    email: str | None
    phone: str | None = Field(description="E.164, visible only to the account owner.")
    full_name: str | None
    status: str
    global_role: str
    mfa_enabled: bool
    created_at: datetime
    memberships: list[MembershipSummary]


class UserUpdate(UpdateModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"full_name": "Amina Bello"}]})

    full_name: Annotated[SafeText, StringConstraints(min_length=1, max_length=200)] | None = None


class PasswordChange(InputModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"current_password": "••••••••••••", "new_password": "••••••••••••••"}]
        }
    )

    current_password: Annotated[SecretStr, Field(min_length=1, max_length=256)]
    new_password: Annotated[SecretStr, Field(min_length=1, max_length=256)]

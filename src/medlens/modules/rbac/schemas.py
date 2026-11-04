from uuid import UUID

from pydantic import ConfigDict

from medlens.core.schemas import InputModel, ReadModel
from medlens.modules.rbac.models import RoleName


class RoleRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "pharmacy_staff",
                    "scope": "organization",
                    "description": "Staff of a pharmacy, PPMV or clinic.",
                    "permissions": ["org:read", "product:read", "scan:create"],
                }
            ]
        }
    )

    name: str
    scope: str
    description: str
    permissions: list[str]


class PermissionRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"name": "scan:create", "description": "Submit a pack photo for verification."}
            ]
        }
    )

    name: str
    description: str


class GlobalRoleAssignment(InputModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"role": "health_worker"}]})

    role: RoleName


class UserRoleRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e30", "global_role": "health_worker"}
            ]
        }
    )

    id: UUID
    global_role: str

"""Roles and permissions. Natural keys ("pharmacy_admin", "scan:create") so rows read clearly in
memberships, audit logs and SQL. Phase 4 fills the permission matrix."""

from datetime import datetime
from enum import StrEnum

from sqlalchemy import ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from medlens.db.base import Base
from medlens.db.types import str_enum


class RoleScope(StrEnum):
    GLOBAL = "global"  # held by a user directly (users.global_role)
    ORGANIZATION = "organization"  # held through a membership


class RoleName(StrEnum):
    CONSUMER = "consumer"
    HEALTH_WORKER = "health_worker"
    PHARMACY_STAFF = "pharmacy_staff"
    PHARMACY_ADMIN = "pharmacy_admin"
    MANUFACTURER_STAFF = "manufacturer_staff"
    MANUFACTURER_ADMIN = "manufacturer_admin"
    REGULATOR_ANALYST = "regulator_analyst"
    REGULATOR_ADMIN = "regulator_admin"
    PLATFORM_ADMIN = "platform_admin"


# Reference rows inserted by the migration that creates the table.
ROLE_DEFINITIONS: dict[RoleName, tuple[RoleScope, str]] = {
    RoleName.CONSUMER: (RoleScope.GLOBAL, "Member of the public verifying a pack."),
    RoleName.HEALTH_WORKER: (RoleScope.GLOBAL, "Health worker verifying packs for patients."),
    RoleName.PHARMACY_STAFF: (RoleScope.ORGANIZATION, "Staff of a pharmacy, PPMV or clinic."),
    RoleName.PHARMACY_ADMIN: (
        RoleScope.ORGANIZATION,
        "Administrator of a pharmacy, PPMV or clinic.",
    ),
    RoleName.MANUFACTURER_STAFF: (RoleScope.ORGANIZATION, "Staff of a manufacturer."),
    RoleName.MANUFACTURER_ADMIN: (RoleScope.ORGANIZATION, "Administrator of a manufacturer."),
    RoleName.REGULATOR_ANALYST: (RoleScope.ORGANIZATION, "Analyst at a regulator."),
    RoleName.REGULATOR_ADMIN: (RoleScope.ORGANIZATION, "Administrator of a regulator."),
    RoleName.PLATFORM_ADMIN: (RoleScope.GLOBAL, "MedLens platform administrator."),
}


class Role(Base):
    __tablename__ = "roles"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    scope: Mapped[RoleScope] = mapped_column(str_enum(RoleScope, "role_scope"))
    description: Mapped[str] = mapped_column(Text, default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Permission(Base):
    __tablename__ = "permissions"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str] = mapped_column(Text, default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class RolePermission(Base):
    __tablename__ = "role_permissions"

    role_name: Mapped[str] = mapped_column(
        ForeignKey("roles.name", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True
    )
    permission_name: Mapped[str] = mapped_column(
        ForeignKey("permissions.name", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True
    )

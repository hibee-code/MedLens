"""Identity: roles and permissions, users, organizations, memberships, branches, auth tokens.

The nine role names are reference data inserted here: users.global_role and memberships.role
reference them, and sign-up needs the "consumer" row in every environment. Permissions and the
role matrix arrive in Phase 4.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import CITEXT

from migrations.helpers import (
    STATE_CODES_ARRAY,
    created_at,
    deleted_at,
    empty_array,
    enum,
    foreign_key,
    id_column,
    point,
    state_enum,
    updated_at,
)

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ROLES = [
    ("consumer", "global", "Member of the public verifying a pack."),
    ("health_worker", "global", "Health worker verifying packs for patients."),
    ("pharmacy_staff", "organization", "Staff of a pharmacy, PPMV or clinic."),
    ("pharmacy_admin", "organization", "Administrator of a pharmacy, PPMV or clinic."),
    ("manufacturer_staff", "organization", "Staff of a manufacturer."),
    ("manufacturer_admin", "organization", "Administrator of a manufacturer."),
    ("regulator_analyst", "organization", "Analyst at a regulator."),
    ("regulator_admin", "organization", "Administrator of a regulator."),
    ("platform_admin", "global", "MedLens platform administrator."),
]


def upgrade() -> None:
    roles = op.create_table(
        "roles",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("scope", enum("role_scope", "global", "organization"), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        created_at(),
    )
    op.bulk_insert(
        roles,
        [{"name": name, "scope": scope, "description": text} for name, scope, text in ROLES],
    )
    op.create_table(
        "permissions",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        created_at(),
    )
    op.create_table(
        "role_permissions",
        sa.Column(
            "role_name",
            sa.String(64),
            sa.ForeignKey("roles.name", ondelete="CASCADE", onupdate="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "permission_name",
            sa.String(64),
            sa.ForeignKey("permissions.name", ondelete="CASCADE", onupdate="CASCADE"),
            primary_key=True,
        ),
    )

    op.create_table(
        "users",
        id_column(),
        sa.Column("phone_e164", sa.Text(), nullable=True),  # AES-GCM ciphertext
        sa.Column("phone_hash", sa.String(64), nullable=True, unique=True),
        sa.Column("email", CITEXT(), nullable=True, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=True),
        sa.Column("full_name", sa.Text(), nullable=True),
        sa.Column(
            "status",
            enum("user_status", "active", "suspended", "deleted"),
            server_default="active",
            nullable=False,
        ),
        sa.Column(
            "global_role",
            sa.String(64),
            sa.ForeignKey("roles.name", onupdate="CASCADE"),
            server_default="consumer",
            nullable=False,
        ),
        sa.Column("mfa_secret", sa.Text(), nullable=True),  # AES-GCM ciphertext
        sa.Column("mfa_enabled", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        created_at(),
        updated_at(),
        sa.CheckConstraint(
            "status = 'deleted' OR phone_hash IS NOT NULL OR email IS NOT NULL", name="identity"
        ),
        sa.CheckConstraint("(phone_e164 IS NULL) = (phone_hash IS NULL)", name="phone_pair"),
    )

    op.create_table(
        "organizations",
        id_column(),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "type",
            enum(
                "organization_type",
                "pharmacy",
                "ppmv",
                "clinic",
                "manufacturer",
                "distributor",
                "regulator",
            ),
            nullable=False,
        ),
        sa.Column("registration_number", sa.String(64), nullable=True),
        sa.Column("state", state_enum(), nullable=True),
        sa.Column("jurisdiction_states", sa.ARRAY(sa.String(5)), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "status",
            enum("organization_status", "pending_verification", "active", "suspended"),
            server_default="pending_verification",
            nullable=False,
        ),
        created_at(),
        updated_at(),
        deleted_at(),
        sa.CheckConstraint(
            "(type = 'regulator') = (jurisdiction_states IS NOT NULL)", name="jurisdiction_owner"
        ),
        sa.CheckConstraint(
            "jurisdiction_states IS NULL OR (cardinality(jurisdiction_states) > 0 "
            f"AND jurisdiction_states <@ {STATE_CODES_ARRAY})",
            name="jurisdiction_states",
        ),
    )
    op.create_index(
        "uq_organizations_type_registration_number",
        "organizations",
        ["type", "registration_number"],
        unique=True,
        postgresql_where=sa.text("registration_number IS NOT NULL"),
    )

    op.create_table(
        "memberships",
        id_column(),
        foreign_key("user_id", "users.id", nullable=False, ondelete="CASCADE"),
        foreign_key("organization_id", "organizations.id", nullable=False, ondelete="CASCADE"),
        sa.Column(
            "role", sa.String(64), sa.ForeignKey("roles.name", onupdate="CASCADE"), nullable=False
        ),
        sa.Column(
            "status",
            enum("membership_status", "invited", "active", "suspended"),
            server_default="active",
            nullable=False,
        ),
        foreign_key("invited_by", "users.id", nullable=True, ondelete="SET NULL"),
        created_at(),
        updated_at(),
        sa.UniqueConstraint("user_id", "organization_id"),
    )
    op.create_index("ix_memberships_organization_id", "memberships", ["organization_id"])

    op.create_table(
        "branches",
        id_column(),
        foreign_key("organization_id", "organizations.id", nullable=False, ondelete="CASCADE"),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("address", sa.Text(), nullable=True),
        sa.Column("state", state_enum(), nullable=True),
        point("location"),
        created_at(),
        updated_at(),
        deleted_at(),
    )
    op.create_index("ix_branches_organization_id", "branches", ["organization_id"])
    op.create_index("ix_branches_location", "branches", ["location"], postgresql_using="gist")

    op.create_table(
        "refresh_tokens",
        id_column(),
        foreign_key("user_id", "users.id", nullable=False, ondelete="CASCADE"),
        sa.Column("family_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        foreign_key("replaced_by_id", "refresh_tokens.id", nullable=True, ondelete="SET NULL"),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("ip_hash", sa.String(64), nullable=True),
        created_at(),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_index("ix_refresh_tokens_family_id", "refresh_tokens", ["family_id"])

    op.create_table(
        "api_keys",
        id_column(),
        foreign_key("organization_id", "organizations.id", nullable=False, ondelete="CASCADE"),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("prefix", sa.String(16), nullable=False, unique=True),
        sa.Column("key_hash", sa.Text(), nullable=False),
        empty_array("scopes", sa.String(64)),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        created_at(),
        updated_at(),
    )
    op.create_index("ix_api_keys_organization_id", "api_keys", ["organization_id"])


def downgrade() -> None:
    for table in (
        "api_keys",
        "refresh_tokens",
        "branches",
        "memberships",
        "organizations",
        "users",
        "role_permissions",
        "permissions",
        "roles",
    ):
        op.drop_table(table)

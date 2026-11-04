"""Phase 5: invitations, product claims, reference-image review, scan codes, case states,
labels, and the product:review permission.

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import CITEXT

from migrations.helpers import created_at, enum, id_column, state_enum, updated_at

revision: str = "0010"
down_revision: str | Sequence[str] | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

REVIEW_STATUSES = ("pending", "approved", "rejected")
PRODUCT_REVIEW = ("product:review", "Review manufacturers' product claims and reference images.")
PRODUCT_REVIEWERS = ("regulator_admin", "platform_admin")


def _user(name: str) -> sa.Column[sa.Uuid]:
    return sa.Column(name, sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"))


def upgrade() -> None:
    op.create_table(
        "invitations",
        id_column(),
        created_at(),
        updated_at(),
        sa.Column(
            "organization_id",
            sa.Uuid(),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "role", sa.String(), sa.ForeignKey("roles.name", onupdate="CASCADE"), nullable=False
        ),
        sa.Column("email", CITEXT()),
        sa.Column("phone_e164", sa.Text()),
        sa.Column("phone_hash", sa.String(64)),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        _user("invited_by"),
        sa.Column("accepted_at", sa.DateTime(timezone=True)),
        _user("accepted_by"),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("(email IS NULL) <> (phone_hash IS NULL)", name="one_recipient"),
        sa.CheckConstraint("(phone_hash IS NULL) = (phone_e164 IS NULL)", name="phone_pair"),
    )
    op.create_index(
        "ix_invitations_organization_id_created_at",
        "invitations",
        ["organization_id", "created_at"],
    )

    op.create_table(
        "product_claims",
        id_column(),
        created_at(),
        updated_at(),
        sa.Column(
            "organization_id",
            sa.Uuid(),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "product_id",
            sa.Uuid(),
            sa.ForeignKey("products.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "status",
            enum("review_status", *REVIEW_STATUSES),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("note", sa.Text()),
        _user("requested_by"),
        _user("reviewed_by"),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.Column("review_note", sa.Text()),
    )
    op.create_index(
        "uq_product_claims_open",
        "product_claims",
        ["organization_id", "product_id"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(
        "ix_product_claims_status_created_at", "product_claims", ["status", "created_at"]
    )

    # Existing reference images were never reviewed: they start as pending.
    op.add_column(
        "reference_images",
        sa.Column(
            "status",
            enum("review_status", *REVIEW_STATUSES),
            server_default="pending",
            nullable=False,
        ),
    )
    op.add_column("reference_images", _user("uploaded_by"))
    op.add_column(
        "reference_images",
        sa.Column(
            "organization_id", sa.Uuid(), sa.ForeignKey("organizations.id", ondelete="SET NULL")
        ),
    )
    op.add_column("reference_images", sa.Column("reviewed_at", sa.DateTime(timezone=True)))

    # Partitioned: columns and indexes added to the parent reach every partition.
    for column in ("reason_codes", "advice_codes"):
        op.add_column(
            "scans",
            sa.Column(
                column,
                sa.ARRAY(sa.String(32)),
                server_default=sa.text("'{}'"),
                nullable=False,
            ),
        )
    op.create_index(
        "ix_scans_device_fingerprint_hash_created_at",
        "scans",
        ["device_fingerprint_hash", "created_at"],
    )

    op.add_column("cases", sa.Column("state", state_enum()))
    op.create_index("ix_cases_state_status", "cases", ["state", "status"])

    op.create_table(
        "labels",
        id_column(),
        created_at(),
        updated_at(),
        sa.Column("scan_id", sa.Uuid(), nullable=False),
        sa.Column("scan_created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "case_id",
            sa.Uuid(),
            sa.ForeignKey("cases.id", ondelete="SET NULL"),
            unique=True,
        ),
        sa.Column(
            "outcome",
            enum("label_outcome", "counterfeit", "genuine", "inconclusive"),
            nullable=False,
        ),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("lab_reference", sa.Text()),
        _user("labelled_by"),
        sa.ForeignKeyConstraint(
            ["scan_id", "scan_created_at"],
            ["scans.id", "scans.created_at"],
            ondelete="RESTRICT",
        ),
    )
    op.create_index("ix_labels_scan_id", "labels", ["scan_id"])

    name, description = PRODUCT_REVIEW
    op.bulk_insert(
        sa.table("permissions", sa.column("name"), sa.column("description")),
        [{"name": name, "description": description}],
    )
    op.bulk_insert(
        sa.table("role_permissions", sa.column("role_name"), sa.column("permission_name")),
        [{"role_name": role, "permission_name": name} for role in PRODUCT_REVIEWERS],
    )


def downgrade() -> None:
    name, _ = PRODUCT_REVIEW
    op.execute(
        sa.text("DELETE FROM role_permissions WHERE permission_name = :name").bindparams(name=name)
    )
    op.execute(sa.text("DELETE FROM permissions WHERE name = :name").bindparams(name=name))
    op.drop_table("labels")
    op.drop_index("ix_cases_state_status", table_name="cases")
    op.drop_column("cases", "state")
    op.drop_index("ix_scans_device_fingerprint_hash_created_at", table_name="scans")
    op.drop_column("scans", "advice_codes")
    op.drop_column("scans", "reason_codes")
    for column in ("reviewed_at", "organization_id", "uploaded_by", "status"):
        op.drop_column("reference_images", column)
    op.drop_table("product_claims")
    op.drop_table("invitations")

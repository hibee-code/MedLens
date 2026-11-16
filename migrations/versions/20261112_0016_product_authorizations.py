"""Organisations a registration holder authorises to declare batches and upload reference
images for a product, recorded by a platform admin.

Revision ID: 0016
Revises: 0015
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.helpers import created_at, id_column, updated_at

revision: str = "0016"
down_revision: str | Sequence[str] | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "product_authorizations",
        id_column(),
        created_at(),
        updated_at(),
        sa.Column(
            "product_id",
            sa.Uuid(),
            sa.ForeignKey("products.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "organization_id",
            sa.Uuid(),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "authorized_by",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "revoked_by",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "uq_product_authorizations_active",
        "product_authorizations",
        ["product_id", "organization_id"],
        unique=True,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )
    op.create_index(
        "ix_product_authorizations_organization_id", "product_authorizations", ["organization_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_product_authorizations_organization_id", "product_authorizations")
    op.drop_index("uq_product_authorizations_active", "product_authorizations")
    op.drop_table("product_authorizations")

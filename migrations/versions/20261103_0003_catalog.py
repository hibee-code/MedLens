"""Catalog: registered products, batches and reference pack images.

Products get GIN trigram indexes for fuzzy lookup. Reference-image embeddings are
vector(MEDLENS_EMBEDDINGS__IMAGE_DIMENSION) with an HNSW cosine index.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from medlens.db.migration_support import migration_settings
from migrations.helpers import (
    created_at,
    deleted_at,
    empty_array,
    enum,
    foreign_key,
    id_column,
    states_check,
    updated_at,
)

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "products",
        id_column(),
        sa.Column("nafdac_number", sa.String(32), nullable=False, unique=True),
        sa.Column("brand_name", sa.Text(), nullable=False),
        sa.Column("generic_name", sa.Text(), nullable=False),
        sa.Column("strength", sa.Text(), nullable=True),
        sa.Column("dosage_form", sa.Text(), nullable=True),
        sa.Column("pack_size", sa.Text(), nullable=True),
        sa.Column("manufacturer_name", sa.Text(), nullable=True),
        foreign_key("manufacturer_org_id", "organizations.id", nullable=True, ondelete="SET NULL"),
        sa.Column(
            "registration_status",
            enum("registration_status", "active", "expired", "suspended", "revoked"),
            nullable=False,
        ),
        sa.Column("registration_expires_on", sa.Date(), nullable=True),
        created_at(),
        updated_at(),
    )
    for column in ("brand_name", "generic_name", "nafdac_number"):
        op.create_index(
            f"ix_products_{column}_trgm",
            "products",
            [column],
            postgresql_using="gin",
            postgresql_ops={column: "gin_trgm_ops"},
        )
    op.create_index("ix_products_manufacturer_org_id", "products", ["manufacturer_org_id"])

    op.create_table(
        "batches",
        id_column(),
        foreign_key("product_id", "products.id", nullable=False, ondelete="CASCADE"),
        sa.Column("batch_number", sa.String(64), nullable=False),
        sa.Column("manufactured_on", sa.Date(), nullable=True),
        sa.Column("expires_on", sa.Date(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=True),
        empty_array("distribution_states", sa.String(5)),
        foreign_key("declared_by_org_id", "organizations.id", nullable=True, ondelete="SET NULL"),
        created_at(),
        updated_at(),
        sa.UniqueConstraint("product_id", "batch_number"),
        sa.CheckConstraint(
            "manufactured_on IS NULL OR expires_on > manufactured_on",
            name="expiry_after_manufacture",
        ),
        sa.CheckConstraint("quantity IS NULL OR quantity >= 0", name="quantity_non_negative"),
        states_check("distribution_states", "distribution_states"),
    )

    dimension = migration_settings().embeddings.image_dimension
    op.create_table(
        "reference_images",
        id_column(),
        foreign_key("product_id", "products.id", nullable=False, ondelete="CASCADE"),
        sa.Column(
            "side", enum("image_side", "front", "back", "blister", "leaflet"), nullable=False
        ),
        sa.Column("s3_key", sa.Text(), nullable=False, unique=True),
        sa.Column("phash", sa.BigInteger(), nullable=True),
        sa.Column("embedding", Vector(dimension), nullable=True),
        sa.Column("embedding_model_version", sa.String(64), nullable=True),
        foreign_key("verified_by", "users.id", nullable=True, ondelete="SET NULL"),
        created_at(),
        updated_at(),
        deleted_at(),
    )
    op.create_index("ix_reference_images_product_id", "reference_images", ["product_id"])
    op.create_index(
        "ix_reference_images_embedding",
        "reference_images",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_table("reference_images")
    op.drop_table("batches")
    op.drop_table("products")

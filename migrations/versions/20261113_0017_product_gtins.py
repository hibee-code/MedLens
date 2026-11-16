"""Products' barcodes (GS1 GTINs) with where each mapping came from (the register or the holder's
declaration), for the cloned-barcode check; and where each product's register entry came from
(a Greenbook snapshot or the hand-typed register file).

Existing products are snapshot rows, except those a hand-typed import added or changed (from the
audit log's register.imported entries), which become manual.

Revision ID: 0017
Revises: 0016
Create Date: 2026-11-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.helpers import created_at, enum, id_column, updated_at

revision: str = "0017"
down_revision: str | Sequence[str] | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "products",
        sa.Column(
            "register_source",
            enum("register_source", "snapshot", "manual"),
            server_default="snapshot",
            nullable=False,
        ),
    )
    op.execute(
        """
        UPDATE products SET register_source = 'manual'
        WHERE nafdac_number IN (
            SELECT jsonb_array_elements_text(metadata->'added') FROM audit_logs
            WHERE action = 'register.imported'
            UNION
            SELECT jsonb_array_elements_text(metadata->'changed') FROM audit_logs
            WHERE action = 'register.imported'
        )
        """
    )
    op.create_table(
        "product_gtins",
        id_column(),
        created_at(),
        updated_at(),
        sa.Column("gtin", sa.String(14), nullable=False),
        sa.Column(
            "product_id",
            sa.Uuid(),
            sa.ForeignKey("products.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source", enum("gtin_source", "register", "manufacturer"), nullable=False),
        sa.UniqueConstraint("gtin"),
        sa.CheckConstraint("gtin ~ '^[0-9]{14}$'", name="gtin_is_14_digits"),
    )
    op.create_index("ix_product_gtins_product_id", "product_gtins", ["product_id"])


def downgrade() -> None:
    op.drop_index("ix_product_gtins_product_id", "product_gtins")
    op.drop_table("product_gtins")
    op.drop_column("products", "register_source")

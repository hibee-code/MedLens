"""Register (Greenbook) fields on products: normalised active ingredients, registration date,
and where and when each entry was read.

Existing rows keep working: the new columns are empty until the next seed or register import
fills them (both upsert by NAFDAC number).

Revision ID: 0015
Revises: 0014
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0015"
down_revision: str | Sequence[str] | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "products",
        sa.Column(
            "active_ingredients",
            JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column("products", sa.Column("registration_date", sa.Date(), nullable=True))
    op.add_column("products", sa.Column("source_url", sa.Text(), nullable=True))
    op.add_column("products", sa.Column("retrieved_on", sa.Date(), nullable=True))
    op.create_check_constraint(
        "active_ingredients_is_list",
        "products",
        "jsonb_typeof(active_ingredients) = 'array'",
    )
    op.create_check_constraint(
        "registration_dates_in_order",
        "products",
        "registration_date IS NULL OR registration_expires_on IS NULL "
        "OR registration_date <= registration_expires_on",
    )


def downgrade() -> None:
    op.drop_constraint("registration_dates_in_order", "products", type_="check")
    op.drop_constraint("active_ingredients_is_list", "products", type_="check")
    op.drop_column("products", "retrieved_on")
    op.drop_column("products", "source_url")
    op.drop_column("products", "registration_date")
    op.drop_column("products", "active_ingredients")

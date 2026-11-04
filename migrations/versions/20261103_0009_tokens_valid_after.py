"""users.tokens_valid_after: the Postgres fallback for access-token revocation.

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | Sequence[str] | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("tokens_valid_after", sa.DateTime(timezone=True)))


def downgrade() -> None:
    op.drop_column("users", "tokens_valid_after")

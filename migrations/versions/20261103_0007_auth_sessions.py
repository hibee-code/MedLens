"""Auth sessions: permission versions on users; organization and auth methods on sessions.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.helpers import empty_array

revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users", sa.Column("perm_ver", sa.Integer(), server_default=sa.text("1"), nullable=False)
    )
    op.add_column(
        "refresh_tokens",
        sa.Column(
            "organization_id",
            sa.Uuid(),
            sa.ForeignKey("organizations.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column("refresh_tokens", empty_array("amr", sa.String(16)))


def downgrade() -> None:
    op.drop_column("refresh_tokens", "amr")
    op.drop_column("refresh_tokens", "organization_id")
    op.drop_column("users", "perm_ver")

"""MFA recovery codes: one-time codes that stand in for the authenticator app, stored hashed.

Revision ID: 0020
Revises: 0019
Create Date: 2026-11-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.helpers import created_at, id_column

revision: str = "0020"
down_revision: str | Sequence[str] | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "mfa_recovery_codes",
        id_column(),
        created_at(),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code_hash", sa.String(64), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("user_id", "code_hash"),
    )


def downgrade() -> None:
    op.drop_table("mfa_recovery_codes")

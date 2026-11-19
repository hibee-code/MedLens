"""E-mail verification for staff accounts: users.email_verified_at. Existing e-mail accounts were
created by the seed or by administrators, so they count as verified.

Revision ID: 0018
Revises: 0017
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: str | Sequence[str] | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users", sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.execute("UPDATE users SET email_verified_at = created_at WHERE email IS NOT NULL")


def downgrade() -> None:
    op.drop_column("users", "email_verified_at")

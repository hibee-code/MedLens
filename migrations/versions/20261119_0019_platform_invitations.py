"""Platform-admin invitations: an invitation with no organization grants the platform_admin
global role (bootstrap and further platform admins); every other invitation names its
organization.

Revision ID: 0019
Revises: 0018
Create Date: 2026-11-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: str | Sequence[str] | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("invitations", "organization_id", existing_type=sa.Uuid(), nullable=True)
    op.create_check_constraint(
        "platform_scope",
        "invitations",
        "(organization_id IS NULL) = (role = 'platform_admin')",
    )


def downgrade() -> None:
    op.execute("DELETE FROM invitations WHERE organization_id IS NULL")
    op.drop_constraint("platform_scope", "invitations", type_="check")
    op.alter_column("invitations", "organization_id", existing_type=sa.Uuid(), nullable=False)

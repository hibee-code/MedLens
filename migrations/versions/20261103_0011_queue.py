"""Phase 6: outbox backoff and dead letters, and the failed_tasks table.

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from migrations.helpers import created_at, id_column, updated_at

revision: str = "0011"
down_revision: str | Sequence[str] | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("outbox_events", sa.Column("next_attempt_at", sa.DateTime(timezone=True)))
    op.add_column("outbox_events", sa.Column("dead_lettered_at", sa.DateTime(timezone=True)))
    op.drop_index("ix_outbox_events_unpublished", table_name="outbox_events")
    op.create_index(
        "ix_outbox_events_unpublished",
        "outbox_events",
        ["created_at"],
        postgresql_where=sa.text("published_at IS NULL AND dead_lettered_at IS NULL"),
    )

    op.create_table(
        "failed_tasks",
        id_column(),
        created_at(),
        updated_at(),
        sa.Column("task_name", sa.String(200), nullable=False),
        sa.Column("task_id", sa.String(64)),
        sa.Column("args", JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("kwargs", JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("args_hash", sa.String(64), nullable=False),
        sa.Column("error_type", sa.String(200), nullable=False),
        sa.Column("error_message", sa.Text()),
        sa.Column("traceback_hash", sa.String(64)),
        sa.Column("failure_count", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("first_failed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_failed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("requeued_at", sa.DateTime(timezone=True)),
        sa.Column("requeue_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.UniqueConstraint("task_name", "args_hash"),
    )


def downgrade() -> None:
    op.drop_table("failed_tasks")
    op.drop_index("ix_outbox_events_unpublished", table_name="outbox_events")
    op.create_index(
        "ix_outbox_events_unpublished",
        "outbox_events",
        ["created_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )
    op.drop_column("outbox_events", "dead_lettered_at")
    op.drop_column("outbox_events", "next_attempt_at")

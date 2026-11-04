"""Platform tables: append-only audit log, transactional outbox, idempotency keys.

audit_logs rejects UPDATE, DELETE and TRUNCATE with a trigger. That applies to every role
except a superuser who disables triggers, so it also protects the log while the app still
connects as the owner (Phase 9 moves the app to a least-privilege login). Migration 0006
additionally withholds those privileges from the app role.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from migrations.helpers import created_at, empty_json, id_column

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_logs",
        id_column(),
        sa.Column("actor_user_id", sa.Uuid(), nullable=True),  # no FK: see the model docstring
        sa.Column("actor_org_id", sa.Uuid(), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("resource_type", sa.String(64), nullable=True),
        sa.Column("resource_id", sa.String(64), nullable=True),
        sa.Column("ip_hash", sa.String(64), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("request_id", sa.String(64), nullable=True),
        empty_json("metadata"),
        created_at(),
    )
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])
    op.create_index(
        "ix_audit_logs_actor_user_id_created_at", "audit_logs", ["actor_user_id", "created_at"]
    )
    op.create_index(
        "ix_audit_logs_resource_type_resource_id", "audit_logs", ["resource_type", "resource_id"]
    )
    op.execute(
        """
        CREATE FUNCTION audit_logs_append_only() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'audit_logs is append-only: % is not allowed', TG_OP
                USING ERRCODE = 'insufficient_privilege';
        END
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER audit_logs_append_only BEFORE UPDATE OR DELETE ON audit_logs "
        "FOR EACH ROW EXECUTE FUNCTION audit_logs_append_only()"
    )
    op.execute(
        "CREATE TRIGGER audit_logs_no_truncate BEFORE TRUNCATE ON audit_logs "
        "FOR EACH STATEMENT EXECUTE FUNCTION audit_logs_append_only()"
    )

    op.create_table(
        "outbox_events",
        id_column(),
        sa.Column("aggregate_type", sa.String(64), nullable=False),
        sa.Column("aggregate_id", sa.String(64), nullable=False),
        sa.Column("event_type", sa.String(128), nullable=False),
        empty_json("payload"),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        created_at(),
    )
    op.create_index(
        "ix_outbox_events_unpublished",
        "outbox_events",
        ["created_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )

    op.create_table(
        "idempotency_keys",
        sa.Column("scope", sa.String(80), primary_key=True),
        sa.Column("key", sa.String(128), primary_key=True),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("response_status", sa.Integer(), nullable=True),
        sa.Column("response_body", JSONB(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        created_at(),
    )
    op.create_index("ix_idempotency_keys_expires_at", "idempotency_keys", ["expires_at"])


def downgrade() -> None:
    op.drop_table("idempotency_keys")
    op.drop_table("outbox_events")
    op.drop_table("audit_logs")  # drops its triggers
    op.execute("DROP FUNCTION audit_logs_append_only()")

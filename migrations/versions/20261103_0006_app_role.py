"""Application role: read/write on every table, but never UPDATE/DELETE/TRUNCATE audit_logs.

medlens_app is a NOLOGIN group role. Phase 9 gives the API and workers a login role that is a
member of it, separate from the migration role that owns the tables; until then the app still
connects as the owner and the audit_logs trigger (migration 0005) is what protects the log.

Creating a role needs CREATEROLE (a superuser locally; the master user on managed Postgres).
Roles are cluster-wide, so downgrade revokes this database's grants but keeps the role.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-03
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "medlens_app"


def upgrade() -> None:
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                CREATE ROLE {APP_ROLE} NOLOGIN;
            END IF;
        END
        $$
        """
    )
    op.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {APP_ROLE}")
    op.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {APP_ROLE}")
    # Tables (and partitions) created later by the role running migrations get the same grants.
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {APP_ROLE}"
    )
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO {APP_ROLE}"
    )
    op.execute(f"REVOKE UPDATE, DELETE, TRUNCATE ON audit_logs FROM {APP_ROLE}, PUBLIC")


def downgrade() -> None:
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f"REVOKE USAGE, SELECT ON SEQUENCES FROM {APP_ROLE}"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f"REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM {APP_ROLE}"
    )
    op.execute(f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM {APP_ROLE}")
    op.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {APP_ROLE}")
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {APP_ROLE}")

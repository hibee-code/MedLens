"""MVP M3/M4: LLM call metering, scan AI and review columns, image vectors sized for the
configured image model, and the reports:read_own permission.

Revision ID: 0012
Revises: 0011
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from medlens.db.migration_support import migration_settings
from migrations.helpers import created_at, enum, id_column

revision: str = "0012"
down_revision: str | Sequence[str] | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

VERDICTS = ("LIKELY_GENUINE", "SUSPICIOUS", "UNREGISTERED", "INCONCLUSIVE")
REPORTS = (
    "reports:read_own",
    "Read aggregated reports about one's own products (no individual scans).",
)
REPORT_READERS = ("manufacturer_staff", "manufacturer_admin", "platform_admin")


def upgrade() -> None:
    op.create_table(
        "llm_calls",
        id_column(),
        created_at(),
        sa.Column("route", sa.String(64), nullable=False),
        sa.Column("tier", sa.String(16), nullable=False),
        sa.Column("requested_model", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_write_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column(
            "outcome",
            enum("llm_call_outcome", "ok", "invalid_output", "refused", "error"),
            nullable=False,
        ),
        sa.Column("error_type", sa.String(100)),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("subject_type", sa.String(32)),
        sa.Column("subject_id", sa.String(64)),
    )
    op.create_index("ix_llm_calls_created_at", "llm_calls", ["created_at"])
    op.create_index("ix_llm_calls_route_created_at", "llm_calls", ["route", "created_at"])

    # Partitioned: columns added to the parent reach every partition.
    op.add_column("scans", sa.Column("verdict_override", enum("verdict_override", *VERDICTS)))
    op.add_column("scans", sa.Column("override_case_id", sa.Uuid()))
    op.add_column("scans", sa.Column("overridden_at", sa.DateTime(timezone=True)))
    op.add_column(
        "scans",
        sa.Column("llm_used", sa.Boolean(), server_default=sa.false(), nullable=False),
    )

    # Vectors of another model cannot be compared: resize and clear them; the embedding task
    # recomputes them (it re-embeds whenever embedding_model_version differs).
    dimension = migration_settings().embeddings.image_dimension
    current = op.get_bind().scalar(
        sa.text(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'reference_images'::regclass AND attname = 'embedding'"
        )
    )
    if current != dimension:
        op.drop_index("ix_reference_images_embedding", table_name="reference_images")
        op.execute(
            f"ALTER TABLE reference_images ALTER COLUMN embedding TYPE vector({int(dimension)}) "
            "USING NULL"
        )
        op.execute("UPDATE reference_images SET embedding_model_version = NULL")
        op.create_index(
            "ix_reference_images_embedding",
            "reference_images",
            ["embedding"],
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )

    name, description = REPORTS
    op.bulk_insert(
        sa.table("permissions", sa.column("name"), sa.column("description")),
        [{"name": name, "description": description}],
    )
    op.bulk_insert(
        sa.table("role_permissions", sa.column("role_name"), sa.column("permission_name")),
        [{"role_name": role, "permission_name": name} for role in REPORT_READERS],
    )


def downgrade() -> None:
    name, _ = REPORTS
    op.execute(
        sa.text("DELETE FROM role_permissions WHERE permission_name = :name").bindparams(name=name)
    )
    op.execute(sa.text("DELETE FROM permissions WHERE name = :name").bindparams(name=name))
    # The image vector size is left as it is: the data in it belongs to the current model.
    for column in ("llm_used", "overridden_at", "override_case_id", "verdict_override"):
        op.drop_column("scans", column)
    op.drop_table("llm_calls")

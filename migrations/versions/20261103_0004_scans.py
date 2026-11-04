"""Scans (RANGE-partitioned by month), cases and hotspot snapshots.

scans' primary key includes the partition key: (id, created_at). Partitions are created for the
current UTC month and the next three, plus a default partition; from Phase 6 a daily Beat task
keeps creating them ahead of time (db/partitions.py).

cases references scans by its full key (scan_id, scan_created_at).

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-03
"""

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from medlens.db.partitions import default_partition_ddl, ensure_monthly_partitions
from migrations.helpers import (
    created_at,
    empty_array,
    empty_json,
    enum,
    foreign_key,
    id_column,
    point,
    state_enum,
    states_check,
    updated_at,
)

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "scans",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        foreign_key("user_id", "users.id", nullable=True, ondelete="SET NULL"),
        foreign_key("organization_id", "organizations.id", nullable=True, ondelete="SET NULL"),
        sa.Column("device_fingerprint_hash", sa.String(64), nullable=True),
        sa.Column("image_s3_key", sa.Text(), nullable=True),
        sa.Column("image_sha256", sa.String(64), nullable=True),
        sa.Column("image_phash", sa.BigInteger(), nullable=True),
        sa.Column("quality", JSONB(), nullable=True),
        sa.Column(
            "status",
            enum("scan_status", "received", "processing", "completed", "failed", "rejected"),
            server_default="received",
            nullable=False,
        ),
        sa.Column("extracted", JSONB(), nullable=True),
        foreign_key("matched_product_id", "products.id", nullable=True, ondelete="SET NULL"),
        foreign_key("matched_batch_id", "batches.id", nullable=True, ondelete="SET NULL"),
        sa.Column("signals", JSONB(), nullable=True),
        sa.Column(
            "verdict",
            enum("verdict", "LIKELY_GENUINE", "SUSPICIOUS", "UNREGISTERED", "INCONCLUSIVE"),
            nullable=True,
        ),
        sa.Column("risk_score", sa.Double(), nullable=True),
        sa.Column("confidence", sa.Double(), nullable=True),
        sa.Column("scorer_version", sa.String(32), nullable=True),
        point("location_coarse"),
        sa.Column("geocell", sa.String(6), nullable=True),
        sa.Column("state", state_enum(), nullable=True),
        sa.Column("idempotency_key", sa.String(128), nullable=True),
        sa.Column("processing_ms", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        updated_at(),
        sa.PrimaryKeyConstraint("id", "created_at"),
        sa.CheckConstraint("risk_score IS NULL OR risk_score BETWEEN 0 AND 1", name="risk_score"),
        sa.CheckConstraint("confidence IS NULL OR confidence BETWEEN 0 AND 1", name="confidence"),
        sa.CheckConstraint("processing_ms IS NULL OR processing_ms >= 0", name="processing_ms"),
        postgresql_partition_by="RANGE (created_at)",
    )
    for name, columns in {
        "ix_scans_verdict_created_at": ["verdict", "created_at"],
        "ix_scans_matched_product_id_matched_batch_id": ["matched_product_id", "matched_batch_id"],
        "ix_scans_image_phash": ["image_phash"],
        "ix_scans_image_sha256": ["image_sha256"],
        "ix_scans_user_id_created_at": ["user_id", "created_at"],
        "ix_scans_organization_id_created_at": ["organization_id", "created_at"],
    }.items():
        op.create_index(name, "scans", columns)
    op.create_index(
        "ix_scans_location_coarse", "scans", ["location_coarse"], postgresql_using="gist"
    )
    ensure_monthly_partitions(op.execute, "scans", today=datetime.now(UTC).date())
    op.execute(default_partition_ddl("scans"))

    op.create_table(
        "cases",
        id_column(),
        sa.Column("scan_id", sa.Uuid(), nullable=False),
        sa.Column("scan_created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "status",
            enum(
                "case_status",
                "open",
                "in_review",
                "confirmed_counterfeit",
                "confirmed_genuine",
                "inconclusive",
                "closed",
            ),
            server_default="open",
            nullable=False,
        ),
        sa.Column(
            "priority",
            enum("case_priority", "low", "normal", "high"),
            server_default="normal",
            nullable=False,
        ),
        foreign_key("assigned_to_user_id", "users.id", nullable=True, ondelete="SET NULL"),
        foreign_key("assigned_org_id", "organizations.id", nullable=True, ondelete="SET NULL"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("lab_result", JSONB(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        created_at(),
        updated_at(),
        sa.ForeignKeyConstraint(
            ["scan_id", "scan_created_at"], ["scans.id", "scans.created_at"], ondelete="RESTRICT"
        ),
        sa.UniqueConstraint("scan_id", "scan_created_at"),
    )
    op.create_index("ix_cases_assigned_org_id_status", "cases", ["assigned_org_id", "status"])
    op.create_index(
        "ix_cases_status_priority_created_at", "cases", ["status", "priority", "created_at"]
    )

    op.create_table(
        "hotspot_snapshots",
        id_column(),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=False),
        point("centroid", nullable=False),
        sa.Column("radius_m", sa.Double(), nullable=False),
        sa.Column("scan_count", sa.Integer(), nullable=False),
        empty_array("product_ids", sa.Uuid()),
        empty_array("states", sa.String(5)),
        sa.Column("severity", enum("severity", "low", "medium", "high"), nullable=False),
        empty_json("algorithm_params"),
        created_at(),
        updated_at(),
        sa.CheckConstraint("window_end > window_start", name="window"),
        sa.CheckConstraint("radius_m >= 0 AND scan_count >= 0", name="non_negative"),
        states_check("states", "states"),
    )
    op.create_index("ix_hotspot_snapshots_window_end", "hotspot_snapshots", ["window_end"])
    op.create_index(
        "ix_hotspot_snapshots_centroid", "hotspot_snapshots", ["centroid"], postgresql_using="gist"
    )


def downgrade() -> None:
    op.drop_table("hotspot_snapshots")
    op.drop_table("cases")
    op.drop_table("scans")  # drops every partition with it

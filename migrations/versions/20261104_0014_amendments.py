"""Decision-log amendments (2026-11-04): official and active knowledge documents for the
published-alert rules; pipeline and final verdicts on scans, with audited, reversible case
overrides; and the image vector column re-checked against the configured model.

Revision ID: 0014
Revises: 0013
Create Date: 2026-11-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from medlens.db.migration_support import migration_settings
from migrations.helpers import enum, resize_vectors

revision: str = "0014"
down_revision: str | Sequence[str] | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

VERDICTS = ("LIKELY_GENUINE", "SUSPICIOUS", "UNREGISTERED", "INCONCLUSIVE")
_OVERRIDE_COMPLETE = (
    "overridden_at IS NULL OR (override_case_id IS NOT NULL "
    "AND override_note IS NOT NULL AND override_lab_result IS NOT NULL)"
)
_FINAL_IS_PIPELINE = (
    "overridden_at IS NOT NULL OR final_verdict IS NOT DISTINCT FROM pipeline_verdict"
)


def upgrade() -> None:
    # Only official, active alerts and recalls trigger scan rules.
    op.add_column(
        "knowledge_documents",
        sa.Column("official", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.add_column(
        "knowledge_documents",
        sa.Column("active", sa.Boolean(), server_default=sa.true(), nullable=False),
    )

    # Scans: the scorer's verdict becomes pipeline_verdict (immutable once set); final_verdict
    # is what clients see. Partitioned table: changes on the parent reach every partition.
    op.alter_column("scans", "verdict", new_column_name="pipeline_verdict")
    op.execute("ALTER TABLE scans RENAME CONSTRAINT ck_scans_verdict TO ck_scans_pipeline_verdict")
    op.add_column("scans", sa.Column("final_verdict", enum("final_verdict", *VERDICTS)))
    op.add_column(
        "scans",
        sa.Column("overridden_by", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL")),
    )
    op.add_column("scans", sa.Column("override_lab_result", JSONB()))
    op.add_column("scans", sa.Column("override_note", sa.Text()))
    # Overrides made before this revision have no note or lab result to show: keep their
    # verdict and record that they predate the rule.
    op.execute(
        """
        UPDATE scans SET final_verdict = COALESCE(verdict_override, pipeline_verdict),
               override_note = CASE WHEN verdict_override IS NOT NULL
                                    THEN 'Recorded before override notes were required.' END,
               override_lab_result = CASE WHEN verdict_override IS NOT NULL
                                          THEN '{}'::jsonb END
        """
    )
    op.drop_column("scans", "verdict_override")
    op.create_check_constraint("override_complete", "scans", _OVERRIDE_COMPLETE)
    op.create_check_constraint("final_is_pipeline_unless_overridden", "scans", _FINAL_IS_PIPELINE)
    op.drop_index("ix_scans_verdict_created_at", table_name="scans")
    op.create_index("ix_scans_final_verdict_created_at", "scans", ["final_verdict", "created_at"])
    op.execute(
        """
        CREATE FUNCTION scans_pipeline_verdict_is_final() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF OLD.pipeline_verdict IS NOT NULL
               AND NEW.pipeline_verdict IS DISTINCT FROM OLD.pipeline_verdict THEN
                RAISE EXCEPTION 'scans.pipeline_verdict cannot change once set'
                    USING ERRCODE = 'integrity_constraint_violation';
            END IF;
            RETURN NEW;
        END
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER scans_pipeline_verdict_is_final BEFORE UPDATE OF pipeline_verdict "
        "ON scans FOR EACH ROW EXECUTE FUNCTION scans_pipeline_verdict_is_final()"
    )

    # Databases migrated with another configured image dimension get the right column size.
    resize_vectors(
        "reference_images",
        "embedding",
        migration_settings().embeddings.image_dimension,
        index="ix_reference_images_embedding",
        model_column="embedding_model_version",
    )


def downgrade() -> None:
    # The vector size is left as it is: the data in it belongs to the current model.
    op.execute("DROP TRIGGER scans_pipeline_verdict_is_final ON scans")
    op.execute("DROP FUNCTION scans_pipeline_verdict_is_final()")
    op.drop_index("ix_scans_final_verdict_created_at", table_name="scans")
    op.create_index("ix_scans_verdict_created_at", "scans", ["pipeline_verdict", "created_at"])
    op.drop_constraint("final_is_pipeline_unless_overridden", "scans", type_="check")
    op.drop_constraint("override_complete", "scans", type_="check")
    op.add_column("scans", sa.Column("verdict_override", enum("verdict_override", *VERDICTS)))
    op.execute("UPDATE scans SET verdict_override = final_verdict WHERE overridden_at IS NOT NULL")
    for column in ("override_note", "override_lab_result", "overridden_by", "final_verdict"):
        op.drop_column("scans", column)
    op.execute("ALTER TABLE scans RENAME CONSTRAINT ck_scans_pipeline_verdict TO ck_scans_verdict")
    op.alter_column("scans", "pipeline_verdict", new_column_name="verdict")
    op.drop_column("knowledge_documents", "active")
    op.drop_column("knowledge_documents", "official")

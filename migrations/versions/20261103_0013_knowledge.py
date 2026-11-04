"""MVP M5: the knowledge base (documents, chunks with text vectors and full-text search,
published alerts) and the knowledge:read / knowledge:manage permissions.

Revision ID: 0013
Revises: 0012
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR

from medlens.db.migration_support import migration_settings
from migrations.helpers import created_at, enum, id_column, updated_at

revision: str = "0013"
down_revision: str | Sequence[str] | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

KINDS = ("alert", "recall", "leaflet", "sop", "other")
PERMISSIONS = {
    "knowledge:read": (
        "Search the knowledge base and ask the assistant.",
        (
            "pharmacy_staff",
            "pharmacy_admin",
            "manufacturer_staff",
            "manufacturer_admin",
            "regulator_analyst",
            "regulator_admin",
            "platform_admin",
        ),
    ),
    "knowledge:manage": (
        "Add documents to the knowledge base.",
        ("regulator_analyst", "regulator_admin", "platform_admin"),
    ),
}


def upgrade() -> None:
    op.create_table(
        "knowledge_documents",
        id_column(),
        created_at(),
        updated_at(),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("kind", enum("kind", *KINDS), nullable=False),
        sa.Column(
            "visibility",
            enum("visibility", "professional", "regulator"),
            server_default="professional",
            nullable=False,
        ),
        sa.Column("source", sa.String(200)),
        sa.Column("source_url", sa.Text()),
        sa.Column("published_on", sa.Date()),
        sa.Column("media_type", sa.String(100), nullable=False),
        sa.Column("s3_key", sa.Text()),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_sha256", sa.String(64)),
        sa.Column(
            "status",
            enum("status", "pending", "indexed", "duplicate", "failed"),
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "duplicate_of_id",
            sa.Uuid(),
            sa.ForeignKey("knowledge_documents.id", ondelete="SET NULL"),
        ),
        sa.Column("page_count", sa.Integer()),
        sa.Column("chunk_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("error_code", sa.String(64)),
        sa.Column("uploaded_by", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("indexed_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_knowledge_documents_content_sha256", "knowledge_documents", ["content_sha256"]
    )
    op.create_index(
        "ix_knowledge_documents_status_created_at",
        "knowledge_documents",
        ["status", "created_at"],
    )

    dimension = int(migration_settings().embeddings.text_dimension)
    op.create_table(
        "knowledge_chunks",
        id_column(),
        created_at(),
        updated_at(),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("knowledge_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer()),
        sa.Column("section", sa.String(300)),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column(
            "nafdac_numbers",
            sa.ARRAY(sa.String(20)),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "batch_numbers",
            sa.ARRAY(sa.String(40)),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("embedding", Vector(dimension)),
        sa.Column("embedding_model_version", sa.String(100)),
        sa.Column(
            "tsv",
            TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=False,
        ),
        sa.UniqueConstraint("document_id", "ordinal"),
    )
    # Written by hand: autogenerate does not render HNSW or GIN options (CLAUDE.md §7).
    op.create_index(
        "ix_knowledge_chunks_embedding",
        "knowledge_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_knowledge_chunks_tsv", "knowledge_chunks", ["tsv"], postgresql_using="gin")
    for column in ("nafdac_numbers", "batch_numbers"):
        op.create_index(
            f"ix_knowledge_chunks_{column}", "knowledge_chunks", [column], postgresql_using="gin"
        )

    op.create_table(
        "published_alerts",
        id_column(),
        created_at(),
        updated_at(),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("knowledge_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_id", sa.Uuid(), sa.ForeignKey("knowledge_chunks.id", ondelete="SET NULL")),
        sa.Column("kind", enum("kind", *KINDS), nullable=False),
        sa.Column("nafdac_number", sa.String(20), nullable=False),
        sa.Column("batch_number", sa.String(40), server_default="", nullable=False),
        sa.Column("published_on", sa.Date()),
        sa.UniqueConstraint("document_id", "nafdac_number", "batch_number"),
    )
    op.create_index("ix_published_alerts_nafdac_number", "published_alerts", ["nafdac_number"])

    op.bulk_insert(
        sa.table("permissions", sa.column("name"), sa.column("description")),
        [{"name": name, "description": text} for name, (text, _) in PERMISSIONS.items()],
    )
    op.bulk_insert(
        sa.table("role_permissions", sa.column("role_name"), sa.column("permission_name")),
        [
            {"role_name": role, "permission_name": name}
            for name, (_, roles) in PERMISSIONS.items()
            for role in roles
        ],
    )


def downgrade() -> None:
    for name in PERMISSIONS:
        op.execute(
            sa.text("DELETE FROM role_permissions WHERE permission_name = :name").bindparams(
                name=name
            )
        )
        op.execute(sa.text("DELETE FROM permissions WHERE name = :name").bindparams(name=name))
    op.drop_table("published_alerts")
    op.drop_table("knowledge_chunks")
    op.drop_table("knowledge_documents")

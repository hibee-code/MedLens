"""Knowledge base: regulatory documents (NAFDAC alerts, recall notices, leaflets, SOPs), their
chunks for retrieval, and the counterfeit alerts and recalls extracted from them."""

from datetime import date, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
    true,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from medlens.db.base import Entity
from medlens.db.types import str_enum


class DocumentKind(StrEnum):
    ALERT = "alert"  # public alert about counterfeit or substandard products
    RECALL = "recall"
    LEAFLET = "leaflet"
    SOP = "sop"
    OTHER = "other"


class DocumentStatus(StrEnum):
    PENDING = "pending"
    INDEXED = "indexed"
    DUPLICATE = "duplicate"  # same content as an earlier document: not indexed again
    FAILED = "failed"


class Visibility(StrEnum):
    PROFESSIONAL = "professional"  # every knowledge:read holder
    REGULATOR = "regulator"  # regulators and platform admins only


# Alerts and recalls feed the published-alert scan signal.
ALERT_KINDS = frozenset({DocumentKind.ALERT, DocumentKind.RECALL})


class KnowledgeDocument(Entity):
    __tablename__ = "knowledge_documents"
    __table_args__ = (
        Index("ix_knowledge_documents_content_sha256", "content_sha256"),
        Index("ix_knowledge_documents_status_created_at", "status", "created_at"),
    )

    title: Mapped[str] = mapped_column(String(300))
    kind: Mapped[DocumentKind] = mapped_column(str_enum(DocumentKind, "kind"))
    visibility: Mapped[Visibility] = mapped_column(
        str_enum(Visibility, "visibility"),
        default=Visibility.PROFESSIONAL,
        server_default="professional",
    )
    source: Mapped[str | None] = mapped_column(String(200))  # e.g. "NAFDAC public alerts"
    # Alerts and recalls trigger scan rules only when official (a regulator vouches for the
    # source) and active (not withdrawn or superseded); both are audited when changed.
    official: Mapped[bool] = mapped_column(default=False, server_default=false())
    active: Mapped[bool] = mapped_column(default=True, server_default=true())
    source_url: Mapped[str | None] = mapped_column(Text)
    published_on: Mapped[date | None]
    media_type: Mapped[str] = mapped_column(String(100))
    s3_key: Mapped[str | None] = mapped_column(Text)
    size_bytes: Mapped[int]
    # Of the extracted, normalised text: the same notice uploaded twice is indexed once.
    content_sha256: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[DocumentStatus] = mapped_column(
        str_enum(DocumentStatus, "status"), default=DocumentStatus.PENDING, server_default="pending"
    )
    duplicate_of_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="SET NULL")
    )
    page_count: Mapped[int | None]
    chunk_count: Mapped[int] = mapped_column(default=0, server_default=sql_text("0"))
    error_code: Mapped[str | None] = mapped_column(String(64))
    uploaded_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    indexed_at: Mapped[datetime | None]


class KnowledgeChunk(Entity):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal"),
        Index(
            "ix_knowledge_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_knowledge_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_knowledge_chunks_nafdac_numbers", "nafdac_numbers", postgresql_using="gin"),
        Index("ix_knowledge_chunks_batch_numbers", "batch_numbers", postgresql_using="gin"),
    )

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE")
    )
    ordinal: Mapped[int]
    page: Mapped[int | None]  # 1-based, PDFs only
    section: Mapped[str | None] = mapped_column(String(300))  # nearest heading
    text: Mapped[str] = mapped_column(Text)  # PII-scrubbed
    token_count: Mapped[int] = mapped_column(Integer)
    nafdac_numbers: Mapped[list[str]] = mapped_column(
        ARRAY(String(20)), default=list, server_default=sql_text("'{}'")
    )
    batch_numbers: Mapped[list[str]] = mapped_column(
        ARRAY(String(40)), default=list, server_default=sql_text("'{}'")
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector())
    embedding_model_version: Mapped[str | None] = mapped_column(String(100))
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )


class PublishedAlert(Entity):
    """A NAFDAC number (and batch, when the notice names one) in an alert or recall notice."""

    __tablename__ = "published_alerts"
    __table_args__ = (
        UniqueConstraint("document_id", "nafdac_number", "batch_number"),
        Index("ix_published_alerts_nafdac_number", "nafdac_number"),
    )

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE")
    )
    chunk_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL")
    )
    kind: Mapped[DocumentKind] = mapped_column(str_enum(DocumentKind, "kind"))
    nafdac_number: Mapped[str] = mapped_column(String(20))
    # Empty string: the notice concerns every batch (NULLs would defeat the unique key).
    batch_number: Mapped[str] = mapped_column(String(40), default="", server_default="")
    published_on: Mapped[date | None]

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from geoalchemy2 import Geography, WKBElement
from sqlalchemy import (
    ARRAY,
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    Text,
    false,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from medlens.core.identifiers import NigerianState
from medlens.db.base import Entity
from medlens.db.types import str_enum


class ScanStatus(StrEnum):
    RECEIVED = "received"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"  # failed the quality gate; verdict INCONCLUSIVE with reason codes


class Verdict(StrEnum):
    """Never "safe" or "100% genuine" (CLAUDE.md §4)."""

    LIKELY_GENUINE = "LIKELY_GENUINE"
    SUSPICIOUS = "SUSPICIOUS"
    UNREGISTERED = "UNREGISTERED"
    INCONCLUSIVE = "INCONCLUSIVE"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Scan(Entity):
    """One verification request. RANGE-partitioned by month on created_at, which is therefore
    part of the primary key (CLAUDE.md §7). Partitions: db/partitions.py."""

    __tablename__ = "scans"
    __table_args__ = (
        CheckConstraint("risk_score IS NULL OR risk_score BETWEEN 0 AND 1", name="risk_score"),
        CheckConstraint("confidence IS NULL OR confidence BETWEEN 0 AND 1", name="confidence"),
        CheckConstraint("processing_ms IS NULL OR processing_ms >= 0", name="processing_ms"),
        Index("ix_scans_verdict_created_at", "verdict", "created_at"),
        Index("ix_scans_location_coarse", "location_coarse", postgresql_using="gist"),
        Index(
            "ix_scans_matched_product_id_matched_batch_id", "matched_product_id", "matched_batch_id"
        ),
        Index("ix_scans_image_phash", "image_phash"),
        Index("ix_scans_image_sha256", "image_sha256"),
        Index("ix_scans_user_id_created_at", "user_id", "created_at"),
        Index(
            "ix_scans_device_fingerprint_hash_created_at", "device_fingerprint_hash", "created_at"
        ),
        Index("ix_scans_organization_id_created_at", "organization_id", "created_at"),
        {"postgresql_partition_by": "RANGE (created_at)"},
    )

    # Set in Python (not only by the server) so the full primary key is known before INSERT.
    created_at: Mapped[datetime] = mapped_column(
        primary_key=True, default=_utcnow, server_default=func.now()
    )

    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    organization_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL")
    )
    device_fingerprint_hash: Mapped[str | None] = mapped_column(String(64))
    image_s3_key: Mapped[str | None] = mapped_column(Text)
    # Exact-duplicate key, computed in the API from the re-encoded JPEG.
    image_sha256: Mapped[str | None] = mapped_column(String(64))
    # Perceptual hash, computed later in the worker (near-duplicates, probing).
    image_phash: Mapped[int | None] = mapped_column(BigInteger)
    quality: Mapped[dict[str, Any] | None]  # blur, brightness, width, height
    status: Mapped[ScanStatus] = mapped_column(
        str_enum(ScanStatus, "scan_status"), default=ScanStatus.RECEIVED, server_default="received"
    )
    # nafdac_number, product_name, batch_number, expiry_date, each with source and confidence.
    extracted: Mapped[dict[str, Any] | None]
    matched_product_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("products.id", ondelete="SET NULL")
    )
    matched_batch_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("batches.id", ondelete="SET NULL")
    )
    signals: Mapped[dict[str, Any] | None]
    # The scorer's verdict: never changed afterwards.
    verdict: Mapped[Verdict | None] = mapped_column(str_enum(Verdict, "verdict"))
    # A confirmed case decision supersedes it (audited); the scorer's verdict stays above.
    verdict_override: Mapped[Verdict | None] = mapped_column(str_enum(Verdict, "verdict_override"))
    override_case_id: Mapped[UUID | None]
    overridden_at: Mapped[datetime | None]
    llm_used: Mapped[bool] = mapped_column(default=False, server_default=false())
    # Stable codes (verification/codes.py); clients translate them.
    reason_codes: Mapped[list[str]] = mapped_column(
        ARRAY(String(32)), default=list, server_default=text("'{}'")
    )
    advice_codes: Mapped[list[str]] = mapped_column(
        ARRAY(String(32)), default=list, server_default=text("'{}'")
    )
    risk_score: Mapped[float | None]
    confidence: Mapped[float | None]
    scorer_version: Mapped[str | None] = mapped_column(String(32))
    # Rounded to about 1 km before storage; raw coordinates are never stored.
    location_coarse: Mapped[WKBElement | None] = mapped_column(
        Geography(geometry_type="POINT", srid=4326, spatial_index=False)
    )
    geocell: Mapped[str | None] = mapped_column(String(6))  # geohash, precision 6
    state: Mapped[NigerianState | None] = mapped_column(str_enum(NigerianState, "state"))
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    processing_ms: Mapped[int | None]
    error_code: Mapped[str | None] = mapped_column(String(64))

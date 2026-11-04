from datetime import date, datetime
from enum import StrEnum
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from medlens.db.base import Entity, SoftDelete
from medlens.db.types import states_array_check, str_enum


class RegistrationStatus(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class ReviewStatus(StrEnum):
    """Human review of something a manufacturer submitted (CLAUDE.md §5)."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ImageSide(StrEnum):
    FRONT = "front"
    BACK = "back"
    BLISTER = "blister"
    LEAFLET = "leaflet"


class Product(Entity):
    """A registered product, standing in for its NAFDAC Greenbook entry."""

    __tablename__ = "products"
    __table_args__ = (
        # Trigram indexes for fuzzy lookup that absorbs OCR errors ("Panad0l Extr").
        Index(
            "ix_products_brand_name_trgm",
            "brand_name",
            postgresql_using="gin",
            postgresql_ops={"brand_name": "gin_trgm_ops"},
        ),
        Index(
            "ix_products_generic_name_trgm",
            "generic_name",
            postgresql_using="gin",
            postgresql_ops={"generic_name": "gin_trgm_ops"},
        ),
        Index(
            "ix_products_nafdac_number_trgm",
            "nafdac_number",
            postgresql_using="gin",
            postgresql_ops={"nafdac_number": "gin_trgm_ops"},
        ),
        Index("ix_products_manufacturer_org_id", "manufacturer_org_id"),
    )

    nafdac_number: Mapped[str] = mapped_column(String(32), unique=True)
    brand_name: Mapped[str] = mapped_column(Text)
    generic_name: Mapped[str] = mapped_column(Text)
    strength: Mapped[str | None] = mapped_column(Text)
    dosage_form: Mapped[str | None] = mapped_column(Text)
    pack_size: Mapped[str | None] = mapped_column(Text)
    # Registration holder as named in the register; kept even when no organization is linked.
    manufacturer_name: Mapped[str | None] = mapped_column(Text)
    # Set once the manufacturer onboards and its claim to the product is approved.
    manufacturer_org_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL")
    )
    registration_status: Mapped[RegistrationStatus] = mapped_column(
        str_enum(RegistrationStatus, "registration_status")
    )
    registration_expires_on: Mapped[date | None]


class Batch(Entity):
    __tablename__ = "batches"
    __table_args__ = (
        UniqueConstraint("product_id", "batch_number"),
        CheckConstraint(
            "manufactured_on IS NULL OR expires_on > manufactured_on",
            name="expiry_after_manufacture",
        ),
        CheckConstraint("quantity IS NULL OR quantity >= 0", name="quantity_non_negative"),
        states_array_check("distribution_states", "distribution_states"),
    )

    product_id: Mapped[UUID] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"))
    batch_number: Mapped[str] = mapped_column(String(64))
    manufactured_on: Mapped[date | None]
    expires_on: Mapped[date]
    quantity: Mapped[int | None]
    distribution_states: Mapped[list[str]] = mapped_column(
        ARRAY(String(5)), default=list, server_default=text("'{}'")
    )
    declared_by_org_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL")
    )


class ReferenceImage(Entity, SoftDelete):
    """A verified photo of a genuine pack, used for visual matching."""

    __tablename__ = "reference_images"
    __table_args__ = (
        Index("ix_reference_images_product_id", "product_id"),
        Index(
            "ix_reference_images_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    product_id: Mapped[UUID] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"))
    # Only approved images are used for visual matching.
    status: Mapped[ReviewStatus] = mapped_column(
        str_enum(ReviewStatus, "review_status"),
        default=ReviewStatus.PENDING,
        server_default="pending",
    )
    uploaded_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    organization_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL")
    )
    reviewed_at: Mapped[datetime | None]
    side: Mapped[ImageSide] = mapped_column(str_enum(ImageSide, "image_side"))
    s3_key: Mapped[str] = mapped_column(Text, unique=True)
    phash: Mapped[int | None] = mapped_column(BigInteger)  # 64-bit perceptual hash
    # The column is vector(N) with N = MEDLENS_EMBEDDINGS__IMAGE_DIMENSION at migration time.
    embedding: Mapped[list[float] | None] = mapped_column(Vector())
    embedding_model_version: Mapped[str | None] = mapped_column(String(64))
    verified_by: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )  # the reviewer, whether they approved or rejected


class ProductClaim(Entity):
    """A manufacturer's request to be linked to a registered product, reviewed by a regulator
    or platform admin. Approval sets ``products.manufacturer_org_id``."""

    __tablename__ = "product_claims"
    __table_args__ = (
        Index(
            "uq_product_claims_open",
            "organization_id",
            "product_id",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
        Index("ix_product_claims_status_created_at", "status", "created_at"),
    )

    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    product_id: Mapped[UUID] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"))
    status: Mapped[ReviewStatus] = mapped_column(
        str_enum(ReviewStatus, "review_status"),
        default=ReviewStatus.PENDING,
        server_default="pending",
    )
    note: Mapped[str | None] = mapped_column(Text)  # the claimant's evidence
    requested_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at: Mapped[datetime | None]
    review_note: Mapped[str | None] = mapped_column(Text)

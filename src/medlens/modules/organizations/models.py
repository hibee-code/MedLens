from datetime import datetime
from enum import StrEnum
from uuid import UUID

from geoalchemy2 import Geography, WKBElement, WKTElement
from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.orm import Mapped, mapped_column

from medlens.core.crypto import phone_lookup_hash
from medlens.core.identifiers import NigerianState
from medlens.db.base import Entity, SoftDelete
from medlens.db.types import STATE_CODES_SQL, EncryptedText, str_enum


class OrganizationType(StrEnum):
    PHARMACY = "pharmacy"
    PPMV = "ppmv"  # patent and proprietary medicine vendor
    CLINIC = "clinic"
    MANUFACTURER = "manufacturer"
    DISTRIBUTOR = "distributor"
    REGULATOR = "regulator"


class OrganizationStatus(StrEnum):
    PENDING_VERIFICATION = "pending_verification"
    ACTIVE = "active"
    SUSPENDED = "suspended"


class MembershipStatus(StrEnum):
    INVITED = "invited"
    ACTIVE = "active"
    SUSPENDED = "suspended"


class Organization(Entity, SoftDelete):
    __tablename__ = "organizations"
    __table_args__ = (
        # Regulators, and only regulators, have a jurisdiction: a non-empty list of state codes.
        # National jurisdiction is spelled out as all 37 codes, never as an empty list.
        CheckConstraint(
            "(type = 'regulator') = (jurisdiction_states IS NOT NULL)", name="jurisdiction_owner"
        ),
        CheckConstraint(
            "jurisdiction_states IS NULL OR (cardinality(jurisdiction_states) > 0 "
            f"AND jurisdiction_states <@ {STATE_CODES_SQL}::varchar[])",
            name="jurisdiction_states",
        ),
        Index(
            "uq_organizations_type_registration_number",
            "type",
            "registration_number",
            unique=True,
            postgresql_where=text("registration_number IS NOT NULL"),
        ),
    )

    name: Mapped[str] = mapped_column(Text)
    type: Mapped[OrganizationType] = mapped_column(str_enum(OrganizationType, "organization_type"))
    # PCN premises licence, NAFDAC or CAC number depending on the type.
    registration_number: Mapped[str | None] = mapped_column(String(64))
    state: Mapped[NigerianState | None] = mapped_column(str_enum(NigerianState, "state"))
    jurisdiction_states: Mapped[list[str] | None] = mapped_column(ARRAY(String(5)))
    verified_at: Mapped[datetime | None]
    status: Mapped[OrganizationStatus] = mapped_column(
        str_enum(OrganizationStatus, "organization_status"),
        default=OrganizationStatus.PENDING_VERIFICATION,
        server_default="pending_verification",
    )


class Membership(Entity):
    __tablename__ = "memberships"
    __table_args__ = (
        UniqueConstraint("user_id", "organization_id"),
        Index("ix_memberships_organization_id", "organization_id"),
    )

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(ForeignKey("roles.name", onupdate="CASCADE"))
    status: Mapped[MembershipStatus] = mapped_column(
        str_enum(MembershipStatus, "membership_status"),
        default=MembershipStatus.ACTIVE,
        server_default="active",
    )
    invited_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class Branch(Entity, SoftDelete):
    __tablename__ = "branches"
    __table_args__ = (
        Index("ix_branches_organization_id", "organization_id"),
        Index("ix_branches_location", "location", postgresql_using="gist"),
    )

    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(Text)
    address: Mapped[str | None] = mapped_column(Text)
    state: Mapped[NigerianState | None] = mapped_column(str_enum(NigerianState, "state"))
    # Read back as WKBElement; written as WKTElement (organizations/branches.py).
    location: Mapped[WKBElement | WKTElement | None] = mapped_column(
        Geography(geometry_type="POINT", srid=4326, spatial_index=False)
    )


class Invitation(Entity):
    """An invitation to join an organization, addressed to a phone number or an e-mail address.

    MedLens sends no messages besides login codes, so the inviter receives the token once and
    passes it on; only its SHA-256 is stored. Accepting needs the token and a signed-in account
    with the invited phone number or e-mail address.
    """

    __tablename__ = "invitations"
    __table_args__ = (
        CheckConstraint("(email IS NULL) <> (phone_hash IS NULL)", name="one_recipient"),
        CheckConstraint("(phone_hash IS NULL) = (phone_e164 IS NULL)", name="phone_pair"),
        Index("ix_invitations_organization_id_created_at", "organization_id", "created_at"),
    )

    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(ForeignKey("roles.name", onupdate="CASCADE"))
    email: Mapped[str | None] = mapped_column(CITEXT)
    phone_e164: Mapped[str | None] = mapped_column(EncryptedText("invitations.phone_e164"))
    phone_hash: Mapped[str | None] = mapped_column(String(64))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime]
    invited_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    accepted_at: Mapped[datetime | None]
    accepted_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    revoked_at: Mapped[datetime | None]

    def set_phone(self, phone_e164: str) -> None:
        self.phone_e164 = phone_e164
        self.phone_hash = phone_lookup_hash(phone_e164)

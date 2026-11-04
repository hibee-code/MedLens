from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import ForeignKey, ForeignKeyConstraint, Index, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from medlens.core.identifiers import NigerianState
from medlens.db.base import Entity
from medlens.db.types import str_enum


class CaseStatus(StrEnum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    CONFIRMED_COUNTERFEIT = "confirmed_counterfeit"
    CONFIRMED_GENUINE = "confirmed_genuine"
    INCONCLUSIVE = "inconclusive"
    CLOSED = "closed"


class CasePriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Case(Entity):
    """Human review of a high-risk scan. Confirmed outcomes become training labels."""

    __tablename__ = "cases"
    __table_args__ = (
        # scans is partitioned, so it is referenced by its full key (id, created_at).
        ForeignKeyConstraint(
            ["scan_id", "scan_created_at"], ["scans.id", "scans.created_at"], ondelete="RESTRICT"
        ),
        UniqueConstraint("scan_id", "scan_created_at"),  # one case per scan
        Index("ix_cases_assigned_org_id_status", "assigned_org_id", "status"),
        Index("ix_cases_status_priority_created_at", "status", "priority", "created_at"),
        Index("ix_cases_state_status", "state", "status"),
    )

    scan_id: Mapped[UUID]
    scan_created_at: Mapped[datetime]
    status: Mapped[CaseStatus] = mapped_column(
        str_enum(CaseStatus, "case_status"), default=CaseStatus.OPEN, server_default="open"
    )
    priority: Mapped[CasePriority] = mapped_column(
        str_enum(CasePriority, "case_priority"),
        default=CasePriority.NORMAL,
        server_default="normal",
    )
    assigned_to_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    assigned_org_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL")
    )
    # The scan's state, copied so jurisdiction filters need no join with partitioned scans.
    state: Mapped[NigerianState | None] = mapped_column(str_enum(NigerianState, "state"))
    notes: Mapped[str | None] = mapped_column(Text)  # SafeText at the API boundary
    lab_result: Mapped[dict[str, Any] | None]
    resolved_at: Mapped[datetime | None]

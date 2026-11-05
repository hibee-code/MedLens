from datetime import datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import ForeignKey, ForeignKeyConstraint, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from medlens.db.base import Entity
from medlens.db.types import str_enum


class LabelOutcome(StrEnum):
    COUNTERFEIT = "counterfeit"
    GENUINE = "genuine"
    INCONCLUSIVE = "inconclusive"


class Label(Entity):
    """A human judgement on a scan, e.g. a regulator's lab result. Training and evaluation data
    for future scorers, compared with the pipeline's verdict in accuracy reports; never changes
    the pipeline's verdict (a case decision may set the final verdict)."""

    __tablename__ = "labels"
    __table_args__ = (
        ForeignKeyConstraint(
            ["scan_id", "scan_created_at"], ["scans.id", "scans.created_at"], ondelete="RESTRICT"
        ),
        Index("ix_labels_scan_id", "scan_id"),
    )

    scan_id: Mapped[UUID]
    scan_created_at: Mapped[datetime]
    case_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("cases.id", ondelete="SET NULL"), unique=True
    )
    outcome: Mapped[LabelOutcome] = mapped_column(str_enum(LabelOutcome, "label_outcome"))
    source: Mapped[str] = mapped_column(String(32))  # e.g. case_review
    lab_reference: Mapped[str | None] = mapped_column(Text)
    labelled_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

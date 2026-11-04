from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from geoalchemy2 import Geography, WKBElement
from sqlalchemy import ARRAY, CheckConstraint, Index, String, Uuid, text
from sqlalchemy.orm import Mapped, mapped_column

from medlens.db.base import Entity
from medlens.db.types import states_array_check, str_enum


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class HotspotSnapshot(Entity):
    """One cluster of suspicious scans found by a periodic recompute."""

    __tablename__ = "hotspot_snapshots"
    __table_args__ = (
        CheckConstraint("window_end > window_start", name="window"),
        CheckConstraint("radius_m >= 0 AND scan_count >= 0", name="non_negative"),
        states_array_check("states", "states"),
        Index("ix_hotspot_snapshots_window_end", "window_end"),
        Index("ix_hotspot_snapshots_centroid", "centroid", postgresql_using="gist"),
    )

    window_start: Mapped[datetime]
    window_end: Mapped[datetime]
    cluster_id: Mapped[int]
    centroid: Mapped[WKBElement] = mapped_column(
        Geography(geometry_type="POINT", srid=4326, spatial_index=False)
    )
    radius_m: Mapped[float]
    scan_count: Mapped[int]
    product_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(Uuid), default=list, server_default=text("'{}'")
    )
    states: Mapped[list[str]] = mapped_column(
        ARRAY(String(5)), default=list, server_default=text("'{}'")
    )
    severity: Mapped[Severity] = mapped_column(str_enum(Severity, "severity"))
    algorithm_params: Mapped[dict[str, Any]] = mapped_column(
        default=dict, server_default=text("'{}'::jsonb")
    )

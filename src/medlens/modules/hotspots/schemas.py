from datetime import datetime
from uuid import UUID

from pydantic import ConfigDict, Field

from medlens.core.identifiers import NigerianState
from medlens.core.schemas import ReadModel
from medlens.modules.hotspots.models import Severity


class HotspotProduct(ReadModel):
    product_id: UUID
    nafdac_number: str
    brand_name: str
    scan_count: int


class Centroid(ReadModel):
    latitude: float
    longitude: float


class Hotspot(ReadModel):
    cluster_id: int
    centroid: Centroid = Field(description="Centre of the cluster's coarse (~1 km) locations.")
    radius_m: float = Field(description="Distance from the centre to the farthest scan, metres.")
    scan_count: int
    severity: Severity = Field(description="low (< 5 scans), medium (5-9), high (10 or more).")
    states: list[NigerianState]
    top_products: list[HotspotProduct] = Field(description="Most flagged matched products.")
    first_seen: datetime
    last_seen: datetime


class HotspotParams(ReadModel):
    days: int
    state: NigerianState | None
    eps_m: float
    min_points: int


class HotspotsRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "generated_at": "2026-11-03T09:00:00Z",
                    "since": "2026-10-27T09:00:00Z",
                    "until": "2026-11-03T09:00:00Z",
                    "params": {"days": 7, "state": None, "eps_m": 2000, "min_points": 3},
                    "clusters": [
                        {
                            "cluster_id": 0,
                            "centroid": {"latitude": 6.52, "longitude": 3.38},
                            "radius_m": 1412.2,
                            "scan_count": 12,
                            "severity": "high",
                            "states": ["NG-LA"],
                            "top_products": [
                                {
                                    "product_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e11",
                                    "nafdac_number": "04-0289",
                                    "brand_name": "Emzor Paracetamol Syrup",
                                    "scan_count": 7,
                                }
                            ],
                            "first_seen": "2026-10-29T14:02:00Z",
                            "last_seen": "2026-11-03T08:41:00Z",
                        }
                    ],
                }
            ]
        }
    )

    generated_at: datetime = Field(description="When the clusters were computed (cached 5 min).")
    since: datetime
    until: datetime
    params: HotspotParams
    clusters: list[Hotspot]

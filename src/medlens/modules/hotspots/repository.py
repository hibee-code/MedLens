"""Hotspot clustering in PostGIS (CLAUDE.md §7): flagged scans' coarse locations are projected
to UTM zone 32N (EPSG:32632, metres across most of Nigeria) and clustered with
ST_ClusterDBSCAN, so ``eps`` is a distance in metres. The same query serves the API (async) and
the scheduled snapshot (sync)."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import TextClause, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy.types import String

FLAGGED_VERDICTS = ("SUSPICIOUS", "UNREGISTERED")
METRIC_SRID = 32632

# EPSG:32632 is METRIC_SRID; written out because the statement is a constant.
_CLUSTERS = text(
    """
WITH points AS (
    SELECT s.id, s.created_at, s.matched_product_id, s.state,
           ST_Transform(s.location_coarse::geometry, 32632) AS geom
    FROM scans s
    WHERE s.created_at >= :since AND s.created_at < :until
      AND s.status = 'completed'
      AND s.final_verdict IN :verdicts
      AND s.location_coarse IS NOT NULL
      AND (CAST(:state AS varchar) IS NULL OR s.state = :state)
      AND (:all_states OR s.state = ANY(:states))
),
clustered AS (
    SELECT p.*, ST_ClusterDBSCAN(p.geom, eps => :eps, minpoints => :min_points) OVER ()
               AS cluster_id
    FROM points p
),
centres AS (
    SELECT cluster_id, ST_Centroid(ST_Collect(geom)) AS centre, count(*) AS scan_count,
           min(created_at) AS first_seen, max(created_at) AS last_seen,
           coalesce(array_agg(DISTINCT state) FILTER (WHERE state IS NOT NULL), '{}')
               AS states
    FROM clustered
    WHERE cluster_id IS NOT NULL
    GROUP BY cluster_id
)
SELECT c.cluster_id, c.scan_count, c.first_seen, c.last_seen, c.states,
       ST_Y(ST_Transform(c.centre, 4326)) AS latitude,
       ST_X(ST_Transform(c.centre, 4326)) AS longitude,
       (SELECT max(ST_Distance(k.geom, c.centre)) FROM clustered k
         WHERE k.cluster_id = c.cluster_id) AS radius_m,
       top.products AS top_products
FROM centres c
LEFT JOIN LATERAL (
    SELECT coalesce(json_agg(json_build_object(
               'product_id', t.product_id, 'nafdac_number', pr.nafdac_number,
               'brand_name', pr.brand_name, 'scan_count', t.n) ORDER BY t.n DESC), '[]')
               AS products
    FROM (SELECT k.matched_product_id AS product_id, count(*) AS n
          FROM clustered k
          WHERE k.cluster_id = c.cluster_id AND k.matched_product_id IS NOT NULL
          GROUP BY k.matched_product_id
          ORDER BY n DESC
          LIMIT :top_products) t
    JOIN products pr ON pr.id = t.product_id
) top ON true
ORDER BY c.scan_count DESC, c.last_seen DESC
LIMIT :max_clusters
"""
).bindparams(
    bindparam("verdicts", expanding=True),
    bindparam("states", type_=ARRAY(String(5))),
)


@dataclass(frozen=True, slots=True)
class ClusterQuery:
    since: datetime
    until: datetime
    eps_m: float
    min_points: int
    state: str | None = None
    states: frozenset[str] | None = None  # None: every state, and scans without one
    top_products: int = 3
    max_clusters: int = 100

    def params(self) -> dict[str, Any]:
        return {
            "since": self.since,
            "until": self.until,
            "verdicts": list(FLAGGED_VERDICTS),
            "state": self.state,
            "all_states": self.states is None,
            "states": sorted(self.states or ()),
            "eps": self.eps_m,
            "min_points": self.min_points,
            "top_products": self.top_products,
            "max_clusters": self.max_clusters,
        }


@dataclass(frozen=True, slots=True)
class Cluster:
    cluster_id: int
    scan_count: int
    first_seen: datetime
    last_seen: datetime
    states: list[str]
    latitude: float
    longitude: float
    radius_m: float
    top_products: list[dict[str, Any]]

    @property
    def product_ids(self) -> list[UUID]:
        return [UUID(str(product["product_id"])) for product in self.top_products]


def _cluster(row: Any) -> Cluster:
    return Cluster(
        cluster_id=int(row.cluster_id),
        scan_count=int(row.scan_count),
        first_seen=row.first_seen,
        last_seen=row.last_seen,
        states=sorted(row.states or []),
        latitude=float(row.latitude),
        longitude=float(row.longitude),
        radius_m=round(float(row.radius_m or 0.0), 1),
        top_products=list(row.top_products or []),
    )


def _statement() -> TextClause:
    return _CLUSTERS


async def find_clusters(session: AsyncSession, query: ClusterQuery) -> list[Cluster]:
    rows = await session.execute(_statement(), query.params())
    return [_cluster(row) for row in rows]


def find_clusters_sync(session: Session, query: ClusterQuery) -> list[Cluster]:
    return [_cluster(row) for row in session.execute(_statement(), query.params())]

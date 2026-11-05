"""Hotspots: clusters of SUSPICIOUS and UNREGISTERED scans (their final verdict, after any
case decision) by place and time, for regulators. Regulators see clusters of scans in their
jurisdiction only; results are cached in Redis for five minutes per parameters and scope."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session

from medlens.cache.aside import cached_json
from medlens.cache.keys import redis_key
from medlens.core.identifiers import NigerianState
from medlens.db.session import UnitOfWork
from medlens.modules.hotspots.models import HotspotSnapshot, Severity
from medlens.modules.hotspots.repository import (
    Cluster,
    ClusterQuery,
    find_clusters,
    find_clusters_sync,
)
from medlens.modules.rbac import policies
from medlens.modules.rbac.access import Actor

CACHE_SECONDS = 300
DEFAULT_DAYS = 7
DEFAULT_EPS_M = 2000.0
DEFAULT_MIN_POINTS = 3


def window_end(now: datetime | None = None) -> datetime:
    """The end of the current minute: stable within a minute (one cache entry) and later than
    every scan made so far."""
    return (now or datetime.now(UTC)).replace(second=0, microsecond=0) + timedelta(minutes=1)


def severity(scan_count: int) -> Severity:
    if scan_count >= 10:
        return Severity.HIGH
    return Severity.MEDIUM if scan_count >= 5 else Severity.LOW


def _cluster_json(cluster: Cluster) -> dict[str, Any]:
    return {
        "cluster_id": cluster.cluster_id,
        "centroid": {"latitude": cluster.latitude, "longitude": cluster.longitude},
        "radius_m": cluster.radius_m,
        "scan_count": cluster.scan_count,
        "severity": severity(cluster.scan_count).value,
        "states": cluster.states,
        "top_products": cluster.top_products,
        "first_seen": cluster.first_seen.isoformat(),
        "last_seen": cluster.last_seen.isoformat(),
    }


class HotspotService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], redis: Redis) -> None:
        self._sessions = sessions
        self._redis = redis

    async def clusters(
        self,
        actor: Actor,
        *,
        days: int = DEFAULT_DAYS,
        state: NigerianState | None = None,
        eps_m: float = DEFAULT_EPS_M,
        min_points: int = DEFAULT_MIN_POINTS,
    ) -> dict[str, Any]:
        scope = policies.hotspot_states(actor)
        until = window_end()
        query = ClusterQuery(
            since=until - timedelta(days=days),
            until=until,
            eps_m=eps_m,
            min_points=min_points,
            state=state.value if state else None,
            states=scope,
        )
        params = {"days": days, "state": query.state, "eps_m": eps_m, "min_points": min_points}
        fingerprint = json.dumps(
            {**params, "scope": None if scope is None else sorted(scope)}, sort_keys=True
        )
        key = redis_key("hotspots", hashlib.sha256(fingerprint.encode()).hexdigest()[:32])

        async def compute() -> dict[str, Any]:
            if scope is not None and (not scope or (query.state and query.state not in scope)):
                found: list[Cluster] = []  # nothing in the caller's jurisdiction to show
            else:
                async with UnitOfWork(self._sessions) as uow:
                    found = await find_clusters(uow.session, query)
            return {
                "generated_at": datetime.now(UTC).isoformat(),
                "since": query.since.isoformat(),
                "until": query.until.isoformat(),
                "params": params,
                "clusters": [_cluster_json(cluster) for cluster in found],
            }

        value, _ = await cached_json(self._redis, key, CACHE_SECONDS, compute)
        result: dict[str, Any] = value
        return result


def record_snapshots(session: Session, *, now: datetime | None = None) -> int:
    """The scheduled recompute (worker): the last week's national clusters with the default
    parameters, kept as HotspotSnapshot rows for trends."""
    until = window_end(now)
    query = ClusterQuery(
        since=until - timedelta(days=DEFAULT_DAYS),
        until=until,
        eps_m=DEFAULT_EPS_M,
        min_points=DEFAULT_MIN_POINTS,
    )
    clusters = find_clusters_sync(session, query)
    for cluster in clusters:
        session.add(
            HotspotSnapshot(
                window_start=query.since,
                window_end=query.until,
                cluster_id=cluster.cluster_id,
                centroid=f"SRID=4326;POINT({cluster.longitude} {cluster.latitude})",
                radius_m=cluster.radius_m,
                scan_count=cluster.scan_count,
                product_ids=cluster.product_ids,
                states=cluster.states,
                severity=severity(cluster.scan_count),
                algorithm_params={
                    "algorithm": "ST_ClusterDBSCAN",
                    "srid": 32632,
                    "eps_m": DEFAULT_EPS_M,
                    "min_points": DEFAULT_MIN_POINTS,
                    "days": DEFAULT_DAYS,
                },
            )
        )
    session.flush()
    return len(clusters)

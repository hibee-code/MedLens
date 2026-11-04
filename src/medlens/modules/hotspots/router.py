from typing import Annotated

from fastapi import Depends, Query

from medlens.core.identifiers import NigerianState
from medlens.core.openapi import problem_responses
from medlens.core.routing import api_v1_router
from medlens.modules.hotspots.schemas import HotspotsRead
from medlens.modules.hotspots.service import (
    DEFAULT_DAYS,
    DEFAULT_EPS_M,
    DEFAULT_MIN_POINTS,
    HotspotService,
)
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.resources import Resources, get_resources

router = api_v1_router("/hotspots", tags=["hotspots"])
Reader = Annotated[Actor, Depends(require_permissions(Permission.HOTSPOT_READ))]


def get_hotspot_service(
    resources: Annotated[Resources, Depends(get_resources)],
) -> HotspotService:
    return HotspotService(resources.session_factory, resources.redis)


@router.get(
    "",
    response_model=HotspotsRead,
    summary="Clusters of flagged scans",
    description=(
        "Clusters of SUSPICIOUS and UNREGISTERED scans (the verdict shown, after any case "
        "decision) from the last `days`, found with PostGIS ST_ClusterDBSCAN on the scans' "
        "coarse (~1 km) locations in a metric projection (EPSG:32632). `eps_m` is the "
        "neighbourhood radius in metres and `min_points` the scans needed to form a cluster. "
        "Regulators see scans in their jurisdiction only. Results are cached for 5 minutes."
    ),
    responses=problem_responses(401, 403, 422),
)
async def list_hotspots(
    actor: Reader,
    service: Annotated[HotspotService, Depends(get_hotspot_service)],
    days: Annotated[int, Query(ge=1, le=90, description="Look-back window.")] = DEFAULT_DAYS,
    state: Annotated[NigerianState | None, Query(description="One state only.")] = None,
    eps_m: Annotated[float, Query(ge=100, le=50_000)] = DEFAULT_EPS_M,
    min_points: Annotated[int, Query(ge=2, le=100)] = DEFAULT_MIN_POINTS,
) -> HotspotsRead:
    return HotspotsRead.model_validate(
        await service.clusters(actor, days=days, state=state, eps_m=eps_m, min_points=min_points)
    )

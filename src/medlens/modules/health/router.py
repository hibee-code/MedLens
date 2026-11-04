"""Liveness and readiness probes.

Served at the root (``/health/...``) rather than under ``/api/v1``: they are for orchestrators and
load balancers, not API clients, and must not change when the API version does.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from medlens.modules.health.deps import get_readiness_service
from medlens.modules.health.schemas import LivenessResponse, ReadinessResponse
from medlens.modules.health.service import ReadinessService

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description=(
        "Returns 200 while the process can serve HTTP. It touches no dependency, so an outage "
        "in Postgres, Redis or S3 never makes the orchestrator restart healthy API processes."
    ),
)
async def liveness(response: Response) -> LivenessResponse:
    response.headers["Cache-Control"] = "no-store"
    return LivenessResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Probes Postgres, Redis and object storage concurrently, each with a short timeout "
        "(`MEDLENS_APP__READINESS_TIMEOUT_SECONDS`), and reports a status per dependency. "
        "Returns 200 when all are reachable, and also when only Redis is down (`degraded`): "
        "the API has fallbacks for Redis, and every instance shares it, so removing them all "
        "from the load balancer would only turn a partial outage into a full one. Returns 503 "
        "when Postgres or object storage is unavailable; load balancers should then stop "
        "routing traffic to the instance."
    ),
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "Postgres or object storage is unavailable.",
        }
    },
)
async def readiness(
    response: Response,
    service: Annotated[ReadinessService, Depends(get_readiness_service)],
) -> ReadinessResponse:
    report = await service.check()
    response.headers["Cache-Control"] = "no-store"
    if report.status == "not_ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report

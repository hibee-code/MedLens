"""Prometheus exposition at /metrics (outside /api/v1, hidden from the OpenAPI document).

One Uvicorn process per container, so the default in-process registry is complete. The endpoint
is for the internal scraper only: the load balancer must not route /metrics publicly.
"""

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

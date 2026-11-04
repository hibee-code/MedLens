"""Routing conventions shared by every module.

Module routers carry their full path, API version included, and ``main.py`` includes them without
an extra prefix. FastAPI no longer flattens included routers: a prefix added at
``include_router()`` time is missing from ``scope["route"].path``, which access logs (and later
metrics) use as the route template.
"""

from fastapi import APIRouter

API_V1_PREFIX = "/api/v1"


def api_v1_router(path: str, *, tags: list[str]) -> APIRouter:
    """Router for one resource under /api/v1, e.g. ``api_v1_router("/scans", tags=["scans"])``."""
    return APIRouter(prefix=f"{API_V1_PREFIX}{path}", tags=list(tags))

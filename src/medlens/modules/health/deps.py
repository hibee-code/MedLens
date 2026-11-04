from functools import partial
from typing import Annotated

from fastapi import Depends

from medlens.adapters.s3 import check_bucket
from medlens.cache.redis import ping_redis
from medlens.core.config import Settings
from medlens.db.session import ping_database
from medlens.modules.health.service import ReadinessService
from medlens.resources import Resources, get_app_settings, get_resources


def readiness_service(resources: Resources, settings: Settings) -> ReadinessService:
    return ReadinessService(
        {
            "postgres": partial(ping_database, resources.db_engine),
            "redis": partial(ping_redis, resources.redis),
            "s3": partial(check_bucket, resources.s3, resources.s3_bucket),
        },
        timeout_seconds=settings.app.readiness_timeout_seconds,
        # Redis outages are survivable: permissions fall back to Postgres, session revocation
        # fails open for consumers and falls back to Postgres for privileged sessions, and
        # sign-in by SMS code answers 503.
        degradable=frozenset({"redis"}),
    )


def get_readiness_service(
    resources: Annotated[Resources, Depends(get_resources)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> ReadinessService:
    return readiness_service(resources, settings)

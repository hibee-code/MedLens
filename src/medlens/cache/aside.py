"""Cache-aside for JSON values. Redis is an optimisation here: when it cannot answer, the value
is computed and returned as if uncached."""

import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

log = structlog.stdlib.get_logger(__name__)


async def cached_json(
    redis: Redis, key: str, ttl_seconds: int, produce: Callable[[], Awaitable[Any]]
) -> tuple[Any, bool]:
    """The cached value under ``key``, or ``produce()`` stored for ``ttl_seconds``. Returns the
    value and whether it came from the cache. ``produce`` must return JSON-compatible data."""
    try:
        cached = await redis.get(key)
    except RedisError as exc:
        log.warning("cache.unavailable", error=type(exc).__name__)
        cached = None
    if cached is not None:
        return json.loads(cached), True
    value = await produce()
    try:
        await redis.set(key, json.dumps(value, separators=(",", ":")), ex=ttl_seconds)
    except RedisError as exc:
        log.warning("cache.unavailable", error=type(exc).__name__)
    return value, False

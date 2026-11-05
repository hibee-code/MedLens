"""Distributed rate limits: a sliding window per key in Redis (a sorted set of request times),
checked and updated atomically by one Lua script on Redis's own clock, so every API container
counts the same requests.

When Redis cannot answer, scan and assistant limits let the request through and count the
skipped check (medlens_ratelimit_checks_skipped_total), the same trade-off as session
revocation (CLAUDE.md decision log). Sign-in limits fail closed: a 503 with Retry-After,
counted in medlens_ratelimit_unchecked_refusals_total, since guessing codes or passwords
without a limit is the attack they exist to stop.
"""

import math
from uuid import uuid4

import structlog
from prometheus_client import Counter
from redis.asyncio import Redis
from redis.exceptions import RedisError

from medlens.cache.keys import redis_key
from medlens.core.errors import DependencyUnavailable

log = structlog.stdlib.get_logger(__name__)

RATE_LIMITED = Counter("medlens_ratelimit_rejections", "Requests refused with 429.", ["bucket"])
CHECKS_SKIPPED = Counter(
    "medlens_ratelimit_checks_skipped", "Rate-limit checks skipped (Redis unavailable).", ["bucket"]
)
UNCHECKED_REFUSALS = Counter(
    "medlens_ratelimit_unchecked_refusals",
    "Requests refused because a fail-closed limit could not be checked (Redis unavailable).",
    ["bucket"],
)
RETRY_WHILE_UNAVAILABLE_SECONDS = 5

# KEYS[1] the window; ARGV: window ms, limit, member. Returns {allowed, retry_after_ms}.
_SLIDING_WINDOW = """
local now_parts = redis.call('TIME')
local now = tonumber(now_parts[1]) * 1000 + math.floor(tonumber(now_parts[2]) / 1000)
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, now - window)
if redis.call('ZCARD', KEYS[1]) < limit then
    redis.call('ZADD', KEYS[1], now, ARGV[3])
    redis.call('PEXPIRE', KEYS[1], window)
    return {1, 0}
end
local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
return {0, window - (now - tonumber(oldest[2]))}
"""


class SlidingWindowLimiter:
    def __init__(self, redis: Redis) -> None:
        self._script = redis.register_script(_SLIDING_WINDOW)

    async def retry_after(
        self,
        bucket: str,
        subject: str,
        *,
        limit: int,
        window_seconds: int = 60,
        fail_open: bool = True,
    ) -> int | None:
        """Records a request; returns None if allowed, else the seconds until it would be.
        Without Redis: allowed if ``fail_open``, else DependencyUnavailable (503)."""
        key = redis_key("ratelimit", bucket, subject)
        try:
            allowed, retry_ms = await self._script(
                keys=[key], args=[window_seconds * 1000, limit, uuid4().hex]
            )
        except RedisError as exc:
            if not fail_open:
                UNCHECKED_REFUSALS.labels(bucket=bucket).inc()
                log.warning("ratelimit.unchecked_refused", bucket=bucket, error=type(exc).__name__)
                raise DependencyUnavailable(
                    "Sign-in is briefly unavailable. Try again shortly.",
                    dependency="redis",
                    retry_after_seconds=RETRY_WHILE_UNAVAILABLE_SECONDS,
                ) from exc
            CHECKS_SKIPPED.labels(bucket=bucket).inc()
            log.warning("ratelimit.check_skipped", bucket=bucket, error=type(exc).__name__)
            return None
        if allowed:
            return None
        RATE_LIMITED.labels(bucket=bucket).inc()
        return max(1, math.ceil(int(retry_ms) / 1000))

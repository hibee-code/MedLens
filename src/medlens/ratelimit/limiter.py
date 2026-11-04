"""Distributed rate limits: a sliding window per key in Redis (a sorted set of request times),
checked and updated atomically by one Lua script on Redis's own clock, so every API container
counts the same requests.

Redis is a dependency of the limit, not of the API: when it cannot answer, the request is let
through and the skipped check is counted (medlens_ratelimit_checks_skipped_total), the same
trade-off as session revocation (CLAUDE.md decision log).
"""

import math
from uuid import uuid4

import structlog
from prometheus_client import Counter
from redis.asyncio import Redis
from redis.exceptions import RedisError

from medlens.cache.keys import redis_key

log = structlog.stdlib.get_logger(__name__)

RATE_LIMITED = Counter("medlens_ratelimit_rejections", "Requests refused with 429.", ["bucket"])
CHECKS_SKIPPED = Counter(
    "medlens_ratelimit_checks_skipped", "Rate-limit checks skipped (Redis unavailable).", ["bucket"]
)

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
        self, bucket: str, subject: str, *, limit: int, window_seconds: int = 60
    ) -> int | None:
        """Records a request; returns None if allowed, else the seconds until it would be."""
        key = redis_key("ratelimit", bucket, subject)
        try:
            allowed, retry_ms = await self._script(
                keys=[key], args=[window_seconds * 1000, limit, uuid4().hex]
            )
        except RedisError as exc:
            CHECKS_SKIPPED.labels(bucket=bucket).inc()
            log.warning("ratelimit.check_skipped", bucket=bucket, error=type(exc).__name__)
            return None
        if allowed:
            return None
        RATE_LIMITED.labels(bucket=bucket).inc()
        return max(1, math.ceil(int(retry_ms) / 1000))

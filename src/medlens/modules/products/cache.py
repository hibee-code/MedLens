"""Product facts cached by the scan pipeline (verification.product_cache, MEDLENS_VERIFICATION__
PRODUCT_CACHE_SECONDS): the key, owned here with the data, and the invalidation every writer of
product facts calls after committing. Today that is the register import and the seed.

Invalidation is best effort: if Redis cannot be reached, entries still expire within the TTL.
"""

import structlog
from redis import Redis
from redis.exceptions import RedisError

from medlens.cache.keys import redis_key

log = structlog.stdlib.get_logger(__name__)
PRODUCT_FACTS_PREFIX = redis_key("catalog", "product")
_BATCH = 500


def product_facts_key(nafdac_number: str) -> str:
    return f"{PRODUCT_FACTS_PREFIX}:{nafdac_number}"


def clear_product_facts(redis: Redis) -> int:
    """Drop every cached product (bulk changes such as a register import); returns how many."""
    removed = 0
    try:
        batch: list[bytes | str] = []
        for key in redis.scan_iter(match=f"{PRODUCT_FACTS_PREFIX}:*", count=_BATCH):
            batch.append(key)
            if len(batch) >= _BATCH:
                removed += int(redis.unlink(*batch))
                batch = []
        if batch:
            removed += int(redis.unlink(*batch))
    except RedisError as exc:
        log.warning("products.cache_clear_failed", error=type(exc).__name__)
    else:
        log.info("products.cache_cleared", removed=removed)
    return removed

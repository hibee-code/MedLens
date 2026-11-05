"""Registered products by NAFDAC number, cached in Redis for the scan pipeline (worker).

Every scan looks its product up; the same few hundred products are scanned over and over.
Misses are cached too (an unregistered number stays unregistered until the next Greenbook
import). Entries expire after MEDLENS_VERIFICATION__PRODUCT_CACHE_SECONDS; Redis errors only
skip the cache."""

import json
from uuid import UUID

from redis import Redis
from redis.exceptions import RedisError

from medlens.cache.keys import redis_key
from medlens.modules.products.models import RegistrationStatus
from medlens.modules.verification.catalog import ProductFacts

_MISSING = "none"


def product_key(nafdac_number: str) -> str:
    return redis_key("catalog", "product", nafdac_number)


class ProductCache:
    def __init__(self, redis: Redis, *, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    def get(self, nafdac_number: str) -> tuple[bool, ProductFacts | None]:
        """(found in the cache, the product or None for an unregistered number)."""
        try:
            value = self._redis.get(product_key(nafdac_number))
        except RedisError:
            return False, None
        if value is None:
            return False, None
        text = value.decode() if isinstance(value, bytes) else str(value)
        if text == _MISSING:
            return True, None
        data = json.loads(text)
        return True, ProductFacts(
            id=UUID(data["id"]),
            nafdac_number=data["nafdac_number"],
            brand_name=data["brand_name"],
            generic_name=data["generic_name"],
            registration_status=RegistrationStatus(data["registration_status"]),
        )

    def put(self, nafdac_number: str, facts: ProductFacts | None) -> None:
        value = (
            _MISSING
            if facts is None
            else json.dumps(
                {
                    "id": str(facts.id),
                    "nafdac_number": facts.nafdac_number,
                    "brand_name": facts.brand_name,
                    "generic_name": facts.generic_name,
                    "registration_status": facts.registration_status.value,
                }
            )
        )
        try:
            self._redis.set(product_key(nafdac_number), value, ex=self._ttl)
        except RedisError:
            return

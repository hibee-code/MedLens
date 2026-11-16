"""Registered products by NAFDAC number, cached in Redis for the scan pipeline (worker).

Every scan looks its product up; the same few hundred products are scanned over and over.
Misses are cached too (an unregistered number stays unregistered until the next Greenbook
import). Entries expire after MEDLENS_VERIFICATION__PRODUCT_CACHE_SECONDS, and writers of product
facts clear them (products.cache); Redis errors only skip the cache."""

import json
from datetime import date
from uuid import UUID

from redis import Redis
from redis.exceptions import RedisError

from medlens.modules.products.cache import product_facts_key
from medlens.modules.products.models import RegistrationStatus
from medlens.modules.products.register import ingredients_from_json
from medlens.modules.verification.catalog import ProductFacts

_MISSING = "none"


class ProductCache:
    def __init__(self, redis: Redis, *, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    def get(self, nafdac_number: str) -> tuple[bool, ProductFacts | None]:
        """(found in the cache, the product or None for an unregistered number)."""
        try:
            value = self._redis.get(product_facts_key(nafdac_number))
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
            active_ingredients=ingredients_from_json(data.get("active_ingredients", [])),
            dosage_form=data.get("dosage_form"),
            holder_name=data.get("holder_name"),
            registration_date=_day(data.get("registration_date")),
            registration_expires_on=_day(data.get("registration_expires_on")),
            source_url=data.get("source_url"),
            retrieved_on=_day(data.get("retrieved_on")),
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
                    "active_ingredients": [item.as_json() for item in facts.active_ingredients],
                    "dosage_form": facts.dosage_form,
                    "holder_name": facts.holder_name,
                    "registration_date": _iso(facts.registration_date),
                    "registration_expires_on": _iso(facts.registration_expires_on),
                    "source_url": facts.source_url,
                    "retrieved_on": _iso(facts.retrieved_on),
                }
            )
        )
        try:
            self._redis.set(product_facts_key(nafdac_number), value, ex=self._ttl)
        except RedisError:
            return


def _iso(day: date | None) -> str | None:
    return day.isoformat() if day else None


def _day(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None

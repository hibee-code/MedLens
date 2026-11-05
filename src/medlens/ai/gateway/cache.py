"""Answer cache for the gateway: a validated output reused for an identical request.

The key covers everything that shapes the answer: model, prompt version, output schema, token
limit, effort and the full message content, which includes the retrieved documents. A document
that changes, or a caller who may see other documents, gives another key. Only routes marked
cacheable use it; Redis errors only skip the cache."""

import hashlib
import json
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from medlens.ai.gateway.providers import ProviderRequest
from medlens.cache.keys import redis_key


def answer_key(request: ProviderRequest, prompt_version: str) -> str:
    material: dict[str, Any] = {
        "model": request.model,
        "prompt": prompt_version,
        "schema": request.schema,
        "max_tokens": request.max_tokens,
        "effort": request.effort,
        "messages": request.messages,
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()
    return redis_key("llm_answer", digest)


class AnswerCache:
    def __init__(self, redis: Redis, *, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    async def get(self, key: str) -> str | None:
        try:
            value = await self._redis.get(key)
        except RedisError:
            return None
        if value is None:
            return None
        return value.decode() if isinstance(value, bytes) else str(value)

    async def put(self, key: str, text: str) -> None:
        try:
            await self._redis.set(key, text, ex=self._ttl)
        except RedisError:
            return

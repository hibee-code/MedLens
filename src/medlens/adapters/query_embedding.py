"""The API's access to the worker's models over Celery (the API image has no models): question
embeddings and reranking. The API waits briefly for the answer; late answers are discarded (the
tasks expire), so a busy queue degrades search instead of slowing it down. Question embeddings
are also cached in Redis, per model, since the same questions recur."""

import contextlib
import hashlib
import json
from collections.abc import Sequence
from typing import Any

import structlog
from celery import Celery
from redis.asyncio import Redis
from redis.exceptions import RedisError
from starlette.concurrency import run_in_threadpool

from medlens.cache.keys import redis_key
from medlens.ports.embedding import QueryEmbeddingPort

log = structlog.stdlib.get_logger(__name__)

EMBED_QUERY_TASK = "medlens.knowledge.embed_query"
RERANK_TASK = "medlens.knowledge.rerank"
QUERY_PRIORITY = 0  # served before reference-image and backfill work on the same queue


async def _call(app: Celery, task: str, args: list[Any], wait_seconds: float) -> Any:
    """Send the task and wait for its result in a thread; it expires when nobody waits."""
    result = app.send_task(task, args=args, priority=QUERY_PRIORITY, expires=wait_seconds)
    return await run_in_threadpool(result.get, timeout=wait_seconds)


class CeleryQueryEmbedder:
    def __init__(self, app: Celery, *, timeout_seconds: float) -> None:
        self._app = app
        self._timeout = timeout_seconds

    async def embed_query(self, text: str) -> list[float] | None:
        try:
            vector = await _call(self._app, EMBED_QUERY_TASK, [text], self._timeout)
        except Exception as exc:  # noqa: BLE001 - search falls back to keywords
            log.warning("knowledge.query_embedding_unavailable", error=type(exc).__name__)
            return None
        if not isinstance(vector, list):
            return None
        return [float(value) for value in vector]


class CeleryReranker:
    def __init__(self, app: Celery, *, timeout_seconds: float) -> None:
        self._app = app
        self._timeout = timeout_seconds

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float] | None:
        try:
            scores = await _call(self._app, RERANK_TASK, [query, list(passages)], self._timeout)
        except Exception as exc:  # noqa: BLE001 - the fused order stands
            log.warning("knowledge.rerank_unavailable", error=type(exc).__name__)
            return None
        if not isinstance(scores, list) or len(scores) != len(passages):
            return None
        return [float(score) for score in scores]


class CachedQueryEmbedder:
    """Question embeddings in Redis for ``ttl_seconds``, keyed by model and the question's
    hash (questions are not stored in clear). Redis errors only skip the cache."""

    def __init__(
        self, inner: QueryEmbeddingPort, redis: Redis, *, model: str, ttl_seconds: int
    ) -> None:
        self._inner = inner
        self._redis = redis
        self._model = model
        self._ttl = ttl_seconds

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(f"{self._model}\n{text}".encode()).hexdigest()
        return redis_key("query_embedding", digest)

    async def embed_query(self, text: str) -> list[float] | None:
        if not self._ttl:
            return await self._inner.embed_query(text)
        key = self._key(text)
        try:
            cached = await self._redis.get(key)
        except RedisError:
            cached = None
        if cached is not None:
            return [float(value) for value in json.loads(cached)]
        vector = await self._inner.embed_query(text)
        if vector is not None:
            with contextlib.suppress(RedisError):
                await self._redis.set(key, json.dumps(vector), ex=self._ttl)
        return vector

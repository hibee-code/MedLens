"""The API's access to the worker's models over Celery (the API image has no models): question
embeddings and reranking. The API waits briefly for the answer; late answers are discarded (the
tasks expire), so a busy queue degrades search instead of slowing it down. Question embeddings
are also cached in Redis, since the same questions recur; an entry made by another model version
than the workers now run is ignored."""

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

from medlens.cache.keys import TEXT_MODEL_VERSION_KEY, redis_key
from medlens.ports.embedding import QueryEmbedding, QueryEmbeddingPort

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

    async def embed_query(self, text: str) -> QueryEmbedding | None:
        try:
            result = await _call(self._app, EMBED_QUERY_TASK, [text], self._timeout)
        except Exception as exc:  # noqa: BLE001 - search falls back to keywords
            log.warning("knowledge.query_embedding_unavailable", error=type(exc).__name__)
            return None
        return _parse(result)


def _parse(value: Any) -> QueryEmbedding | None:
    """The worker's (or the cache's) ``{"model_version": ..., "vector": [...]}``."""
    if not isinstance(value, dict):
        return None
    version, vector = value.get("model_version"), value.get("vector")
    if not isinstance(version, str) or not isinstance(vector, list):
        return None
    return QueryEmbedding(version, [float(item) for item in vector])


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
    """Question embeddings in Redis for ``ttl_seconds``, keyed by the configured model and the
    question's hash (questions are not stored in clear). Each entry records the model version
    that made it; one made by another version than the workers now run is ignored (read in
    the same round trip), so a model change invalidates the cache. Redis errors only skip it."""

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

    async def embed_query(self, text: str) -> QueryEmbedding | None:
        if not self._ttl:
            return await self._inner.embed_query(text)
        key = self._key(text)
        try:
            cached, current = await self._redis.mget([key, TEXT_MODEL_VERSION_KEY])
        except RedisError:
            cached = current = None
        hit = _parse(json.loads(cached)) if cached is not None else None
        if hit is not None and (current is None or _text(current) == hit.model_version):
            return hit
        embedding = await self._inner.embed_query(text)
        if embedding is not None:
            value = {"model_version": embedding.model_version, "vector": embedding.vector}
            with contextlib.suppress(RedisError):
                await self._redis.set(key, json.dumps(value), ex=self._ttl)
        return embedding


def _text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value

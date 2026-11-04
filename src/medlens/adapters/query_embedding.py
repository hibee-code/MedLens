"""QueryEmbeddingPort over Celery: the API asks a worker that has the text model loaded and
waits briefly for the answer. Late answers are discarded (the task expires), so a busy queue
degrades search to keywords instead of slowing it down."""

import structlog
from celery import Celery
from starlette.concurrency import run_in_threadpool

log = structlog.stdlib.get_logger(__name__)

EMBED_QUERY_TASK = "medlens.knowledge.embed_query"
QUERY_PRIORITY = 0  # served before reference-image and backfill work on the same queue


class CeleryQueryEmbedder:
    def __init__(self, app: Celery, *, timeout_seconds: float) -> None:
        self._app = app
        self._timeout = timeout_seconds

    async def embed_query(self, text: str) -> list[float] | None:
        try:
            result = self._app.send_task(
                EMBED_QUERY_TASK, args=[text], priority=QUERY_PRIORITY, expires=self._timeout
            )
            vector = await run_in_threadpool(result.get, timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001 - search falls back to keywords
            log.warning("knowledge.query_embedding_unavailable", error=type(exc).__name__)
            return None
        if not isinstance(vector, list):
            return None
        return [float(value) for value in vector]

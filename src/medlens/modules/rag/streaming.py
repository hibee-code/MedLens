"""Server-sent events for the assistant: a status event as each stage starts, comments to keep
proxies from closing a quiet connection while the model works, then one final event.

The answer is never streamed token by token: it is shown only after its citations are checked
(CLAUDE.md §5, grounded or silent), so clients see progress, then the verified answer.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import structlog

from medlens.core.errors import PROBLEM_TYPE_PREFIX
from medlens.modules.rag.service import AssistantReply, Progress

log = structlog.stdlib.get_logger(__name__)
KEEPALIVE_SECONDS = 10.0


def event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'), default=str)}\n\n"


async def stream(
    run: Callable[[Progress], Awaitable[AssistantReply]],
    finish: Callable[[AssistantReply], dict[str, Any]],
    *,
    keepalive_seconds: float = KEEPALIVE_SECONDS,
) -> AsyncIterator[str]:
    """Run ``run`` with a progress callback and yield its stages as they happen, then the
    final ``answer`` event (``finish`` renders it), or an ``error`` event."""
    stages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def progress(stage: str, data: dict[str, object]) -> None:
        await stages.put({"stage": stage, **data})

    work = asyncio.ensure_future(run(progress))
    try:
        while True:
            getter = asyncio.ensure_future(stages.get())
            done, _ = await asyncio.wait(
                {getter, work}, timeout=keepalive_seconds, return_when=asyncio.FIRST_COMPLETED
            )
            if getter in done:
                yield event("status", getter.result())
                continue
            getter.cancel()
            if work in done:
                break
            yield ": keepalive\n\n"
        while not stages.empty():
            yield event("status", stages.get_nowait())
        yield event("answer", finish(work.result()))
    except Exception as exc:
        log.exception("assistant.stream_failed", error=type(exc).__name__)
        yield event(
            "error",
            {
                "type": f"{PROBLEM_TYPE_PREFIX}internal-error",
                "title": "Internal error",
                "status": 500,
            },
        )
    finally:
        if not work.done():
            work.cancel()

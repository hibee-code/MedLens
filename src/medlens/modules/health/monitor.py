"""Background probing, so ``medlens_dependency_up`` (and the alerts built on it) stays current
however often, or rarely, an orchestrator calls /health/ready."""

import asyncio
from collections.abc import Sequence
from contextlib import suppress

import structlog

from medlens.modules.health.service import ReadinessService
from medlens.queue.backpressure import QueueDepth

log = structlog.stdlib.get_logger(__name__)


async def monitor_dependencies(
    service: ReadinessService, *, interval_seconds: float, queues: Sequence[QueueDepth] = ()
) -> None:
    """Probe every dependency, and read the queue depths (medlens_queue_depth), once per
    interval until cancelled."""
    while True:
        await asyncio.sleep(interval_seconds)
        await service.check()
        for queue in queues:
            await queue.depth()


def start_monitor(
    service: ReadinessService, *, interval_seconds: float, queues: Sequence[QueueDepth] = ()
) -> asyncio.Task[None]:
    return asyncio.create_task(
        monitor_dependencies(service, interval_seconds=interval_seconds, queues=queues),
        name="dependency-monitor",
    )


async def stop_monitor(task: asyncio.Task[None]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

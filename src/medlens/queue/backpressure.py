"""Backpressure for scan submission: how many scans wait in the ``scans.pipeline`` queue.

The depth is read from the broker (the length of the queue's Redis lists, one per priority
step) and cached for a few seconds, so a burst of submissions costs one Redis round trip.
Above the soft limit the estimated wait grows; above the hard limit new scans are refused with
503 and Retry-After. If the depth cannot be read, submissions are not blocked.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from medlens.core.errors import DependencyUnavailable

log = structlog.stdlib.get_logger(__name__)

SCANS_QUEUE = "scans.pipeline"
PRIORITY_STEPS = tuple(range(10))
PRIORITY_SEP = ":"
BASE_WAIT_SECONDS = 10


def queue_keys(queue: str) -> list[str]:
    """Redis lists holding a queue's messages: kombu keeps one list per priority step."""
    return [queue if step == 0 else f"{queue}{PRIORITY_SEP}{step}" for step in PRIORITY_STEPS]


class QueueDepth:
    def __init__(
        self,
        broker: Redis,
        queue: str = SCANS_QUEUE,
        *,
        cache_seconds: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._broker = broker
        self._keys = queue_keys(queue)
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._value: int | None = None
        self._read_at = float("-inf")

    async def depth(self) -> int | None:
        now = self._clock()
        if now - self._read_at < self._cache_seconds:
            return self._value
        try:
            async with self._broker.pipeline(transaction=False) as pipe:
                for key in self._keys:
                    pipe.llen(key)
                lengths = await pipe.execute()
            self._value = int(sum(lengths))
        except RedisError as exc:
            log.warning("queue.depth_unavailable", error=type(exc).__name__)
            self._value = None
        self._read_at = now
        return self._value


@dataclass(frozen=True, slots=True)
class Backpressure:
    soft_limit: int
    hard_limit: int
    seconds_per_queued_scan: float

    def estimated_wait(self, depth: int | None) -> int:
        """Seconds until a new scan is likely done; refuses new scans above the hard limit."""
        if depth is None or depth <= self.soft_limit:
            return BASE_WAIT_SECONDS
        wait = BASE_WAIT_SECONDS + round(depth * self.seconds_per_queued_scan)
        if depth > self.hard_limit:
            raise DependencyUnavailable(
                "Verification is very busy right now. Try again shortly.",
                dependency="scan_queue",
                retry_after_seconds=min(wait, 300),
            )
        return wait

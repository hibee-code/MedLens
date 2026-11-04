"""Readiness: probe every dependency concurrently, each with its own short timeout.

A failed *degradable* dependency leaves the instance serving ("degraded"): the application has
fallbacks for it, and since every instance shares it, taking them all out of the load balancer
would turn a partial outage into a full one.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping

import structlog

from medlens.modules.health.metrics import DEPENDENCY_UP
from medlens.modules.health.schemas import (
    CheckStatus,
    DependencyCheck,
    ReadinessResponse,
    ReadinessStatus,
)

Probe = Callable[[], Awaitable[None]]

log = structlog.stdlib.get_logger(__name__)


class ReadinessService:
    def __init__(
        self,
        probes: Mapping[str, Probe],
        *,
        timeout_seconds: float,
        degradable: frozenset[str] = frozenset(),
    ) -> None:
        self._probes = dict(probes)
        self._timeout_seconds = timeout_seconds
        self._degradable = degradable

    async def check(self) -> ReadinessResponse:
        names = list(self._probes)
        results = await asyncio.gather(*(self._run(name) for name in names))
        checks = dict(zip(names, results, strict=True))
        for name, check in checks.items():
            DEPENDENCY_UP.labels(dependency=name).set(1 if check.status == "ok" else 0)
        failed = {name for name, check in checks.items() if check.status != "ok"}
        status: ReadinessStatus = "ready"
        if failed:
            status = "degraded" if failed <= self._degradable else "not_ready"
        return ReadinessResponse(status=status, checks=checks)

    async def _run(self, name: str) -> DependencyCheck:
        started = time.perf_counter()
        status: CheckStatus = "ok"
        error: str | None = None
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await self._probes[name]()
        except TimeoutError:
            status = "timeout"
        except Exception as exc:  # noqa: BLE001 - any failure means "not ready", reported below
            status, error = "error", type(exc).__name__
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        if status != "ok":
            log.warning(
                "health.dependency_unavailable", dependency=name, status=status, error=error
            )
        return DependencyCheck(status=status, latency_ms=latency_ms, error=error)

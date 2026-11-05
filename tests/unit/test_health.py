import asyncio

import httpx
import pytest
from fastapi import FastAPI
from prometheus_client import REGISTRY
from structlog.testing import capture_logs

from medlens.core.config import Settings
from medlens.main import create_app
from medlens.modules.health.deps import get_readiness_service
from medlens.modules.health.monitor import start_monitor, stop_monitor
from medlens.modules.health.service import ReadinessService


async def ok() -> None:
    return None


async def refused() -> None:
    raise ConnectionRefusedError("connection refused by 10.0.0.12")


async def hangs() -> None:
    await asyncio.sleep(10)


def use_probes(app: FastAPI, **probes) -> None:
    service = ReadinessService(probes, timeout_seconds=0.05, degradable=frozenset({"redis"}))
    app.dependency_overrides[get_readiness_service] = lambda: service


def up(dependency: str) -> float | None:
    return REGISTRY.get_sample_value("medlens_dependency_up", {"dependency": dependency})


async def test_ready_when_every_dependency_answers(app: FastAPI, client: httpx.AsyncClient) -> None:
    use_probes(app, postgres=ok, redis=ok, s3=ok)

    response = await client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert {name: check["status"] for name, check in body["checks"].items()} == {
        "postgres": "ok",
        "redis": "ok",
        "s3": "ok",
    }


async def test_not_ready_reports_each_dependency(app: FastAPI, client: httpx.AsyncClient) -> None:
    use_probes(app, postgres=ok, redis=hangs, s3=refused)

    with capture_logs() as logs:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    checks = response.json()["checks"]
    assert checks["postgres"]["status"] == "ok"
    assert checks["redis"]["status"] == "timeout"
    assert checks["redis"]["latency_ms"] == pytest.approx(50, abs=40)
    assert checks["s3"]["status"] == "error"
    assert checks["s3"]["error"] == "ConnectionRefusedError"
    assert "10.0.0.12" not in response.text
    assert {
        log["dependency"] for log in logs if log["event"] == "health.dependency_unavailable"
    } == {
        "redis",
        "s3",
    }


async def test_probes_run_concurrently() -> None:
    service = ReadinessService({"a": hangs, "b": hangs, "c": hangs}, timeout_seconds=0.1)

    started = asyncio.get_running_loop().time()
    report = await service.check()
    elapsed = asyncio.get_running_loop().time() - started

    assert report.status == "not_ready"
    assert elapsed < 0.25


async def test_redis_outage_is_degraded_but_still_serving(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    use_probes(app, postgres=ok, redis=refused, s3=ok)

    response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["redis"]["status"] == "error"


async def test_probes_set_the_dependency_gauge() -> None:
    service = ReadinessService({"gauge-up": ok, "gauge-down": refused}, timeout_seconds=0.05)

    await service.check()

    assert up("gauge-up") == 1
    assert up("gauge-down") == 0


async def test_the_monitor_probes_until_stopped() -> None:
    calls = 0

    async def counted() -> None:
        nonlocal calls
        calls += 1

    task = start_monitor(
        ReadinessService({"monitored": counted}, timeout_seconds=0.05), interval_seconds=0.01
    )
    await asyncio.sleep(0.05)
    await stop_monitor(task)
    seen = calls
    await asyncio.sleep(0.03)

    assert seen >= 2
    assert calls == seen
    assert task.cancelled()


async def test_the_app_runs_the_monitor_only_when_enabled(
    settings_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[float] = []
    monkeypatch.setattr(
        "medlens.main.start_monitor",
        lambda service, *, interval_seconds, queues: (
            started.append(interval_seconds) or asyncio.create_task(asyncio.sleep(3600))
        ),
    )
    for interval in (0.0, 7.5):
        settings: Settings = settings_factory(
            observability={"dependency_probe_interval_seconds": interval}
        )
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            pass

    assert started == [7.5]


async def test_metrics_are_exposed_for_the_scraper(client: httpx.AsyncClient) -> None:
    response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "medlens_auth_revocation_checks_skipped_total" in response.text

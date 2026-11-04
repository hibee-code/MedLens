import socket
from collections.abc import AsyncIterator

import httpx
import pytest
from botocore.exceptions import ClientError

from medlens.adapters.s3 import open_s3_client
from medlens.core.config import S3Settings
from medlens.main import create_app

pytestmark = pytest.mark.integration


async def ensure_bucket(s3: dict[str, str]) -> None:
    async with open_s3_client(S3Settings(**s3)) as client:
        try:
            await client.create_bucket(Bucket=s3["bucket"])  # type: ignore[attr-defined]
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                raise


def closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def client_for(settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


async def readiness(settings) -> httpx.Response:
    async for client in client_for(settings):
        return await client.get("/health/ready")
    raise AssertionError("unreachable")


async def test_ready_when_all_dependencies_are_up(settings_factory, live_groups) -> None:
    await ensure_bucket(live_groups["s3"])

    response = await readiness(settings_factory(**live_groups))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ready"
    assert {name: check["status"] for name, check in body["checks"].items()} == {
        "postgres": "ok",
        "redis": "ok",
        "s3": "ok",
    }


async def test_missing_bucket_is_not_ready(settings_factory, live_groups) -> None:
    live_groups["s3"]["bucket"] = "bucket-that-does-not-exist"

    response = await readiness(settings_factory(**live_groups))

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["s3"] == checks["s3"] | {"status": "error", "error": "ClientError"}
    assert checks["postgres"]["status"] == checks["redis"]["status"] == "ok"


async def test_unreachable_database_is_not_ready(settings_factory, live_groups) -> None:
    await ensure_bucket(live_groups["s3"])
    live_groups["database"] |= {"host": "127.0.0.1", "port": closed_port()}

    response = await readiness(settings_factory(**live_groups))

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["postgres"]["status"] in {"error", "timeout"}
    assert checks["redis"]["status"] == checks["s3"]["status"] == "ok"


async def test_pgbouncer_mode_still_queries(settings_factory, live_groups) -> None:
    """The transaction-mode connect options (no statement cache, unique names) are accepted."""
    await ensure_bucket(live_groups["s3"])
    live_groups["database"]["pgbouncer"] = True

    response = await readiness(settings_factory(**live_groups))

    assert response.json()["checks"]["postgres"]["status"] == "ok"


async def test_unreachable_redis_is_degraded_but_ready(settings_factory, live_groups) -> None:
    await ensure_bucket(live_groups["s3"])
    live_groups["redis"] |= {"host": "127.0.0.1", "port": closed_port()}

    response = await readiness(settings_factory(**live_groups))

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["redis"]["status"] in {"error", "timeout"}

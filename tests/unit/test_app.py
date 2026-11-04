import uuid

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from medlens.main import create_app
from medlens.resources import Resources
from tests.unit.test_config import PRODUCTION_SECRETS


async def test_app_factory_boots_and_manages_resources(app: FastAPI) -> None:
    async with app.router.lifespan_context(app):
        resources = app.state.resources
        assert isinstance(resources, Resources)
        assert isinstance(resources.db_engine, AsyncEngine)
        assert resources.s3_bucket == "medlens-local"

    assert not hasattr(app.state, "resources")


async def test_liveness(client: httpx.AsyncClient) -> None:
    response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["cache-control"] == "no-store"


async def test_request_id_is_generated_as_uuid7(client: httpx.AsyncClient) -> None:
    response = await client.get("/health/live")

    assert uuid.UUID(response.headers["x-request-id"]).version == 7


async def test_valid_request_id_is_echoed(client: httpx.AsyncClient) -> None:
    incoming = str(uuid.uuid4())

    response = await client.get("/health/live", headers={"X-Request-ID": incoming})

    assert response.headers["x-request-id"] == incoming


@pytest.mark.parametrize("incoming", ["not-a-uuid", "1; DROP TABLE scans", "a" * 500])
async def test_invalid_request_id_is_replaced(client: httpx.AsyncClient, incoming: str) -> None:
    response = await client.get("/health/live", headers={"X-Request-ID": incoming})

    returned = response.headers["x-request-id"]
    assert returned != incoming
    assert uuid.UUID(returned).version == 7


async def test_unknown_route_returns_problem_json(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "about:blank",
        "title": "Not Found",
        "status": 404,
        "instance": "/api/v1/does-not-exist",
        "request_id": response.headers["x-request-id"],
    }


async def test_wrong_method_returns_problem_json_with_allow(client: httpx.AsyncClient) -> None:
    response = await client.post("/health/live")

    assert response.status_code == 405
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["allow"] == "GET"


async def test_docs_served_outside_production(client: httpx.AsyncClient) -> None:
    assert (await client.get("/docs")).status_code == 200
    schema = (await client.get("/openapi.json")).json()
    assert "/health/ready" in schema["paths"]


async def test_docs_disabled_in_production(settings_factory) -> None:
    settings = settings_factory(app={"environment": "production"}, **PRODUCTION_SECRETS)
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert (await client.get(path)).status_code == 404


async def test_cors_allows_configured_origin_only(settings_factory) -> None:
    app = create_app(settings_factory(app={"cors_origins": ["https://medlens.example"]}))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        allowed = await client.get("/health/live", headers={"Origin": "https://medlens.example"})
        other = await client.get("/health/live", headers={"Origin": "https://evil.example"})

    assert allowed.headers["access-control-allow-origin"] == "https://medlens.example"
    assert "X-Request-ID" in allowed.headers["access-control-expose-headers"]
    assert "access-control-allow-origin" not in other.headers

"""Error mapping and middleware, exercised through a throwaway router mounted on the real app."""

import json
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI, Query, Request
from structlog.testing import capture_logs

from medlens.core.config import Settings
from medlens.core.errors import (
    Conflict,
    DependencyUnavailable,
    DomainError,
    Forbidden,
    NotFound,
    RateLimited,
    Unauthorized,
    ValidationFailed,
)
from medlens.core.routing import api_v1_router
from medlens.main import create_app

DOMAIN_ERRORS: dict[str, DomainError] = {
    "not-found": NotFound("Scan not found."),
    "conflict": Conflict("Idempotency key reused with a different body."),
    "forbidden": Forbidden(),
    "unauthorized": Unauthorized(),
    "validation-failed": ValidationFailed(
        "Batch rows are invalid.", errors=[{"row": 3, "msg": "expiry before manufacture"}]
    ),
    "rate-limited": RateLimited(retry_after_seconds=30),
    "dependency-unavailable": DependencyUnavailable(dependency="ocr", retry_after_seconds=5),
}


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    app = create_app(settings)
    router = api_v1_router("/_test", tags=["test"])

    @router.get("/domain/{slug}")
    async def raise_domain_error(slug: str) -> None:
        raise DOMAIN_ERRORS[slug]

    @router.get("/boom")
    async def boom() -> None:
        raise RuntimeError("password=hunter2 leaked into an exception message")

    @router.get("/items/{item_id}")
    async def get_item(item_id: int, limit: int = Query(ge=1, le=10)) -> dict[str, int]:
        return {"item_id": item_id, "limit": limit}

    @router.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    app.include_router(router)
    return app


@pytest.mark.parametrize("slug", DOMAIN_ERRORS)
async def test_domain_errors_render_as_problem_json(client: httpx.AsyncClient, slug: str) -> None:
    error = DOMAIN_ERRORS[slug]

    response = await client.get(f"/api/v1/_test/domain/{slug}")

    assert response.status_code == error.status_code
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["type"] == f"urn:medlens:problem:{slug}"
    assert body["title"] == error.title
    assert body["status"] == error.status_code
    assert body["instance"] == f"/api/v1/_test/domain/{slug}"
    assert body["request_id"] == response.headers["x-request-id"]


async def test_error_specific_headers_and_extensions(client: httpx.AsyncClient) -> None:
    unauthorized = await client.get("/api/v1/_test/domain/unauthorized")
    limited = await client.get("/api/v1/_test/domain/rate-limited")
    invalid = await client.get("/api/v1/_test/domain/validation-failed")
    unavailable = await client.get("/api/v1/_test/domain/dependency-unavailable")

    assert unauthorized.headers["www-authenticate"] == "Bearer"
    assert limited.headers["retry-after"] == "30"
    assert invalid.json()["errors"] == [{"row": 3, "msg": "expiry before manufacture"}]
    assert unavailable.headers["retry-after"] == "5"
    assert "ocr" not in unavailable.text


async def test_request_validation_error_lists_fields_without_input(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/api/v1/_test/items/abc", params={"limit": 999})

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["type"] == "urn:medlens:problem:validation-failed"
    assert {tuple(error["loc"]) for error in body["errors"]} == {
        ("path", "item_id"),
        ("query", "limit"),
    }
    rendered_errors = json.dumps(body["errors"])
    assert "abc" not in rendered_errors
    assert "999" not in rendered_errors


async def test_unhandled_exception_is_logged_and_hidden(client: httpx.AsyncClient) -> None:
    with capture_logs() as logs:
        response = await client.get("/api/v1/_test/boom")

    assert response.status_code == 500
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["title"] == "Internal Server Error"
    assert body["request_id"] == response.headers["x-request-id"]
    assert "hunter2" not in response.text
    assert "Traceback" not in response.text

    [entry] = [log for log in logs if log["event"] == "http.unhandled_exception"]
    assert entry["log_level"] == "error"
    assert entry["exc_info"] is True
    assert entry["route"] == "/api/v1/_test/boom"


async def test_access_log_uses_route_template(client: httpx.AsyncClient) -> None:
    with capture_logs() as logs:
        await client.get("/api/v1/_test/items/7", params={"limit": 2})
        await client.get("/nowhere/123")

    access = [(log["route"], log["status"]) for log in logs if log["event"] == "http.request"]
    assert access == [("/api/v1/_test/items/{item_id}", 200), ("<unmatched>", 404)]


async def test_body_within_limit_is_accepted(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/_test/echo", content=b"x" * 1024)

    assert response.status_code == 200
    assert response.json() == {"size": 1024}


async def test_body_over_declared_limit_is_rejected(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/_test/echo", content=b"x" * 1025)

    assert response.status_code == 413
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["type"] == "urn:medlens:problem:payload-too-large"
    assert response.headers["x-request-id"]


async def test_streamed_body_over_limit_is_rejected(client: httpx.AsyncClient) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(4):
            yield b"x" * 512  # no Content-Length: the limit is enforced while reading

    response = await client.post("/api/v1/_test/echo", content=chunks())

    assert response.status_code == 413
    assert response.json()["detail"] == "The request body exceeds the limit of 1024 bytes."

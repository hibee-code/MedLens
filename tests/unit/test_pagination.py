import base64
import json
from datetime import UTC, datetime, timedelta, timezone
from typing import Annotated
from uuid import UUID

import httpx
import pytest
from fastapi import Depends, FastAPI

from medlens.core.config import Settings
from medlens.core.ids import uuid7
from medlens.core.pagination import (
    Cursor,
    InvalidCursorError,
    Page,
    PageParams,
    build_page,
    decode_cursor,
    encode_cursor,
    page_params,
)
from medlens.core.routing import api_v1_router
from medlens.core.schemas import ReadModel
from medlens.main import create_app


def test_cursor_round_trip() -> None:
    cursor = Cursor(created_at=datetime(2026, 11, 2, 15, 4, 5, 123456, tzinfo=UTC), id=uuid7())

    token = encode_cursor(cursor)

    assert decode_cursor(token) == cursor
    assert set(token) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def test_cursor_normalises_timezones_to_utc() -> None:
    lagos = timezone(timedelta(hours=1))
    cursor = Cursor(created_at=datetime(2026, 11, 2, 16, 0, tzinfo=lagos), id=uuid7())

    decoded = decode_cursor(encode_cursor(cursor))

    assert decoded.created_at == datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
    assert decoded.created_at.tzinfo is UTC


def test_naive_timestamps_cannot_be_encoded() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        encode_cursor(Cursor(created_at=datetime(2026, 11, 2), id=uuid7()))  # noqa: DTZ001


def _token(payload: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


@pytest.mark.parametrize(
    "token",
    [
        "not-base64!!",
        "e30",  # {}
        _token([1, 2, 3]),
        _token({"v": 2, "t": "2026-11-02T15:00:00+00:00", "id": str(uuid7())}),
        _token({"v": 1, "t": "2026-11-02T15:00:00", "id": str(uuid7())}),  # naive
        _token({"v": 1, "t": "yesterday", "id": str(uuid7())}),
        _token({"v": 1, "t": "2026-11-02T15:00:00+00:00", "id": "not-a-uuid"}),
        base64.urlsafe_b64encode(b"\xff\xfe").decode(),
    ],
)
def test_malformed_cursors_are_rejected(token: str) -> None:
    with pytest.raises(InvalidCursorError):
        decode_cursor(token)


class Item(ReadModel):
    id: UUID
    created_at: datetime


def _rows(count: int) -> list[Item]:
    start = datetime(2026, 11, 2, tzinfo=UTC)
    return [Item(id=uuid7(), created_at=start - timedelta(minutes=i)) for i in range(count)]


def _cursor_of(item: Item) -> Cursor:
    return Cursor(created_at=item.created_at, id=item.id)


def test_build_page_with_more_rows() -> None:
    rows = _rows(4)  # limit + 1 fetched

    page = build_page(rows, limit=3, cursor_of=_cursor_of, to_item=lambda row: row)

    assert page.items == rows[:3]
    assert page.has_more is True
    assert page.next_cursor is not None
    assert decode_cursor(page.next_cursor) == _cursor_of(rows[2])


@pytest.mark.parametrize("count", [0, 2, 3])
def test_build_page_last_page(count: int) -> None:
    page = build_page(_rows(count), limit=3, cursor_of=_cursor_of, to_item=lambda row: row)

    assert len(page.items) == count
    assert page.has_more is False
    assert page.next_cursor is None


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    app = create_app(settings)
    router = api_v1_router("/_test", tags=["products"])

    @router.get("/items", response_model=Page[Item])
    async def list_items(params: Annotated[PageParams, Depends(page_params)]) -> Page[Item]:
        rows = _rows(params.limit + 1)
        return build_page(rows, limit=params.limit, cursor_of=_cursor_of, to_item=lambda r: r)

    @router.get("/params")
    async def echo_params(params: Annotated[PageParams, Depends(page_params)]) -> dict[str, str]:
        return {"limit": str(params.limit), "after": str(params.after.id if params.after else "")}

    app.include_router(router)
    return app


async def test_page_endpoint_defaults_and_shape(client: httpx.AsyncClient) -> None:
    body = (await client.get("/api/v1/_test/items")).json()

    assert set(body) == {"items", "next_cursor", "has_more"}
    assert len(body["items"]) == 20
    assert body["has_more"] is True


async def test_cursor_is_accepted_by_the_dependency(client: httpx.AsyncClient) -> None:
    cursor = Cursor(created_at=datetime(2026, 11, 2, tzinfo=UTC), id=uuid7())

    response = await client.get(
        "/api/v1/_test/params", params={"limit": 5, "cursor": encode_cursor(cursor)}
    )

    assert response.json() == {"limit": "5", "after": str(cursor.id)}


@pytest.mark.parametrize("limit", [0, 101, -1])
async def test_limit_out_of_range_is_rejected(client: httpx.AsyncClient, limit: int) -> None:
    response = await client.get("/api/v1/_test/items", params={"limit": limit})

    assert response.status_code == 422
    assert response.json()["errors"][0]["loc"] == ["query", "limit"]


async def test_invalid_cursor_is_a_validation_problem(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/_test/items", params={"cursor": "garbage"})

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["errors"] == [
        {"loc": ["query", "cursor"], "msg": "Invalid cursor", "type": "cursor"}
    ]

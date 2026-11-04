"""Cursor (keyset) pagination for every list endpoint.

Lists are ordered newest first by ``(created_at, id)``. The cursor is an opaque, URL-safe token
holding the position of the last item returned. The next page is the rows strictly before it:

    WHERE (created_at, id) < (:created_at, :id) ORDER BY created_at DESC, id DESC LIMIT :limit + 1

Fetch one row more than the limit and pass everything to ``build_page``; the extra row only
tells us whether another page exists. Tampering with a cursor only moves the position within
results the caller may already see (tenant filters are applied by the repository regardless).
"""

import base64
import binascii
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import Query
from pydantic import Field

from medlens.core.errors import ValidationFailed
from medlens.core.schemas import ReadModel

DEFAULT_LIMIT = 20
MAX_LIMIT = 100
_CURSOR_VERSION = 1
_MAX_CURSOR_LENGTH = 200


@dataclass(frozen=True, slots=True)
class Cursor:
    created_at: datetime
    id: UUID


class InvalidCursorError(ValueError):
    pass


def encode_cursor(cursor: Cursor) -> str:
    if cursor.created_at.tzinfo is None:
        raise ValueError("cursor timestamps must be timezone-aware")
    payload = {
        "v": _CURSOR_VERSION,
        "t": cursor.created_at.astimezone(UTC).isoformat(),
        "id": str(cursor.id),
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def decode_cursor(token: str) -> Cursor:
    try:
        padded = token + "=" * (-len(token) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
        version, created_at, cursor_id = (
            payload["v"],
            datetime.fromisoformat(payload["t"]),
            UUID(payload["id"]),
        )
    # JSON and Unicode decode errors are ValueErrors; TypeError covers a non-object payload.
    except (binascii.Error, ValueError, KeyError, TypeError) as exc:
        raise InvalidCursorError("malformed cursor") from exc
    if version != _CURSOR_VERSION:
        raise InvalidCursorError("unsupported cursor version")
    if created_at.tzinfo is None:
        raise InvalidCursorError("cursor timestamp has no timezone")
    return Cursor(created_at=created_at.astimezone(UTC), id=cursor_id)


@dataclass(frozen=True, slots=True)
class PageParams:
    limit: int
    after: Cursor | None


def page_params(
    limit: Annotated[
        int, Query(ge=1, le=MAX_LIMIT, description=f"Items per page (1-{MAX_LIMIT}).")
    ] = DEFAULT_LIMIT,
    cursor: Annotated[
        str | None,
        Query(
            max_length=_MAX_CURSOR_LENGTH,
            description="Opaque `next_cursor` from the previous page. Omit for the first page.",
        ),
    ] = None,
) -> PageParams:
    """FastAPI dependency for list endpoints."""
    if cursor is None:
        return PageParams(limit=limit, after=None)
    try:
        return PageParams(limit=limit, after=decode_cursor(cursor))
    except InvalidCursorError:
        raise ValidationFailed(
            "The pagination cursor is invalid. Start again without a cursor.",
            errors=[{"loc": ["query", "cursor"], "msg": "Invalid cursor", "type": "cursor"}],
        ) from None


class Page[ItemT](ReadModel):
    """One page of a list. Pass ``next_cursor`` back as ``cursor`` to get the next page."""

    items: list[ItemT]
    next_cursor: str | None = Field(description="Cursor for the next page; null on the last.")
    has_more: bool = Field(description="Whether another page exists.")


def build_page[RowT, ItemT](
    rows: Sequence[RowT],
    *,
    limit: int,
    cursor_of: Callable[[RowT], Cursor],
    to_item: Callable[[RowT], ItemT],
) -> Page[ItemT]:
    """Turn ``limit + 1`` fetched rows into a page."""
    has_more = len(rows) > limit
    page_rows = rows[:limit]
    next_cursor = encode_cursor(cursor_of(page_rows[-1])) if has_more and page_rows else None
    return Page(
        items=[to_item(row) for row in page_rows], next_cursor=next_cursor, has_more=has_more
    )

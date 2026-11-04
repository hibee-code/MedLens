"""Idempotency-Key handling (draft-ietf-httpapi-idempotency-key-header).

The key is claimed in the same transaction as the work it protects, so a retry either replays
the stored response or waits for the first request to commit. The same key with a different
request body is a 409; keys are kept for ``IDEMPOTENCY_TTL`` and purged afterwards.
"""

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from medlens.core.errors import IdempotencyKeyReused, ValidationFailed
from medlens.db.models import IdempotencyKey

IDEMPOTENCY_TTL = timedelta(hours=24)
_KEY = re.compile(r"^[A-Za-z0-9._:\-]{8,128}$")


@dataclass(frozen=True, slots=True)
class StoredResponse:
    status: int
    body: dict[str, Any]


def validate_key(key: str) -> str:
    if not _KEY.fullmatch(key):
        raise ValidationFailed(
            "Idempotency-Key must be 8-128 letters, digits or . _ : -",
            errors=[{"loc": ["header", "Idempotency-Key"], "msg": "invalid", "type": "value"}],
        )
    return key


def request_fingerprint(*parts: str | bytes | None) -> str:
    digest = hashlib.sha256()
    for part in parts:
        data = part.encode() if isinstance(part, str) else (part or b"")
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


async def claim(
    session: AsyncSession, *, scope: str, key: str, request_hash: str
) -> StoredResponse | None:
    """None: the key is new and now held by this transaction. Otherwise the stored response
    to replay. Raises IdempotencyKeyReused for a different request or one still running."""
    now = datetime.now(UTC)
    inserted = await session.scalar(
        insert(IdempotencyKey)
        .values(scope=scope, key=key, request_hash=request_hash, expires_at=now + IDEMPOTENCY_TTL)
        .on_conflict_do_nothing(index_elements=["scope", "key"])
        .returning(IdempotencyKey.key)
    )
    if inserted is not None:
        return None
    existing = await session.scalar(
        select(IdempotencyKey).where(IdempotencyKey.scope == scope, IdempotencyKey.key == key)
    )
    if existing is None or existing.expires_at <= now:
        raise IdempotencyKeyReused("This Idempotency-Key has expired; use a new one.")
    if existing.request_hash != request_hash:
        raise IdempotencyKeyReused(
            "This Idempotency-Key was already used with a different request."
        )
    if existing.response_status is None or existing.response_body is None:
        raise IdempotencyKeyReused("A request with this Idempotency-Key is still in progress.")
    return StoredResponse(existing.response_status, existing.response_body)


async def store(
    session: AsyncSession, *, scope: str, key: str, status: int, body: dict[str, Any]
) -> None:
    await session.execute(
        update(IdempotencyKey)
        .where(IdempotencyKey.scope == scope, IdempotencyKey.key == key)
        .values(response_status=status, response_body=body)
    )

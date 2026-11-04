"""Transactional outbox (CLAUDE.md §3): side effects that must survive a crash are written as
rows in the caller's transaction, then relayed to Celery (``queue/relay.py``). An event exists
if and only if the change it announces was committed.

Each event also records the request context (request id, user, organization) of the request
that caused it, so the task it triggers logs under the same request id."""

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from medlens.core.context import snapshot
from medlens.db.models import OutboxEvent

# Event types; the relay maps each to a task.
SCAN_RECEIVED = "scan.received"
SCAN_COMPLETED = "scan.completed"
REFERENCE_IMAGE_UPLOADED = "reference_image.uploaded"
SESSIONS_REVOKED = "auth.sessions_revoked"
KNOWLEDGE_DOCUMENT_UPLOADED = "knowledge.document_uploaded"


def _event(
    event_type: str, aggregate_type: str, aggregate_id: UUID, payload: dict[str, Any]
) -> OutboxEvent:
    context = snapshot()
    return OutboxEvent(
        event_type=event_type,
        aggregate_type=aggregate_type,
        aggregate_id=str(aggregate_id),
        payload={**payload, "context": context} if context else payload,
    )


async def emit(
    session: AsyncSession,
    event_type: str,
    *,
    aggregate_type: str,
    aggregate_id: UUID,
    payload: dict[str, Any] | None = None,
) -> None:
    session.add(_event(event_type, aggregate_type, aggregate_id, payload or {}))


def emit_sync(
    session: Session,
    event_type: str,
    *,
    aggregate_type: str,
    aggregate_id: UUID,
    payload: dict[str, Any] | None = None,
) -> None:
    session.add(_event(event_type, aggregate_type, aggregate_id, payload or {}))

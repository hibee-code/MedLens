"""The outbox relay: moves committed outbox events to Celery.

Run by Beat every second. Each run claims a batch of due events with
``SELECT ... FOR UPDATE SKIP LOCKED`` (so concurrent relays never take the same event), hands
each to ``dispatch`` and marks it published, all in one transaction. A failed dispatch is
retried with exponential backoff; after ``max_attempts`` the event is dead-lettered and kept
for inspection.

Delivery to Celery is at least once (a crash between dispatch and commit re-sends the event),
so every task is idempotent. From the database's point of view each event is published exactly
once.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from medlens.db.models import OutboxEvent

log = structlog.stdlib.get_logger(__name__)

BACKOFF_BASE_SECONDS = 2
BACKOFF_MAX_SECONDS = 600
ERROR_MAX_LENGTH = 500

Dispatch = Callable[[OutboxEvent], None]


@dataclass(frozen=True, slots=True)
class RelayResult:
    published: int = 0
    failed: int = 0
    dead_lettered: int = 0


def backoff(attempts: int) -> timedelta:
    """2 s, 4 s, 8 s ... capped at 10 minutes."""
    return timedelta(seconds=min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS**attempts))


def relay_once(
    session: Session,
    dispatch: Dispatch,
    *,
    batch_size: int = 100,
    max_attempts: int = 10,
    now: datetime | None = None,
) -> RelayResult:
    """One relay run inside the caller's transaction (commit afterwards)."""
    now = now or datetime.now(UTC)
    events = session.scalars(
        select(OutboxEvent)
        .where(
            OutboxEvent.published_at.is_(None),
            OutboxEvent.dead_lettered_at.is_(None),
            or_(OutboxEvent.next_attempt_at.is_(None), OutboxEvent.next_attempt_at <= now),
        )
        .order_by(OutboxEvent.created_at, OutboxEvent.id)
        .limit(batch_size)
        .with_for_update(skip_locked=True)
    ).all()
    published = failed = dead = 0
    for event in events:
        try:
            dispatch(event)
        except Exception as exc:  # noqa: BLE001 - any dispatch error is retried, then dead-lettered
            event.attempts += 1
            event.last_error = f"{type(exc).__name__}: {exc}"[:ERROR_MAX_LENGTH]
            if event.attempts >= max_attempts:
                event.dead_lettered_at = now
                dead += 1
                log.error(
                    "outbox.dead_lettered",
                    event_id=str(event.id),
                    event_type=event.event_type,
                    attempts=event.attempts,
                )
            else:
                event.next_attempt_at = now + backoff(event.attempts)
                failed += 1
                log.warning(
                    "outbox.dispatch_failed",
                    event_id=str(event.id),
                    event_type=event.event_type,
                    attempts=event.attempts,
                    error=type(exc).__name__,
                )
            continue
        event.published_at = now
        published += 1
    if events:
        log.info("outbox.relayed", published=published, failed=failed, dead_lettered=dead)
    return RelayResult(published, failed, dead)

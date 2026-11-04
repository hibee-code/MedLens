"""Cross-cutting infrastructure tables (not owned by a domain module)."""

from datetime import datetime
from typing import Any

from sqlalchemy import Index, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from medlens.db.base import Base, CreatedAt, Entity, UUIDv7PrimaryKey


class OutboxEvent(UUIDv7PrimaryKey, CreatedAt, Base):
    """Transactional outbox: written in the same transaction as the change it announces and
    relayed to Celery afterwards (Phase 6), so no event is lost or sent for a rolled-back
    change."""

    __tablename__ = "outbox_events"
    __table_args__ = (
        Index(
            "ix_outbox_events_unpublished",
            "created_at",
            postgresql_where=text("published_at IS NULL AND dead_lettered_at IS NULL"),
        ),
    )
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012

    aggregate_type: Mapped[str] = mapped_column(String(64))
    aggregate_id: Mapped[str] = mapped_column(String(64))
    event_type: Mapped[str] = mapped_column(String(128))  # e.g. scan.received
    payload: Mapped[dict[str, Any]] = mapped_column(
        default=dict, server_default=text("'{}'::jsonb")
    )
    published_at: Mapped[datetime | None]
    attempts: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    last_error: Mapped[str | None] = mapped_column(Text)
    next_attempt_at: Mapped[datetime | None]  # backoff after a failed dispatch
    dead_lettered_at: Mapped[datetime | None]  # gave up after too many attempts


class IdempotencyKey(CreatedAt, Base):
    """Stored response for an Idempotency-Key, per caller. A retry with the same key and body
    replays the response; the same key with a different body is a 409."""

    __tablename__ = "idempotency_keys"
    __table_args__ = (Index("ix_idempotency_keys_expires_at", "expires_at"),)

    # Whose key it is: a user id or a device fingerprint hash, prefixed with "user:" / "device:".
    scope: Mapped[str] = mapped_column(String(80), primary_key=True)
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    response_status: Mapped[int | None]  # null while the first request is still running
    response_body: Mapped[dict[str, Any] | None]
    expires_at: Mapped[datetime]


class FailedTask(Entity):
    """A background task that failed after all its retries: one row per task name and
    arguments, kept for inspection and requeueing (workers/failures.py)."""

    __tablename__ = "failed_tasks"
    __table_args__ = (UniqueConstraint("task_name", "args_hash"),)

    task_name: Mapped[str] = mapped_column(String(200))
    task_id: Mapped[str | None] = mapped_column(String(64))
    args: Mapped[list[Any]] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))
    kwargs: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=text("'{}'::jsonb"))
    args_hash: Mapped[str] = mapped_column(String(64))
    error_type: Mapped[str] = mapped_column(String(200))
    error_message: Mapped[str | None] = mapped_column(Text)
    traceback_hash: Mapped[str | None] = mapped_column(String(64))
    failure_count: Mapped[int] = mapped_column(default=1, server_default=text("1"))
    first_failed_at: Mapped[datetime]
    last_failed_at: Mapped[datetime]
    requeued_at: Mapped[datetime | None]
    requeue_count: Mapped[int] = mapped_column(default=0, server_default=text("0"))

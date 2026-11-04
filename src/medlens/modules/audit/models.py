from typing import Any
from uuid import UUID

from sqlalchemy import Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from medlens.db.base import Base, CreatedAt, UUIDv7PrimaryKey


class AuditLog(UUIDv7PrimaryKey, CreatedAt, Base):
    """Append-only record of security-relevant actions.

    UPDATE, DELETE and TRUNCATE are blocked by a trigger and not granted to the app role (see
    the migration). Actor ids are plain columns, not foreign keys: a cascading SET NULL would
    be an UPDATE, which the trigger forbids, and the log must outlive the users it mentions.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_created_at", "created_at"),
        Index("ix_audit_logs_actor_user_id_created_at", "actor_user_id", "created_at"),
        Index("ix_audit_logs_resource_type_resource_id", "resource_type", "resource_id"),
    )
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012

    actor_user_id: Mapped[UUID | None]
    actor_org_id: Mapped[UUID | None]
    action: Mapped[str] = mapped_column(String(64))  # e.g. auth.login_failed
    resource_type: Mapped[str | None] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(String(64))
    ip_hash: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(Text)
    request_id: Mapped[str | None] = mapped_column(String(64))
    # "metadata" is reserved on declarative classes, hence the trailing underscore.
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", default=dict, server_default=text("'{}'::jsonb")
    )

"""Declarative base, constraint naming convention and column mixins.

The naming convention gives every index and constraint a deterministic name, so Alembic diffs
are stable and migrations can drop constraints by name.
"""

from datetime import date, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import ARRAY, Date, DateTime, MetaData, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from medlens.core.ids import uuid7

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {  # noqa: RUF012 - SQLAlchemy reads this class attribute
        datetime: DateTime(timezone=True),  # timestamptz, always UTC
        date: Date(),
        dict[str, Any]: JSONB(),
        list[str]: ARRAY(Text()),
    }


class UUIDv7PrimaryKey:
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)


class CreatedAt:
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Timestamps(CreatedAt):
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class SoftDelete:
    deleted_at: Mapped[datetime | None] = mapped_column(default=None)


class Entity(UUIDv7PrimaryKey, Timestamps, Base):
    """Base for tables with a UUIDv7 id and created/updated timestamps."""

    __abstract__ = True
    # Read server-generated values (timestamps) back with RETURNING on INSERT and UPDATE, so
    # async code never triggers a lazy load to refresh them.
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012

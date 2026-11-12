"""Monthly RANGE partitions on ``created_at`` (UTC months).

Used by the scans migration and, from Phase 6, by the daily Beat task that keeps partitions
ahead of time. Each month is ``<table>_yYYYYmMM``; ``<table>_default`` catches anything outside.

Create partitions *before* their month starts: once the default partition holds rows for a
month, creating that month's partition fails until those rows are moved out.
"""

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import ColumnElement
from sqlalchemy.orm import InstrumentedAttribute

from medlens.core.ids import uuid7_time

PARTITION_MONTHS_AHEAD = 3
# created_at and the UUIDv7 id are taken from the same clock when a row is made; the margin only
# absorbs clock adjustments between the two calls.
ID_TIME_MARGIN = timedelta(minutes=5)


def id_window(
    id_column: InstrumentedAttribute[UUID],
    created_column: InstrumentedAttribute[Any],
    row_id: UUID,
) -> ColumnElement[bool]:
    """``id = row_id`` plus a ``created_at`` window around the time encoded in the UUIDv7, so
    Postgres prunes a lookup by id to one monthly partition and uses the primary key."""
    try:
        made = uuid7_time(row_id)
    except ValueError:
        return id_column == row_id  # not a UUIDv7 of ours: a full search finds nothing anyway
    return (
        (id_column == row_id)
        & (created_column >= made - ID_TIME_MARGIN)
        & (created_column <= made + ID_TIME_MARGIN)
    )


def month_start(day: date) -> date:
    return day.replace(day=1)


def add_months(month: date, count: int) -> date:
    index = month.year * 12 + (month.month - 1) + count
    return date(index // 12, index % 12 + 1, 1)


def partition_name(table: str, month: date) -> str:
    return f"{table}_y{month.year}m{month.month:02d}"


def monthly_partition_ddl(table: str, month: date) -> str:
    start, end = month_start(month), add_months(month_start(month), 1)
    return (
        f'CREATE TABLE IF NOT EXISTS "{partition_name(table, start)}" PARTITION OF "{table}" '
        f"FOR VALUES FROM ('{start.isoformat()} 00:00:00+00') TO ('{end.isoformat()} 00:00:00+00')"
    )


def default_partition_ddl(table: str) -> str:
    return f'CREATE TABLE IF NOT EXISTS "{table}_default" PARTITION OF "{table}" DEFAULT'


def ensure_monthly_partitions(
    execute: Callable[[str], object],
    table: str,
    *,
    today: date,
    months_ahead: int = PARTITION_MONTHS_AHEAD,
) -> list[str]:
    """Create the current month's partition and the next ``months_ahead``. Idempotent."""
    first = month_start(today)
    months = [add_months(first, offset) for offset in range(months_ahead + 1)]
    for month in months:
        execute(monthly_partition_ddl(table, month))
    return [partition_name(table, month) for month in months]

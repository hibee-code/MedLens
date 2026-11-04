"""Identifier helpers.

IDs are UUIDv7 (RFC 9562): a 48-bit Unix millisecond timestamp followed by random bits, so new
rows land at the end of B-tree indexes and IDs sort by creation time.
"""

import os
import time
import uuid
from datetime import UTC, datetime

_UNIX_MS_MASK = (1 << 48) - 1
_RAND_B_MASK = (1 << 62) - 1


def uuid7() -> uuid.UUID:
    """Return a new UUIDv7 (Python 3.12 has no ``uuid.uuid7``)."""
    unix_ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")  # 80 random bits
    rand_a = rand >> 68  # top 12 bits
    rand_b = rand & _RAND_B_MASK  # low 62 bits
    value = (
        (unix_ms & _UNIX_MS_MASK) << 80
        | 0x7 << 76  # version
        | rand_a << 64
        | 0b10 << 62  # RFC 9562 variant
        | rand_b
    )
    return uuid.UUID(int=value)


def uuid7_time(value: uuid.UUID) -> datetime:
    """When a UUIDv7 was made (millisecond precision). Lets queries on tables partitioned by
    ``created_at`` prune to one partition when only the id is known."""
    if value.version != 7:
        raise ValueError("not a UUIDv7")
    return datetime.fromtimestamp((value.int >> 80) / 1000, UTC)

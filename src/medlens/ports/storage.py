from typing import Protocol


class StorageUnavailableError(Exception):
    """Object storage could not be reached (after retries) or its circuit breaker is open."""


class ObjectNotFoundError(Exception):
    """No object under that key."""


class StoragePort(Protocol):
    """Object storage for the API (async). Keys are opaque; callers build them."""

    async def put(self, key: str, data: bytes, *, content_type: str) -> None: ...

    async def get(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None: ...


class SyncStoragePort(Protocol):
    """Object storage for Celery workers, which are synchronous (CLAUDE.md §7)."""

    def get(self, key: str) -> bytes: ...

    def put(self, key: str, data: bytes, *, content_type: str) -> None: ...

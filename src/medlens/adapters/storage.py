"""Object storage on S3-compatible services: ``S3Storage`` (async, API) and ``SyncS3Storage``
(boto3, workers).

botocore retries transient errors itself (standard mode, ``MEDLENS_S3__MAX_ATTEMPTS``) within the
client's connect and read timeouts. A circuit breaker per bucket opens after five failed calls,
so an outage fails fast with ``StorageUnavailableError``; a missing key is
``ObjectNotFoundError`` and does not count as a failure.
"""

from typing import Any

import structlog
from botocore.exceptions import BotoCoreError, ClientError
from purgatory import AsyncCircuitBreakerFactory, SyncCircuitBreakerFactory
from purgatory.domain.model import OpenedState

from medlens.adapters.s3 import S3Client
from medlens.ports.storage import ObjectNotFoundError, StorageUnavailableError

log = structlog.stdlib.get_logger(__name__)

BREAKER_THRESHOLD = 5
BREAKER_RESET_SECONDS = 30
_MISSING = frozenset({"NoSuchKey", "404", "NotFound"})


def _missing(error: ClientError) -> bool:
    return str(error.response.get("Error", {}).get("Code")) in _MISSING


_EXCLUDE: list[Any] = [ObjectNotFoundError]


class S3Storage:
    def __init__(
        self,
        client: S3Client,
        bucket: str,
        breakers: AsyncCircuitBreakerFactory | None = None,
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._breakers = breakers or AsyncCircuitBreakerFactory(
            default_threshold=BREAKER_THRESHOLD, default_ttl=BREAKER_RESET_SECONDS
        )

    async def put(self, key: str, data: bytes, *, content_type: str) -> None:
        await self._call("put_object", Key=key, Body=data, ContentType=content_type)

    async def get(self, key: str) -> bytes:
        response = await self._call("get_object", Key=key)
        async with response["Body"] as stream:
            return bytes(await stream.read())

    async def delete(self, key: str) -> None:
        await self._call("delete_object", Key=key)

    async def _call(self, operation: str, **kwargs: Any) -> Any:
        try:
            async with await self._breakers.get_breaker(f"s3:{self._bucket}", exclude=_EXCLUDE):
                try:
                    return await getattr(self._client, operation)(Bucket=self._bucket, **kwargs)
                except ClientError as exc:
                    if _missing(exc):
                        raise ObjectNotFoundError(kwargs.get("Key", "")) from None
                    raise
        except OpenedState:
            raise StorageUnavailableError("object storage circuit breaker is open") from None
        except (BotoCoreError, ClientError) as exc:
            log.warning("storage.unavailable", operation=operation, error=type(exc).__name__)
            raise StorageUnavailableError(type(exc).__name__) from None


class SyncS3Storage:
    """For Celery workers. ``client`` is a boto3 S3 client."""

    def __init__(
        self,
        client: Any,
        bucket: str,
        breakers: SyncCircuitBreakerFactory | None = None,
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._breakers = breakers or SyncCircuitBreakerFactory(
            default_threshold=BREAKER_THRESHOLD, default_ttl=BREAKER_RESET_SECONDS
        )

    def get(self, key: str) -> bytes:
        response = self._call("get_object", Key=key)
        return bytes(response["Body"].read())

    def put(self, key: str, data: bytes, *, content_type: str) -> None:
        self._call("put_object", Key=key, Body=data, ContentType=content_type)

    def _call(self, operation: str, **kwargs: Any) -> Any:
        try:
            with self._breakers.get_breaker(f"s3:{self._bucket}", exclude=_EXCLUDE):
                try:
                    return getattr(self._client, operation)(Bucket=self._bucket, **kwargs)
                except ClientError as exc:
                    if _missing(exc):
                        raise ObjectNotFoundError(kwargs.get("Key", "")) from None
                    raise
        except OpenedState:
            raise StorageUnavailableError("object storage circuit breaker is open") from None
        except (BotoCoreError, ClientError) as exc:
            log.warning("storage.unavailable", operation=operation, error=type(exc).__name__)
            raise StorageUnavailableError(type(exc).__name__) from None

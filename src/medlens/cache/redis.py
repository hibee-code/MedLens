"""Redis clients. Cache, Celery broker and Celery results live in separate logical databases.

Every client gets the same bounded failure behaviour: short timeouts and at most
MEDLENS_REDIS__RETRIES quick retries. redis-py's default (ten retries with backoff) would turn a
Redis outage into multi-second requests, although the callers are built to fail open.
"""

from redis import Redis as SyncRedis
from redis.asyncio import Redis
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import ExponentialWithJitterBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.retry import Retry

from medlens.core.config import RedisSettings

RETRY_ON = (RedisConnectionError, RedisTimeoutError)
_BACKOFF_BASE_SECONDS = 0.01
_BACKOFF_CAP_SECONDS = 0.1


def _backoff() -> ExponentialWithJitterBackoff:
    return ExponentialWithJitterBackoff(base=_BACKOFF_BASE_SECONDS, cap=_BACKOFF_CAP_SECONDS)


def create_redis_client(settings: RedisSettings, *, db: int) -> Redis:
    password = settings.password.get_secret_value() if settings.password else None
    return Redis(
        host=settings.host,
        port=settings.port,
        db=db,
        password=password,
        ssl=settings.tls,
        socket_timeout=settings.socket_timeout_seconds,
        socket_connect_timeout=settings.socket_connect_timeout_seconds,
        retry=AsyncRetry(_backoff(), settings.retries, supported_errors=RETRY_ON),
        max_connections=settings.max_connections,
        health_check_interval=30,
    )


def create_sync_redis_client(settings: RedisSettings, *, db: int) -> SyncRedis:
    """For Celery workers (synchronous)."""
    password = settings.password.get_secret_value() if settings.password else None
    return SyncRedis(
        host=settings.host,
        port=settings.port,
        db=db,
        password=password,
        ssl=settings.tls,
        socket_timeout=settings.socket_timeout_seconds,
        socket_connect_timeout=settings.socket_connect_timeout_seconds,
        retry=Retry(_backoff(), settings.retries, supported_errors=RETRY_ON),
        max_connections=settings.max_connections,
        health_check_interval=30,
    )


async def ping_redis(client: Redis) -> None:
    await client.ping()

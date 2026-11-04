"""Redis clients. Cache, Celery broker and Celery results live in separate logical databases."""

from redis.asyncio import Redis

from medlens.core.config import RedisSettings


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
        max_connections=settings.max_connections,
        health_check_interval=30,
    )


async def ping_redis(client: Redis) -> None:
    await client.ping()

"""Long-lived clients shared by all requests, opened and closed by the app lifespan."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import cast

from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from medlens.adapters.query_embedding import CeleryQueryEmbedder
from medlens.adapters.s3 import S3Client, open_s3_client
from medlens.adapters.sms import create_sms_adapter
from medlens.adapters.storage import S3Storage
from medlens.ai.gateway.gateway import AsyncLLMGateway
from medlens.ai.gateway.ledger import AsyncLedger
from medlens.ai.gateway.providers import AsyncAnthropicProvider
from medlens.cache.redis import create_redis_client
from medlens.core.config import Settings
from medlens.core.security import JwtKeyRing
from medlens.db.session import create_db_engine, create_session_factory
from medlens.ports.embedding import QueryEmbeddingPort
from medlens.ports.sms import SMSPort
from medlens.ports.storage import StoragePort
from medlens.queue.backpressure import QueueDepth
from medlens.ratelimit.limiter import SlidingWindowLimiter


@dataclass(frozen=True, slots=True)
class Resources:
    db_engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis: Redis
    s3: S3Client
    s3_bucket: str
    keyring: JwtKeyRing
    sms: SMSPort
    storage: StoragePort
    broker_redis: Redis  # Celery's broker database: queue depths
    scan_queue: QueueDepth
    llm: AsyncLLMGateway  # the assistant's gateway (the worker has its own, synchronous)
    query_embedder: QueryEmbeddingPort  # question embeddings, computed by a worker
    rate_limiter: SlidingWindowLimiter


@asynccontextmanager
async def open_resources(settings: Settings) -> AsyncIterator[Resources]:
    """Create clients; close them in reverse order on exit, even if startup fails halfway."""
    async with AsyncExitStack() as stack:
        engine = create_db_engine(
            settings.database, application_name=f"{settings.observability.service_name}-api"
        )
        stack.push_async_callback(engine.dispose)
        redis = create_redis_client(settings.redis, db=settings.redis.cache_db)
        stack.push_async_callback(redis.aclose)
        s3 = await stack.enter_async_context(open_s3_client(settings.s3))
        sms = create_sms_adapter(settings.sms)
        stack.push_async_callback(sms.aclose)
        broker_redis = create_redis_client(settings.redis, db=settings.redis.broker_db)
        stack.push_async_callback(broker_redis.aclose)
        session_factory = create_session_factory(engine)
        provider = AsyncAnthropicProvider(settings.llm)
        stack.push_async_callback(provider.aclose)
        from medlens.queue.celery_app import celery_app  # Celery loads its config lazily

        yield Resources(
            db_engine=engine,
            session_factory=session_factory,
            redis=redis,
            s3=s3,
            s3_bucket=settings.s3.bucket,
            keyring=JwtKeyRing(settings.auth),
            sms=sms,
            storage=S3Storage(s3, settings.s3.bucket),
            broker_redis=broker_redis,
            scan_queue=QueueDepth(broker_redis),
            llm=AsyncLLMGateway(settings.llm, provider, AsyncLedger(session_factory, redis)),
            query_embedder=CeleryQueryEmbedder(
                celery_app, timeout_seconds=settings.knowledge.query_embedding_timeout_seconds
            ),
            rate_limiter=SlidingWindowLimiter(redis),
        )


def get_resources(request: Request) -> Resources:
    return cast(Resources, request.app.state.resources)


def get_storage(request: Request) -> StoragePort:
    """Object storage, as a dependency so tests can swap it."""
    return get_resources(request).storage


def get_app_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)

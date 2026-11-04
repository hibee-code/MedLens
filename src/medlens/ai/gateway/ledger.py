"""Metering: one ``llm_calls`` row per model call, and today's spend for the budget check.

Today's spend is the sum of ``cost_usd`` since 00:00 UTC, cached in Redis for a minute and
dropped whenever a call is recorded, so the check is cheap and never stale by more than the
calls in flight. Without Redis it reads Postgres every time.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, time
from decimal import Decimal
from typing import Protocol
from uuid import UUID

import structlog
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import RedisError
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker

from medlens.ai.gateway.pricing import TokenUsage
from medlens.cache.keys import redis_key
from medlens.modules.control_plane.models import LLMCall, LLMCallOutcome

log = structlog.stdlib.get_logger(__name__)

SPEND_CACHE_SECONDS = 60


@dataclass(frozen=True, slots=True)
class CallRecord:
    route: str
    tier: str
    requested_model: str
    model: str
    prompt_version: str
    usage: TokenUsage
    cost_usd: Decimal
    latency_ms: int
    attempts: int
    outcome: LLMCallOutcome
    error_type: str | None = None
    user_id: UUID | None = None
    subject_type: str | None = None
    subject_id: str | None = None

    def row(self) -> LLMCall:
        return LLMCall(
            route=self.route,
            tier=self.tier,
            requested_model=self.requested_model,
            model=self.model,
            prompt_version=self.prompt_version,
            input_tokens=self.usage.input,
            output_tokens=self.usage.output,
            cache_read_tokens=self.usage.cache_read,
            cache_write_tokens=self.usage.cache_write,
            cost_usd=self.cost_usd,
            latency_ms=self.latency_ms,
            attempts=self.attempts,
            outcome=self.outcome,
            error_type=self.error_type,
            user_id=self.user_id,
            subject_type=self.subject_type,
            subject_id=self.subject_id,
        )


class SpendLedger(Protocol):
    def spent_today(self) -> Decimal: ...

    def record(self, call: CallRecord) -> None: ...


class AsyncSpendLedger(Protocol):
    async def spent_today(self) -> Decimal: ...

    async def record(self, call: CallRecord) -> None: ...


def day_start(now: datetime | None = None) -> datetime:
    return datetime.combine((now or datetime.now(UTC)).date(), time.min, tzinfo=UTC)


def _spend_key() -> str:
    return redis_key("llm_spend", day_start().date().isoformat())


def _spend_query() -> Select[Decimal]:
    return select(func.coalesce(func.sum(LLMCall.cost_usd), 0)).where(
        LLMCall.created_at >= day_start()
    )


class SyncLedger:
    def __init__(self, sessions: sessionmaker[Session], redis: Redis | None = None) -> None:
        self._sessions = sessions
        self._redis = redis

    def spent_today(self) -> Decimal:
        key = _spend_key()
        if self._redis is not None:
            try:
                cached = self._redis.get(key)
                if cached is not None:
                    return Decimal(cached.decode() if isinstance(cached, bytes) else str(cached))
            except RedisError as exc:
                log.warning("llm.spend_cache_unavailable", error=type(exc).__name__)
        with self._sessions() as session:
            spent = Decimal(session.scalar(_spend_query()) or 0)
        self._cache(key, spent)
        return spent

    def record(self, call: CallRecord) -> None:
        with self._sessions() as session, session.begin():
            session.add(call.row())
        if self._redis is not None:
            try:
                self._redis.delete(_spend_key())
            except RedisError as exc:
                log.warning("llm.spend_cache_unavailable", error=type(exc).__name__)

    def _cache(self, key: str, spent: Decimal) -> None:
        if self._redis is None:
            return
        try:
            self._redis.set(key, str(spent), ex=SPEND_CACHE_SECONDS)
        except RedisError as exc:
            log.warning("llm.spend_cache_unavailable", error=type(exc).__name__)


class AsyncLedger:
    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], redis: AsyncRedis | None = None
    ) -> None:
        self._sessions = sessions
        self._redis = redis

    async def spent_today(self) -> Decimal:
        key = _spend_key()
        if self._redis is not None:
            try:
                cached = await self._redis.get(key)
                if cached is not None:
                    return Decimal(cached.decode() if isinstance(cached, bytes) else str(cached))
            except RedisError as exc:
                log.warning("llm.spend_cache_unavailable", error=type(exc).__name__)
        async with self._sessions() as session:
            spent = Decimal(await session.scalar(_spend_query()) or 0)
        if self._redis is not None:
            try:
                await self._redis.set(key, str(spent), ex=SPEND_CACHE_SECONDS)
            except RedisError as exc:
                log.warning("llm.spend_cache_unavailable", error=type(exc).__name__)
        return spent

    async def record(self, call: CallRecord) -> None:
        async with self._sessions() as session, session.begin():
            session.add(call.row())
        if self._redis is not None:
            try:
                await self._redis.delete(_spend_key())
            except RedisError as exc:
                log.warning("llm.spend_cache_unavailable", error=type(exc).__name__)

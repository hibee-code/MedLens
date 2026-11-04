"""The gateway's ledger on Postgres and Redis: one llm_calls row per call, today's spend cached
in Redis and dropped on every new call, and Postgres alone when Redis is down."""

from decimal import Decimal
from uuid import uuid4

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from sqlalchemy import select

from medlens.adapters.fakes import FakeLLM, FakeLLMProvider
from medlens.ai.gateway.gateway import ImagePart, LLMGateway, Subject
from medlens.ai.gateway.ledger import AsyncLedger, CallRecord, SyncLedger, _spend_key
from medlens.ai.gateway.pricing import TokenUsage
from medlens.core.config import LLMSettings, Settings
from medlens.db.session import UnitOfWork
from medlens.modules.control_plane.models import LLMCall, LLMCallOutcome
from medlens.modules.verification.pipeline import PackExtraction
from tests.integration.api_support import Api

pytestmark = pytest.mark.integration


def call(cost: str, subject_id: str) -> CallRecord:
    return CallRecord(
        route="pack_extraction",
        tier="fast",
        requested_model="claude-haiku-5-5",
        model="claude-haiku-5-5",
        prompt_version="pack_extraction@v1",
        usage=TokenUsage(input=100, output=10),
        cost_usd=Decimal(cost),
        latency_ms=12,
        attempts=1,
        outcome=LLMCallOutcome.OK,
        subject_type="test",
        subject_id=subject_id,
    )


async def test_the_async_ledger_caches_spend_and_drops_it_on_record(api: Api) -> None:
    ledger = AsyncLedger(api.resources.session_factory, api.resources.redis)

    before = await ledger.spent_today()
    assert await api.resources.redis.get(_spend_key()) is not None  # cached
    await ledger.record(call("0.250000", uuid4().hex))
    assert await api.resources.redis.get(_spend_key()) is None  # dropped
    after = await ledger.spent_today()

    assert after - before >= Decimal("0.25")


async def test_the_async_ledger_works_without_redis(api: Api) -> None:
    down = AsyncRedis(host="127.0.0.1", port=1, socket_connect_timeout=0.2)
    ledger = AsyncLedger(api.resources.session_factory, down)
    subject = uuid4().hex

    before = await ledger.spent_today()
    await ledger.record(call("0.100000", subject))

    assert await ledger.spent_today() - before >= Decimal("0.1")
    await down.aclose()


async def test_the_gateway_writes_one_row_per_call(api: Api, app_settings: Settings) -> None:
    redis = Redis.from_url(app_settings.redis.url(app_settings.redis.cache_db))
    ledger = SyncLedger(api.sync_sessions, redis)
    provider = FakeLLMProvider(script=["not json", FakeLLM.reading(nafdac_number="A4-1234")])
    gateway = LLMGateway(LLMSettings(enabled=True), provider, ledger)
    subject = uuid4().hex

    gateway.run(
        "pack_extraction", [ImagePart(b"jpeg")], PackExtraction, subject=Subject("scan", subject)
    )

    async with UnitOfWork(api.resources.session_factory) as uow:
        rows = (
            await uow.session.scalars(select(LLMCall).where(LLMCall.subject_id == subject))
        ).all()
    [row] = rows
    assert row.outcome is LLMCallOutcome.OK
    assert row.attempts == 2
    assert (row.input_tokens, row.output_tokens) == (2000, 400)
    assert row.cost_usd == Decimal("0.000400")
    assert row.prompt_version == "pack_extraction@v1"
    assert ledger.spent_today() >= Decimal("0.0004")
    down = SyncLedger(
        api.sync_sessions, Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2)
    )
    assert down.spent_today() == ledger.spent_today()
    down.record(call("0", subject))
    redis.close()

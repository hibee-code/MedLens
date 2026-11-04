"""Metered model use (llm_calls), aggregated for the AI usage report."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from medlens.modules.control_plane.models import LLMCall, LLMCallOutcome


@dataclass(frozen=True, slots=True)
class UsageRow:
    day: date
    route: str
    calls: int
    failed: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: Decimal


async def usage_by_day_and_route(
    session: AsyncSession, since: datetime, until: datetime
) -> list[UsageRow]:
    day = cast(func.timezone("UTC", LLMCall.created_at), Date)
    statement = (
        select(
            day,
            LLMCall.route,
            func.count(),
            func.count().filter(LLMCall.outcome != LLMCallOutcome.OK),
            func.coalesce(func.sum(LLMCall.input_tokens), 0),
            func.coalesce(func.sum(LLMCall.output_tokens), 0),
            func.coalesce(func.sum(LLMCall.cache_read_tokens), 0),
            func.coalesce(func.sum(LLMCall.cost_usd), 0),
        )
        .where(LLMCall.created_at >= since, LLMCall.created_at < until)
        .group_by(day, LLMCall.route)
        .order_by(day.desc(), LLMCall.route)
    )
    return [
        UsageRow(value, route, int(calls), int(failed), int(i), int(o), int(c), Decimal(cost))
        for value, route, calls, failed, i, o, c, cost in await session.execute(statement)
    ]

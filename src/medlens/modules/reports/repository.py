"""Aggregate queries over scans (no rows about individual people leave this module). Counts use
the final verdict (after case decisions); accuracy compares the pipeline's verdict with lab
labels."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import ColumnElement, Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from medlens.modules.control_plane.models import LLMCall
from medlens.modules.hitl.models import Label
from medlens.modules.products.models import Product
from medlens.modules.scans.models import Scan, ScanStatus, Verdict

FLAGGED = (Verdict.SUSPICIOUS, Verdict.UNREGISTERED)


@dataclass(frozen=True, slots=True)
class SummaryCounts:
    by_verdict: dict[str, int]
    by_state: list[tuple[str | None, int, int]]  # state, total, flagged
    by_day: list[tuple[date, int, int]]
    llm_scans: int
    llm_calls: int
    llm_cost_usd: Decimal


async def summary(
    session: AsyncSession, visible: ColumnElement[bool], since: datetime, until: datetime
) -> SummaryCounts:
    window = (
        Scan.created_at >= since,
        Scan.created_at < until,
        Scan.status == ScanStatus.COMPLETED,
        visible,
    )
    verdict = Scan.final_verdict
    flagged = func.count().filter(verdict.in_([v.value for v in FLAGGED]))
    by_verdict = {
        str(name): int(count)
        for name, count in await session.execute(
            select(verdict, func.count()).where(*window).group_by(verdict)
        )
    }
    by_state = [
        (state.value if state else None, int(total), int(hits))
        for state, total, hits in await session.execute(
            select(Scan.state, func.count(), flagged)
            .where(*window)
            .group_by(Scan.state)
            .order_by(func.count().desc())
        )
    ]
    day = cast(func.timezone("UTC", Scan.created_at), Date)
    by_day = [
        (value, int(total), int(hits))
        for value, total, hits in await session.execute(
            select(day, func.count(), flagged).where(*window).group_by(day).order_by(day)
        )
    ]
    llm_scans = await session.scalar(
        select(func.count()).select_from(Scan).where(*window, Scan.llm_used)
    )
    calls, cost = (
        await session.execute(
            select(func.count(), func.coalesce(func.sum(LLMCall.cost_usd), 0)).where(
                LLMCall.created_at >= since, LLMCall.created_at < until
            )
        )
    ).one()
    return SummaryCounts(
        by_verdict=by_verdict,
        by_state=by_state,
        by_day=by_day,
        llm_scans=int(llm_scans or 0),
        llm_calls=int(calls),
        llm_cost_usd=Decimal(cost),
    )


@dataclass(frozen=True, slots=True)
class ProductCell:
    product_id: UUID
    nafdac_number: str
    brand_name: str
    state: str | None
    iso_week: str
    scan_count: int


async def flagged_by_product_state_week(
    session: AsyncSession, manufacturer_org_id: UUID, since: datetime, until: datetime
) -> list[ProductCell]:
    week = func.to_char(func.timezone("UTC", Scan.created_at), 'IYYY-"W"IW')
    statement = (
        select(
            Product.id, Product.nafdac_number, Product.brand_name, Scan.state, week, func.count()
        )
        .join(Product, Product.id == Scan.matched_product_id)
        .where(
            Product.manufacturer_org_id == manufacturer_org_id,
            Scan.created_at >= since,
            Scan.created_at < until,
            Scan.status == ScanStatus.COMPLETED,
            Scan.final_verdict.in_(FLAGGED),
        )
        .group_by(Product.id, Product.nafdac_number, Product.brand_name, Scan.state, week)
        .order_by(week.desc(), Product.brand_name, Scan.state)
    )
    return [
        ProductCell(product_id, number, brand, state.value if state else None, label, int(count))
        for product_id, number, brand, state, label, count in await session.execute(statement)
    ]


@dataclass(frozen=True, slots=True)
class LabelledCell:
    scorer_version: str | None
    pipeline_verdict: str | None
    lab_outcome: str
    count: int


async def pipeline_verdicts_against_labels(
    session: AsyncSession, visible: ColumnElement[bool], since: datetime, until: datetime
) -> list[LabelledCell]:
    """Labelled scans by scorer version, pipeline verdict and lab outcome (labels recorded in
    the window)."""
    statement = (
        select(Scan.scorer_version, Scan.pipeline_verdict, Label.outcome, func.count())
        .join(Scan, (Scan.id == Label.scan_id) & (Scan.created_at == Label.scan_created_at))
        .where(Label.created_at >= since, Label.created_at < until, visible)
        .group_by(Scan.scorer_version, Scan.pipeline_verdict, Label.outcome)
        .order_by(Scan.scorer_version, Scan.pipeline_verdict, Label.outcome)
    )
    return [
        LabelledCell(
            version,
            verdict.value if verdict else None,
            outcome.value,
            int(count),
        )
        for version, verdict, outcome, count in await session.execute(statement)
    ]

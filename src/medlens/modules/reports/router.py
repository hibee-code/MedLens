from typing import Annotated
from uuid import UUID

from fastapi import Depends, Query

from medlens.core.openapi import problem_responses
from medlens.core.routing import api_v1_router
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.modules.reports.schemas import (
    DayCount,
    LLMUsage,
    ManufacturerReportRead,
    ProductWeekCell,
    StateCount,
    StatsSummary,
)
from medlens.modules.reports.service import MIN_CELL, ReportService
from medlens.resources import Resources, get_resources

stats_router = api_v1_router("/stats", tags=["reports"])
manufacturer_router = api_v1_router("/manufacturer", tags=["reports"])


def get_report_service(resources: Annotated[Resources, Depends(get_resources)]) -> ReportService:
    return ReportService(resources.session_factory)


ServiceDep = Annotated[ReportService, Depends(get_report_service)]


@stats_router.get(
    "/summary",
    response_model=StatsSummary,
    summary="Scan statistics",
    description=(
        "Completed scans in the last `days` by verdict (after case decisions), state and UTC "
        "day, the share that needed the vision LLM, and the LLM calls and cost in the window. "
        "Regulators see scans in their jurisdiction; platform admins see all."
    ),
    responses=problem_responses(401, 403, 422),
)
async def stats_summary(
    actor: Annotated[Actor, Depends(require_permissions(Permission.SCAN_READ_ALL))],
    service: ServiceDep,
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> StatsSummary:
    result = await service.summary(actor, days)
    counts = result.counts
    total = sum(counts.by_verdict.values())
    return StatsSummary(
        since=result.since,
        until=result.until,
        total_scans=total,
        by_verdict=counts.by_verdict,
        by_state=[
            StateCount(state=state, total=all_, flagged=hits)  # type: ignore[arg-type]
            for state, all_, hits in counts.by_state
        ],
        by_day=[DayCount(day=day, total=all_, flagged=hits) for day, all_, hits in counts.by_day],
        llm=LLMUsage(
            scans_needing_llm=counts.llm_scans,
            share_of_scans=round(counts.llm_scans / total, 4) if total else 0.0,
            calls=counts.llm_calls,
            cost_usd=counts.llm_cost_usd,
        ),
    )


@manufacturer_router.get(
    "/scans",
    response_model=ManufacturerReportRead,
    summary="Flagged scans of your products",
    description=(
        "SUSPICIOUS and UNREGISTERED scans matched to your organization's products, counted by "
        "product, state and ISO week. No individual scans, people or locations finer than a "
        "state; cells with fewer than 5 scans are left out. Platform admins pass "
        "`organization_id`."
    ),
    responses=problem_responses(401, 403, 422),
)
async def manufacturer_scans(
    actor: Annotated[Actor, Depends(require_permissions(Permission.REPORTS_READ_OWN))],
    service: ServiceDep,
    weeks: Annotated[int, Query(ge=1, le=52)] = 12,
    organization_id: Annotated[UUID | None, Query(description="Platform admins only.")] = None,
) -> ManufacturerReportRead:
    report = await service.manufacturer_report(actor, weeks, organization_id)
    return ManufacturerReportRead(
        organization_id=report.organization_id,
        since=report.since,
        until=report.until,
        min_cell_size=MIN_CELL,
        suppressed_cells=report.suppressed_cells,
        cells=[
            ProductWeekCell(
                product_id=cell.product_id,
                nafdac_number=cell.nafdac_number,
                brand_name=cell.brand_name,
                state=cell.state,  # type: ignore[arg-type]
                iso_week=cell.iso_week,
                flagged_scans=cell.scan_count,
            )
            for cell in report.cells
        ],
    )

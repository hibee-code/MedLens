from datetime import timedelta
from decimal import Decimal
from typing import Annotated

from fastapi import Depends, Query

from medlens.ai.gateway.ledger import AsyncLedger, day_start
from medlens.core.config import Settings
from medlens.core.openapi import problem_responses
from medlens.core.routing import api_v1_router
from medlens.db.session import UnitOfWork
from medlens.modules.control_plane.repository import usage_by_day_and_route
from medlens.modules.control_plane.schemas import AIUsage, Budget, UsageDay
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.resources import Resources, get_app_settings, get_resources

router = api_v1_router("/admin/ai", tags=["admin"])


@router.get(
    "/usage",
    response_model=AIUsage,
    summary="AI usage and budget",
    description=(
        "Metered model calls by UTC day and route (tokens, cost, failures) for the last "
        "`days`, whether AI features are switched on, the model behind each tier, and how much "
        "of today's budget is spent. Platform admins only."
    ),
    responses=problem_responses(401, 403, 422),
)
async def ai_usage(
    _: Annotated[Actor, Depends(require_permissions(Permission.PLATFORM_ADMIN))],
    resources: Annotated[Resources, Depends(get_resources)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    days: Annotated[int, Query(ge=1, le=90)] = 7,
) -> AIUsage:
    until = day_start() + timedelta(days=1)
    since = until - timedelta(days=days)
    async with UnitOfWork(resources.session_factory) as uow:
        rows = await usage_by_day_and_route(uow.session, since, until)
    spent = await AsyncLedger(resources.session_factory, resources.redis).spent_today()
    daily = Decimal(str(settings.llm.daily_budget_usd))
    return AIUsage(
        ai_enabled=settings.llm.enabled,
        assistant_enabled=settings.knowledge.assistant_enabled,
        models={"fast": settings.llm.model_fast, "balanced": settings.llm.model_balanced},
        budget=Budget(
            daily_usd=daily,
            spent_today_usd=spent,
            remaining_usd=max(daily - spent, Decimal(0)),
            share_used=round(float(spent / daily), 4) if daily else 1.0,
        ),
        since=since,
        until=until,
        days=[
            UsageDay(
                day=row.day,
                route=row.route,
                calls=row.calls,
                failed=row.failed,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                cache_read_tokens=row.cache_read_tokens,
                cost_usd=row.cost_usd,
            )
            for row in rows
        ],
    )

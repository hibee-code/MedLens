"""Reports: the scan summary for regulators and platform admins, and the aggregated report a
manufacturer gets about its own products (CLAUDE.md decision log: no raw scans for
manufacturers; cells under five scans are suppressed so no single buyer can be singled out)."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.errors import Forbidden, ValidationFailed
from medlens.db.session import UnitOfWork
from medlens.modules.organizations.models import OrganizationType
from medlens.modules.rbac import policies
from medlens.modules.rbac.access import Actor
from medlens.modules.reports.repository import (
    ProductCell,
    SummaryCounts,
    flagged_by_product_state_week,
    summary,
)

MIN_CELL = 5  # smaller cells are suppressed


@dataclass(frozen=True, slots=True)
class Summary:
    since: datetime
    until: datetime
    counts: SummaryCounts


@dataclass(frozen=True, slots=True)
class ManufacturerReport:
    organization_id: UUID
    since: datetime
    until: datetime
    cells: list[ProductCell]
    suppressed_cells: int


def _window(days: int) -> tuple[datetime, datetime]:
    until = datetime.now(UTC)
    return until - timedelta(days=days), until


class ReportService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def summary(self, actor: Actor, days: int) -> Summary:
        since, until = _window(days)
        async with UnitOfWork(self._sessions) as uow:
            counts = await summary(uow.session, policies.scan_visibility(actor), since, until)
        return Summary(since, until, counts)

    async def manufacturer_report(
        self, actor: Actor, weeks: int, organization_id: UUID | None
    ) -> ManufacturerReport:
        organization = actor.organization
        if actor.is_platform_admin:
            if organization_id is None:
                raise ValidationFailed(
                    "Platform admins choose the manufacturer with organization_id.",
                    errors=[
                        {"loc": ["query", "organization_id"], "msg": "required", "type": "missing"}
                    ],
                )
            target = organization_id
        elif organization is not None and organization.type is OrganizationType.MANUFACTURER:
            if organization_id not in (None, organization.id):
                raise Forbidden("You can only see reports about your own products.")
            target = organization.id
        else:
            raise Forbidden("Only manufacturers have product reports.")
        since, until = _window(weeks * 7)
        async with UnitOfWork(self._sessions) as uow:
            cells = await flagged_by_product_state_week(uow.session, target, since, until)
        shown = [cell for cell in cells if cell.scan_count >= MIN_CELL]
        return ManufacturerReport(target, since, until, shown, len(cells) - len(shown))

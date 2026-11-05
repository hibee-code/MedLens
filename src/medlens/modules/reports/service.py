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
    LabelledCell,
    ProductCell,
    SummaryCounts,
    flagged_by_product_state_week,
    pipeline_verdicts_against_labels,
    summary,
)

MIN_CELL = 5  # smaller cells are suppressed


@dataclass(frozen=True, slots=True)
class Summary:
    since: datetime
    until: datetime
    counts: SummaryCounts


@dataclass(frozen=True, slots=True)
class Accuracy:
    """The pipeline's verdicts against lab results. Flagged = SUSPICIOUS or UNREGISTERED;
    INCONCLUSIVE scans and inconclusive lab results are counted but not scored."""

    since: datetime
    until: datetime
    cells: list[LabelledCell]

    def _count(self, verdicts: set[str], outcome: str) -> int:
        return sum(
            cell.count
            for cell in self.cells
            if cell.pipeline_verdict in verdicts and cell.lab_outcome == outcome
        )

    @property
    def confusion(self) -> dict[str, int]:
        flagged, passed = {"SUSPICIOUS", "UNREGISTERED"}, {"LIKELY_GENUINE"}
        return {
            "flagged_counterfeit": self._count(flagged, "counterfeit"),
            "flagged_genuine": self._count(flagged, "genuine"),
            "passed_counterfeit": self._count(passed, "counterfeit"),
            "passed_genuine": self._count(passed, "genuine"),
        }

    @property
    def precision(self) -> float | None:
        c = self.confusion
        flagged = c["flagged_counterfeit"] + c["flagged_genuine"]
        return round(c["flagged_counterfeit"] / flagged, 4) if flagged else None

    @property
    def recall(self) -> float | None:
        c = self.confusion
        counterfeit = c["flagged_counterfeit"] + c["passed_counterfeit"]
        return round(c["flagged_counterfeit"] / counterfeit, 4) if counterfeit else None

    @property
    def agreement(self) -> float | None:
        c = self.confusion
        total = sum(c.values())
        right = c["flagged_counterfeit"] + c["passed_genuine"]
        return round(right / total, 4) if total else None


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

    async def accuracy(self, actor: Actor, days: int) -> Accuracy:
        since, until = _window(days)
        async with UnitOfWork(self._sessions) as uow:
            cells = await pipeline_verdicts_against_labels(
                uow.session, policies.scan_visibility(actor), since, until
            )
        return Accuracy(since, until, cells)

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

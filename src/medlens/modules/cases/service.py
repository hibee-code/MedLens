"""Review cases for high-risk scans.

Opening (worker, on ``scan.completed``): a SUSPICIOUS high-priority or UNREGISTERED scan opens
one case, routed to the active regulator whose jurisdiction covers the scan's state; the most
local one wins (a state agency before the national one). Scans without a state go to a national
regulator. Opening is idempotent: one case per scan.

Reviewing (API, regulators): claim or assign, move through the workflow, and record the lab
result. A lab result is also stored as a training label. With it, the reviewer may override the
verdict shown for the scan, within what the outcome allows; the pipeline's verdict is kept and
the override is audited. Nothing else changes a verdict (CLAUDE.md §5).
"""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from sqlalchemy import ColumnElement, Select, any_, func, literal, select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session

from medlens.core.errors import Conflict, NotFound, ValidationFailed
from medlens.core.identifiers import NigerianState
from medlens.core.pagination import PageParams
from medlens.core.request_meta import RequestMeta
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.cases.models import Case, CasePriority, CaseStatus
from medlens.modules.hitl.models import Label, LabelOutcome
from medlens.modules.organizations.models import (
    Membership,
    MembershipStatus,
    Organization,
    OrganizationStatus,
    OrganizationType,
)
from medlens.modules.rbac import policies
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.models import RolePermission
from medlens.modules.rbac.permissions import Permission
from medlens.modules.scans.models import Scan, Verdict
from medlens.modules.scans.repository import id_window
from medlens.modules.scans.service import override_verdict
from medlens.modules.verification.service import needs_case
from medlens.ports.storage import ObjectNotFoundError, StoragePort

_NOT_FOUND = "Case not found."
_OPEN = frozenset({CaseStatus.OPEN, CaseStatus.IN_REVIEW})
ReviewStatus = Literal["in_review", "inconclusive", "closed"]
_OUTCOME_STATUS = {
    LabelOutcome.COUNTERFEIT: CaseStatus.CONFIRMED_COUNTERFEIT,
    LabelOutcome.GENUINE: CaseStatus.CONFIRMED_GENUINE,
    LabelOutcome.INCONCLUSIVE: CaseStatus.INCONCLUSIVE,
}
# Verdicts a lab outcome may set (never "safe": a genuine pack is LIKELY_GENUINE).
_ALLOWED_OVERRIDES = {
    LabelOutcome.COUNTERFEIT: frozenset({Verdict.SUSPICIOUS, Verdict.UNREGISTERED}),
    LabelOutcome.GENUINE: frozenset({Verdict.LIKELY_GENUINE}),
    LabelOutcome.INCONCLUSIVE: frozenset[Verdict](),
}


def _now() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- opening, in the worker
def route_regulator(session: Session, state: NigerianState | None) -> UUID | None:
    """The most local active regulator covering ``state`` (national ones for no state)."""
    size = func.cardinality(Organization.jurisdiction_states)
    statement = select(Organization.id).where(
        Organization.type == OrganizationType.REGULATOR,
        Organization.status == OrganizationStatus.ACTIVE,
        Organization.deleted_at.is_(None),
    )
    if state is None:
        statement = statement.where(size == len(NigerianState))
    else:
        statement = statement.where(literal(state.value) == any_(Organization.jurisdiction_states))
    return session.scalar(
        statement.order_by(size, Organization.verified_at, Organization.id).limit(1)
    )


def open_case_for_scan(session: Session, scan_id: UUID) -> Case | None:
    """Open the scan's case if it needs one; the existing case if it was opened already."""
    scan = session.scalar(select(Scan).where(id_window(scan_id)))
    if scan is None or not needs_case(scan):
        return None
    session.execute(
        insert(Case)
        .values(
            scan_id=scan.id,
            scan_created_at=scan.created_at,
            priority=CasePriority.HIGH,
            state=scan.state,
            assigned_org_id=route_regulator(session, scan.state),
        )
        .on_conflict_do_nothing(index_elements=["scan_id", "scan_created_at"])
    )
    return session.scalar(
        select(Case).where(Case.scan_id == scan.id, Case.scan_created_at == scan.created_at)
    )


# --------------------------------------------------------------------------- reviewing, in the API
class CaseService:
    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], storage: StoragePort | None = None
    ) -> None:
        self._sessions = sessions
        self._storage = storage

    async def list_page(
        self, actor: Actor, params: PageParams, *filters: ColumnElement[bool]
    ) -> list[tuple[Case, Scan]]:
        statement = _with_scan().where(policies.case_visibility(actor), *filters)
        if params.after is not None:
            position = tuple_(params.after.created_at, params.after.id)
            statement = statement.where(tuple_(Case.created_at, Case.id) < position)
        statement = statement.order_by(Case.created_at.desc(), Case.id.desc()).limit(
            params.limit + 1
        )
        async with UnitOfWork(self._sessions) as uow:
            return [(case, scan) for case, scan in await uow.session.execute(statement)]

    async def get(self, actor: Actor, case_id: UUID) -> tuple[Case, Scan]:
        async with UnitOfWork(self._sessions) as uow:
            return await _visible(uow.session, actor, case_id)

    async def claim(self, actor: Actor, case_id: UUID, meta: RequestMeta) -> tuple[Case, Scan]:
        async with UnitOfWork(self._sessions) as uow:
            case, scan = await _visible(uow.session, actor, case_id, lock=True)
            _require_open(case)
            case.assigned_to_user_id = actor.user_id
            if (
                actor.organization is not None
                and actor.organization.type is OrganizationType.REGULATOR
            ):
                case.assigned_org_id = actor.organization.id
            case.status = CaseStatus.IN_REVIEW
            await _audit(uow.session, "case.claimed", actor, case, meta)
            await uow.commit()
            return case, scan

    async def assign(
        self, actor: Actor, case_id: UUID, user_id: UUID, meta: RequestMeta
    ) -> tuple[Case, Scan]:
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            case, scan = await _visible(session, actor, case_id, lock=True)
            _require_open(case)
            organization_id = case.assigned_org_id or (
                actor.organization.id if actor.organization else None
            )
            if organization_id is None or not await _can_review(session, user_id, organization_id):
                raise ValidationFailed(
                    "Assign the case to a reviewer of the regulator handling it.",
                    errors=[{"loc": ["body", "user_id"], "msg": "not a reviewer", "type": "value"}],
                )
            case.assigned_org_id = organization_id
            case.assigned_to_user_id = user_id
            await _audit(session, "case.assigned", actor, case, meta, {"to": str(user_id)})
            await uow.commit()
            return case, scan

    async def update(
        self,
        actor: Actor,
        case_id: UUID,
        status: ReviewStatus | None,
        notes: str | None,
        meta: RequestMeta,
    ) -> tuple[Case, Scan]:
        async with UnitOfWork(self._sessions) as uow:
            case, scan = await _visible(uow.session, actor, case_id, lock=True)
            _require_open(case)
            previous = case.status
            if status is not None:
                case.status = CaseStatus(status)
                if case.status not in _OPEN:
                    case.resolved_at = _now()
            if notes is not None:
                case.notes = notes
            await _audit(
                uow.session,
                "case.updated",
                actor,
                case,
                meta,
                {"from": previous.value, "to": case.status.value},
            )
            await uow.commit()
            return case, scan

    async def record_lab_result(
        self,
        actor: Actor,
        case_id: UUID,
        outcome: LabelOutcome,
        lab_reference: str | None,
        notes: str | None,
        meta: RequestMeta,
        *,
        override: Verdict | None = None,
    ) -> tuple[Case, Scan]:
        if override is not None and override not in _ALLOWED_OVERRIDES[outcome]:
            raise ValidationFailed(
                f"A {outcome.value} lab result cannot set the verdict to {override.value}.",
                errors=[
                    {"loc": ["body", "override_verdict"], "msg": "not allowed", "type": "value"}
                ],
            )
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            case, scan = await _visible(session, actor, case_id, lock=True)
            _require_open(case)
            now = _now()
            case.lab_result = {
                "outcome": outcome.value,
                "lab_reference": lab_reference,
                "recorded_by": str(actor.user_id),
                "recorded_at": now.isoformat(),
            }
            case.status = _OUTCOME_STATUS[outcome]
            case.resolved_at = now
            if notes is not None:
                case.notes = notes
            session.add(
                Label(
                    scan_id=case.scan_id,
                    scan_created_at=case.scan_created_at,
                    case_id=case.id,
                    outcome=outcome,
                    source="case_review",
                    lab_reference=lab_reference,
                    labelled_by=actor.user_id,
                )
            )
            await _audit(session, "case.lab_result", actor, case, meta, {"outcome": outcome.value})
            if override is not None:
                shown = await override_verdict(session, scan, override, case_id=case.id)
                await AuditService(session).record(
                    "scan.verdict_overridden",
                    actor_user_id=actor.user_id,
                    actor_org_id=actor.organization.id if actor.organization else None,
                    resource_type="scan",
                    resource_id=scan.id,
                    meta=meta,
                    details={
                        "case_id": str(case.id),
                        "from": shown.value if shown else None,
                        "to": override.value,
                        "pipeline_verdict": scan.verdict.value if scan.verdict else None,
                    },
                )
            await uow.commit()
            return case, scan

    async def image(self, actor: Actor, case_id: UUID, meta: RequestMeta) -> bytes:
        """The scan's stored photo (already re-encoded, EXIF stripped). Audited."""
        if self._storage is None:
            raise NotFound("The photo is not available.")
        async with UnitOfWork(self._sessions) as uow:
            case, scan = await _visible(uow.session, actor, case_id)
            if scan.image_s3_key is None:
                raise NotFound("The photo is no longer kept.")
            try:
                image = await self._storage.get(scan.image_s3_key)
            except ObjectNotFoundError:
                raise NotFound("The photo is no longer kept.") from None
            await _audit(uow.session, "case.image_viewed", actor, case, meta)
            await uow.commit()
        return image


def _with_scan() -> Select[Case, Scan]:
    return select(Case, Scan).join(
        Scan, (Scan.id == Case.scan_id) & (Scan.created_at == Case.scan_created_at)
    )


async def _visible(
    session: AsyncSession, actor: Actor, case_id: UUID, *, lock: bool = False
) -> tuple[Case, Scan]:
    statement = _with_scan().where(Case.id == case_id, policies.case_visibility(actor))
    if lock:
        statement = statement.with_for_update(of=Case)
    row = (await session.execute(statement)).first()
    if row is None:
        raise NotFound(_NOT_FOUND)
    case, scan = row
    return case, scan


def _require_open(case: Case) -> None:
    if case.status not in _OPEN:
        raise Conflict(f"The case is already {case.status.value}.")


async def _can_review(session: AsyncSession, user_id: UUID, organization_id: UUID) -> bool:
    """An active member of the organization whose role may review cases."""
    statement = (
        select(func.count())
        .select_from(Membership)
        .join(RolePermission, RolePermission.role_name == Membership.role)
        .where(
            Membership.user_id == user_id,
            Membership.organization_id == organization_id,
            Membership.status == MembershipStatus.ACTIVE,
            RolePermission.permission_name == Permission.CASE_REVIEW.value,
        )
    )
    return bool(await session.scalar(statement))


async def _audit(
    session: AsyncSession,
    action: str,
    actor: Actor,
    case: Case,
    meta: RequestMeta,
    details: dict[str, object] | None = None,
) -> None:
    await AuditService(session).record(
        action,
        actor_user_id=actor.user_id,
        actor_org_id=actor.organization.id if actor.organization else None,
        resource_type="case",
        resource_id=case.id,
        meta=meta,
        details=details,
    )

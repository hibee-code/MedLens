"""Review cases for high-risk scans.

Opening (worker, on ``scan.completed``): a SUSPICIOUS high-priority or UNREGISTERED scan opens
one case, routed to the active regulator whose jurisdiction covers the scan's state; the most
local one wins (a state agency before the national one). Scans without a state go to a national
regulator. Opening is idempotent: one case per scan.

Reviewing (API, regulators): claim or assign, move through the workflow, and record the lab
result. A lab result is also stored as a training label.

Overrides: once a case has a lab result, a regulator or platform admin may set the scan's final
verdict, within what the outcome allows, with a required note. The pipeline's verdict never
changes; every override and every reversal is audited. Nothing else changes a verdict
(CLAUDE.md §5).
"""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import ColumnElement, Select, any_, func, literal, select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session

from medlens.core.errors import Conflict, Forbidden, NotFound, ValidationFailed
from medlens.core.identifiers import NigerianState
from medlens.core.pagination import PageParams
from medlens.core.request_meta import RequestMeta
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.cases.models import Case, CasePriority, CaseStatus
from medlens.modules.hitl.models import Label, LabelOutcome
from medlens.modules.hotspots.service import invalidate_hotspots
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
from medlens.modules.scans.service import Override, override_verdict, revert_override
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
        self,
        sessions: async_sessionmaker[AsyncSession],
        storage: StoragePort | None = None,
        redis: Redis | None = None,
    ) -> None:
        self._sessions = sessions
        self._storage = storage
        self._redis = redis  # cached hotspots are invalidated when a final verdict changes

    async def _verdict_changed(self) -> None:
        if self._redis is not None:
            await invalidate_hotspots(self._redis)

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
        override_note: str | None = None,
    ) -> tuple[Case, Scan]:
        if override is not None:
            _require_override_rights(actor)
            _check_allowed(outcome, override, "override_verdict")
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
                await _override(session, actor, case, scan, override, override_note or "", meta)
            await uow.commit()
        if override is not None:
            await self._verdict_changed()
        return case, scan

    async def override(
        self, actor: Actor, case_id: UUID, verdict: Verdict, note: str, meta: RequestMeta
    ) -> tuple[Case, Scan]:
        """Set (or change) the scan's final verdict after the lab result, also on a resolved
        case."""
        _require_override_rights(actor)
        async with UnitOfWork(self._sessions) as uow:
            case, scan = await _visible(uow.session, actor, case_id, lock=True)
            if not case.lab_result:
                raise Conflict("Record the lab result before deciding the verdict.")
            outcome = LabelOutcome(case.lab_result["outcome"])
            _check_allowed(outcome, verdict, "verdict")
            await _override(uow.session, actor, case, scan, verdict, note, meta)
            await uow.commit()
        await self._verdict_changed()
        return case, scan

    async def revert_override(
        self, actor: Actor, case_id: UUID, note: str, meta: RequestMeta
    ) -> tuple[Case, Scan]:
        """Back to the pipeline's verdict. The audit log keeps the reverted decision."""
        _require_override_rights(actor)
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            case, scan = await _visible(session, actor, case_id, lock=True)
            if scan.override_case_id != case.id:
                raise Conflict("This case has not decided the scan's verdict.")
            reverted = await revert_override(session, scan)
            await _scan_audit(
                session,
                "scan.verdict_override_reverted",
                actor,
                scan,
                meta,
                {
                    "case_id": str(case.id),
                    "from": reverted.value if reverted else None,
                    "to": scan.final_verdict.value if scan.final_verdict else None,
                    "note": note,
                },
            )
            await uow.commit()
        await self._verdict_changed()
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


def _require_override_rights(actor: Actor) -> None:
    """Regulators who review cases, and platform admins."""
    organization = actor.organization
    regulator = organization is not None and organization.type is OrganizationType.REGULATOR
    if not (actor.is_platform_admin or (regulator and actor.can(Permission.CASE_REVIEW))):
        raise Forbidden("Only regulators and platform admins decide verdicts.")


def _check_allowed(outcome: LabelOutcome, verdict: Verdict, field: str) -> None:
    if verdict not in _ALLOWED_OVERRIDES[outcome]:
        raise ValidationFailed(
            f"A {outcome.value} lab result cannot set the verdict to {verdict.value}.",
            errors=[{"loc": ["body", field], "msg": "not allowed", "type": "value"}],
        )


async def _override(
    session: AsyncSession,
    actor: Actor,
    case: Case,
    scan: Scan,
    verdict: Verdict,
    note: str,
    meta: RequestMeta,
) -> None:
    lab_result = dict(case.lab_result or {})
    before = await override_verdict(
        session,
        scan,
        Override(verdict=verdict, case_id=case.id, lab_result=lab_result, note=note),
        by=actor.user_id,
    )
    await _scan_audit(
        session,
        "scan.verdict_overridden",
        actor,
        scan,
        meta,
        {
            "case_id": str(case.id),
            "from": before.value if before else None,
            "to": verdict.value,
            "pipeline_verdict": scan.pipeline_verdict.value if scan.pipeline_verdict else None,
            "lab_outcome": lab_result.get("outcome"),
            "note": note,
        },
    )


async def _scan_audit(
    session: AsyncSession,
    action: str,
    actor: Actor,
    scan: Scan,
    meta: RequestMeta,
    details: dict[str, object],
) -> None:
    await AuditService(session).record(
        action,
        actor_user_id=actor.user_id,
        actor_org_id=actor.organization.id if actor.organization else None,
        resource_type="scan",
        resource_id=scan.id,
        meta=meta,
        details=details,
    )


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

from typing import Annotated
from uuid import UUID

from fastapi import Depends, Query, Response
from sqlalchemy import ColumnElement

from medlens.core.openapi import problem_responses
from medlens.core.pagination import Cursor, Page, PageParams, build_page, page_params
from medlens.core.routing import api_v1_router
from medlens.modules.auth.deps import Meta as RequestMetaDep
from medlens.modules.cases.models import Case, CasePriority, CaseStatus
from medlens.modules.cases.schemas import (
    CaseAssignment,
    CaseRead,
    CaseScan,
    CaseUpdate,
    LabResult,
    OverrideRevert,
    VerdictOverride,
)
from medlens.modules.cases.service import CaseService
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.modules.scans.models import Scan
from medlens.ports.storage import StoragePort
from medlens.resources import Resources, get_resources, get_storage

router = api_v1_router("/cases", tags=["cases"])
Reader = Annotated[Actor, Depends(require_permissions(Permission.CASE_READ))]
Reviewer = Annotated[
    Actor, Depends(require_permissions(Permission.CASE_READ, Permission.CASE_REVIEW))
]
Assigner = Annotated[
    Actor, Depends(require_permissions(Permission.CASE_READ, Permission.CASE_ASSIGN))
]


def get_case_service(
    resources: Annotated[Resources, Depends(get_resources)],
    storage: Annotated[StoragePort, Depends(get_storage)],
) -> CaseService:
    return CaseService(resources.session_factory, storage, resources.redis)


ServiceDep = Annotated[CaseService, Depends(get_case_service)]


def _read(row: tuple[Case, Scan]) -> CaseRead:
    case, scan = row
    return CaseRead(
        id=case.id,
        status=case.status,
        priority=case.priority,
        state=case.state,
        assigned_org_id=case.assigned_org_id,
        assigned_to_user_id=case.assigned_to_user_id,
        notes=case.notes,
        lab_result=case.lab_result,
        scan=CaseScan(
            id=scan.id,
            pipeline_verdict=scan.pipeline_verdict,
            final_verdict=scan.final_verdict,
            reviewed=scan.reviewed,
            reason_codes=list(scan.reason_codes),
            signals=scan.signals,
            extracted=scan.extracted,
            has_image=scan.image_s3_key is not None,
            created_at=scan.created_at,
        ),
        created_at=case.created_at,
        updated_at=case.updated_at,
        resolved_at=case.resolved_at,
    )


@router.get(
    "",
    response_model=Page[CaseRead],
    summary="List review cases",
    description=(
        "Cases in your jurisdiction (national regulators: all) and cases routed to your "
        "organization, newest first."
    ),
    responses=problem_responses(401, 403, 422),
)
async def list_cases(
    actor: Reader,
    service: ServiceDep,
    params: Annotated[PageParams, Depends(page_params)],
    case_status: Annotated[CaseStatus | None, Query(alias="status")] = None,
    priority: CasePriority | None = None,
    assigned_to_me: Annotated[bool, Query(description="Only cases assigned to you.")] = False,
) -> Page[CaseRead]:
    filters: list[ColumnElement[bool]] = []
    if case_status is not None:
        filters.append(Case.status == case_status)
    if priority is not None:
        filters.append(Case.priority == priority)
    if assigned_to_me:
        filters.append(Case.assigned_to_user_id == actor.user_id)
    rows = await service.list_page(actor, params, *filters)
    return build_page(
        rows,
        limit=params.limit,
        cursor_of=lambda row: Cursor(created_at=row[0].created_at, id=row[0].id),
        to_item=_read,
    )


@router.get(
    "/{case_id}",
    response_model=CaseRead,
    summary="Get a case",
    description="Cases outside your jurisdiction answer 404.",
    responses=problem_responses(401, 403, 404),
)
async def get_case(case_id: UUID, actor: Reader, service: ServiceDep) -> CaseRead:
    return _read(await service.get(actor, case_id))


@router.post(
    "/{case_id}/claim",
    response_model=CaseRead,
    summary="Take a case",
    description="Assigns an open case to you and moves it to `in_review`.",
    responses=problem_responses(401, 403, 404, 409),
)
async def claim_case(
    case_id: UUID, actor: Reviewer, service: ServiceDep, meta: RequestMetaDep
) -> CaseRead:
    return _read(await service.claim(actor, case_id, meta))


@router.post(
    "/{case_id}/assign",
    response_model=CaseRead,
    summary="Assign a case to a reviewer",
    description=(
        "Assigns an open case to an active member of the regulator handling it whose role may "
        "review cases."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def assign_case(
    case_id: UUID,
    body: CaseAssignment,
    actor: Assigner,
    service: ServiceDep,
    meta: RequestMetaDep,
) -> CaseRead:
    return _read(await service.assign(actor, case_id, body.user_id, meta))


@router.patch(
    "/{case_id}",
    response_model=CaseRead,
    summary="Update a case's status or notes",
    description=(
        "Moves an open case to `in_review`, `inconclusive` or `closed`, or edits its notes. "
        "Closed and confirmed cases can no longer change."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def update_case(
    case_id: UUID, body: CaseUpdate, actor: Reviewer, service: ServiceDep, meta: RequestMetaDep
) -> CaseRead:
    return _read(await service.update(actor, case_id, body.status, body.notes, meta))


@router.post(
    "/{case_id}/lab-result",
    response_model=CaseRead,
    summary="Record the lab result",
    description=(
        "Resolves the case as confirmed counterfeit, confirmed genuine or inconclusive, and "
        "stores the outcome as a training label. With `override_verdict`, the verdict shown for "
        "the scan changes (audited); the pipeline's verdict is kept alongside. This is the only "
        "way a verdict changes."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def record_lab_result(
    case_id: UUID, body: LabResult, actor: Reviewer, service: ServiceDep, meta: RequestMetaDep
) -> CaseRead:
    return _read(
        await service.record_lab_result(
            actor,
            case_id,
            body.outcome,
            body.lab_reference,
            body.notes,
            meta,
            override=body.override_verdict,
            override_note=body.override_note,
        )
    )


@router.post(
    "/{case_id}/override",
    response_model=CaseRead,
    summary="Decide the scan's final verdict",
    description=(
        "After the lab result, sets (or changes) the final verdict clients see, with a required "
        "note: counterfeit allows SUSPICIOUS or UNREGISTERED, genuine allows LIKELY_GENUINE. "
        "Also on resolved cases. The pipeline's verdict is kept; the decision is audited and can "
        "be reverted. Regulators and platform admins only."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def override_verdict(
    case_id: UUID, body: VerdictOverride, actor: Reviewer, service: ServiceDep, meta: RequestMetaDep
) -> CaseRead:
    return _read(await service.override(actor, case_id, body.verdict, body.note, meta))


@router.post(
    "/{case_id}/override/revert",
    response_model=CaseRead,
    summary="Revert a verdict decision",
    description=(
        "Returns the scan to the pipeline's verdict, with a required note. The reverted "
        "decision stays in the audit log. Regulators and platform admins only."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def revert_override(
    case_id: UUID, body: OverrideRevert, actor: Reviewer, service: ServiceDep, meta: RequestMetaDep
) -> CaseRead:
    return _read(await service.revert_override(actor, case_id, body.note, meta))


@router.get(
    "/{case_id}/image",
    response_class=Response,
    summary="Download the scan's photo",
    description=(
        "The re-encoded JPEG the scan was verified from (EXIF stripped), for the case's "
        "reviewers. Every download is audited. Photos past the retention period are gone (404)."
    ),
    responses={
        200: {
            "content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}},
            "description": "The photo (JPEG).",
        },
        **problem_responses(401, 403, 404),
    },
)
async def download_case_image(
    case_id: UUID, actor: Reader, service: ServiceDep, meta: RequestMetaDep
) -> Response:
    image = await service.image(actor, case_id, meta)
    return Response(
        image,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="case-{case_id}.jpg"',
        },
    )

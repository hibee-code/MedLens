from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, Form, Header, Query, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy import ColumnElement

from medlens.core.config import Settings
from medlens.core.devices import device_hash
from medlens.core.errors import Unauthorized, ValidationFailed
from medlens.core.idempotency import validate_key
from medlens.core.identifiers import NigerianState
from medlens.core.openapi import problem_responses
from medlens.core.pagination import Cursor, Page, PageParams, build_page, page_params
from medlens.core.routing import api_v1_router
from medlens.core.schemas import Latitude, Longitude, NafdacNumber
from medlens.core.uploads import ValidatedImage, validate_image_upload
from medlens.modules.products.models import Product
from medlens.modules.rbac.deps import CurrentActor, OptionalActor
from medlens.modules.scans.models import Scan, ScanStatus, Verdict
from medlens.modules.scans.schemas import ProductSummary, ScanAccepted, ScanAnalysis, ScanRead
from medlens.modules.scans.service import (
    RedisScanWaiter,
    ScanService,
    ScanSubmission,
    Submitter,
)
from medlens.ports.storage import StoragePort
from medlens.ratelimit.deps import limit_scans
from medlens.resources import Resources, get_app_settings, get_resources, get_storage

router = api_v1_router("/scans", tags=["scans"])
_DUPLICATE: dict[int | str, dict[str, Any]] = {
    200: {
        "model": ScanAccepted,
        "description": (
            "The same photo was already scanned (`duplicate`), or the verdict was ready within "
            "the sync wait (`result`)."
        ),
    }
}
DeviceHash = Annotated[str | None, Depends(device_hash)]


def get_scan_service(
    resources: Annotated[Resources, Depends(get_resources)],
    storage: Annotated[StoragePort, Depends(get_storage)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> ScanService:
    return ScanService(
        resources.session_factory,
        storage,
        settings.verification,
        queue=settings.queue,
        queue_depth=resources.scan_queue,
        waiter=RedisScanWaiter(resources.redis),
    )


ServiceDep = Annotated[ScanService, Depends(get_scan_service)]


def _read(scan: Scan, product: Product | None, *, analysis: bool) -> ScanRead:
    return ScanRead(
        id=scan.id,
        status=scan.status,
        final_verdict=scan.final_verdict,
        reviewed=scan.reviewed,
        confidence=scan.confidence,
        reason_codes=list(scan.reason_codes),
        advice_codes=list(scan.advice_codes),
        product=(
            ProductSummary(
                nafdac_number=product.nafdac_number,
                brand_name=product.brand_name,
                generic_name=product.generic_name,
            )
            if product
            else None
        ),
        created_at=scan.created_at,
        updated_at=scan.updated_at,
        analysis=(
            ScanAnalysis(
                risk_score=scan.risk_score,
                scorer_version=scan.scorer_version,
                signals=scan.signals,
                extracted=scan.extracted,
                quality=scan.quality,
                state=scan.state,
                geocell=scan.geocell,
                processing_ms=scan.processing_ms,
                error_code=scan.error_code,
                llm_used=scan.llm_used,
                pipeline_verdict=scan.pipeline_verdict,
                override_case_id=scan.override_case_id,
                overridden_at=scan.overridden_at,
                override_note=scan.override_note,
            )
            if analysis
            else None
        ),
    )


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ScanAccepted,
    summary="Submit a pack photo for verification",
    description=(
        "Multipart form: `image` (JPEG, PNG or WebP) and optionally `latitude`/`longitude`, "
        "`state` and the `nafdac_number` printed on the pack. Open to guests, who must send "
        "`X-Device-Id`. Returns 202 with the scan id; poll `status_url` for the verdict.\n\n"
        "Send an `Idempotency-Key` header to retry safely: the same key and request replay the "
        "first response; the same key with a different request is a 409. The same photo from "
        "the same person or device within 24 hours returns the earlier scan with 200 and "
        "`duplicate: true`. With the sync fast path on (`MEDLENS_QUEUE__SYNC_WAIT_SECONDS`) "
        "and a short queue, the request may wait a few seconds and answer 200 with `result`. "
        "When the queue is very long it answers 503 with Retry-After.\n\n"
        "Privacy: the photo is re-encoded (metadata removed) before storage; coordinates are "
        "rounded to about 1 km and never stored raw."
    ),
    responses=_DUPLICATE | problem_responses(401, 403, 409, 413, 415, 422, 429, 503),
    dependencies=[Depends(limit_scans)],
)
async def submit_scan(
    image: Annotated[ValidatedImage, Depends(validate_image_upload)],
    actor: OptionalActor,
    device: DeviceHash,
    service: ServiceDep,
    latitude: Annotated[Latitude | None, Form()] = None,
    longitude: Annotated[Longitude | None, Form()] = None,
    state: Annotated[
        NigerianState | None, Form(description="State where the pack was found (code or name).")
    ] = None,
    nafdac_number: Annotated[
        NafdacNumber | None, Form(description="The NAFDAC number printed on the pack, if typed.")
    ] = None,
    idempotency_key: Annotated[
        str | None, Header(alias="Idempotency-Key", description="8-128 characters.")
    ] = None,
) -> Response:
    if (latitude is None) != (longitude is None):
        raise ValidationFailed(
            "Send latitude and longitude together.",
            errors=[{"loc": ["body", "longitude"], "msg": "pair required", "type": "missing"}],
        )
    result = await service.submit(
        Submitter(actor=actor, device_hash=device),
        ScanSubmission(
            image=image,
            latitude=latitude,
            longitude=longitude,
            state=state,
            claimed_nafdac_number=nafdac_number,
            idempotency_key=validate_key(idempotency_key) if idempotency_key else None,
        ),
    )
    accepted = ScanAccepted.model_validate(result.body)
    status_code = result.status_code
    if result.finished:
        scan, product, analysis = await service.get(actor, device, result.scan_id)
        accepted = accepted.model_copy(
            update={
                "status": scan.status,
                "estimated_wait_seconds": 0,
                "result": _read(scan, product, analysis=analysis),
            }
        )
        status_code = status.HTTP_200_OK
    return JSONResponse(accepted.model_dump(mode="json"), status_code=status_code)


@router.get(
    "/{scan_id}",
    response_model=ScanRead,
    summary="Get a scan's status and verdict",
    description=(
        "Your own scans (guests: scans made with the same `X-Device-Id`), your organization's "
        "or, for regulators, scans in their jurisdiction. Everyone gets the verdict, confidence, "
        "reason codes and advice codes; regulators and platform admins also get `analysis`, "
        "the evidence behind it. Other scans answer 404."
    ),
    responses=problem_responses(401, 404, 422),
)
async def get_scan(
    scan_id: UUID, actor: OptionalActor, device: DeviceHash, service: ServiceDep
) -> ScanRead:
    if actor is None and device is None:
        raise Unauthorized("Sign in, or send the X-Device-Id the scan was made with.")
    scan, product, analysis = await service.get(actor, device, scan_id)
    return _read(scan, product, analysis=analysis)


@router.get(
    "",
    response_model=Page[ScanRead],
    summary="List scans",
    description=(
        "Newest first. Consumers see their own scans, pharmacy staff their organization's, "
        "regulators those in their jurisdiction, platform admins all."
    ),
    responses=problem_responses(401, 403, 422),
)
async def list_scans(
    actor: CurrentActor,
    service: ServiceDep,
    params: Annotated[PageParams, Depends(page_params)],
    verdict: Annotated[Verdict | None, Query(description="The final verdict.")] = None,
    scan_status: Annotated[ScanStatus | None, Query(alias="status")] = None,
    state: NigerianState | None = None,
    since: Annotated[datetime | None, Query(description="Created at or after (ISO 8601).")] = None,
    until: Annotated[datetime | None, Query(description="Created before (ISO 8601).")] = None,
) -> Page[ScanRead]:
    filters: list[ColumnElement[bool]] = []
    if verdict is not None:
        filters.append(Scan.final_verdict == verdict)
    if scan_status is not None:
        filters.append(Scan.status == scan_status)
    if state is not None:
        filters.append(Scan.state == state)
    if since is not None:
        filters.append(Scan.created_at >= since)
    if until is not None:
        filters.append(Scan.created_at < until)
    rows = await service.list_page(actor, params, *filters)
    return build_page(
        rows,
        limit=params.limit,
        cursor_of=lambda row: Cursor(created_at=row[0].created_at, id=row[0].id),
        to_item=lambda row: _read(row[0], row[1], analysis=False),
    )

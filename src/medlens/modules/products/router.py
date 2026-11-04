from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, Form, Path, Query, Response, status

from medlens.core.openapi import problem_responses
from medlens.core.pagination import Cursor, Page, PageParams, build_page, page_params
from medlens.core.routing import api_v1_router
from medlens.core.uploads import CsvUpload, ValidatedImage, read_csv_upload, validate_image_upload
from medlens.modules.auth.deps import Meta as RequestMetaDep
from medlens.modules.products.models import (
    ImageSide,
    Product,
    ProductClaim,
    ReferenceImage,
    ReviewStatus,
)
from medlens.modules.products.schemas import (
    BatchImportResult,
    BatchRead,
    BatchWrite,
    ClaimCreate,
    ClaimRead,
    ProductRead,
    ReferenceImageRead,
    ReviewDecision,
)
from medlens.modules.products.service import (
    BatchService,
    ProductClaimService,
    ProductService,
    ReferenceImageService,
)
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import CurrentActor, require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.ports.storage import StoragePort
from medlens.resources import Resources, get_resources, get_storage

router = api_v1_router("/products", tags=["products"])
reference_images_router = api_v1_router("/reference-images", tags=["products"])
claims_router = api_v1_router("/product-claims", tags=["products"])

ResourcesDep = Annotated[Resources, Depends(get_resources)]
StorageDep = Annotated[StoragePort, Depends(get_storage)]
Params = Annotated[PageParams, Depends(page_params)]
BatchWriter = Annotated[Actor, Depends(require_permissions(Permission.BATCH_WRITE))]
ImageWriter = Annotated[Actor, Depends(require_permissions(Permission.REFERENCE_IMAGE_WRITE))]
Reviewer = Annotated[Actor, Depends(require_permissions(Permission.PRODUCT_REVIEW, fresh=True))]
Claimant = Annotated[Actor, Depends(require_permissions(Permission.ORG_MANAGE))]
_BATCH_CREATED: dict[int | str, dict[str, Any]] = {
    201: {"description": "Batch created.", "model": BatchRead}
}
NafdacPath = Annotated[
    str,
    Path(max_length=32, description="NAFDAC number, e.g. 04-0005 (case and separators are free)."),
]


def _product(product: Product, score: float | None = None) -> ProductRead:
    return ProductRead(
        nafdac_number=product.nafdac_number,
        brand_name=product.brand_name,
        generic_name=product.generic_name,
        strength=product.strength,
        dosage_form=product.dosage_form,
        pack_size=product.pack_size,
        manufacturer_name=product.manufacturer_name,
        registration_status=product.registration_status,
        registration_expires_on=product.registration_expires_on,
        manufacturer_onboarded=product.manufacturer_org_id is not None,
        match_score=score,
    )


def _reference(image: ReferenceImage, nafdac_number: str) -> ReferenceImageRead:
    return ReferenceImageRead(
        id=image.id,
        nafdac_number=nafdac_number,
        side=image.side,
        status=image.status,
        created_at=image.created_at,
        reviewed_at=image.reviewed_at,
    )


def _cursor(row: Product | ReferenceImage) -> Cursor:
    return Cursor(created_at=row.created_at, id=row.id)


# ------------------------------------------------------------------ public product lookup
@router.get(
    "",
    response_model=Page[ProductRead],
    summary="Search registered products",
    description=(
        "With `query`: an exact NAFDAC number match first, otherwise products whose brand or "
        "generic name is similar (trigram similarity, best first), at most `limit` results and "
        "no further pages. Without `query`: every product, newest first, cursor paginated. "
        "Public: no authentication needed."
    ),
    responses=problem_responses(422),
)
async def list_products(
    resources: ResourcesDep,
    params: Params,
    query: Annotated[
        str | None,
        Query(min_length=2, max_length=100, description="NAFDAC number or (part of) a name."),
    ] = None,
) -> Page[ProductRead]:
    service = ProductService(resources.session_factory)
    if query is not None:
        hits = await service.search(query, params.limit)
        return Page(
            items=[_product(hit.product, hit.score) for hit in hits],
            next_cursor=None,
            has_more=False,
        )
    rows = await service.list_page(params)
    return build_page(rows, limit=params.limit, cursor_of=_cursor, to_item=_product)


@router.get(
    "/{nafdac_number}",
    response_model=ProductRead,
    summary="Look up a product by NAFDAC number",
    description="The registration record. Public: no authentication needed.",
    responses=problem_responses(404),
)
async def get_product(nafdac_number: NafdacPath, resources: ResourcesDep) -> ProductRead:
    return _product(await ProductService(resources.session_factory).get(nafdac_number))


# ------------------------------------------------------------------ batches
@router.get(
    "/{nafdac_number}/batches",
    response_model=Page[BatchRead],
    summary="List a product's declared batches",
    description=(
        "Batches its manufacturer declared, newest first. Visible to the manufacturer, "
        "regulators and platform admins only (batch numbers would help counterfeiters)."
    ),
    responses=problem_responses(401, 403, 404, 422),
)
async def list_batches(
    nafdac_number: NafdacPath, actor: CurrentActor, resources: ResourcesDep, params: Params
) -> Page[BatchRead]:
    rows = await BatchService(resources.session_factory).list_page(actor, nafdac_number, params)
    return build_page(
        rows,
        limit=params.limit,
        cursor_of=lambda batch: Cursor(created_at=batch.created_at, id=batch.id),
        to_item=BatchRead.model_validate,
    )


@router.put(
    "/{nafdac_number}/batches/{batch_number}",
    response_model=BatchRead,
    summary="Declare or update a batch",
    description=(
        "Creates the batch (201) or replaces its details (200). Only the product's registered "
        "manufacturer (an approved product claim) can declare its batches."
    ),
    responses=_BATCH_CREATED | problem_responses(401, 403, 404, 422),
)
async def put_batch(
    nafdac_number: NafdacPath,
    batch_number: str,
    body: BatchWrite,
    actor: BatchWriter,
    resources: ResourcesDep,
    meta: RequestMetaDep,
    response: Response,
) -> BatchRead:
    batch, created = await BatchService(resources.session_factory).put(
        actor, nafdac_number, batch_number, body, meta
    )
    if created:
        response.status_code = status.HTTP_201_CREATED
    return BatchRead.model_validate(batch)


@router.post(
    "/{nafdac_number}/batches/import",
    response_model=BatchImportResult,
    summary="Import batches from a CSV file",
    description=(
        "UTF-8 CSV with the header `batch_number,expires_on[,manufactured_on][,quantity]"
        "[,distribution_states]`; dates as YYYY-MM-DD, states as ISO codes or names separated "
        "by `;`. Up to 5,000 rows and 2 MB. All or nothing: if any row is invalid nothing is "
        "imported and the 422 response lists every problem, with `loc` "
        '`["body", "file", <row>, <column>]` (row 1 is the header).'
    ),
    responses=problem_responses(401, 403, 404, 413, 422),
)
async def import_batches(
    nafdac_number: NafdacPath,
    upload: Annotated[CsvUpload, Depends(read_csv_upload)],
    actor: BatchWriter,
    resources: ResourcesDep,
    meta: RequestMetaDep,
) -> BatchImportResult:
    result = await BatchService(resources.session_factory).import_csv(
        actor, nafdac_number, upload.data, meta
    )
    return BatchImportResult(rows=result.rows, created=result.created, updated=result.updated)


# ------------------------------------------------------------------ reference images
@router.get(
    "/{nafdac_number}/reference-images",
    response_model=Page[ReferenceImageRead],
    summary="List a product's reference images",
    description=(
        "Photos of the genuine pack, newest first, with their review status. Visible to the "
        "product's manufacturer and to reviewers (`product:review`)."
    ),
    responses=problem_responses(401, 403, 404, 422),
)
async def list_reference_images(
    nafdac_number: NafdacPath,
    actor: CurrentActor,
    resources: ResourcesDep,
    storage: StorageDep,
    params: Params,
) -> Page[ReferenceImageRead]:
    service = ReferenceImageService(resources.session_factory, storage)
    images, number = await service.list_page(actor, nafdac_number, params)
    return build_page(
        images,
        limit=params.limit,
        cursor_of=_cursor,
        to_item=lambda image: _reference(image, number),
    )


@router.post(
    "/{nafdac_number}/reference-images",
    status_code=status.HTTP_201_CREATED,
    response_model=ReferenceImageRead,
    summary="Upload a reference image of the genuine pack",
    description=(
        "Multipart upload (`image`, `side`) by the product's registered manufacturer. The image "
        "is validated and re-encoded like scan photos. It is used for visual matching only "
        "after a reviewer approves it."
    ),
    responses=problem_responses(401, 403, 404, 413, 415, 422, 503),
)
async def upload_reference_image(
    nafdac_number: NafdacPath,
    image: Annotated[ValidatedImage, Depends(validate_image_upload)],
    side: Annotated[ImageSide, Form(description="Which side of the pack the photo shows.")],
    actor: ImageWriter,
    resources: ResourcesDep,
    storage: StorageDep,
    meta: RequestMetaDep,
) -> ReferenceImageRead:
    service = ReferenceImageService(resources.session_factory, storage)
    reference, number = await service.upload(actor, nafdac_number, image, side, meta)
    return _reference(reference, number)


@reference_images_router.post(
    "/{image_id}/review",
    response_model=ReferenceImageRead,
    summary="Approve or reject a reference image",
    description=(
        "A human decision (`product:review`); approved images are used for visual matching. "
        "Each image is reviewed once."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def review_reference_image(
    image_id: UUID,
    body: ReviewDecision,
    actor: Reviewer,
    resources: ResourcesDep,
    storage: StorageDep,
    meta: RequestMetaDep,
) -> ReferenceImageRead:
    service = ReferenceImageService(resources.session_factory, storage)
    reference, number = await service.review(actor, image_id, body.status, body.note, meta)
    return _reference(reference, number)


# ------------------------------------------------------------------ product claims
def _claim(row: tuple[ProductClaim, str]) -> ClaimRead:
    claim, number = row
    return ClaimRead(
        id=claim.id,
        organization_id=claim.organization_id,
        nafdac_number=number,
        status=claim.status,
        note=claim.note,
        review_note=claim.review_note,
        created_at=claim.created_at,
        reviewed_at=claim.reviewed_at,
    )


@claims_router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=ClaimRead,
    summary="Claim a registered product for your organization",
    description=(
        "A verified manufacturer's administrator asks to be linked to a product, so it can "
        "declare batches and upload reference images. A regulator or platform admin reviews "
        "the claim."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def create_claim(
    body: ClaimCreate, actor: Claimant, resources: ResourcesDep, meta: RequestMetaDep
) -> ClaimRead:
    service = ProductClaimService(resources.session_factory)
    return _claim(await service.create(actor, body.nafdac_number, body.note, meta))


@claims_router.get(
    "",
    response_model=Page[ClaimRead],
    summary="List product claims",
    description=(
        "Reviewers see every claim; manufacturers see their own organization's. Newest first."
    ),
    responses=problem_responses(401, 403, 422),
)
async def list_claims(
    actor: CurrentActor,
    resources: ResourcesDep,
    params: Params,
    claim_status: Annotated[
        ReviewStatus | None, Query(alias="status", description="Only claims in this status.")
    ] = None,
) -> Page[ClaimRead]:
    rows = await ProductClaimService(resources.session_factory).list_page(
        actor, claim_status, params
    )
    return build_page(
        rows,
        limit=params.limit,
        cursor_of=lambda row: Cursor(created_at=row[0].created_at, id=row[0].id),
        to_item=_claim,
    )


@claims_router.post(
    "/{claim_id}/review",
    response_model=ClaimRead,
    summary="Approve or reject a product claim",
    description=(
        "A human decision (`product:review`). Approval links the product to the claiming "
        "manufacturer, replacing any earlier link (recorded in the audit log)."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def review_claim(
    claim_id: UUID,
    body: ReviewDecision,
    actor: Reviewer,
    resources: ResourcesDep,
    meta: RequestMetaDep,
) -> ClaimRead:
    service = ProductClaimService(resources.session_factory)
    return _claim(await service.review(actor, claim_id, body.status, body.note, meta))

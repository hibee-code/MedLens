from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, Form, Path, Query, Response, status

from medlens.core.openapi import problem_responses
from medlens.core.pagination import Cursor, Page, PageParams, build_page, page_params
from medlens.core.routing import api_v1_router
from medlens.core.uploads import CsvUpload, ValidatedImage, read_csv_upload, validate_image_upload
from medlens.modules.auth.deps import Meta as RequestMetaDep
from medlens.modules.products.models import (
    INACTIVE_REGISTRATIONS,
    ImageSide,
    Product,
    ProductAuthorization,
    ProductClaim,
    ReferenceImage,
    ReviewStatus,
)
from medlens.modules.products.schemas import (
    AuthorizationCreate,
    AuthorizationRead,
    BatchImportResult,
    BatchRead,
    BatchWrite,
    ClaimCreate,
    ClaimRead,
    IngredientRead,
    ProductRead,
    ReferenceImageRead,
    ReviewDecision,
)
from medlens.modules.products.service import (
    BatchService,
    ProductAuthorizationService,
    ProductClaimService,
    ProductLabel,
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


def _product(
    product: Product, gtins: dict[UUID, list[str]], score: float | None = None
) -> ProductRead:
    return ProductRead(
        nafdac_number=product.nafdac_number,
        brand_name=product.brand_name,
        generic_name=product.generic_name,
        strength=product.strength,
        dosage_form=product.dosage_form,
        pack_size=product.pack_size,
        manufacturer_name=product.manufacturer_name,
        registration_status=product.registration_status,
        registration_inactive=product.registration_status in INACTIVE_REGISTRATIONS,
        registration_date=product.registration_date,
        registration_expires_on=product.registration_expires_on,
        active_ingredients=[
            IngredientRead.model_validate(item) for item in product.active_ingredients or []
        ],
        source_url=product.source_url,
        retrieved_on=product.retrieved_on,
        gtins=gtins.get(product.id, []),
        manufacturer_onboarded=product.manufacturer_org_id is not None,
        match_score=score,
    )


def _reference(image: ReferenceImage, label: ProductLabel) -> ReferenceImageRead:
    return ReferenceImageRead(
        id=image.id,
        nafdac_number=label.nafdac_number,
        side=image.side,
        status=image.status,
        created_at=image.created_at,
        reviewed_at=image.reviewed_at,
        registration_inactive=label.registration_inactive,
    )


def _batch(batch: Any, label: ProductLabel) -> BatchRead:
    read = BatchRead.model_validate(batch)
    return read.model_copy(update={"registration_inactive": label.registration_inactive})


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
        gtins = await service.gtins([hit.product for hit in hits])
        return Page(
            items=[_product(hit.product, gtins, hit.score) for hit in hits],
            next_cursor=None,
            has_more=False,
        )
    rows = await service.list_page(params)
    gtins = await service.gtins(rows)
    return build_page(
        rows, limit=params.limit, cursor_of=_cursor, to_item=lambda row: _product(row, gtins)
    )


@router.get(
    "/{nafdac_number}",
    response_model=ProductRead,
    summary="Look up a product by NAFDAC number",
    description="The registration record. Public: no authentication needed.",
    responses=problem_responses(404),
)
async def get_product(nafdac_number: NafdacPath, resources: ResourcesDep) -> ProductRead:
    service = ProductService(resources.session_factory)
    product = await service.get(nafdac_number)
    return _product(product, await service.gtins([product]))


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
    service = BatchService(resources.session_factory)
    rows, label = await service.list_page(actor, nafdac_number, params)
    return build_page(
        rows,
        limit=params.limit,
        cursor_of=lambda batch: Cursor(created_at=batch.created_at, id=batch.id),
        to_item=lambda batch: _batch(batch, label),
    )


@router.put(
    "/{nafdac_number}/batches/{batch_number}",
    response_model=BatchRead,
    summary="Declare or update a batch",
    description=(
        "Creates the batch (201) or replaces its details (200). Only the product's registration "
        "holder (linked by an approved product claim) or an organisation it authorised "
        "(recorded by a platform admin) can declare its batches."
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
    batch, created, label = await BatchService(resources.session_factory).put(
        actor, nafdac_number, batch_number, body, meta
    )
    if created:
        response.status_code = status.HTTP_201_CREATED
    return _batch(batch, label)


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
    images, label = await service.list_page(actor, nafdac_number, params)
    return build_page(
        images,
        limit=params.limit,
        cursor_of=_cursor,
        to_item=lambda image: _reference(image, label),
    )


@router.post(
    "/{nafdac_number}/reference-images",
    status_code=status.HTTP_201_CREATED,
    response_model=ReferenceImageRead,
    summary="Upload a reference image of the genuine pack",
    description=(
        "Multipart upload (`image`, `side`) by the product's registration holder or an "
        "organisation it authorised. The image "
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
    reference, label = await service.upload(actor, nafdac_number, image, side, meta)
    return _reference(reference, label)


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
    reference, label = await service.review(actor, image_id, body.status, body.note, meta)
    return _reference(reference, label)


# ------------------------------------------------------------------ authorisations
def _authorization(row: ProductAuthorization, label: ProductLabel) -> AuthorizationRead:
    return AuthorizationRead(
        id=row.id,
        nafdac_number=label.nafdac_number,
        organization_id=row.organization_id,
        note=row.note,
        active=row.revoked_at is None,
        created_at=row.created_at,
        revoked_at=row.revoked_at,
    )


@router.post(
    "/{nafdac_number}/authorizations",
    status_code=status.HTTP_201_CREATED,
    response_model=AuthorizationRead,
    summary="Authorise another organisation for a product",
    description=(
        "A platform admin records that the registration holder authorises an organisation "
        "(a contract manufacturer, an importer) to declare batches and upload reference images "
        "for this product; the holder's evidence goes in `note`. The product must have a linked "
        "holder (an approved product claim). Audited."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def grant_authorization(
    nafdac_number: NafdacPath,
    body: AuthorizationCreate,
    actor: CurrentActor,
    resources: ResourcesDep,
    meta: RequestMetaDep,
) -> AuthorizationRead:
    service = ProductAuthorizationService(resources.session_factory)
    row, label = await service.grant(actor, nafdac_number, body.organization_id, body.note, meta)
    return _authorization(row, label)


@router.get(
    "/{nafdac_number}/authorizations",
    response_model=list[AuthorizationRead],
    summary="List a product's authorised organisations",
    description=(
        "Current and revoked authorisations, newest first. For platform admins and the "
        "registration holder."
    ),
    responses=problem_responses(401, 403, 404, 422),
)
async def list_authorizations(
    nafdac_number: NafdacPath, actor: CurrentActor, resources: ResourcesDep
) -> list[AuthorizationRead]:
    service = ProductAuthorizationService(resources.session_factory)
    rows, label = await service.list_for_product(actor, nafdac_number)
    return [_authorization(row, label) for row in rows]


@router.post(
    "/{nafdac_number}/authorizations/{authorization_id}/revoke",
    response_model=AuthorizationRead,
    summary="Revoke an authorisation",
    description=(
        "The organisation can no longer declare batches or upload reference images for the "
        "product; what it declared stays. Platform admins only; audited."
    ),
    responses=problem_responses(401, 403, 404, 409, 422),
)
async def revoke_authorization(
    nafdac_number: NafdacPath,
    authorization_id: UUID,
    actor: CurrentActor,
    resources: ResourcesDep,
    meta: RequestMetaDep,
) -> AuthorizationRead:
    service = ProductAuthorizationService(resources.session_factory)
    row, label = await service.revoke(actor, nafdac_number, authorization_id, meta)
    return _authorization(row, label)


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

"""Registered products, batches, reference images and manufacturers' product claims.

Products are public. Batches are not: batch numbers would help counterfeiters label fakes, so
only the product's manufacturer, regulators and platform admins see them. Reference images and
claims change what the pipeline trusts, so a person with ``product:review`` approves each one
(CLAUDE.md §5); only approved reference images are used for visual matching.
"""

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import structlog
from pydantic import ValidationError
from sqlalchemy import ColumnElement, true
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.errors import Conflict, Forbidden, NotFound, ValidationFailed
from medlens.core.identifiers import normalize_batch_number, normalize_nafdac_number
from medlens.core.ids import uuid7
from medlens.core.pagination import PageParams
from medlens.core.request_meta import RequestMeta
from medlens.core.uploads import ValidatedImage
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.organizations.models import (
    Organization,
    OrganizationStatus,
    OrganizationType,
)
from medlens.modules.products.models import (
    INACTIVE_REGISTRATIONS,
    Batch,
    ImageSide,
    Product,
    ProductAuthorization,
    ProductClaim,
    ReferenceImage,
    RegistrationStatus,
    ReviewStatus,
)
from medlens.modules.products.repository import (
    BatchRepository,
    ProductAuthorizationRepository,
    ProductClaimRepository,
    ProductRepository,
    ReferenceImageRepository,
)
from medlens.modules.products.schemas import BatchWrite
from medlens.modules.rbac import policies
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.permissions import Permission
from medlens.ports.storage import StoragePort
from medlens.queue import outbox

PRODUCT_NOT_FOUND = "No registered product has this NAFDAC number."
MAX_IMPORT_ROWS = 5000
IMPORT_COLUMNS = (
    "batch_number",
    "expires_on",
    "manufactured_on",
    "quantity",
    "distribution_states",  # codes or names, separated by ; or ,
)

log = structlog.stdlib.get_logger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ProductLabel:
    """What responses about a product's batches and images show of the product itself."""

    nafdac_number: str
    registration_status: RegistrationStatus

    @property
    def registration_inactive(self) -> bool:
        """Suspended or revoked: its batches and images are kept but flagged."""
        return self.registration_status in INACTIVE_REGISTRATIONS

    @classmethod
    def of(cls, product: Product) -> "ProductLabel":
        return cls(product.nafdac_number, product.registration_status)


async def _authorized(session: AsyncSession, product: Product) -> frozenset[UUID]:
    return await ProductAuthorizationRepository(session).active_organizations(product.id)


def _warn_without_holder(product: Product, action: str) -> None:
    """Products whose holder has no linked organisation keep working (platform admins may still
    write to them), but every such write is logged."""
    if product.manufacturer_org_id is None:
        log.warning("products.no_holder_link", nafdac_number=product.nafdac_number, action=action)


def _org_id(actor: Actor) -> UUID | None:
    return actor.organization.id if actor.organization else None


async def _product(session: AsyncSession, nafdac_number: str) -> Product:
    try:
        number = normalize_nafdac_number(nafdac_number)
    except ValueError:
        raise NotFound(PRODUCT_NOT_FOUND) from None
    product = await ProductRepository(session).get_by_nafdac_number(number)
    if product is None:
        raise NotFound(PRODUCT_NOT_FOUND)
    return product


# --------------------------------------------------------------------------- products
@dataclass(frozen=True, slots=True)
class ProductHit:
    product: Product
    score: float | None = None


class ProductService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def search(self, query: str, limit: int) -> list[ProductHit]:
        """An exact NAFDAC number first; otherwise names by trigram similarity, best first."""
        async with UnitOfWork(self._sessions) as uow:
            repo = ProductRepository(uow.session)
            try:
                number = normalize_nafdac_number(query)
            except ValueError:
                number = None
            if number is not None:
                product = await repo.get_by_nafdac_number(number)
                if product is not None:
                    return [ProductHit(product, 1.0)]
            return [
                ProductHit(product, round(score, 3))
                for product, score in await repo.search_names(query, limit=limit)
            ]

    async def list_page(self, params: PageParams) -> list[Product]:
        async with UnitOfWork(self._sessions) as uow:
            return await ProductRepository(uow.session).list_page(params)

    async def get(self, nafdac_number: str) -> Product:
        async with UnitOfWork(self._sessions) as uow:
            return await _product(uow.session, nafdac_number)

    async def gtins(self, products: Sequence[Product]) -> dict[UUID, list[str]]:
        """The products' barcodes (GTINs), by product id."""
        async with UnitOfWork(self._sessions) as uow:
            return await ProductRepository(uow.session).gtins([p.id for p in products])


# --------------------------------------------------------------------------- batches
@dataclass(frozen=True, slots=True)
class ImportResult:
    rows: int
    created: int
    updated: int


class BatchService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def list_page(
        self, actor: Actor, nafdac_number: str, params: PageParams
    ) -> tuple[list[Batch], ProductLabel]:
        async with UnitOfWork(self._sessions) as uow:
            product = await _product(uow.session, nafdac_number)
            authorized = await _authorized(uow.session, product)
            if not policies.can_read_batches(actor, product, authorized):
                raise NotFound(PRODUCT_NOT_FOUND)  # do not confirm the product has batches
            batches = await BatchRepository(uow.session).list_page(
                params, Batch.product_id == product.id
            )
            return batches, ProductLabel.of(product)

    async def put(
        self,
        actor: Actor,
        nafdac_number: str,
        batch_number: str,
        body: BatchWrite,
        meta: RequestMeta,
    ) -> tuple[Batch, bool, ProductLabel]:
        """Create or replace one batch; returns it, whether it was created, and the product."""
        try:
            number = normalize_batch_number(batch_number)
        except ValueError as exc:
            raise ValidationFailed(
                "The batch number is invalid.",
                errors=[{"loc": ["path", "batch_number"], "msg": str(exc), "type": "value"}],
            ) from None
        async with UnitOfWork(self._sessions) as uow:
            product = await self._writable(uow.session, actor, nafdac_number)
            created = await self._upsert(uow.session, actor, product, {number: body})
            batch = await BatchRepository(uow.session).by_number(product.id, number)
            if batch is None:  # just written in this transaction
                raise RuntimeError(f"batch {number} vanished after the upsert")
            await AuditService(uow.session).record(
                "batch.created" if created else "batch.updated",
                actor_user_id=actor.user_id,
                actor_org_id=_org_id(actor),
                resource_type="batch",
                resource_id=batch.id,
                meta=meta,
                details={"nafdac_number": product.nafdac_number, "batch_number": number},
            )
            label = ProductLabel.of(product)
            await uow.commit()
        return batch, bool(created), label

    async def import_csv(
        self, actor: Actor, nafdac_number: str, data: bytes, meta: RequestMeta
    ) -> ImportResult:
        """All or nothing: any invalid row rejects the file with a report of every problem."""
        rows = parse_batch_csv(data)
        async with UnitOfWork(self._sessions) as uow:
            product = await self._writable(uow.session, actor, nafdac_number)
            created = await self._upsert(uow.session, actor, product, rows)
            await AuditService(uow.session).record(
                "batch.imported",
                actor_user_id=actor.user_id,
                actor_org_id=_org_id(actor),
                resource_type="product",
                resource_id=product.id,
                meta=meta,
                details={"rows": len(rows), "created": created},
            )
            await uow.commit()
        return ImportResult(rows=len(rows), created=created, updated=len(rows) - created)

    @staticmethod
    async def _writable(session: AsyncSession, actor: Actor, nafdac_number: str) -> Product:
        product = await _product(session, nafdac_number)
        if not policies.can_write_batches(actor, product, await _authorized(session, product)):
            raise Forbidden(
                "Only the product's registration holder, or an organisation it authorised, "
                "can declare its batches."
            )
        _warn_without_holder(product, "batch_write")
        return product

    @staticmethod
    async def _upsert(
        session: AsyncSession, actor: Actor, product: Product, rows: dict[str, BatchWrite]
    ) -> int:
        existing = await BatchRepository(session).by_numbers(product.id, list(rows))
        for number, body in rows.items():
            batch = existing.get(number)
            if batch is None:
                batch = Batch(product_id=product.id, batch_number=number)
                session.add(batch)
            batch.manufactured_on = body.manufactured_on
            batch.expires_on = body.expires_on
            batch.quantity = body.quantity
            batch.distribution_states = sorted({state.value for state in body.distribution_states})
            batch.declared_by_org_id = _org_id(actor)
        await session.flush()
        return len(rows) - len(existing)


def parse_batch_csv(data: bytes) -> dict[str, BatchWrite]:
    """Batch number -> validated row. Raises ValidationFailed listing every bad row and field;
    ``loc`` is ["body", "file", <row>, <column>] with rows numbered as in a spreadsheet."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValidationFailed("The file must be UTF-8 encoded CSV.") from None
    reader = csv.DictReader(io.StringIO(text))
    header = [name.strip().lower() for name in reader.fieldnames or []]
    missing = [column for column in ("batch_number", "expires_on") if column not in header]
    unknown = [column for column in header if column not in IMPORT_COLUMNS]
    if missing or unknown:
        raise ValidationFailed(
            "The header needs batch_number and expires_on, and may only use: "
            f"{', '.join(IMPORT_COLUMNS)}.",
            errors=[
                {"loc": ["body", "file", 1, column], "msg": "missing column", "type": "missing"}
                for column in missing
            ]
            + [
                {"loc": ["body", "file", 1, column], "msg": "unknown column", "type": "extra"}
                for column in unknown
            ],
        )
    rows: dict[str, BatchWrite] = {}
    errors: list[dict[str, object]] = []
    for line, raw in enumerate(reader, start=2):
        if line - 1 > MAX_IMPORT_ROWS:
            raise ValidationFailed(f"The file has more than {MAX_IMPORT_ROWS} rows; split it.")
        record = {key.strip().lower(): (value or "").strip() for key, value in raw.items() if key}
        try:
            number = normalize_batch_number(record.get("batch_number", ""))
        except ValueError as exc:
            errors.append(_row_error(line, "batch_number", str(exc), "value"))
            continue
        if number in rows:
            errors.append(_row_error(line, "batch_number", "duplicate batch number", "duplicate"))
            continue
        states = [
            part.strip()
            for part in record.get("distribution_states", "").replace(",", ";").split(";")
        ]
        try:
            rows[number] = BatchWrite.model_validate(
                {
                    "expires_on": record.get("expires_on") or None,
                    "manufactured_on": record.get("manufactured_on") or None,
                    "quantity": record.get("quantity") or None,
                    "distribution_states": [state for state in states if state],
                }
            )
        except ValidationError as exc:
            for item in exc.errors(include_url=False, include_input=False, include_context=False):
                column = str(item["loc"][0]) if item["loc"] else "row"
                message = item["msg"].removeprefix("Value error, ")
                errors.append(_row_error(line, column, message, item["type"]))
    if errors:
        raise ValidationFailed(
            f"{len(errors)} problems found; nothing was imported.", errors=errors
        )
    if not rows:
        raise ValidationFailed("The file has no rows.")
    return rows


def _row_error(line: int, column: str, message: str, kind: str) -> dict[str, object]:
    return {"loc": ["body", "file", line, column], "msg": message, "type": kind}


# --------------------------------------------------------------------------- reference images
class ReferenceImageService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], storage: StoragePort) -> None:
        self._sessions = sessions
        self._storage = storage

    async def upload(
        self,
        actor: Actor,
        nafdac_number: str,
        image: ValidatedImage,
        side: ImageSide,
        meta: RequestMeta,
    ) -> tuple[ReferenceImage, ProductLabel]:
        async with UnitOfWork(self._sessions) as uow:
            product = await _product(uow.session, nafdac_number)
            authorized = await _authorized(uow.session, product)
            if not policies.can_upload_reference_images(actor, product, authorized):
                raise Forbidden(
                    "Only the product's registration holder, or an organisation it authorised, "
                    "can upload reference images."
                )
            _warn_without_holder(product, "reference_image_upload")
            product_id, number, label = product.id, product.nafdac_number, ProductLabel.of(product)
        image_id = uuid7()
        key = f"reference-images/{product_id}/{image_id}.jpg"
        # Stored first: a crash before the commit leaves an unreferenced object (swept by the
        # bucket lifecycle), never a row pointing at nothing.
        await self._storage.put(key, image.jpeg, content_type="image/jpeg")
        async with UnitOfWork(self._sessions) as uow:
            reference = ReferenceImage(
                id=image_id,
                product_id=product_id,
                side=side,
                s3_key=key,
                uploaded_by=actor.user_id,
                organization_id=_org_id(actor),
            )
            uow.session.add(reference)
            await outbox.emit(
                uow.session,
                outbox.REFERENCE_IMAGE_UPLOADED,
                aggregate_type="reference_image",
                aggregate_id=image_id,
            )
            await AuditService(uow.session).record(
                "reference_image.uploaded",
                actor_user_id=actor.user_id,
                actor_org_id=_org_id(actor),
                resource_type="reference_image",
                resource_id=image_id,
                meta=meta,
                details={"nafdac_number": number, "side": side.value},
            )
            await uow.commit()
        return reference, label

    async def list_page(
        self, actor: Actor, nafdac_number: str, params: PageParams
    ) -> tuple[list[ReferenceImage], ProductLabel]:
        async with UnitOfWork(self._sessions) as uow:
            product = await _product(uow.session, nafdac_number)
            authorized = await _authorized(uow.session, product)
            if not policies.can_read_reference_images(actor, product, authorized):
                raise NotFound(PRODUCT_NOT_FOUND)
            images = await ReferenceImageRepository(uow.session).list_page(
                params, ReferenceImage.product_id == product.id
            )
            return images, ProductLabel.of(product)

    async def review(
        self,
        actor: Actor,
        image_id: UUID,
        status: ReviewStatus,
        note: str | None,
        meta: RequestMeta,
    ) -> tuple[ReferenceImage, ProductLabel]:
        if not policies.can_review_products(actor):
            raise Forbidden("Reviewing reference images needs the product:review permission.")
        async with UnitOfWork(self._sessions) as uow:
            reference = await ReferenceImageRepository(uow.session).for_update(image_id)
            if reference is None:
                raise NotFound("Reference image not found.")
            if reference.status is not ReviewStatus.PENDING:
                raise Conflict(f"This image was already {reference.status.value}.")
            reference.status = status
            reference.verified_by = actor.user_id
            reference.reviewed_at = _now()
            product = await uow.session.get(Product, reference.product_id)
            await AuditService(uow.session).record(
                f"reference_image.{status.value}",
                actor_user_id=actor.user_id,
                actor_org_id=_org_id(actor),
                resource_type="reference_image",
                resource_id=image_id,
                meta=meta,
                details={"note": note} if note else None,
            )
            if product is None:
                raise NotFound(PRODUCT_NOT_FOUND)
            label = ProductLabel.of(product)
            await uow.commit()
        return reference, label


# --------------------------------------------------------------------------- claims
class ProductClaimService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(
        self, actor: Actor, nafdac_number: str, note: str | None, meta: RequestMeta
    ) -> tuple[ProductClaim, str]:
        organization = actor.organization
        if (
            organization is None
            or organization.type is not OrganizationType.MANUFACTURER
            or not actor.can(Permission.ORG_MANAGE)
        ):
            raise Forbidden("Product claims are made by a manufacturer's administrators.")
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            org = await session.get(Organization, organization.id)
            if org is None or org.status is not OrganizationStatus.ACTIVE:
                raise Conflict("The organization must be verified before it can claim products.")
            product = await _product(session, nafdac_number)
            if product.manufacturer_org_id == org.id:
                raise Conflict("This product is already linked to your organization.")
            claim = ProductClaim(
                organization_id=org.id,
                product_id=product.id,
                note=note,
                requested_by=actor.user_id,
            )
            session.add(claim)
            try:
                await session.flush()
            except IntegrityError:
                raise Conflict("A claim for this product is already pending review.") from None
            await AuditService(session).record(
                "product_claim.created",
                actor_user_id=actor.user_id,
                actor_org_id=org.id,
                resource_type="product_claim",
                resource_id=claim.id,
                meta=meta,
                details={"nafdac_number": product.nafdac_number},
            )
            await uow.commit()
            return claim, product.nafdac_number

    async def list_page(
        self, actor: Actor, status: ReviewStatus | None, params: PageParams
    ) -> list[tuple[ProductClaim, str]]:
        scope: ColumnElement[bool]
        if policies.can_review_products(actor) or actor.is_platform_admin:
            scope = true()
        elif actor.organization is not None and actor.can(Permission.ORG_READ):
            scope = ProductClaim.organization_id == actor.organization.id
        else:
            raise Forbidden("You cannot see product claims.")
        filters = [scope]
        if status is not None:
            filters.append(ProductClaim.status == status)
        async with UnitOfWork(self._sessions) as uow:
            return await ProductClaimRepository(uow.session).page_with_products(params, *filters)

    async def review(
        self,
        actor: Actor,
        claim_id: UUID,
        status: ReviewStatus,
        note: str | None,
        meta: RequestMeta,
    ) -> tuple[ProductClaim, str]:
        if not policies.can_review_products(actor):
            raise Forbidden("Reviewing product claims needs the product:review permission.")
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            claim = await ProductClaimRepository(session).for_update(claim_id)
            if claim is None:
                raise NotFound("Product claim not found.")
            if claim.status is not ReviewStatus.PENDING:
                raise Conflict(f"This claim was already {claim.status.value}.")
            product = await session.get(Product, claim.product_id, with_for_update=True)
            if product is None:
                raise NotFound(PRODUCT_NOT_FOUND)
            previous = product.manufacturer_org_id
            claim.status = status
            claim.reviewed_by = actor.user_id
            claim.reviewed_at = _now()
            claim.review_note = note
            if status is ReviewStatus.APPROVED:
                product.manufacturer_org_id = claim.organization_id
            await AuditService(session).record(
                f"product_claim.{status.value}",
                actor_user_id=actor.user_id,
                actor_org_id=_org_id(actor),
                resource_type="product_claim",
                resource_id=claim.id,
                meta=meta,
                details={
                    "nafdac_number": product.nafdac_number,
                    "organization_id": str(claim.organization_id),
                    "previous_organization_id": str(previous) if previous else None,
                },
            )
            await uow.commit()
            return claim, product.nafdac_number


# --------------------------------------------------------------------------- authorisations
class ProductAuthorizationService:
    """Organisations a registration holder authorises to declare batches and upload reference
    images for one product, recorded (and revoked) by a platform admin on the holder's behalf.
    Audited; revoking keeps the record."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def grant(
        self,
        actor: Actor,
        nafdac_number: str,
        organization_id: UUID,
        note: str | None,
        meta: RequestMeta,
    ) -> tuple[ProductAuthorization, ProductLabel]:
        if not actor.is_platform_admin:
            raise Forbidden("Only platform admins record authorisations.")
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            product = await _product(session, nafdac_number)
            if product.manufacturer_org_id is None:
                raise Conflict(
                    "This product has no linked registration holder: approve the holder's "
                    "product claim first."
                )
            if organization_id == product.manufacturer_org_id:
                raise Conflict("The registration holder needs no authorisation.")
            organization = await session.get(Organization, organization_id)
            if organization is None or organization.status is not OrganizationStatus.ACTIVE:
                raise NotFound("No verified organisation with this id.")
            repository = ProductAuthorizationRepository(session)
            if await repository.active(product.id, organization_id) is not None:
                raise Conflict("This organisation is already authorised for the product.")
            authorization = ProductAuthorization(
                product_id=product.id,
                organization_id=organization_id,
                note=note,
                authorized_by=actor.user_id,
            )
            session.add(authorization)
            await session.flush()
            await AuditService(session).record(
                "product.authorization_granted",
                actor_user_id=actor.user_id,
                actor_org_id=_org_id(actor),
                resource_type="product",
                resource_id=product.id,
                meta=meta,
                details={
                    "nafdac_number": product.nafdac_number,
                    "organization_id": str(organization_id),
                    "holder_organization_id": str(product.manufacturer_org_id),
                },
            )
            label = ProductLabel.of(product)
            await uow.commit()
        return authorization, label

    async def list_for_product(
        self, actor: Actor, nafdac_number: str
    ) -> tuple[list[ProductAuthorization], ProductLabel]:
        async with UnitOfWork(self._sessions) as uow:
            product = await _product(uow.session, nafdac_number)
            if not (actor.is_platform_admin or policies.owns_product(actor, product)):
                raise Forbidden("Only platform admins and the registration holder can see these.")
            rows = await ProductAuthorizationRepository(uow.session).for_product(product.id)
            return rows, ProductLabel.of(product)

    async def revoke(
        self, actor: Actor, nafdac_number: str, authorization_id: UUID, meta: RequestMeta
    ) -> tuple[ProductAuthorization, ProductLabel]:
        if not actor.is_platform_admin:
            raise Forbidden("Only platform admins revoke authorisations.")
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            product = await _product(session, nafdac_number)
            authorization = await ProductAuthorizationRepository(session).for_update(
                authorization_id
            )
            if authorization is None or authorization.product_id != product.id:
                raise NotFound("Authorisation not found.")
            if authorization.revoked_at is not None:
                raise Conflict("This authorisation was already revoked.")
            authorization.revoked_at = _now()
            authorization.revoked_by = actor.user_id
            await AuditService(session).record(
                "product.authorization_revoked",
                actor_user_id=actor.user_id,
                actor_org_id=_org_id(actor),
                resource_type="product",
                resource_id=product.id,
                meta=meta,
                details={
                    "nafdac_number": product.nafdac_number,
                    "organization_id": str(authorization.organization_id),
                },
            )
            label = ProductLabel.of(product)
            await uow.commit()
        return authorization, label

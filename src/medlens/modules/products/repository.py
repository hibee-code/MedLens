from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import ColumnElement, func, or_, select, tuple_

from medlens.core.pagination import PageParams
from medlens.db.repository import Repository
from medlens.modules.products.models import Batch, Product, ProductClaim, ReferenceImage

DEFAULT_SIMILARITY_THRESHOLD = 0.3


class ProductRepository(Repository[Product]):
    model = Product

    async def get_by_nafdac_number(
        self, nafdac_number: str, *, for_update: bool = False
    ) -> Product | None:
        """Exact match on the normalised number (see core/identifiers.py)."""
        statement = self._select().where(Product.nafdac_number == nafdac_number)
        return await self._first(statement.with_for_update() if for_update else statement)

    async def search_names(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[tuple[Product, float]]:
        """Fuzzy match on brand and generic name, best first, absorbing OCR errors.

        Uses pg_trgm's ``%`` operator, which the GIN trigram indexes serve; the threshold is
        set for this transaction only.
        """
        await self.session.execute(
            select(func.set_config("pg_trgm.similarity_threshold", str(threshold), True))
        )
        score = func.greatest(
            func.similarity(Product.brand_name, query),
            func.similarity(Product.generic_name, query),
        ).label("score")
        statement = (
            select(Product, score)
            .where(or_(Product.brand_name.op("%")(query), Product.generic_name.op("%")(query)))
            .order_by(score.desc(), Product.brand_name)
            .limit(limit)
        )
        return [(product, float(value)) for product, value in await self.session.execute(statement)]


class BatchRepository(Repository[Batch]):
    model = Batch

    async def by_number(self, product_id: UUID, batch_number: str) -> Batch | None:
        return await self._first(
            self._select().where(Batch.product_id == product_id, Batch.batch_number == batch_number)
        )

    async def by_numbers(self, product_id: UUID, numbers: Sequence[str]) -> dict[str, Batch]:
        statement = self._select().where(
            Batch.product_id == product_id, Batch.batch_number.in_(list(numbers))
        )
        return {batch.batch_number: batch for batch in await self.session.scalars(statement)}


class ReferenceImageRepository(Repository[ReferenceImage]):
    model = ReferenceImage

    async def for_update(self, image_id: UUID) -> ReferenceImage | None:
        return await self._first(
            self._select().where(ReferenceImage.id == image_id).with_for_update()
        )


class ProductClaimRepository(Repository[ProductClaim]):
    model = ProductClaim

    async def for_update(self, claim_id: UUID) -> ProductClaim | None:
        return await self._first(
            self._select().where(ProductClaim.id == claim_id).with_for_update()
        )

    async def page_with_products(
        self, params: PageParams, *filters: ColumnElement[bool]
    ) -> list[tuple[ProductClaim, str]]:
        statement = (
            select(ProductClaim, Product.nafdac_number)
            .join(Product, Product.id == ProductClaim.product_id)
            .where(*filters)
        )
        if params.after is not None:
            position = tuple_(params.after.created_at, params.after.id)
            statement = statement.where(tuple_(ProductClaim.created_at, ProductClaim.id) < position)
        statement = statement.order_by(
            ProductClaim.created_at.desc(), ProductClaim.id.desc()
        ).limit(params.limit + 1)
        return [(claim, number) for claim, number in await self.session.execute(statement)]

"""Scans are partitioned by month on ``created_at``. Lookups by id add a window around the time
encoded in the UUIDv7 id, so Postgres prunes to one partition and uses the primary key."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ColumnElement, Select, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from medlens.core.pagination import PageParams
from medlens.modules.products.models import Product
from medlens.modules.scans.models import Scan


def id_window(scan_id: UUID) -> ColumnElement[bool]:
    return Scan.id_window(scan_id)


def _with_product() -> Select[Scan, Product]:
    """Product is None at runtime when the scan matched none (outer join)."""
    return select(Scan, Product).outerjoin(Product, Product.id == Scan.matched_product_id)


class ScanRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(
        self, scan_id: UUID, visibility: ColumnElement[bool]
    ) -> tuple[Scan, Product | None] | None:
        result = await self.session.execute(_with_product().where(id_window(scan_id), visibility))
        row = result.first()
        if row is None:
            return None
        scan, product = row
        return scan, product

    async def page(
        self, visibility: ColumnElement[bool], params: PageParams, *filters: ColumnElement[bool]
    ) -> list[tuple[Scan, Product | None]]:
        statement = _with_product().where(visibility, *filters)
        if params.after is not None:
            position = tuple_(params.after.created_at, params.after.id)
            statement = statement.where(tuple_(Scan.created_at, Scan.id) < position)
        statement = statement.order_by(Scan.created_at.desc(), Scan.id.desc()).limit(
            params.limit + 1
        )
        return [(scan, product) for scan, product in await self.session.execute(statement)]

    async def recent_duplicate(
        self, *, sha256: str, owner: ColumnElement[bool], since: datetime
    ) -> Scan | None:
        """The latest scan of exactly this photo by the same person or device."""
        statement = (
            select(Scan)
            .where(Scan.image_sha256 == sha256, owner, Scan.created_at >= since)
            .order_by(Scan.created_at.desc())
            .limit(1)
        )
        return (await self.session.scalars(statement)).first()

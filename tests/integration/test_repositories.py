"""Repository base classes against a real database: tenant scoping, soft delete, keyset pages."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from medlens.core.pagination import Cursor, PageParams, build_page, decode_cursor
from medlens.core.tenancy import TenantContext
from medlens.modules.organizations.models import Branch, Organization, OrganizationType
from medlens.modules.organizations.repository import BranchRepository
from medlens.modules.products.models import Product, RegistrationStatus
from medlens.modules.products.repository import ProductRepository

pytestmark = pytest.mark.integration


async def organization(session: AsyncSession, name: str) -> Organization:
    org = Organization(name=name, type=OrganizationType.PHARMACY)
    session.add(org)
    await session.flush()
    return org


async def test_tenants_only_see_their_own_rows(db_session: AsyncSession) -> None:
    alpha, beta = await organization(db_session, "Alpha"), await organization(db_session, "Beta")
    repository = BranchRepository(db_session)
    alpha_ctx = TenantContext(user_id=None, organization_id=alpha.id)
    beta_ctx = TenantContext(user_id=None, organization_id=beta.id)
    branch = await repository.add(alpha_ctx, Branch(organization_id=alpha.id, name="Ikeja"))

    assert (await repository.get(alpha_ctx, branch.id)) is branch
    assert (await repository.get(beta_ctx, branch.id)) is None  # looks like "not found"
    assert await repository.list_page(beta_ctx, PageParams(limit=10, after=None)) == []
    no_org = TenantContext(user_id=None, organization_id=None)
    assert (await repository.get(no_org, branch.id)) is None
    assert (await repository.get(TenantContext.system(), branch.id)) is branch


async def test_tenants_cannot_add_rows_for_another_tenant(db_session: AsyncSession) -> None:
    alpha, beta = await organization(db_session, "Alpha"), await organization(db_session, "Beta")
    alpha_ctx = TenantContext(user_id=None, organization_id=alpha.id)

    with pytest.raises(PermissionError):
        await BranchRepository(db_session).add(
            alpha_ctx, Branch(organization_id=beta.id, name="Not yours")
        )


async def test_soft_deleted_rows_are_hidden(db_session: AsyncSession) -> None:
    alpha = await organization(db_session, "Alpha")
    ctx = TenantContext(user_id=None, organization_id=alpha.id)
    repository = BranchRepository(db_session)
    branch = await repository.add(ctx, Branch(organization_id=alpha.id, name="Closed"))

    branch.deleted_at = datetime.now(UTC)
    await db_session.flush()

    assert (await repository.get(ctx, branch.id)) is None


async def test_keyset_pagination_walks_every_row_once(db_session: AsyncSession) -> None:
    start = datetime(2026, 11, 1, tzinfo=UTC)
    products = [
        Product(
            nafdac_number=f"A4-{1000 + index}",
            brand_name=f"Brand {index}",
            generic_name="Paracetamol",
            registration_status=RegistrationStatus.ACTIVE,
            # Pairs share a timestamp, so ties must be broken by id.
            created_at=start - timedelta(minutes=index // 2),
        )
        for index in range(25)
    ]
    db_session.add_all(products)
    await db_session.flush()
    repository = ProductRepository(db_session)

    seen: list[str] = []
    after = None
    for _ in range(10):
        rows = await repository.list_page(
            PageParams(limit=10, after=after), Product.nafdac_number.like("A4-1%")
        )
        page = build_page(
            rows,
            limit=10,
            cursor_of=_cursor,
            to_item=lambda row: row.nafdac_number,
        )
        seen.extend(page.items)
        if not page.has_more:
            break
        assert page.next_cursor is not None
        after = decode_cursor(page.next_cursor)

    assert sorted(seen) == sorted(p.nafdac_number for p in products)
    assert len(seen) == len(set(seen)) == 25


def _cursor(product: Product) -> Cursor:
    return Cursor(created_at=product.created_at, id=product.id)

"""Repository base classes: the only place that writes SQL (CLAUDE.md §3).

``Repository``        data visible to everyone who may read it at all (products, batches).
``TenantRepository``  tenant data. Every public method takes a ``TenantContext`` and the
                      subclass's ``tenant_clause`` is applied to every query, so forgetting the
                      tenant is a type error rather than a data leak.

Both page with keyset (cursor) pagination, newest first by (created_at, id), and hide rows with
``deleted_at`` set for soft-deletable models.
"""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from sqlalchemy import ColumnElement, Select, false, select, true, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from medlens.core.pagination import PageParams
from medlens.core.tenancy import TenantContext
from medlens.db.base import Entity


class _RepositoryBase[ModelT: Entity]:
    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _select(self) -> Select[ModelT]:
        statement = select(self.model)
        deleted_at = getattr(self.model, "deleted_at", None)  # models with the SoftDelete mixin
        if deleted_at is not None:
            statement = statement.where(deleted_at.is_(None))
        return statement

    async def _first(self, statement: Select[ModelT]) -> ModelT | None:
        return (await self.session.scalars(statement.limit(1))).first()

    async def _page(self, statement: Select[ModelT], params: PageParams) -> list[ModelT]:
        """Up to ``limit + 1`` rows; pass them to ``core.pagination.build_page``."""
        created_at, row_id = self.model.created_at, self.model.id
        if params.after is not None:
            position = tuple_(params.after.created_at, params.after.id)
            statement = statement.where(tuple_(created_at, row_id) < position)
        statement = statement.order_by(created_at.desc(), row_id.desc()).limit(params.limit + 1)
        return list(await self.session.scalars(statement))

    async def _add(self, entity: ModelT) -> ModelT:
        self.session.add(entity)
        await self.session.flush()
        return entity


class Repository[ModelT: Entity](_RepositoryBase[ModelT]):
    """For data that is not owned by a tenant."""

    async def get(self, entity_id: UUID) -> ModelT | None:
        return await self._first(self._select().where(self.model.id == entity_id))

    async def list_page(self, params: PageParams, *filters: ColumnElement[bool]) -> list[ModelT]:
        return await self._page(self._select().where(*filters), params)

    async def add(self, entity: ModelT) -> ModelT:
        return await self._add(entity)


class TenantRepository[ModelT: Entity](_RepositoryBase[ModelT], ABC):
    """For tenant-owned data. Subclasses define which rows a tenant may see."""

    @abstractmethod
    def tenant_clause(self, tenant: TenantContext) -> ColumnElement[bool]:
        """SQL condition selecting the rows visible to ``tenant``."""

    @abstractmethod
    def belongs_to(self, tenant: TenantContext, entity: ModelT) -> bool:
        """Whether ``tenant`` may create or modify ``entity``."""

    def scoped(self, tenant: TenantContext) -> Select[ModelT]:
        return self._select().where(self.tenant_clause(tenant))

    async def get(self, tenant: TenantContext, entity_id: UUID) -> ModelT | None:
        """None both when the row does not exist and when it belongs to another tenant."""
        return await self._first(self.scoped(tenant).where(self.model.id == entity_id))

    async def list_page(
        self, tenant: TenantContext, params: PageParams, *filters: ColumnElement[bool]
    ) -> list[ModelT]:
        return await self._page(self.scoped(tenant).where(*filters), params)

    async def add(self, tenant: TenantContext, entity: ModelT) -> ModelT:
        if not self.belongs_to(tenant, entity):
            raise PermissionError("entity belongs to another tenant")
        return await self._add(entity)


class OrganizationScopedRepository[ModelT: Entity](TenantRepository[ModelT]):
    """Tenant data owned by one organization through an ``organization_id`` column."""

    def _organization_column(self) -> Any:
        return self.model.organization_id  # type: ignore[attr-defined]

    def tenant_clause(self, tenant: TenantContext) -> ColumnElement[bool]:
        if tenant.unrestricted:
            return true()
        if tenant.organization_id is None:
            return false()
        clause: ColumnElement[bool] = self._organization_column() == tenant.organization_id
        return clause

    def belongs_to(self, tenant: TenantContext, entity: ModelT) -> bool:
        if tenant.unrestricted:
            return True
        owner = getattr(entity, "organization_id", None)
        return tenant.organization_id is not None and owner == tenant.organization_id

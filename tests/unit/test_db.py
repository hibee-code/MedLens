"""Database layer pieces that need no database."""

from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql

from medlens.core.config import Settings
from medlens.core.tenancy import TenantContext
from medlens.db.all_models import Base
from medlens.db.migration_support import migration_settings
from medlens.db.partitions import (
    default_partition_ddl,
    ensure_monthly_partitions,
    monthly_partition_ddl,
)
from medlens.db.session import (
    async_connect_args,
    async_database_url,
    sync_connect_args,
    sync_database_url,
)
from medlens.modules.organizations.models import Branch
from medlens.modules.organizations.repository import BranchRepository
from medlens.modules.products.models import RegistrationStatus
from medlens.seed import read_products

ROOT = Path(__file__).resolve().parents[2]


def test_monthly_partition_ddl_uses_utc_month_bounds() -> None:
    ddl = monthly_partition_ddl("scans", date(2026, 12, 17))

    assert ddl == (
        'CREATE TABLE IF NOT EXISTS "scans_y2026m12" PARTITION OF "scans" '
        "FOR VALUES FROM ('2026-12-01 00:00:00+00') TO ('2027-01-01 00:00:00+00')"
    )
    assert default_partition_ddl("scans").endswith('PARTITION OF "scans" DEFAULT')


def test_ensure_monthly_partitions_covers_current_and_three_ahead() -> None:
    executed: list[str] = []

    names = ensure_monthly_partitions(executed.append, "scans", today=date(2026, 11, 3))

    assert names == ["scans_y2026m11", "scans_y2026m12", "scans_y2027m01", "scans_y2027m02"]
    assert len(executed) == 4


def test_constraint_names_follow_the_convention() -> None:
    users = Base.metadata.tables["users"]
    names = {constraint.name for constraint in users.constraints}

    assert {"pk_users", "ck_users_user_status", "ck_users_identity", "uq_users_phone_hash"} <= names
    assert "fk_users_global_role_roles" in names


def test_every_table_has_a_primary_key_and_named_constraints() -> None:
    for table in Base.metadata.tables.values():
        assert table.primary_key.columns, table.name
        assert all(constraint.name for constraint in table.constraints), table.name


def _sql(clause: object) -> str:
    return str(clause.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))  # type: ignore[attr-defined]


def test_organization_scoped_tenant_clause() -> None:
    repository = BranchRepository(session=None)  # type: ignore[arg-type]
    no_org = TenantContext(user_id=None, organization_id=None)
    member = TenantContext(user_id=None, organization_id=uuid4())

    assert _sql(repository.tenant_clause(TenantContext.system())) == "true"
    assert _sql(repository.tenant_clause(no_org)) == "false"
    assert "branches.organization_id =" in _sql(repository.tenant_clause(member))
    assert repository.belongs_to(member, Branch(organization_id=member.organization_id, name="x"))
    assert not repository.belongs_to(member, Branch(organization_id=None, name="x"))
    assert repository.belongs_to(TenantContext.system(), Branch(name="x"))


def test_pgbouncer_disables_prepared_statements(settings_factory) -> None:
    direct: Settings = settings_factory()
    pooled: Settings = settings_factory(database={"pgbouncer": True})

    assert "prepared_statement_cache_size" not in str(async_database_url(direct.database))
    assert async_database_url(pooled.database).query == {"prepared_statement_cache_size": "0"}
    assert sync_database_url(pooled.database).drivername == "postgresql+psycopg"
    pooled_async = async_connect_args(pooled.database, application_name="t")
    assert pooled_async["statement_cache_size"] == 0
    assert (
        pooled_async["prepared_statement_name_func"]()
        != pooled_async["prepared_statement_name_func"]()
    )
    assert "statement_cache_size" not in async_connect_args(direct.database, application_name="t")
    assert sync_connect_args(pooled.database, application_name="t")["prepare_threshold"] is None
    assert "prepare_threshold" not in sync_connect_args(direct.database, application_name="t")


def test_database_urls_hide_the_password(settings_factory) -> None:
    url = async_database_url(settings_factory().database)

    assert "medlens-local-only" not in repr(url)


def test_seed_csv_is_valid_greenbook_data() -> None:
    products = read_products(ROOT / "seed" / "products.csv")

    assert 30 <= len(products) <= 50
    assert len({p["nafdac_number"] for p in products}) == len(products)
    assert all(isinstance(p["registration_status"], RegistrationStatus) for p in products)
    panadol = next(p for p in products if p["nafdac_number"] == "04-0005")
    assert panadol["brand_name"] == "Panadol Extra Caplet"


def test_migration_settings_require_alembic() -> None:
    with pytest.raises(RuntimeError, match=r"migrations/env\.py"):
        migration_settings()

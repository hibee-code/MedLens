"""Migrations: full round trip, models match the schema, partitions, roles, extensions."""

from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

import pytest
from alembic import command
from sqlalchemy import Engine, create_engine, inspect, text

from medlens.core.config import Settings
from medlens.db.partitions import add_months, ensure_monthly_partitions, month_start
from medlens.db.session import sync_database_url
from medlens.modules.rbac.models import RoleName
from tests.integration.support import alembic_config, make_settings

pytestmark = pytest.mark.integration


def sync_engine(settings: Settings, **kwargs: Any) -> Engine:
    return create_engine(sync_database_url(settings.database), **kwargs)


@pytest.fixture
def fresh_database(database_groups: dict[str, Any]) -> Iterator[Settings]:
    """An empty database in the shared container, dropped afterwards."""
    name = f"migrations_{uuid4().hex[:8]}"
    admin = sync_engine(
        make_settings(database={**database_groups, "name": "postgres"}),
        isolation_level="AUTOCOMMIT",
    )
    with admin.connect() as connection:
        connection.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        yield make_settings(database={**database_groups, "name": name})
    finally:
        with admin.connect() as connection:
            connection.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


def public_tables(settings: Settings) -> set[str]:
    engine = sync_engine(settings)
    try:
        return set(inspect(engine).get_table_names()) - {"alembic_version", "spatial_ref_sys"}
    finally:
        engine.dispose()


def test_upgrade_downgrade_upgrade(fresh_database: Settings) -> None:
    config = alembic_config(fresh_database)

    command.upgrade(config, "head")
    assert {"users", "scans", "audit_logs", "reference_images"} <= public_tables(fresh_database)

    command.downgrade(config, "base")
    assert public_tables(fresh_database) == set()

    command.upgrade(config, "head")
    assert len(public_tables(fresh_database)) >= 18


def test_models_match_migrations(migrated_settings: Settings) -> None:
    """Fails if a model changed without a migration (or a migration drifted from the models)."""
    command.check(alembic_config(migrated_settings))


def test_extensions_are_installed(migrated_settings: Settings) -> None:
    engine = sync_engine(migrated_settings)
    with engine.connect() as connection:
        installed = set(connection.scalars(text("SELECT extname FROM pg_extension")))
    engine.dispose()

    assert {"postgis", "vector", "pg_trgm", "citext", "btree_gist"} <= installed


def test_scans_partitions_cover_current_and_next_three_months(migrated_settings: Settings) -> None:
    engine = sync_engine(migrated_settings)
    with engine.connect() as connection:
        partitions = set(
            connection.scalars(
                text(
                    "SELECT inhrelid::regclass::text FROM pg_inherits "
                    "WHERE inhparent = 'scans'::regclass"
                )
            )
        )
        primary_key = inspect(connection).get_pk_constraint("scans")["constrained_columns"]
    engine.dispose()

    first = month_start(datetime.now(UTC).date())
    expected = {f"scans_y{m.year}m{m.month:02d}" for m in (add_months(first, i) for i in range(4))}
    assert partitions == expected | {"scans_default"}
    assert primary_key == ["id", "created_at"]


def test_partition_helper_is_idempotent(migrated_settings: Settings) -> None:
    engine = sync_engine(migrated_settings)
    later = add_months(month_start(datetime.now(UTC).date()), 8)
    with engine.begin() as connection:
        created = ensure_monthly_partitions(
            connection.exec_driver_sql, "scans", today=later, months_ahead=1
        )
        again = ensure_monthly_partitions(
            connection.exec_driver_sql, "scans", today=later, months_ahead=1
        )
        exists = connection.scalar(
            text("SELECT to_regclass(:name) IS NOT NULL"), {"name": created[1]}
        )
        connection.rollback()
    engine.dispose()

    assert created == again
    assert created[0] == f"scans_y{later.year}m{later.month:02d}"
    assert exists


def test_role_reference_rows(migrated_settings: Settings) -> None:
    engine = sync_engine(migrated_settings)
    with engine.connect() as connection:
        roles = dict(connection.execute(text("SELECT name, scope FROM roles")).all())
    engine.dispose()

    assert set(roles) == {role.value for role in RoleName}
    assert roles["consumer"] == "global"
    assert roles["regulator_admin"] == "organization"


def test_month_arithmetic_rolls_over_years() -> None:
    assert add_months(date(2026, 11, 1), 3) == date(2027, 2, 1)
    assert add_months(date(2026, 12, 1), 1) == date(2027, 1, 1)
    assert month_start(date(2026, 11, 30)) == date(2026, 11, 1)

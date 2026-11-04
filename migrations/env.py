"""Alembic environment.

Connection settings come from MedLens settings (``MEDLENS_DATABASE__*``). Code that runs
migrations programmatically (tests) can pass a Settings object in
``config.attributes["settings"]``; migrations read it through
``medlens.db.migration_support.migration_settings()``.

Autogenerate is a starting point only (CLAUDE.md §7): extensions, partitions, HNSW/trigram
indexes, roles and triggers are written by hand, and every generated revision is reviewed.
"""

import asyncio
import re
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy.engine import Connection

from medlens.core.config import Settings, get_settings
from medlens.db.all_models import Base
from medlens.db.session import async_database_url, create_db_engine

config = context.config
settings: Settings = config.attributes.setdefault("settings", None) or get_settings()
config.attributes["settings"] = settings
if config.config_file_name is not None and config.attributes.get("configure_logging", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Tables that exist in the database but deliberately not in the models.
_PARTITION = re.compile(r"^[a-z_]+_(y\d{4}m\d{2}|default)$")
_EXTENSION_TABLES = {"spatial_ref_sys"}  # owned by PostGIS


def include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    if not reflected or compare_to is not None:
        return True
    if type_ == "table" and name is not None:
        return not (_PARTITION.match(name) or name in _EXTENSION_TABLES)
    if type_ == "foreign_key_constraint":
        # Postgres adds an internal copy of a foreign key for each partition it points at.
        return not _PARTITION.match(obj.referred_table.name)
    return True


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of connecting (``alembic upgrade head --sql``)."""
    context.configure(
        url=async_database_url(settings.database),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_db_engine(settings.database, application_name="medlens-migrations")
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

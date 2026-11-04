"""Helpers for hand-written Alembic migrations."""

from alembic import context

from medlens.core.config import Settings


def migration_settings() -> Settings:
    """The settings the running migration was started with (set by migrations/env.py)."""
    try:
        settings = context.config.attributes.get("settings")
    except (AttributeError, NameError):  # the alembic context only exists during a migration
        settings = None
    if not isinstance(settings, Settings):
        raise RuntimeError("migrations must run through migrations/env.py")
    return settings

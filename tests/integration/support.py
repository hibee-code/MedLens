"""Helpers shared by integration tests and their fixtures."""

from pathlib import Path

from alembic.config import Config

from medlens.core.config import Settings
from tests.support import build_settings

ROOT = Path(__file__).resolve().parents[2]
POSTGRES_IMAGE = "medlens-postgres:17"  # built from docker/postgres, same tag as compose
REDIS_IMAGE = "redis:8.8.4-alpine"
MINIO_IMAGE = (
    "cgr.dev/chainguard/minio:latest"
    "@sha256:0a01100085e30fb08d2f39e39c9f5c9d83a95707f220b007d813b690cd41622d"
)
MINIO_USER = "medlens-test"
MINIO_PASSWORD = "medlens-test-secret"  # pragma: allowlist secret
DB_PASSWORD = "medlens-test-password"  # pragma: allowlist secret


make_settings = build_settings


def alembic_config(settings: Settings) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["settings"] = settings
    config.attributes["configure_logging"] = False  # keep structlog's handlers
    return config

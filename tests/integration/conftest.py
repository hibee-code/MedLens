"""Real Postgres (our PostGIS + pgvector image), Redis and MinIO in throwaway containers.

Needs Docker. Containers are shared by the whole session; the main database is migrated to head
once, and each test using ``db_session`` runs inside a transaction that is rolled back.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any

import docker
import pytest
from alembic import command
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.redis import RedisContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import HttpWaitStrategy

from medlens.core.config import Settings
from medlens.core.crypto import configure_crypto
from medlens.db.session import create_db_engine
from tests.integration.api_support import Api, open_api
from tests.integration.support import (
    DB_PASSWORD,
    MINIO_IMAGE,
    MINIO_PASSWORD,
    MINIO_USER,
    POSTGRES_IMAGE,
    REDIS_IMAGE,
    ROOT,
    alembic_config,
    make_settings,
)


@pytest.fixture(scope="session")
def postgres() -> Iterator[PostgresContainer]:
    docker.from_env().images.build(path=str(ROOT / "docker" / "postgres"), tag=POSTGRES_IMAGE)
    container = PostgresContainer(
        POSTGRES_IMAGE, username="medlens", password=DB_PASSWORD, dbname="medlens", driver=None
    )
    with container:
        yield container


@pytest.fixture(scope="session")
def database_groups(postgres: PostgresContainer) -> dict[str, Any]:
    return {
        "host": postgres.get_container_host_ip(),
        "port": int(postgres.get_exposed_port(5432)),
        "user": "medlens",
        "password": DB_PASSWORD,
        "name": "medlens",
    }


@pytest.fixture(scope="session")
def migrated_settings(database_groups: dict[str, Any]) -> Settings:
    """Settings for the main test database, migrated to head once per session."""
    settings = make_settings(database=database_groups)
    command.upgrade(alembic_config(settings), "head")
    configure_crypto(settings.crypto)
    return settings


@pytest.fixture
async def db_engine(migrated_settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = create_db_engine(migrated_settings.database, application_name="medlens-tests")
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """A session whose work is rolled back after the test (commits become savepoints)."""
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(
            bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()


@pytest.fixture(scope="session")
def redis_server() -> Iterator[RedisContainer]:
    with RedisContainer(REDIS_IMAGE) as container:
        yield container


@pytest.fixture(scope="session")
def minio() -> Iterator[DockerContainer]:
    container = (
        DockerContainer(MINIO_IMAGE)
        .with_command("server /tmp/data")
        .with_env("MINIO_ROOT_USER", MINIO_USER)
        .with_env("MINIO_ROOT_PASSWORD", MINIO_PASSWORD)
        .with_exposed_ports(9000)
        .waiting_for(HttpWaitStrategy(9000, "/minio/health/live"))
    )
    with container:
        yield container


@pytest.fixture
def live_groups(
    database_groups: dict[str, Any], redis_server: RedisContainer, minio: DockerContainer
) -> dict[str, dict[str, Any]]:
    """Settings groups pointing at the containers (bucket: medlens-it, created by the tests)."""
    return {
        "database": dict(database_groups),
        "redis": {
            "host": redis_server.get_container_host_ip(),
            "port": int(redis_server.get_exposed_port(6379)),
        },
        "s3": {
            "endpoint_url": (
                f"http://{minio.get_container_host_ip()}:{minio.get_exposed_port(9000)}"
            ),
            "access_key_id": MINIO_USER,
            "secret_access_key": MINIO_PASSWORD,
            "bucket": "medlens-it",
        },
    }


@pytest.fixture(scope="session")
def redis_groups(redis_server: RedisContainer) -> dict[str, Any]:
    return {
        "host": redis_server.get_container_host_ip(),
        "port": int(redis_server.get_exposed_port(6379)),
    }


@pytest.fixture
def app_settings(
    migrated_settings: Settings, database_groups: dict[str, Any], redis_groups: dict[str, Any]
) -> Settings:
    """The migrated database plus a real Redis; SMS goes to the in-memory console adapter."""
    return make_settings(database=database_groups, redis=redis_groups)


@pytest.fixture
async def api(app_settings: Settings) -> AsyncIterator[Api]:
    async with open_api(app_settings) as harness:
        yield harness

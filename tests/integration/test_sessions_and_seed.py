"""Transaction handling (per-request session, unit of work, sync scope) and the seed command."""

from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from medlens.core.config import Settings
from medlens.core.errors import Conflict
from medlens.core.routing import api_v1_router
from medlens.core.security import verify_password
from medlens.db.session import (
    DbSession,
    UnitOfWork,
    create_session_factory,
    create_sync_engine,
    create_sync_session_factory,
    sync_session_scope,
)
from medlens.main import create_app
from medlens.modules.rbac.models import Permission
from medlens.modules.users.models import User
from medlens.seed import TEST_USERS, run_seed
from tests.integration.support import ROOT

pytestmark = pytest.mark.integration

STAFF_PASSWORD = "seed-staff-passphrase-2026"  # pragma: allowlist secret


async def permission_exists(engine: AsyncEngine, name: str) -> bool:
    async with engine.connect() as connection:
        count = await connection.scalar(
            select(func.count()).select_from(Permission).where(Permission.name == name)
        )
    return bool(count)


@pytest.fixture
async def cleanup(db_engine: AsyncEngine) -> AsyncIterator[list[str]]:
    """Names of committed test permissions, deleted after the test."""
    names: list[str] = []
    yield names
    async with db_engine.begin() as connection:
        await connection.execute(delete(Permission).where(Permission.name.in_(names)))


@pytest.fixture
async def db_client(migrated_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app(migrated_settings)
    router = api_v1_router("/_test", tags=["admin"])

    @router.post("/permissions/{name}")
    async def create_permission(
        name: str, session: DbSession, fail: bool = False
    ) -> dict[str, bool]:
        session.add(Permission(name=name))
        await session.flush()
        if fail:
            raise Conflict("Deliberate failure after writing.")
        return {"created": True}

    app.include_router(router)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


async def test_request_session_commits_before_the_response(
    db_client: httpx.AsyncClient, db_engine: AsyncEngine, cleanup: list[str]
) -> None:
    name = f"test:{uuid4().hex[:8]}"
    cleanup.append(name)

    response = await db_client.post(f"/api/v1/_test/permissions/{name}")

    assert response.status_code == 200
    # Visible from another connection as soon as the client has the response.
    assert await permission_exists(db_engine, name)


async def test_request_session_rolls_back_on_domain_errors(
    db_client: httpx.AsyncClient, db_engine: AsyncEngine, cleanup: list[str]
) -> None:
    name = f"test:{uuid4().hex[:8]}"
    cleanup.append(name)

    response = await db_client.post(f"/api/v1/_test/permissions/{name}", params={"fail": True})

    assert response.status_code == 409
    assert not await permission_exists(db_engine, name)


async def test_unit_of_work_commits_only_when_asked(
    db_engine: AsyncEngine, cleanup: list[str]
) -> None:
    kept, discarded = f"test:{uuid4().hex[:8]}", f"test:{uuid4().hex[:8]}"
    cleanup.extend([kept, discarded])
    factory = create_session_factory(db_engine)

    async with UnitOfWork(factory) as uow:
        uow.session.add(Permission(name=discarded))
        await uow.session.flush()
    async with UnitOfWork(factory) as uow:
        uow.session.add(Permission(name=kept))
        await uow.commit()

    assert not await permission_exists(db_engine, discarded)
    assert await permission_exists(db_engine, kept)


def test_unit_of_work_outside_context_is_an_error(db_engine: AsyncEngine) -> None:
    with pytest.raises(RuntimeError, match="outside 'async with'"):
        _ = UnitOfWork(create_session_factory(db_engine)).session


def test_seed_is_idempotent(migrated_settings: Settings) -> None:
    engine = create_sync_engine(migrated_settings.database, application_name="medlens-tests")
    factory = create_sync_session_factory(engine)
    try:
        with factory() as session, session.begin():
            first = run_seed(session, ROOT / "seed", staff_password=STAFF_PASSWORD)
            second = run_seed(session, ROOT / "seed")
            seed_emails = {user.email for user in TEST_USERS if user.email}
            found = set(session.scalars(select(User.email).where(User.email.in_(seed_emails))))
            hashes = set(
                session.scalars(select(User.password_hash).where(User.email.in_(seed_emails)))
            )
            session.rollback()  # leave the shared database as it was
    finally:
        engine.dispose()

    # Counts are whole-table (other tests commit users), so check idempotence and the seed's rows.
    assert first == second
    assert first.products >= 43
    assert first.batches >= 10
    assert found == seed_emails
    assert len(hashes) == 1  # one shared staff password, set once
    assert verify_password(STAFF_PASSWORD, hashes.pop())[0]


def test_sync_session_scope_commits_and_rolls_back(
    migrated_settings: Settings, cleanup: list[str]
) -> None:
    kept, failed = f"test:{uuid4().hex[:8]}", f"test:{uuid4().hex[:8]}"
    cleanup.extend([kept, failed])
    engine = create_sync_engine(migrated_settings.database, application_name="medlens-tests")
    factory = create_sync_session_factory(engine)
    try:
        with sync_session_scope(factory) as session:
            session.add(Permission(name=kept))

        def fail_after_writing() -> None:
            with sync_session_scope(factory) as session:
                session.add(Permission(name=failed))
                session.flush()
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            fail_after_writing()
        with factory() as session:
            names = set(session.scalars(select(Permission.name)))
    finally:
        engine.dispose()

    assert kept in names
    assert failed not in names

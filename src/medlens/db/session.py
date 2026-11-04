"""Database engines and sessions.

API (async, asyncpg):
    ``DbSession`` -- one transaction per request, committed before the response is sent and
    rolled back if the endpoint raises (including domain errors that become 4xx responses).
    ``UnitOfWork`` -- explicit transaction control for services that need it.

Celery workers (sync, psycopg 3):
    ``create_sync_engine`` / ``sync_session_scope`` -- workers are not async (CLAUDE.md §7).

Both engines work through PgBouncer in transaction mode when ``MEDLENS_DATABASE__PGBOUNCER`` is
true: server-side prepared statements are disabled, because consecutive statements may run on
different server connections.
"""

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Annotated, Any, Self
from uuid import uuid4

from fastapi import Depends, Request
from sqlalchemy import URL, Engine, create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from medlens.core.config import DatabaseSettings


def _url(settings: DatabaseSettings, driver: str, query: dict[str, str]) -> URL:
    return URL.create(
        driver,
        username=settings.user,
        password=settings.password.get_secret_value(),
        host=settings.host,
        port=settings.port,
        database=settings.name,
        query=query,
    )


# ------------------------------------------------------------------ async engine, for the API
def async_database_url(settings: DatabaseSettings) -> URL:
    # PgBouncer transaction mode: SQLAlchemy's own prepared-statement cache must be off too.
    query = {"prepared_statement_cache_size": "0"} if settings.pgbouncer else {}
    return _url(settings, "postgresql+asyncpg", query)


def _unique_statement_name() -> str:
    return f"__asyncpg_{uuid4()}__"


def async_connect_args(settings: DatabaseSettings, *, application_name: str) -> dict[str, Any]:
    connect_args: dict[str, Any] = {
        "timeout": settings.connect_timeout_seconds,
        "command_timeout": settings.command_timeout_seconds,
        "server_settings": {"application_name": application_name},
    }
    if settings.pgbouncer:
        # CLAUDE.md §7: disable asyncpg's statement cache and give each prepared statement a
        # unique name so statements never collide on a shared server connection.
        connect_args["statement_cache_size"] = 0
        connect_args["prepared_statement_name_func"] = _unique_statement_name
    return connect_args


def create_db_engine(settings: DatabaseSettings, *, application_name: str) -> AsyncEngine:
    return create_async_engine(
        async_database_url(settings),
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout_seconds,
        pool_recycle=settings.pool_recycle_seconds,
        pool_pre_ping=True,
        connect_args=async_connect_args(settings, application_name=application_name),
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: objects stay readable after commit without a lazy (sync) reload.
    return async_sessionmaker(engine, expire_on_commit=False)


async def ping_database(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


async def _request_session(request: Request) -> AsyncIterator[AsyncSession]:
    session_factory: async_sessionmaker[AsyncSession] = request.app.state.resources.session_factory
    async with session_factory() as session, session.begin():
        yield session


# scope="function": the transaction ends when the endpoint returns, *before* the response is
# sent. With the default ("request") the commit would run after the client already got a 2xx.
DbSession = Annotated[AsyncSession, Depends(_request_session, scope="function")]


class UnitOfWork:
    """Explicit transaction: nothing is committed unless ``commit()`` is called; leaving the
    block discards anything uncommitted. Objects loaded inside stay readable afterwards.

    async with UnitOfWork(session_factory) as uow:
        uow.session.add(...)
        await uow.commit()
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None

    @property
    def session(self) -> AsyncSession:
        if self._session is None:
            raise RuntimeError("UnitOfWork used outside 'async with'")
        return self._session

    async def __aenter__(self) -> Self:
        self._session = self._session_factory()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        session, self._session = self.session, None
        # close() rolls back anything uncommitted and detaches objects *without* expiring them,
        # so values loaded inside the block stay readable after it (an explicit rollback()
        # would expire them and make every later attribute access fail).
        await session.close()

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()


# ------------------------------------------------------------------ sync engine, for workers
def sync_database_url(settings: DatabaseSettings) -> URL:
    return _url(settings, "postgresql+psycopg", {})


def sync_connect_args(settings: DatabaseSettings, *, application_name: str) -> dict[str, Any]:
    connect_args: dict[str, Any] = {
        "connect_timeout": max(1, round(settings.connect_timeout_seconds)),
        "application_name": application_name,
    }
    if settings.pgbouncer:
        connect_args["prepare_threshold"] = None  # psycopg 3: never prepare server-side
    return connect_args


def create_sync_engine(settings: DatabaseSettings, *, application_name: str) -> Engine:
    return create_engine(
        sync_database_url(settings),
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout_seconds,
        pool_recycle=settings.pool_recycle_seconds,
        pool_pre_ping=True,
        connect_args=sync_connect_args(settings, application_name=application_name),
    )


def create_sync_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


@contextmanager
def sync_session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """One transaction: committed if the block succeeds, rolled back if it raises."""
    with session_factory() as session, session.begin():
        yield session

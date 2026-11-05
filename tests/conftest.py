import os
from collections.abc import AsyncIterator, Callable, Iterator

import httpx
import pytest
from fastapi import FastAPI

from medlens.core import config
from medlens.core.config import Settings
from medlens.main import create_app
from tests.support import build_settings


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tests marked ``live`` call paid external services (the model API): skipped unless
    MEDLENS_LIVE_TESTS=1 and an API key are set by hand. CI never sets them."""
    if os.environ.get("MEDLENS_LIVE_TESTS") == "1" and os.environ.get("ANTHROPIC_API_KEY"):
        return
    skip = pytest.mark.skip(reason="live: set MEDLENS_LIVE_TESTS=1 and ANTHROPIC_API_KEY")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


SettingsFactory = Callable[..., Settings]


@pytest.fixture(scope="session", autouse=True)
def _isolated_environment() -> Iterator[None]:
    """Tests (and session fixtures) never see MEDLENS_* variables from the developer's shell,
    nor the developer's .env: code that loads settings itself (``get_settings()``, e.g. the
    Celery app's configuration) gets test settings, so the suite runs the same in CI."""
    with pytest.MonkeyPatch.context() as patch:
        for name in list(os.environ):
            if name.startswith("MEDLENS_"):
                patch.delenv(name)
        patch.setattr(config, "load_settings", build_settings)
        config.get_settings.cache_clear()
        yield
        config.get_settings.cache_clear()


@pytest.fixture
def settings_factory() -> SettingsFactory:
    """Build settings from explicit groups only (no .env file), defaulting to the test env."""
    return build_settings


@pytest.fixture
def settings(settings_factory: SettingsFactory) -> Settings:
    return settings_factory(app={"max_request_body_bytes": 1024, "max_upload_bytes": 512})


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client

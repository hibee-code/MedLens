"""The ``python -m medlens.seed`` wrapper. The seeding itself is tested against a real
database in tests/integration/test_sessions_and_seed.py."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from medlens.seed import SeedSummary
from medlens.seed import __main__ as cli
from tests.unit.test_config import PRODUCTION_SECRETS


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_refuses_deployed_environments(
    monkeypatch: pytest.MonkeyPatch, settings_factory, environment: str, capsys
) -> None:
    settings = settings_factory(app={"environment": environment}, **PRODUCTION_SECRETS)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr("sys.argv", ["medlens.seed"])

    assert cli.main() == 1
    assert f"refusing to seed a {environment} database" in capsys.readouterr().err


@pytest.mark.parametrize(("flags", "with_images"), [([], True), (["--no-images"], False)])
def test_seeds_through_the_sync_engine(
    monkeypatch: pytest.MonkeyPatch,
    settings_factory,
    capsys,
    tmp_path: Path,
    flags: list[str],
    with_images: bool,
) -> None:
    calls: dict[str, Any] = {}

    class FakeEngine:
        def dispose(self) -> None:
            calls["disposed"] = True

    @contextmanager
    def fake_scope(factory: object):  # type: ignore[no-untyped-def]
        yield "session"

    def fake_run_seed(
        session: object,
        data_dir: Path,
        staff_password: str | None,
        reset_staff_logins: bool,
        storage: object | None,
    ) -> SeedSummary:
        calls["seeded"] = (session, data_dir)
        calls["images"] = storage is not None
        return SeedSummary(products=43, organizations=35, batches=10, users=9, memberships=6)

    monkeypatch.setattr(cli, "get_settings", lambda: settings_factory(app={"environment": "local"}))
    monkeypatch.setattr(cli, "create_sync_engine", lambda *a, **k: FakeEngine())
    monkeypatch.setattr(cli, "create_sync_session_factory", lambda engine: "factory")
    monkeypatch.setattr(cli, "create_sync_s3_client", lambda settings: "s3")
    monkeypatch.setattr(cli, "sync_session_scope", fake_scope)
    monkeypatch.setattr(cli, "run_seed", fake_run_seed)
    monkeypatch.setattr("sys.argv", ["medlens.seed", "--data-dir", str(tmp_path), *flags])

    assert cli.main() == 0
    assert calls == {"seeded": ("session", tmp_path), "images": with_images, "disposed": True}
    assert "seeded 43 products, 35 organizations" in capsys.readouterr().out


def test_rejects_a_weak_staff_password(
    monkeypatch: pytest.MonkeyPatch, settings_factory, capsys
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: settings_factory(app={"environment": "local"}))
    monkeypatch.setattr("sys.argv", ["medlens.seed", "--staff-password", "short"])

    assert cli.main() == 2
    assert "at least 12 characters" in capsys.readouterr().err

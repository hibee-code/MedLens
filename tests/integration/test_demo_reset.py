"""`make demo-reset` (seed/demo_reset.py): local only, report unless confirmed, and the deletes in
foreign-key order. The deletes run in a transaction that is rolled back: the test database is
shared, so this test never empties it."""

from pathlib import Path

import pytest
from sqlalchemy import func, select

from medlens.core.config import Environment, Settings
from medlens.modules.cases.models import Case
from medlens.modules.scans.models import Scan
from medlens.seed import demo_reset
from tests.integration.api_support import Api
from tests.integration.test_cases import open_case, unregistered_scan

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def local(settings: Settings) -> Settings:
    app = settings.app.model_copy(update={"environment": Environment.LOCAL})
    return settings.model_copy(update={"app": app})


def test_it_refuses_anywhere_but_local(
    migrated_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("medlens.core.config.get_settings", lambda: migrated_settings)

    assert migrated_settings.app.environment is Environment.TEST
    assert demo_reset.main(["--confirm"]) == 1
    assert "local demo data only; this environment is test" in capsys.readouterr().err


async def test_without_confirm_it_only_reports(
    api: Api,
    migrated_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await unregistered_scan(api, "NG-KO")
    with api.sync_sessions() as session:
        before = session.scalar(select(func.count()).select_from(Scan))
    monkeypatch.setattr("medlens.core.config.get_settings", lambda: local(migrated_settings))

    code = demo_reset.main([])

    out = capsys.readouterr()
    with api.sync_sessions() as session:
        after = session.scalar(select(func.count()).select_from(Scan))
    assert code == 0
    assert after == before  # nothing deleted
    assert "Would delete:" in out.out
    assert "Nothing deleted. Run again with CONFIRM=1" in out.err


async def test_the_reset_deletes_scans_and_what_hangs_off_them(api: Api) -> None:
    scan_id = await unregistered_scan(api, "NG-KO")
    open_case(api, scan_id)

    with api.sync_sessions() as session:
        found = demo_reset.reset(session)  # in a transaction, rolled back below
        left = {
            "scans": session.scalar(select(func.count()).select_from(Scan)),
            "cases": session.scalar(select(func.count()).select_from(Case)),
        }
        session.rollback()

    assert found.rows["scans"] >= 1
    assert found.rows["cases"] >= 1
    assert left == {"scans": 0, "cases": 0}
    assert list(found.rows) == list(demo_reset.TABLES)
    assert "scan photos in object storage" in demo_reset.render(found)


# ------------------------------------------------------------------------- demo barcodes
def test_demo_barcodes_are_local_only(
    migrated_settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from medlens.seed import demo_gtins

    monkeypatch.setattr("medlens.core.config.get_settings", lambda: migrated_settings)

    assert demo_gtins.main([str(ROOT / "seed" / "test_packs" / "gtins.csv")]) == 1


async def test_demo_barcodes_load_only_synthetic_rows(api: Api, tmp_path: Path) -> None:
    from medlens.modules.products.models import ProductGtin
    from medlens.seed.demo_gtins import load

    product = await api.create_product()
    good = tmp_path / "good.csv"
    good.write_text(
        f"nafdac_number,gtin,source,synthetic\n{product.nafdac_number},06291041500299,register,yes\n"
    )
    real = tmp_path / "real.csv"
    real.write_text(
        f"nafdac_number,gtin,source,synthetic\n{product.nafdac_number},06291041500299,register,\n"
    )

    with api.sync_sessions() as session:
        loaded = load(session, good)
        owner = session.scalar(
            select(ProductGtin.product_id).where(ProductGtin.gtin == "06291041500299")
        )
        refused = load(session, real)
        session.rollback()

    assert loaded == (1, [])
    assert owner == product.id
    assert refused == (0, ["line 2: not marked synthetic (refused)"])

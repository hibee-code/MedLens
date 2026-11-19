"""`make demo-reset` (seed/demo_reset.py): local only, report unless confirmed, and the deletes in
foreign-key order. The deletes run in a transaction that is rolled back: the test database is
shared, so this test never empties it."""

from pathlib import Path
from uuid import uuid4

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


async def test_accounts_erases_the_demo_accounts_only(api: Api) -> None:
    from medlens.modules.auth.models import RefreshToken
    from medlens.modules.organizations.models import Membership, OrganizationType
    from medlens.modules.rbac.models import RoleName
    from medlens.modules.users.models import User, UserStatus

    demo = await api.create_user(email=f"demo.{uuid4().hex[:8]}@medlens.test")
    await api.add_member(
        demo, await api.create_org(OrganizationType.PHARMACY), RoleName.PHARMACY_STAFF
    )
    await api.staff_login(demo)
    other = await api.create_user(email=f"other.{uuid4().hex[:8]}@example.ng")

    with api.sync_sessions() as session:
        listed = demo_reset.demo_accounts(session)
        erased = demo_reset.erase_accounts(session)
        after = session.get(User, demo.id)
        kept = session.get(User, other.id)
        assert after is not None
        assert kept is not None
        memberships = session.scalar(
            select(func.count()).select_from(Membership).where(Membership.user_id == demo.id)
        )
        open_sessions = session.scalar(
            select(func.count())
            .select_from(RefreshToken)
            .where(RefreshToken.user_id == demo.id, RefreshToken.revoked_at.is_(None))
        )
        state = (after.status, after.email, after.password_hash, after.mfa_enabled, kept.status)
        session.rollback()

    assert demo.id in listed
    assert other.id not in listed
    assert erased >= 1
    assert state == (UserStatus.DELETED, None, None, False, UserStatus.ACTIVE)
    assert memberships == 0
    assert open_sessions == 0


def test_erasing_accounts_forgets_the_saved_demo_sessions(tmp_path: Path) -> None:
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "pharmacist").write_text("refresh")
    (tmp_path / "someone@medlens.test.totp").write_text("secret")
    (tmp_path / "someone@medlens.test.recovery").write_text("codes")
    (tmp_path / "password").write_text("kept")

    demo_reset._forget_demo_sessions(tmp_path)

    assert sorted(path.name for path in tmp_path.rglob("*")) == ["password", "sessions"]


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

"""Importing a verified Greenbook snapshot into products."""

import random
from pathlib import Path

import pytest
from sqlalchemy import delete, select

from medlens.core.config import Settings
from medlens.db.session import create_sync_engine, create_sync_session_factory
from medlens.modules.organizations.models import Organization, OrganizationType
from medlens.modules.products.models import Product, RegistrationStatus
from medlens.seed import greenbook
from medlens.seed.greenbook import Manifest, SnapshotError, encode_snapshot, import_snapshot
from tests.unit.test_greenbook import manifest_for, record

pytestmark = pytest.mark.integration


def unique_numbers(count: int) -> list[str]:
    serials = random.sample(range(100000, 999999), count)
    return [f"Z9-{serial}" for serial in serials]


def test_import_upserts_and_keeps_manufacturer_links(migrated_settings: Settings) -> None:
    first, second = unique_numbers(2)
    snapshot = [record(NAFDAC=first), record(NAFDAC=second, status="Inactive")]
    data = encode_snapshot(snapshot)
    refreshed = [record(NAFDAC=first, product_name="Panadol Extra (renamed)")]
    refreshed_data = encode_snapshot(refreshed)
    engine = create_sync_engine(migrated_settings.database, application_name="medlens-tests")
    factory = create_sync_session_factory(engine)
    try:
        with factory() as session, session.begin():
            new, updated = import_snapshot(session, data, manifest_for(data, 2))
            manufacturer = Organization(
                name="Linked Manufacturer", type=OrganizationType.MANUFACTURER
            )
            session.add(manufacturer)
            session.flush()
            product = session.scalars(select(Product).where(Product.nafdac_number == first)).one()
            product.manufacturer_org_id = manufacturer.id  # an approved claim
            session.flush()

            again = import_snapshot(session, refreshed_data, manifest_for(refreshed_data, 1))

            session.refresh(product)
            suspended = session.scalars(
                select(Product.registration_status).where(Product.nafdac_number == second)
            ).one()
            session.execute(delete(Product).where(Product.nafdac_number.in_([first, second])))
    finally:
        engine.dispose()

    assert (new, updated) == (2, 0)
    assert again == (0, 1)
    assert product.brand_name == "Panadol Extra (renamed)"
    assert product.manufacturer_org_id == manufacturer.id  # the import never unlinks
    assert suspended is RegistrationStatus.SUSPENDED


def test_cli_import_loads_a_local_snapshot(
    migrated_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (number,) = unique_numbers(1)
    data = encode_snapshot([record(NAFDAC=number)])
    snapshot = tmp_path / "snapshot.csv.gz"
    snapshot.write_bytes(data)
    manifest_for(data, 1).write(tmp_path / "manifest.json")
    monkeypatch.setattr(greenbook, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr("medlens.core.config.get_settings", lambda: migrated_settings)

    assert greenbook.main(["import", "--source", str(snapshot)]) == 0

    engine = create_sync_engine(migrated_settings.database, application_name="medlens-tests")
    try:
        with create_sync_session_factory(engine)() as session, session.begin():
            found = session.scalars(select(Product).where(Product.nafdac_number == number)).one()
            session.delete(found)
    finally:
        engine.dispose()


def test_import_refuses_a_snapshot_that_does_not_match(migrated_settings: Settings) -> None:
    data = encode_snapshot([record(NAFDAC=unique_numbers(1)[0])])
    wrong = Manifest(file="x.csv.gz", sha256="0" * 64, rows=1, fetched_on="2026-11-03")
    engine = create_sync_engine(migrated_settings.database, application_name="medlens-tests")
    try:
        with create_sync_session_factory(engine)() as session, pytest.raises(SnapshotError):
            import_snapshot(session, data, wrong)
    finally:
        engine.dispose()

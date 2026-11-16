"""Hand-typed register rows against a database: the diff report, all-or-nothing apply, the audit
entry, kept holder links, and the command line (seed/register_import.py)."""

import csv
import hashlib
import random
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete, select

from medlens.core.config import Settings
from medlens.db.session import create_sync_engine, create_sync_session_factory
from medlens.modules.audit.models import AuditLog
from medlens.modules.organizations.models import Organization, OrganizationType
from medlens.modules.products.models import Product, RegisterSource, RegistrationStatus
from medlens.seed import register_import
from medlens.seed.register_import import MANUAL_COLUMNS, apply, diff, read_rows, render
from tests.integration.test_greenbook_import import unique_numbers
from tests.unit.test_register_import import GOOD

pytestmark = pytest.mark.integration
TODAY = date(2026, 11, 12)


def write(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANUAL_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def sessions(migrated_settings: Settings) -> Any:
    engine = create_sync_engine(migrated_settings.database, application_name="medlens-tests")
    yield create_sync_session_factory(engine)
    engine.dispose()


def test_the_report_then_apply_upserts_audits_and_keeps_holder_links(
    sessions: Any, tmp_path: Path
) -> None:
    new, existing, untouched = unique_numbers(3)
    first_file = write(
        tmp_path / "first.csv",
        [{**GOOD, "nafdac_number": existing}, {**GOOD, "nafdac_number": untouched}],
    )
    second_file = write(
        tmp_path / "second.csv",
        [
            {**GOOD, "nafdac_number": new, "brand_name": "Lonart DS"},
            {**GOOD, "nafdac_number": existing, "strength": "80 mg; 480 mg",
             "registration_status": "suspended", "retrieved_on": "2026-11-11"},
            {**GOOD, "nafdac_number": untouched},
        ],
    )  # fmt: skip
    with sessions() as session:
        rows, _ = read_rows(first_file, TODAY)
        first = diff(session, rows)
        apply(session, first, file_name="first.csv", sha256="a" * 64)
        holder = Organization(name="Novartis Nigeria", type=OrganizationType.MANUFACTURER)
        session.add(holder)
        session.flush()
        product = session.scalars(select(Product).where(Product.nafdac_number == existing)).one()
        product.manufacturer_org_id = holder.id  # linked by an approved claim
        session.flush()

        rows, _ = read_rows(second_file, TODAY)
        second = diff(session, rows)
        apply(session, second, file_name="second.csv", sha256="b" * 64)
        session.refresh(product)
        stored = {
            "strength": product.strength,
            "ingredients": product.active_ingredients,
            "retrieved_on": product.retrieved_on,
            "holder": product.manufacturer_org_id,
        }
        holder_id = holder.id
        audited = session.scalars(
            select(AuditLog)
            .where(AuditLog.action == "register.imported")
            .order_by(AuditLog.id.desc())
        ).first()
        audit = dict(audited.metadata_) if audited else None
        session.rollback()  # leave the shared database as it was

    assert [row.nafdac_number for row in first.added] == [existing, untouched]
    assert [row.nafdac_number for row in second.added] == [new]
    assert [row.nafdac_number for row in second.unchanged] == [untouched]
    [(changed, fields)] = second.changed
    assert changed.nafdac_number == existing
    assert set(fields) == {"strength", "active_ingredients", "registration_status", "retrieved_on"}
    assert fields["registration_status"] == ("active", "suspended")
    assert stored["strength"] == "80 mg; 480 mg"
    assert stored["ingredients"][1] == {
        "name": "lumefantrine", "strength_value": 480.0, "strength_unit": "mg"
    }  # fmt: skip
    assert stored["retrieved_on"] == date(2026, 11, 11)
    assert stored["holder"] == holder_id  # never unlinked by an import
    assert audit is not None
    assert audit["added"] == [new]
    assert audit["changed"] == [existing]
    assert existing not in second.not_in_file
    assert untouched not in second.not_in_file


def test_the_command_reports_without_applying_and_refuses_invalid_files(
    migrated_settings: Settings, sessions: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    from medlens.core import config

    monkeypatch.setattr(config, "get_settings", lambda: migrated_settings)
    cleared: list[bool] = []
    monkeypatch.setattr(
        "medlens.seed.greenbook.clear_cached_products", lambda settings: cleared.append(True)
    )
    number, other = unique_numbers(2)
    valid = write(tmp_path / "valid.csv", [{**GOOD, "nafdac_number": number}])
    invalid = write(
        tmp_path / "invalid.csv",
        [{**GOOD, "nafdac_number": other}, {**GOOD, "nafdac_number": "nonsense"}],
    )

    report_only = register_import.main([str(valid)])
    refused = register_import.main([str(invalid), "--confirm"])
    out = capsys.readouterr()
    with sessions() as session:
        stored_before = session.scalar(select(Product.id).where(Product.nafdac_number == number))
    applied = register_import.main([str(valid), "--confirm"])
    with sessions() as session, session.begin():
        stored = session.scalars(select(Product).where(Product.nafdac_number == number)).one()
        refused_stored = session.scalar(select(Product.id).where(Product.nafdac_number == other))
        values = (stored.brand_name, stored.source_url)
        session.execute(delete(Product).where(Product.nafdac_number == number))

    assert report_only == 0
    assert stored_before is None  # the report changed nothing
    assert "added 1, changed 0, unchanged 0, invalid 0" in out.out
    assert refused == 2
    assert "line 3 nonsense: nafdac_number 'nonsense' is not a NAFDAC number" in out.out
    assert "Nothing applied" in out.err
    assert refused_stored is None  # all or nothing: the valid row was not applied either
    assert applied == 0
    assert values == ("Coartem Dispersible", GOOD["source_url"])
    assert cleared == [True]  # cached product facts are cleared after applying


def random_gtin() -> str:
    body = "0" + "".join(random.SystemRandom().choice("0123456789") for _ in range(12))
    return body + str(_check_digit(body))


def test_gtins_are_imported_with_their_source_and_never_shared(
    sessions: Any, tmp_path: Path
) -> None:
    from medlens.modules.verification.repository import SqlCatalog

    owner, other = unique_numbers(2)
    gtin, replacement = random_gtin(), random_gtin()
    declared = {**GOOD, "nafdac_number": owner, "gtin": gtin, "gtin_source": "manufacturer"}
    first = write(tmp_path / "a.csv", [declared])
    clash = write(tmp_path / "b.csv", [{**declared, "nafdac_number": other}])
    replaced = write(
        tmp_path / "c.csv", [{**declared, "gtin": replacement, "gtin_source": "register"}]
    )
    with sessions() as session:
        rows, _ = read_rows(first, TODAY)
        apply(session, diff(session, rows), file_name="a.csv", sha256="c" * 64)
        found = SqlCatalog(session).gtin_owner(gtin)
        missing = SqlCatalog(session).gtin_owner("0" * 13 + "0")
        rows, _ = read_rows(clash, TODAY)
        refused = diff(session, rows)
        rows, _ = read_rows(replaced, TODAY)
        report = diff(session, rows)
        apply(session, report, file_name="c.csv", sha256="d" * 64)
        after = SqlCatalog(session).gtin_owner(gtin), SqlCatalog(session).gtin_owner(replacement)
        session.rollback()

    assert found is not None
    assert (found.product.nafdac_number, found.source) == (owner, "manufacturer")
    assert missing is None
    [invalid] = refused.invalid
    assert invalid.problems == [f"GTIN {gtin} is registered to {owner}"]
    [(_, changes)] = report.changed
    assert changes == {"gtins": ([f"{gtin} (manufacturer)"], [f"{replacement} (register)"])}
    assert after[0] is None
    assert after[1] is not None
    assert after[1].source == "register"


def snapshot_of(records: list[dict[str, str]], fetched_on: str) -> tuple[bytes, Any]:
    from medlens.seed.greenbook import Manifest, encode_snapshot

    data = encode_snapshot([{**record, "fetched_on": fetched_on} for record in records])
    manifest = Manifest(
        file="s.csv.gz",
        sha256=hashlib.sha256(data).hexdigest(),
        rows=len(records),
        fetched_on=fetched_on,
    )
    return data, manifest


def test_snapshot_and_hand_typed_rows_the_most_recent_wins(sessions: Any, tmp_path: Path) -> None:
    """Owner's rule: the row retrieved most recently wins, on the same day the snapshot; a
    snapshot never deletes a hand-typed row; every conflict is reported before applying."""
    from medlens.seed.greenbook import import_snapshot, render_conflicts
    from tests.unit.test_greenbook import record

    older_hand, newer_hand, same_day, hand_only = unique_numbers(4)
    hand_typed = write(
        tmp_path / "manual.csv",
        [  # retrieved_on 2026-11-10 (GOOD)
            {**GOOD, "nafdac_number": older_hand},
            {**GOOD, "nafdac_number": newer_hand, "brand_name": "Hand-typed name"},
            {**GOOD, "nafdac_number": same_day, "brand_name": "Hand-typed name"},
            {**GOOD, "nafdac_number": hand_only},
        ],
    )
    later = snapshot_of(  # retrieved two days after the hand-typed rows: they lose
        [record(NAFDAC=older_hand, status="Inactive")], "2026-11-12"
    )
    earlier = snapshot_of([record(NAFDAC=newer_hand, product_name="Snapshot name")], "2026-11-05")
    tie = snapshot_of([record(NAFDAC=same_day, product_name="Snapshot name")], "2026-11-10")
    with sessions() as session:
        rows, _ = read_rows(hand_typed, TODAY)
        apply(session, diff(session, rows), file_name="manual.csv", sha256="e" * 64)
        report_only, nothing = import_snapshot(session, *later, apply=False)
        untouched = session.scalar(
            select(Product.registration_status).where(Product.nafdac_number == older_hand)
        )
        _, applied = import_snapshot(session, *later)
        kept, _ = import_snapshot(session, *earlier)
        tied, _ = import_snapshot(session, *tie)
        stored = {
            number: (status, name, source)
            for number, status, name, source in session.execute(
                select(
                    Product.nafdac_number,
                    Product.registration_status,
                    Product.brand_name,
                    Product.register_source,
                ).where(Product.nafdac_number.in_([older_hand, newer_hand, same_day, hand_only]))
            )
        }
        session.rollback()

    # A newer snapshot suspends a hand-typed product: reported first, then applied.
    assert nothing is None
    assert untouched is RegistrationStatus.ACTIVE  # the report applied nothing
    [conflict] = report_only.conflicts
    assert conflict.incoming_wins
    assert conflict.fields["registration_status"] == (
        RegistrationStatus.ACTIVE,
        RegistrationStatus.SUSPENDED,
    )
    assert "registration_status: 'active' (stored) vs 'suspended' (snapshot)" in "\n".join(
        render_conflicts(report_only.conflicts, "snapshot")
    )
    assert applied is not None
    assert (applied.updated, applied.kept) == (1, 0)
    assert stored[older_hand][0] is RegistrationStatus.SUSPENDED
    assert stored[older_hand][2] is RegisterSource.SNAPSHOT
    # An older snapshot keeps the newer hand-typed row, and says so.
    assert kept.kept == [newer_hand]
    assert [c.incoming_wins for c in kept.conflicts] == [False]
    assert stored[newer_hand][1:] == ("Hand-typed name", RegisterSource.MANUAL)
    # Same day: the snapshot wins.
    assert [c.incoming_wins for c in tied.conflicts] == [True]
    assert stored[same_day][1:] == ("Snapshot name", RegisterSource.SNAPSHOT)
    # Never deleted by a snapshot.
    assert stored[hand_only][2] is RegisterSource.MANUAL


def test_the_hand_typed_report_lists_conflicts_and_never_overrides_a_newer_snapshot(
    sessions: Any, tmp_path: Path
) -> None:
    from medlens.seed.greenbook import import_snapshot
    from tests.unit.test_greenbook import record

    older_file, newer_file = unique_numbers(2)
    snapshot = snapshot_of([record(NAFDAC=older_file), record(NAFDAC=newer_file)], "2026-11-11")
    hand_typed = write(
        tmp_path / "manual.csv",
        [
            {**GOOD, "nafdac_number": older_file, "strength": "80 mg; 480 mg"},  # 2026-11-10
            {**GOOD, "nafdac_number": newer_file, "retrieved_on": "2026-11-12"},
        ],
    )
    with sessions() as session:
        import_snapshot(session, *snapshot)
        rows, _ = read_rows(hand_typed, TODAY)
        report = diff(session, rows)
        output = render(report, [])
        apply(session, report, file_name="manual.csv", sha256="f" * 64)
        stored = dict(
            session.execute(
                select(Product.nafdac_number, Product.register_source).where(
                    Product.nafdac_number.in_([older_file, newer_file])
                )
            )
            .tuples()
            .all()
        )
        strength = session.scalar(
            select(Product.strength).where(Product.nafdac_number == older_file)
        )
        session.rollback()

    winners = {c.nafdac_number: c.incoming_wins for c in report.conflicts}
    assert winners == {older_file: False, newer_file: True}
    assert [row.nafdac_number for row in report.superseded] == [older_file]
    assert [row.nafdac_number for row, _ in report.changed] == [newer_file]
    assert f"<> {older_file}: stored snapshot retrieved 2026-11-11, file retrieved 2026-11-10" in (
        output
    )
    assert "strength: '500 mg/65 mg' (stored) vs '80 mg; 480 mg' (file)" in output
    assert strength == "500 mg/65 mg"  # the newer snapshot's facts stay
    assert stored == {older_file: RegisterSource.SNAPSHOT, newer_file: RegisterSource.MANUAL}


def _check_digit(body: str) -> int:
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return (10 - total % 10) % 10

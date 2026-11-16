"""Validation of hand-typed register rows (seed/register_import.py). Diff and apply run against
a database: tests/integration/test_register_import.py."""

import csv
from datetime import date
from pathlib import Path

import pytest

from medlens.modules.products.models import GtinSource
from medlens.seed.register_import import MANUAL_COLUMNS, Report, read_rows, render, validate_row

ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 11, 12)
GOOD = {
    "nafdac_number": "A4-1234",
    "brand_name": "Coartem Dispersible",
    "generic_name": "Artemether; Lumefantrine",
    "strength": "20 mg; 120 mg",
    "dosage_form": "Dispersible tablet",
    "pack_size": "1 x 6",
    "manufacturer": "Novartis Nigeria Limited",
    "category": "Drugs",
    "registration_status": "active",
    "registration_date": "2023-04-25",
    "registration_expires_on": "2028-04-25",
    "greenbook_product_id": "123",
    "source_url": "https://greenbook.nafdac.gov.ng/?q=A4-1234",
    "retrieved_on": "2026-11-10",
}


def row(**changes: str) -> dict[str, str]:
    return {**GOOD, **changes}


def test_a_complete_row_is_valid_and_parsed() -> None:
    result = validate_row(2, row(nafdac_number="a4 1234", registration_status="Active"), TODAY)

    assert result.problems == []
    assert result.nafdac_number == "A4-1234"
    assert result.values is not None
    assert result.values["registration_status"] == "active"
    assert result.values["registration_date"] == date(2023, 4, 25)
    assert result.values["retrieved_on"] == date(2026, 11, 10)
    assert result.values["source_url"] == GOOD["source_url"]
    assert result.values["manufacturer_name"] == "Novartis Nigeria Limited"
    assert result.values["active_ingredients"] == [
        {"name": "artemether", "strength_value": 20.0, "strength_unit": "mg"},
        {"name": "lumefantrine", "strength_value": 120.0, "strength_unit": "mg"},
    ]


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"nafdac_number": "XYZ"}, "is not a NAFDAC number"),
        ({"brand_name": ""}, "brand_name is required"),
        ({"manufacturer": ""}, "manufacturer is required"),
        ({"source_url": ""}, "source_url is required"),
        ({"registration_status": "inactive"}, "is not active, expired, suspended or revoked"),
        ({"registration_expires_on": "25/04/2028"}, "is not a date"),
        ({"registration_date": "2029-01-01"}, "registration_date is after registration_expires_on"),
        ({"retrieved_on": "2026-12-01"}, "retrieved_on is in the future"),
        ({"registration_expires_on": "2026-01-01"}, "expired before it was retrieved"),
        ({"registration_status": "expired"}, "had not passed when retrieved"),
        ({"source_url": "https://example.com/coartem"}, "must be an https://greenbook"),
        ({"source_url": "http://greenbook.nafdac.gov.ng/"}, "must be an https://greenbook"),
        ({"strength": "20 mg"}, "2 ingredient(s) but 1 strength(s)"),
        ({"strength": "lots"}, "does not parse"),
    ],
)
def test_invalid_rows_say_why(changes: dict[str, str], problem: str) -> None:
    result = validate_row(7, row(**changes), TODAY)

    assert result.values is None
    assert any(problem in reason for reason in result.problems), result.problems


def test_an_unknown_dosage_form_is_a_warning_not_an_error() -> None:
    result = validate_row(3, row(dosage_form="Implant rod"), TODAY)

    assert result.problems == []
    assert result.warnings == ["dosage_form 'Implant rod' is not a known form (kept as is)"]


def write(
    path: Path, rows: list[dict[str, str]], columns: tuple[str, ...] = MANUAL_COLUMNS
) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_duplicated_numbers_are_invalid_on_every_line(tmp_path: Path) -> None:
    rows, problems = read_rows(
        write(tmp_path / "f.csv", [row(), row(brand_name="Other"), row(nafdac_number="04-0289")]),
        TODAY,
    )

    assert problems == []
    assert [r.values is None for r in rows] == [True, True, False]
    assert rows[0].problems == ["NAFDAC number on more than one line: 2, 3"]


def test_files_without_the_required_columns_are_refused(tmp_path: Path) -> None:
    rows, problems = read_rows(
        write(tmp_path / "f.csv", [row()], columns=("nafdac_number", "brand_name")), TODAY
    )

    assert rows == []
    assert problems[0].startswith("missing columns: generic_name, dosage_form, manufacturer")
    assert "cannot be read" in render(Report(), problems)


def test_snapshot_files_with_fetched_on_are_accepted(tmp_path: Path) -> None:
    columns = tuple(c if c != "retrieved_on" else "fetched_on" for c in MANUAL_COLUMNS)
    record = {**row(), "fetched_on": "2026-11-03"}

    rows, problems = read_rows(write(tmp_path / "f.csv", [record], columns), TODAY)

    assert problems == []
    assert rows[0].values is not None
    assert rows[0].values["retrieved_on"] == date(2026, 11, 3)


def test_the_committed_manual_file_is_valid() -> None:
    rows, problems = read_rows(ROOT / "seed" / "greenbook_manual.csv", TODAY)

    assert problems == []
    assert rows
    assert all(r.problems == [] for r in rows), [r.problems for r in rows]


# --------------------------------------------------------------------------- GTINs
def test_gtins_are_normalised_and_checked() -> None:
    result = validate_row(
        2, row(gtin="6291041500213; 06291041500220", gtin_source="Manufacturer"), TODAY
    )
    broken = validate_row(3, row(gtin="6291041500214", gtin_source="register"), TODAY)
    no_source = validate_row(4, row(gtin="6291041500213"), TODAY)
    cleared = validate_row(5, row(gtin="", gtin_source=""), TODAY)
    no_column = validate_row(6, row(), TODAY)

    assert result.values is not None
    assert result.gtins == {
        "06291041500213": GtinSource.MANUFACTURER,
        "06291041500220": GtinSource.MANUFACTURER,
    }
    assert broken.values is None
    assert "gtin '6291041500214' is not a GTIN" in broken.problems[0]
    assert no_source.problems == [
        "gtin_source must be register or manufacturer when a gtin is given"
    ]
    assert cleared.gtins == {}  # an empty cell clears the product's GTINs
    assert no_column.values is not None
    assert no_column.gtins is None  # files without the column leave GTINs alone


def test_a_gtin_on_two_products_in_one_file_is_invalid(tmp_path: Path) -> None:
    columns = (*MANUAL_COLUMNS,)
    rows, _ = read_rows(
        write(
            tmp_path / "f.csv",
            [
                row(gtin="6291041500213", gtin_source="register"),
                row(nafdac_number="04-0289", gtin="06291041500213", gtin_source="register"),
            ],
            columns,
        ),
        TODAY,
    )

    assert all(r.values is None for r in rows)
    assert "GTIN 06291041500213 is on more than one product" in rows[0].problems[0]

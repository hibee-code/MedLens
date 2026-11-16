"""Reading register fields from pack text (verification/pack_details.py): golden fixtures in
tests/fixtures/pack_text (OCR text and expected.json), unreadable cases included, and the
fields the vision LLM may add."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from medlens.modules.verification.extraction import FieldSource, from_gs1, from_ocr
from medlens.modules.verification.pack_details import details_from_readings, read_pack_details

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pack_text"
EXPECTED: dict[str, dict[str, Any]] = json.loads((FIXTURES / "expected.json").read_text())


def value(extraction_json: dict[str, Any], name: str) -> Any:
    item = extraction_json.get(name)
    return item["value"] if item else None


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_golden_pack_texts(case: str) -> None:
    expected = EXPECTED[case]
    text = (FIXTURES / f"{case}.txt").read_text()

    read = read_pack_details(text, 0.9, expected.get("extra_names", [])).as_json()

    for name in ("active_ingredients", "strengths", "dosage_form", "companies"):
        assert value(read, name) == expected[name], name
    for item in read.values():
        assert item["source"] == "ocr"
        assert item["confidence"] == 0.9  # no line measured: the whole text's confidence


def test_each_field_takes_the_measured_confidence_of_its_lines() -> None:
    """The owner's 0.85 bar compares the OCR engine's own confidence for the lines a value
    was read from (the lowest of them), not a fixed discount."""
    text = (
        "EMZOR PARACETAMOL SYRUP\n"
        "Each 5 ml contains:\n"
        "Paracetamol 250 mg\n"
        "Oral suspension\n"
        "Manufactured by: Emzor Pharmaceutical Industries Ltd"
    )
    lines = (
        ("EMZOR PARACETAMOL SYRUP", 0.95),
        ("Each 5 ml contains:", 0.97),
        ("Paracetamol 250 mg", 0.71),
        ("Manufactured by: Emzor Pharmaceutical Industries Ltd", 0.92),
    )  # "Oral suspension" was not measured: the whole text's 0.9

    read = read_pack_details(text, 0.9, (), lines)

    assert read.active_ingredients is not None
    assert read.active_ingredients.confidence == 0.71
    assert read.strengths is not None
    assert read.strengths.confidence == 0.71  # the lower of the volume line and its own
    assert read.dosage_form is not None
    assert read.dosage_form.confidence == 0.9
    assert read.companies is not None
    assert read.companies.confidence == 0.92


def test_a_manufacture_date_takes_its_line_confidence() -> None:
    text = "NAFDAC REG NO: A4-1234\nMFG: 03/2025 EXP: 02/2028"

    measured = from_ocr(text, 0.9, (("MFG: 03/2025 EXP: 02/2028", 0.62),))
    unmeasured = from_ocr(text, 0.9)

    assert measured.manufactured_on is not None
    assert measured.manufactured_on.confidence == 0.62
    assert unmeasured.manufactured_on is not None
    assert unmeasured.manufactured_on.confidence == 0.9


def test_every_fixture_has_an_expectation() -> None:
    texts = {path.stem for path in FIXTURES.glob("*.txt")}

    assert texts == set(EXPECTED)


def test_empty_text_reads_nothing() -> None:
    assert read_pack_details("   ", 0.9).as_json() == {}


def test_manufacture_dates_from_the_label_and_the_barcode() -> None:
    printed = from_ocr("NAFDAC REG NO: A4-1234\nMFG: 03/2025 EXP: 02/2028", 0.9)
    full = from_ocr("Date of manufacture: 14/03/2025", 0.9)
    barcode = from_gs1("(01)06291041500213(11)250300(17)280229(10)B1")

    assert printed.manufactured_on is not None
    assert printed.manufactured_on.value == date(2025, 3, 1)  # month only: its first day
    assert printed.expiry_date is not None
    assert printed.expiry_date.value == date(2028, 2, 29)
    assert full.manufactured_on is not None
    assert full.manufactured_on.value == date(2025, 3, 14)
    assert barcode.manufactured_on is not None
    assert barcode.manufactured_on.value == date(2025, 3, 1)  # GS1 day 00
    assert from_ocr("Manufactured by Emzor", 0.9).manufactured_on is None


def test_the_vision_llm_may_add_register_fields() -> None:
    read = details_from_readings(
        ingredients=("Artemether; Lumefantrine", 0.95),
        strengths=("20 mg; 120 mg", 0.9),
        dosage_form=("Dispersible tablets", 0.9),
        manufacturer=("Novartis Pharma AG, Basel", 0.7),
        cap=0.8,
    ).as_json()

    assert value(read, "active_ingredients") == ["artemether", "lumefantrine"]
    assert value(read, "strengths") == [
        {"strength_value": 20.0, "strength_unit": "mg"},
        {"strength_value": 120.0, "strength_unit": "mg"},
    ]
    assert value(read, "dosage_form") == "tablet"
    assert value(read, "companies") == ["novartis pharmaceutical ag"]
    assert {item["source"] for item in read.values()} == {FieldSource.LLM.value}
    assert max(item["confidence"] for item in read.values()) == 0.8  # capped


def test_fields_the_llm_could_not_read_stay_unread() -> None:
    read = details_from_readings(
        ingredients=(None, 0.0),
        strengths=("lots", 0.9),
        dosage_form=("an unknown form", 0.9),
        manufacturer=("X", 0.9),
        cap=0.8,
    )

    assert read.as_json() == {}


def test_ocr_fields_win_over_less_confident_llm_fields() -> None:
    ocr = read_pack_details("Paracetamol 500 mg tablets\ntablets", 0.95)
    llm = details_from_readings(
        ingredients=("Ibuprofen", 0.6),
        strengths=(None, 0.0),
        dosage_form=(None, 0.0),
        manufacturer=(None, 0.0),
        cap=0.8,
    )

    merged = ocr.merge(llm)

    assert merged.active_ingredients is not None
    assert merged.active_ingredients.value == ("paracetamol",)


def test_the_nafdac_number_takes_its_line_confidence() -> None:
    """The barcode guard compares the number's own reading, not the page's average, which
    noise elsewhere on the pack pulls down."""
    text = "Augmentin 1 G\nNAFDAC REG. NO: 04-8655\nSYNTHETIC TEST _, Ween ee eg"
    lines = (("NAFDAC REG. NO: 04-8655", 0.945), ("SYNTHETIC TEST _, Ween ee eg", 0.31))

    measured = from_ocr(text, 0.82, lines)
    unmeasured = from_ocr(text, 0.82)
    bare = from_ocr("Augmentin 04-8655", 0.9, (("Augmentin 04-8655", 0.9),))

    assert measured.nafdac_number is not None
    assert measured.nafdac_number.confidence == 0.945
    assert unmeasured.nafdac_number is not None
    assert unmeasured.nafdac_number.confidence == 0.82
    assert bare.nafdac_number is not None
    assert bare.nafdac_number.confidence == 0.63  # unlabelled: 0.9 x 0.7


def test_the_nafdac_number_takes_its_own_words_confidence() -> None:
    """A clear label on the same line cannot lift a doubtful number: the number's own words
    decide (the lowest of them)."""
    text = "NAFDAC REG. NO: 04-8655\nBATCH NO: B1"
    lines = (("NAFDAC REG. NO: 04-8655", 0.86), ("BATCH NO: B1", 0.95))
    words = (("NAFDAC REG. NO: 04-8655", (0.97, 0.96, 0.95, 0.58)), ("BATCH NO: B1", (0.95,) * 3))

    doubtful = from_ocr(text, 0.9, lines, words)
    split = from_ocr(
        "NAFDAC NO: A4 1234", 0.9, (), (("NAFDAC NO: A4 1234", (0.97, 0.96, 0.9, 0.7)),)
    )
    unaligned = from_ocr(text, 0.9, lines, (("NAFDAC REG. NO: 04-8655", (0.9,)),))

    assert doubtful.nafdac_number is not None
    assert doubtful.nafdac_number.confidence == 0.58  # not the line's 0.86
    assert split.nafdac_number is not None
    assert split.nafdac_number.value == "A4-1234"
    assert split.nafdac_number.confidence == 0.7  # the lowest of the number's two words
    assert unaligned.nafdac_number is not None
    assert unaligned.nafdac_number.confidence == 0.86  # words do not fit the line: the line's

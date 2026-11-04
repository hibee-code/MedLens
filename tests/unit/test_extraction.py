"""Reading fields from GS1 barcodes and OCR text."""

from datetime import date

import pytest

from medlens.modules.verification.extraction import (
    ExtractedField,
    Extraction,
    FieldSource,
    from_gs1,
    from_ocr,
    gs1_date,
    name_presence,
    parse_expiry,
    parse_gs1,
    valid_gtin,
)

GS = "\x1d"
GTIN = "06291041500213"  # valid check digit


# ------------------------------------------------------------------------- GS1
def test_scanned_gs1_strings_are_parsed() -> None:
    elements = parse_gs1(f"]d201{GTIN}17270331{GS}10AB1234{GS}21SERIAL9")

    assert elements == {"01": GTIN, "17": "270331", "10": "AB1234", "21": "SERIAL9"}


def test_human_readable_gs1_strings_are_parsed() -> None:
    assert parse_gs1(f"(01){GTIN}(17)270300(10)AB1234") == {
        "01": GTIN,
        "17": "270300",
        "10": "AB1234",
    }


def test_parsing_stops_at_an_unknown_identifier() -> None:
    assert parse_gs1(f"01{GTIN}99internal") == {"01": GTIN}


@pytest.mark.parametrize(
    ("gtin", "valid"),
    [(GTIN, True), ("06291041500214", False), ("abc", False), ("4006381333931", True)],
)
def test_gtin_check_digits(gtin: str, valid: bool) -> None:
    assert valid_gtin(gtin) is valid


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("270331", date(2027, 3, 31)),
        ("270300", date(2027, 3, 31)),  # day 00: end of month
        ("990101", date(1999, 1, 1)),
        ("271301", None),
        ("270230", None),
        ("2703", None),
    ],
)
def test_gs1_dates(value: str, expected: date | None) -> None:
    assert gs1_date(value) == expected


def test_barcode_fields_are_normalised() -> None:
    extraction = from_gs1(f"01{GTIN}17270331{GS}10 ab-1234")

    assert extraction.gtin == ExtractedField(GTIN, FieldSource.BARCODE, 0.98)
    assert extraction.batch_number is not None
    assert extraction.batch_number.value == "AB-1234"
    assert extraction.expiry_date is not None
    assert extraction.expiry_date.value == date(2027, 3, 31)


def test_a_bad_gtin_discards_the_read() -> None:
    assert from_gs1("010629104150021417270331") == Extraction()


# ------------------------------------------------------------------------- OCR
PACK_TEXT = """
Panadol Extra  Paracetamol 500mg Caffeine 65mg
NAFDAC REG. NO: A4-1234
BATCH NO: B2309-14   MFG: 01/2025   EXP: 12/2027
"""


def test_labelled_fields_are_read_from_ocr_text() -> None:
    extraction = from_ocr(PACK_TEXT, 0.9)

    assert extraction.nafdac_number == ExtractedField("A4-1234", FieldSource.OCR, 0.9)
    assert extraction.batch_number == ExtractedField("B2309-14", FieldSource.OCR, 0.9)
    assert extraction.expiry_date == ExtractedField(date(2027, 12, 31), FieldSource.OCR, 0.9)


def test_unlabelled_and_corrected_numbers_are_less_certain() -> None:
    bare = from_ocr("some text 04-1234 more", 1.0).nafdac_number
    corrected = from_ocr("NAFDAC No. A4-I234", 1.0).nafdac_number

    assert bare is not None
    assert bare.confidence == pytest.approx(0.7)
    assert corrected is not None
    assert corrected.value == "A4-1234"
    assert corrected.confidence == pytest.approx(0.85)


def test_text_without_fields_yields_nothing() -> None:
    extraction = from_ocr("TAKE TWO TABLETS DAILY", 0.9)

    assert extraction.missing(0.5) == ["nafdac_number", "batch_number", "expiry_date"]
    assert extraction.text == "TAKE TWO TABLETS DAILY"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12/2027", date(2027, 12, 31)),
        ("02/27", date(2027, 2, 28)),
        ("MAR 2027", date(2027, 3, 31)),
        ("SEPT-27", date(2027, 9, 30)),
        ("15/06/2026", date(2026, 6, 15)),
        ("2027-04", date(2027, 4, 30)),
        ("13/2027", None),
        ("31/02/2027", None),
        ("SOON", None),
        ("12/123", None),
    ],
)
def test_expiry_formats(raw: str, expected: date | None) -> None:
    assert parse_expiry(raw) == expected


# ------------------------------------------------------------------------- merging
def test_merge_keeps_the_most_confident_value() -> None:
    claimed = Extraction(nafdac_number=ExtractedField("A4-1111", FieldSource.CLAIMED, 0.5))
    read = from_ocr(PACK_TEXT, 0.9)

    merged = claimed.merge(read)

    assert merged.nafdac_number is not None
    assert merged.nafdac_number.value == "A4-1234"
    assert claimed.merge(Extraction()).nafdac_number == claimed.nafdac_number


def test_extraction_json_holds_values_not_text() -> None:
    data = from_ocr(PACK_TEXT, 0.9).as_json()

    assert data["expiry_date"] == {"value": "2027-12-31", "source": "ocr", "confidence": 0.9}
    assert "Panadol" not in str(data)


# ------------------------------------------------------------------------- names
def test_name_presence_ignores_dosage_words() -> None:
    assert name_presence("Panadol Extra Caplets", PACK_TEXT) == 1.0
    assert name_presence("Coartem 20/120 Tablets", PACK_TEXT) == 0.0
    assert name_presence("Tablets 500 mg", PACK_TEXT) is None
    assert name_presence("Panadol", "   ") is None

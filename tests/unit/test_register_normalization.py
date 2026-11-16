"""Normalising register and pack text (modules/products/register.py)."""

import csv
from decimal import Decimal
from pathlib import Path

import pytest

from medlens.modules.products.register import (
    Strength,
    dosage_form_family,
    normalize_company,
    normalize_ingredient,
    normalize_nafdac_number,
    parse_ingredients,
    parse_strength,
    parse_strengths,
    same_strength,
)

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- strengths
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("500mg", "500 mg"),
        ("500 mg", "0.5 g"),
        ("0.5g", "500 MG"),
        ("1000 mcg", "1 mg"),
        ("250 µg", "0.25 mg"),
        ("125 mg/5 ml", "25 mg/mL"),
        ("125mg/5mL", "125 mg per 5 ml"),
        ("600 mg/2 mL", "300 mg/ml"),
        ("10 IU/mL", "10 i.u./ml"),
        ("100 mcg/Metered-Dose", "100 mcg/dose"),
        ("1%", "1 %"),
        ("2 % w/w", "2%"),
        ("0,5 g", "500 mg"),
    ],
)
def test_equal_strengths(a: str, b: str) -> None:
    first, second = parse_strength(a), parse_strength(b)

    assert first is not None
    assert second is not None
    assert same_strength(first, second)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("500 mg", "250 mg"),
        ("125 mg/5 mL", "250 mg/5 mL"),
        ("125 mg/5 mL", "125 mg"),  # a concentration is not an amount per tablet
        ("10 IU", "10 mg"),
        ("1%", "1 mg"),
    ],
)
def test_different_strengths(a: str, b: str) -> None:
    first, second = parse_strength(a), parse_strength(b)

    assert first is not None
    assert second is not None
    assert not same_strength(first, second)


@pytest.mark.parametrize("text", ["", "mg", "five hundred mg", "500", "500 mg/5 bananas", "12 kg"])
def test_text_that_is_not_a_strength(text: str) -> None:
    assert parse_strength(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("80 mg; 480 mg", [("80", "mg"), ("480", "mg")]),
        ("20/120 mg", [("20", "mg"), ("120", "mg")]),
        ("20mg/120mg", [("20", "mg"), ("120", "mg")]),
        ("500 mg + 125 mg", [("500", "mg"), ("125", "mg")]),
        ("875 mg, 125 mg", [("875", "mg"), ("125", "mg")]),
        ("1000 mg/20 mL; 45 mg/20 mL", [("1000", "mg"), ("45", "mg")]),
        ("", []),
    ],
)
def test_strength_lists(text: str, expected: list[tuple[str, str]]) -> None:
    parsed = parse_strengths(text)

    assert parsed is not None
    assert [(strength.value, strength.unit) for strength in parsed] == [
        (Decimal(value), unit) for value, unit in expected
    ]


def test_a_list_with_an_unreadable_part_is_unreadable() -> None:
    assert parse_strengths("500 mg; lots") is None
    assert parse_strengths("20/120") is None  # no unit anywhere


def test_strengths_serialise_for_the_register() -> None:
    syrup = parse_strength("125 mg/5 mL")
    tablet = parse_strength("0.5 g")
    inhaler = parse_strength("100 mcg/Metered-Dose")

    assert syrup is not None
    assert tablet is not None
    assert inhaler is not None
    assert syrup.as_json() == {"strength_value": 125.0, "strength_unit": "mg/5 mL"}
    assert tablet.as_json() == {"strength_value": 500.0, "strength_unit": "mg"}
    assert inhaler.as_json() == {"strength_value": 0.1, "strength_unit": "mg/dose"}
    assert Strength(Decimal(10), "IU", Decimal(1), "mL").as_json()["strength_unit"] == "IU/mL"


# --------------------------------------------------------------------------- dosage forms
@pytest.mark.parametrize(
    ("text", "family"),
    [
        ("Tablet", "tablet"),
        ("Caplet", "tablet"),
        ("Film-coated tablets", "tablet"),
        ("Prolonged-release tablet", "tablet"),
        ("Chewable tablet", "tablet"),
        ("Capsule", "capsule"),
        ("Soft gelatin capsules", "capsule"),
        ("Syrup", "oral liquid"),
        ("Oral suspension", "oral liquid"),
        ("Suspension", "oral liquid"),
        ("Injection", "injection"),
        ("Solution for injection", "injection"),
        ("Powder for injection", "injection"),
        ("Intravenous infusion", "injection"),
        ("Eye drops", "drops"),
        ("Ophthalmic solution", "drops"),
        ("Oral powder", "powder"),
        ("Aerosol, Metered-Dose", "inhaler"),
        ("Cream", "cream"),
        ("Suppository", "suppository"),
    ],
)
def test_dosage_form_families(text: str, family: str) -> None:
    assert dosage_form_family(text) == family


def test_an_unknown_dosage_form_has_no_family() -> None:
    assert dosage_form_family("Implant rod") is None
    assert dosage_form_family("") is None


# --------------------------------------------------------------------------- ingredients
@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Amoxicillin (Amoxicillin Trihydrate)", "amoxicillin"),
        ("amoxicillin trihydrate", "amoxicillin"),
        ("AMOXYCILLIN", "amoxicillin"),
        ("Diclofenac Sodium (Diclofenac Diethylamine)", "diclofenac"),
        ("Diclofenac Potassium", "diclofenac"),
        ("Metformin Hydrochloride", "metformin"),
        ("Diphenhydramine HCl", "diphenhydramine"),
        ("Clavulanic Acid (Potassium Clavulanate)", "clavulanic acid"),
        ("Potassium clavulanate", "clavulanic acid"),
        ("Acetaminophen", "paracetamol"),
        ("Acetylsalicylic acid", "aspirin"),
        ("Magnesium hydroxide", "magnesium hydroxide"),
        ("Magnesium trisilicate", "magnesium trisilicate"),
        ("Aluminum hydroxide", "aluminium hydroxide"),
        ("Ferrous sulphate", "ferrous sulfate"),
        ("Sodium citrate", "sodium citrate"),
        ("Salbutamol (Salbutamol Sulfate)", "salbutamol"),
        ("Cefuroxime (Cefuroxime Axetil)", "cefuroxime"),
        ("Vitamin C", "ascorbic acid"),
        ("Artenimol", "dihydroartemisinin"),
        ("", ""),
    ],
)
def test_ingredient_names(text: str, name: str) -> None:
    assert normalize_ingredient(text) == name


def test_multi_ingredient_registrations_pair_names_and_strengths() -> None:
    coartem = parse_ingredients("Artemether; Lumefantrine", "20 mg; 120 mg")
    pack = parse_ingredients("Artemether; Lumefantrine", "20/120 mg")
    cotrimoxazole = parse_ingredients("Co-trimoxazole", "")

    assert [item.as_json() for item in coartem] == [
        {"name": "artemether", "strength_value": 20.0, "strength_unit": "mg"},
        {"name": "lumefantrine", "strength_value": 120.0, "strength_unit": "mg"},
    ]
    assert [(item.name, item.strength) for item in pack] == [
        (item.name, item.strength) for item in coartem
    ]
    assert [item.name for item in cotrimoxazole] == ["sulfamethoxazole", "trimethoprim"]
    assert cotrimoxazole[0].as_json()["strength_value"] is None


@pytest.mark.parametrize(
    ("names", "strengths", "message"),
    [
        ("Artemether; Lumefantrine", "20 mg", "2 ingredient(s) but 1 strength(s)"),
        ("Paracetamol", "lots", "does not parse"),
        ("", "500 mg", "no active ingredient"),
    ],
)
def test_register_rows_that_do_not_pair_up_are_refused(
    names: str, strengths: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        parse_ingredients(names, strengths)


def test_every_row_of_the_committed_sample_normalises() -> None:
    with (ROOT / "seed" / "products.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        assert normalize_nafdac_number(row["nafdac_number"]) == row["nafdac_number"]
        assert parse_ingredients(row["generic_name"], row["strength"]), row
        assert dosage_form_family(row["dosage_form"]) is not None, row["dosage_form"]


# --------------------------------------------------------------------------- companies
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("GlaxoSmithKline Consumer Nigeria PLC", "Glaxosmithkline Consumer Nig. Ltd."),
        ("Emzor Pharmaceutical Industries Limited", "EMZOR PHARMACEUTICAL INDUSTRIES LTD"),
        ("May & Baker Nigeria PLC", "May and Baker Nigeria Plc."),
        ("Phillips Pharmaceuticals (Nigeria) Limited", "Phillips Pharmaceuticals Nigeria Ltd"),
        ("Fidson Healthcare PLC", "Fidson Healthcare Public Limited Company"),
        ("Swiss Pharma Nigeria Limited", "Swiss Pharmaceutical Nig Ltd"),
    ],
)
def test_equal_company_names(a: str, b: str) -> None:
    assert normalize_company(a) == normalize_company(b)


def test_different_companies_stay_different() -> None:
    assert normalize_company("Emzor Pharmaceutical Industries") != normalize_company("Fidson")
    assert normalize_company("Copac Nig Ltd") == "copac nigeria"


def test_gtins_are_padded_to_14_digits_and_check_digits_verified() -> None:
    from medlens.core.identifiers import normalize_gtin, valid_gtin

    assert normalize_gtin("6291041500213") == "06291041500213"
    assert normalize_gtin("0629-1041-500213") == "06291041500213"
    assert valid_gtin("06291041500213")
    for bad in ("6291041500214", "12345", "abcdefghijklm"):
        with pytest.raises(ValueError, match="valid check digit"):
            normalize_gtin(bad)

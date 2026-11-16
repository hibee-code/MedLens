"""The register check (verification/register_check.py) and its scoring (weighted-rules-v3, off
unless MEDLENS_VERIFICATION__REGISTER_CHECK_SCORING)."""

from datetime import date
from uuid import uuid4

import pytest

from medlens.modules.products.models import RegistrationStatus
from medlens.modules.products.register import parse_ingredients, parse_strengths
from medlens.modules.scans.models import Verdict
from medlens.modules.verification.catalog import BatchFacts, GtinOwner, ProductFacts
from medlens.modules.verification.codes import ReasonCode
from medlens.modules.verification.extraction import ExtractedField, Extraction, FieldSource
from medlens.modules.verification.register_check import (
    MIN_MISMATCH_CONFIDENCE,
    Outcome,
    barcode_signals,
    check_register,
    register_signals,
)
from medlens.modules.verification.scoring import (
    REGISTER_WEIGHTS,
    UNAVAILABLE,
    WEIGHTS,
    ScoreInput,
    Signal,
    SignalName,
    WeightedRuleScorer,
)

COARTEM = ProductFacts(
    id=uuid4(),
    nafdac_number="A4-1234",
    brand_name="Coartem Dispersible",
    generic_name="Artemether; Lumefantrine",
    registration_status=RegistrationStatus.ACTIVE,
    active_ingredients=tuple(parse_ingredients("Artemether; Lumefantrine", "20 mg; 120 mg")),
    dosage_form="Dispersible tablet",
    holder_name="Novartis Nigeria Limited",
    registration_date=date(2022, 1, 10),
    registration_expires_on=date(2027, 1, 9),
)


def ocr[T](value: T, confidence: float = 0.9) -> ExtractedField[T]:
    return ExtractedField(value, FieldSource.OCR, confidence)


def pack(**fields: object) -> Extraction:
    defaults: dict[str, object] = {"text": "COARTEM Dispersible"}
    return Extraction(**{**defaults, **fields})  # type: ignore[arg-type]


def strengths(text: str) -> ExtractedField[tuple[object, ...]]:
    parsed = parse_strengths(text)
    assert parsed is not None
    return ocr(tuple(parsed))


def test_a_pack_that_agrees_with_its_registration() -> None:
    check = check_register(
        COARTEM,
        pack(
            active_ingredients=ocr(("artemether", "lumefantrine")),
            strengths=strengths("20/120 mg"),
            dosage_form=ocr("tablet"),
            companies=ocr(("novartis pharmaceutical ag",)),
            manufactured_on=ocr(date(2025, 3, 1)),
        ),
    )

    assert check.mismatched == []
    assert check.as_json()["strength"] == {
        "outcome": "matched",
        "pack": ["20 mg", "120 mg"],
        "register": ["20 mg", "120 mg"],
        "confidence": 0.9,
        "counted": True,
    }
    assert check.holder.outcome is Outcome.MATCHED  # same group, other company


@pytest.mark.parametrize(
    ("fields", "mismatched"),
    [
        ({"active_ingredients": ocr(("artemether", "paracetamol"))}, ["active_ingredients"]),
        ({"strengths": strengths("80 mg; 480 mg")}, ["strength"]),
        ({"dosage_form": ocr("oral liquid")}, ["dosage_form"]),
        ({"companies": ocr(("emzor pharmaceutical industry",))}, ["holder"]),
        ({"manufactured_on": ocr(date(2021, 6, 1))}, ["registration_window"]),  # before
        ({"manufactured_on": ocr(date(2027, 3, 1))}, ["registration_window"]),  # lapsed
        ({"text": "Coartem Disp"}, ["product_name"]),  # partly read: evidence
        ({"text": "CORTEX Disp"}, []),  # none of the name read: no evidence either way
    ],
)
def test_mismatches_field_by_field(fields: dict[str, object], mismatched: list[str]) -> None:
    assert check_register(COARTEM, pack(**fields)).mismatched == mismatched


def test_unread_fields_and_fields_the_register_lacks() -> None:
    bare = ProductFacts(uuid4(), "04-0001", "Bare", "Bare", RegistrationStatus.ACTIVE)

    unread = check_register(COARTEM, pack(text=""))
    missing = check_register(bare, pack(active_ingredients=ocr(("paracetamol",))))

    assert {item["outcome"] for item in unread.as_json().values()} == {"unread"}
    assert missing.active_ingredients.outcome is Outcome.NOT_IN_REGISTER
    assert missing.registration_window.outcome is Outcome.NOT_IN_REGISTER
    assert missing.mismatched == []


def test_reading_only_some_ingredients_is_not_a_mismatch() -> None:
    check = check_register(COARTEM, pack(active_ingredients=ocr(("lumefantrine",))))

    assert check.active_ingredients.outcome is Outcome.MATCHED


def test_a_combination_printed_as_its_total_strength_matches() -> None:
    septrin = ProductFacts(
        uuid4(), "04-0002", "Septrin", "Co-trimoxazole", RegistrationStatus.ACTIVE,
        active_ingredients=tuple(parse_ingredients("Co-trimoxazole", "")),
    )  # fmt: skip
    registered = ProductFacts(
        uuid4(), "04-0003", "Septrin", "Sulfamethoxazole; Trimethoprim",
        RegistrationStatus.ACTIVE,
        active_ingredients=tuple(
            parse_ingredients("Sulfamethoxazole; Trimethoprim", "400 mg; 80 mg")
        ),
    )  # fmt: skip

    total = check_register(registered, pack(strengths=strengths("480 mg")))
    no_strengths = check_register(septrin, pack(strengths=strengths("480 mg")))

    assert total.strength.outcome is Outcome.MATCHED
    assert no_strengths.strength.outcome is Outcome.NOT_IN_REGISTER


def test_concentrations_compare_per_unit() -> None:
    syrup = ProductFacts(
        uuid4(), "04-0289", "Emzor Paracetamol Syrup", "Paracetamol", RegistrationStatus.ACTIVE,
        active_ingredients=tuple(parse_ingredients("Paracetamol", "125 mg/5 mL")),
        dosage_form="Syrup",
    )  # fmt: skip

    same = check_register(
        syrup, pack(strengths=strengths("25 mg/ml"), dosage_form=ocr("oral liquid"))
    )
    double = check_register(syrup, pack(strengths=strengths("250 mg/5 ml")))

    assert same.mismatched == []
    assert double.mismatched == ["strength"]


def test_the_batch_manufacture_date_is_used_when_the_pack_shows_none() -> None:
    batch = BatchFacts(uuid4(), "B1", date(2028, 1, 31), frozenset(), date(2021, 1, 5))

    check = check_register(COARTEM, pack(), batch)

    assert check.registration_window.outcome is Outcome.MISMATCHED
    assert check.registration_window.pack == "2021-01-05"


def test_a_mismatch_read_below_the_bar_is_shown_but_not_counted() -> None:
    """Owner's guard: a mismatch counts only when read with confidence >= 0.85."""
    unsure = check_register(COARTEM, pack(strengths=ocr(strengths("80 mg; 480 mg").value, 0.84)))
    sure = check_register(COARTEM, pack(strengths=ocr(strengths("80 mg; 480 mg").value, 0.85)))

    assert MIN_MISMATCH_CONFIDENCE == 0.85
    assert unsure.as_json()["strength"]["counted"] is False
    assert unsure.as_json()["strength"]["not_counted_because"] == "read_with_low_confidence"
    assert unsure.mismatched == ["strength"]  # regulators still see it
    assert register_signals(unsure)[SignalName.STRENGTH_MISMATCH] is UNAVAILABLE
    assert register_signals(sure)[SignalName.STRENGTH_MISMATCH] == Signal(1.0)


def test_held_back_mismatches_name_their_signal_and_reason() -> None:
    from medlens.modules.verification.register_check import held_back

    unsure = check_register(
        COARTEM,
        pack(
            strengths=ocr(strengths("80 mg; 480 mg").value, 0.7),
            dosage_form=ocr("oral liquid", 0.95),  # counted: not held back
        ),
    )

    assert held_back(unsure) == [(SignalName.STRENGTH_MISMATCH, "read_with_low_confidence")]


def test_a_declared_manufacture_date_counts_as_read() -> None:
    batch = BatchFacts(uuid4(), "B1", date(2028, 1, 31), frozenset(), date(2021, 1, 5))

    check = check_register(COARTEM, pack(), batch)

    assert check.registration_window.confidence == 1.0
    assert register_signals(check)[SignalName.REGISTERED_AFTER_OR_LAPSED_AT_MANUFACTURE] == Signal(
        1.0
    )


def barcode_pack(gtin_source: FieldSource = FieldSource.BARCODE, number: float = 0.9) -> Extraction:
    return pack(
        nafdac_number=ocr("A4-1234", number),
        gtin=ExtractedField("06291041500213", gtin_source, 0.98),
    )


def test_barcode_guards() -> None:
    other = ProductFacts(uuid4(), "04-0005", "Panadol", "Paracetamol", RegistrationStatus.ACTIVE)
    owner = GtinOwner(other, "manufacturer")

    counted = check_register(COARTEM, barcode_pack(), gtin_owner=owner)
    typed = check_register(COARTEM, barcode_pack(FieldSource.OCR), gtin_owner=owner)
    unsure = check_register(COARTEM, barcode_pack(number=0.84), gtin_owner=owner)
    other_number = check_register(
        COARTEM, pack(nafdac_number=ocr("A4-9999"), gtin=barcode_pack().gtin), gtin_owner=owner
    )

    assert barcode_signals(counted)[SignalName.BARCODE_PRODUCT_MISMATCH] == Signal(1.0)
    assert typed.barcode.not_counted == "gtin_not_from_barcode"
    assert unsure.barcode.not_counted == "nafdac_number_not_read_confidently"
    # The product was not identified by the printed number (a name match, say).
    assert other_number.barcode.not_counted == "nafdac_number_not_read_confidently"
    for check in (typed, unsure, other_number):
        assert all(signal is UNAVAILABLE for signal in barcode_signals(check).values())


def test_signals_never_raise_risk_for_unread_fields() -> None:
    check = check_register(
        COARTEM, pack(text="", strengths=strengths("80 mg; 480 mg"), dosage_form=ocr("tablet"))
    )

    signals = register_signals(check)

    assert signals[SignalName.STRENGTH_MISMATCH] == Signal(1.0)
    assert signals[SignalName.DOSAGE_FORM_MISMATCH] == Signal(0.0)
    assert signals[SignalName.INGREDIENT_MISMATCH] is UNAVAILABLE
    assert signals[SignalName.HOLDER_MISMATCH] is UNAVAILABLE


# --------------------------------------------------------------------------- scoring
def facts(**signals: Signal) -> ScoreInput:
    base = dict.fromkeys(WEIGHTS, Signal(0.0))
    return ScoreInput(
        product_identified=True,
        nafdac_confidence=0.9,
        registration_status=RegistrationStatus.ACTIVE,
        signals={**base, **{SignalName(name): value for name, value in signals.items()}},
        extraction_confidence=0.9,
    )


def test_register_signals_are_ignored_until_switched_on() -> None:
    v2, v3 = WeightedRuleScorer(), WeightedRuleScorer(register_check=True)
    mismatch = facts(strength_mismatch=Signal(1.0))

    off, on = v2.score(mismatch), v3.score(mismatch)

    assert (off.verdict, off.risk_score, off.scorer_version) == (
        Verdict.LIKELY_GENUINE, 0.0, "weighted-rules-v2"
    )  # fmt: skip
    assert ReasonCode.DETAILS_DO_NOT_MATCH_REGISTER not in off.reason_codes
    assert on.verdict is Verdict.SUSPICIOUS
    assert on.scorer_version == "weighted-rules-v3"
    assert on.risk_score == REGISTER_WEIGHTS[SignalName.STRENGTH_MISMATCH]
    # Consumers see one coarse reason, whatever the field.
    assert ReasonCode.DETAILS_DO_NOT_MATCH_REGISTER in on.reason_codes
    assert not any("STRENGTH" in code for code in on.reason_codes)


def test_v2_confidence_is_unchanged_by_the_new_signals() -> None:
    plain = facts()

    assert WeightedRuleScorer().score(plain).confidence == 0.9


def test_unread_register_fields_lower_confidence_only_when_scored() -> None:
    v3 = WeightedRuleScorer(register_check=True)
    all_matched = facts(**dict.fromkeys((n.value for n in REGISTER_WEIGHTS), Signal(0.0)))
    all_unread = facts(**dict.fromkeys((n.value for n in REGISTER_WEIGHTS), UNAVAILABLE))

    matched, unread = v3.score(all_matched), v3.score(all_unread)

    assert matched.risk_score == unread.risk_score == 0.0
    assert unread.confidence is not None
    assert matched.confidence is not None
    assert unread.confidence < matched.confidence


def test_an_ingredient_mismatch_alone_makes_a_pack_high_priority() -> None:
    score = WeightedRuleScorer(register_check=True).score(facts(ingredient_mismatch=Signal(1.0)))

    assert score.high_priority
    assert score.verdict is Verdict.SUSPICIOUS


def test_the_approved_weights() -> None:
    from medlens.modules.verification.scoring import BARCODE_WEIGHTS

    assert {name.value: weight for name, weight in REGISTER_WEIGHTS.items()} == {
        "ingredient_mismatch": 0.60,
        "strength_mismatch": 0.50,
        "dosage_form_mismatch": 0.25,
        "holder_mismatch": 0.10,
        "registered_after_or_lapsed_at_manufacture": 0.25,
    }
    assert {name.value: weight for name, weight in BARCODE_WEIGHTS.items()} == {
        "barcode_product_mismatch": 0.60,
        "barcode_same_holder_mismatch": 0.25,
    }


def test_a_name_mismatch_is_reported_as_details_do_not_match_register() -> None:
    score = WeightedRuleScorer().score(facts(name_mismatch=Signal(1.0)))

    assert score.reason_codes == (ReasonCode.DETAILS_DO_NOT_MATCH_REGISTER,)
    assert ReasonCode.NAME_MISMATCH not in score.reason_codes


def test_the_barcode_check_has_its_own_switch_and_weight() -> None:
    from medlens.modules.verification.scoring import BARCODE_WEIGHTS

    cloned = facts(barcode_product_mismatch=Signal(1.0))

    off = WeightedRuleScorer().score(cloned)
    on = WeightedRuleScorer(barcode_check=True).score(cloned)
    both = WeightedRuleScorer(register_check=True, barcode_check=True)

    assert off.risk_score == 0.0
    assert on.risk_score == BARCODE_WEIGHTS[SignalName.BARCODE_PRODUCT_MISMATCH]
    assert on.high_priority
    assert ReasonCode.DETAILS_DO_NOT_MATCH_REGISTER in on.reason_codes
    assert both.version == "weighted-rules-v3+barcode"

"""The register check: what the pack shows, field by field, against the product's registration
(its NAFDAC Greenbook entry). Pure functions, no I/O.

Each field is one of:

    matched          the pack agrees with the register
    mismatched       the pack shows something the register does not have
    unread           the pack's value could not be read (lowers confidence, never raises risk)
    not_in_register  the register has no value to compare with

Rules, chosen so that reading errors and legitimate labelling never count as mismatches:

* product_name: the registered brand's distinctive words on the pack (the existing name check);
* active_ingredients: every ingredient read must be registered; reading only some is a match;
* strength: every strength read must equal a registered strength, or the registered strengths'
  total ("Co-trimoxazole 480 mg" for 400 mg + 80 mg); compared per unit (125 mg/5 mL = 25 mg/mL);
* dosage_form: the same family (a film-coated tablet is a tablet, a syrup an oral liquid);
* holder: a company named on the pack shares the holder's distinctive name word (packs often name
  the manufacturer or a distributor of the same group: "Novartis Pharma AG" for "Novartis
  Nigeria Ltd");
* registration_window: the manufacture date (from the pack, else the declared batch's) falls
  within the registration's validity;
* barcode: the GTIN in the pack's barcode belongs to the product the printed NAFDAC number
  identifies, not to another registered product (a cloned barcode). GTINs come from the
  register import, each with its source (register or the holder's declaration, shown to
  regulators); a GTIN no product has is not_in_register. When the GTIN's product has the same
  registration holder, it is more likely a re-registration than a clone: a weaker signal.

Owner's guards (CLAUDE.md decision log), so that reading errors never raise risk:

* a register-field mismatch counts only when the pack's value was read with confidence of at
  least MIN_MISMATCH_CONFIDENCE (0.85); otherwise it is shown with counted: false and scored as
  unread (lowers confidence, never raises risk);
* the barcode check counts only when the GTIN came from a decoded barcode and the printed NAFDAC
  number, the one that identified the product, was read with confidence of at least
  MIN_NUMBER_CONFIDENCE (0.85).

Regulators and platform admins see the whole check (scans.signals); consumers see only the
coarse reason DETAILS_DO_NOT_MATCH_REGISTER, so the check does not teach which field to fix.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any

from medlens.modules.products.register import (
    Strength,
    dosage_form_family,
    normalize_company,
    same_strength,
)
from medlens.modules.verification.catalog import BatchFacts, GtinOwner, ProductFacts
from medlens.modules.verification.extraction import (
    ExtractedField,
    Extraction,
    FieldSource,
    name_presence,
)
from medlens.modules.verification.scoring import UNAVAILABLE, Signal, SignalName

NAME_MATCH = 0.8  # share of the brand's distinctive words that makes a name match
MIN_MISMATCH_CONFIDENCE = 0.85  # a mismatch read less surely than this is scored as unread
MIN_NUMBER_CONFIDENCE = 0.85  # the printed NAFDAC number, for the barcode check to count
DECLARED = 1.0  # confidence of a date the holder declared for the batch (not read)
# Why a mismatch was recorded but not counted.
LOW_CONFIDENCE = "read_with_low_confidence"
GTIN_NOT_FROM_BARCODE = "gtin_not_from_barcode"
NUMBER_NOT_READ_SURELY = "nafdac_number_not_read_confidently"


class Outcome(StrEnum):
    MATCHED = "matched"
    MISMATCHED = "mismatched"
    UNREAD = "unread"
    NOT_IN_REGISTER = "not_in_register"


@dataclass(frozen=True, slots=True)
class FieldCheck:
    outcome: Outcome
    pack: object = None
    register: object = None
    confidence: float | None = None  # of the pack's value
    not_counted: str | None = None  # why this result is not scored (a guard)

    @property
    def counted(self) -> bool:
        return self.not_counted is None

    def as_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "outcome": self.outcome.value,
            "pack": self.pack,
            "register": self.register,
        }
        if self.confidence is not None:
            data["confidence"] = round(self.confidence, 3)
        if self.outcome in {Outcome.MATCHED, Outcome.MISMATCHED}:
            data["counted"] = self.counted
            if self.not_counted:
                data["not_counted_because"] = self.not_counted
        return data


def _compared(
    outcome: Outcome, pack: object, register: object, read: ExtractedField[Any] | None
) -> FieldCheck:
    """A matched or mismatched field, with the reading's confidence; a mismatch read with less
    than MIN_MISMATCH_CONFIDENCE is not counted."""
    confidence = read.confidence if read is not None else None
    low = (
        outcome is Outcome.MISMATCHED
        and confidence is not None
        and confidence < MIN_MISMATCH_CONFIDENCE
    )
    return FieldCheck(outcome, pack, register, confidence, LOW_CONFIDENCE if low else None)


@dataclass(frozen=True, slots=True)
class RegisterCheck:
    product_name: FieldCheck
    active_ingredients: FieldCheck
    strength: FieldCheck
    dosage_form: FieldCheck
    holder: FieldCheck
    registration_window: FieldCheck
    barcode: FieldCheck

    def as_json(self) -> dict[str, Any]:
        return {name: getattr(self, name).as_json() for name in self.__slots__}

    @property
    def mismatched(self) -> list[str]:
        return [
            name for name in self.__slots__ if getattr(self, name).outcome is Outcome.MISMATCHED
        ]


def _strength_text(strength: Strength) -> str:
    data = strength.as_json()
    value = Decimal(str(data["strength_value"])).normalize()
    return f"{format(value, 'f')} {data['strength_unit']}"


def _check_name(product: ProductFacts, extraction: Extraction) -> FieldCheck:
    text = extraction.text
    if extraction.product_name:
        text = f"{text} {extraction.product_name.value}"
    presence = name_presence(product.brand_name, text)
    if presence is None or presence == 0.0:
        return FieldCheck(Outcome.UNREAD, None, product.brand_name)
    outcome = Outcome.MATCHED if presence >= NAME_MATCH else Outcome.MISMATCHED
    return FieldCheck(outcome, round(presence, 3), product.brand_name)


def _check_ingredients(product: ProductFacts, extraction: Extraction) -> FieldCheck:
    registered = [item.name for item in product.active_ingredients]
    read = list(extraction.active_ingredients.value) if extraction.active_ingredients else []
    if not registered:
        return FieldCheck(Outcome.NOT_IN_REGISTER, read or None, None)
    if not read:
        return FieldCheck(Outcome.UNREAD, None, registered)
    outcome = Outcome.MATCHED if set(read) <= set(registered) else Outcome.MISMATCHED
    return _compared(outcome, read, registered, extraction.active_ingredients)


def _total(strengths: list[Strength]) -> Strength | None:
    """The registered strengths added up, when they share a unit (co-trimoxazole's 480 mg)."""
    if len(strengths) < 2 or len({(s.unit, s.per_value, s.per_unit) for s in strengths}) != 1:
        return None
    first = strengths[0]
    return Strength(sum((s.value for s in strengths), Decimal(0)), first.unit, first.per_value,
                    first.per_unit)  # fmt: skip


def _check_strength(product: ProductFacts, extraction: Extraction) -> FieldCheck:
    registered = [item.strength for item in product.active_ingredients if item.strength]
    read = list(extraction.strengths.value) if extraction.strengths else []
    shown_register = [_strength_text(s) for s in registered] or None
    if not registered:
        return FieldCheck(Outcome.NOT_IN_REGISTER, [_strength_text(s) for s in read] or None, None)
    if not read:
        return FieldCheck(Outcome.UNREAD, None, shown_register)
    allowed = registered + ([total] if (total := _total(registered)) else [])
    agrees = all(any(same_strength(r, a) for a in allowed) for r in read)
    outcome = Outcome.MATCHED if agrees else Outcome.MISMATCHED
    return _compared(
        outcome, [_strength_text(s) for s in read], shown_register, extraction.strengths
    )


def _check_dosage_form(product: ProductFacts, extraction: Extraction) -> FieldCheck:
    registered = dosage_form_family(product.dosage_form) if product.dosage_form else None
    read = extraction.dosage_form.value if extraction.dosage_form else None
    if registered is None:
        return FieldCheck(Outcome.NOT_IN_REGISTER, read, product.dosage_form)
    if read is None:
        return FieldCheck(Outcome.UNREAD, None, registered)
    outcome = Outcome.MATCHED if read == registered else Outcome.MISMATCHED
    return _compared(outcome, read, registered, extraction.dosage_form)


# Words that say nothing about which company it is.
_GENERIC_COMPANY_WORDS = frozenset(
    [
        "pharmaceutical",
        "pharmaceuticals",
        "pharmacy",
        "pharmacia",
        "healthcare",
        "health",
        "care",
        "consumer",
        "nigeria",
        "global",
        "group",
        "holdings",
        "laboratories",
        "laboratory",
        "labs",
        "industry",
        "medical",
        "medicals",
        "products",
        "drugs",
        "and",
        "the",
        "of",
        "company",
        "ag",
        "sa",
        "plc",
        "ltd",
        "limited",
    ]
)


def _distinctive(name: str) -> set[str]:
    return {w for w in name.split() if len(w) >= 3 and w not in _GENERIC_COMPANY_WORDS}


def _check_holder(product: ProductFacts, extraction: Extraction) -> FieldCheck:
    holder = normalize_company(product.holder_name) if product.holder_name else ""
    read = list(extraction.companies.value) if extraction.companies else []
    if not holder:
        return FieldCheck(Outcome.NOT_IN_REGISTER, read or None, None)
    if not read:
        return FieldCheck(Outcome.UNREAD, None, product.holder_name)
    wanted = _distinctive(holder)
    agrees = any(company == holder or (wanted & _distinctive(company)) for company in read)
    outcome = Outcome.MATCHED if agrees else Outcome.MISMATCHED
    return _compared(outcome, read, product.holder_name, extraction.companies)


def _check_window(
    product: ProductFacts, extraction: Extraction, batch: BatchFacts | None
) -> FieldCheck:
    reading: ExtractedField[date] | None = extraction.manufactured_on
    if reading is None and batch is not None and batch.manufactured_on is not None:
        reading = ExtractedField(batch.manufactured_on, FieldSource.CLAIMED, DECLARED)
    made = reading.value if reading else None
    start, end = product.registration_date, product.registration_expires_on
    register = {
        "registered_on": start.isoformat() if start else None,
        "expires_on": end.isoformat() if end else None,
    }
    if start is None and end is None:
        return FieldCheck(Outcome.NOT_IN_REGISTER, made.isoformat() if made else None, None)
    if made is None:
        return FieldCheck(Outcome.UNREAD, None, register)
    outside = (start is not None and made < start) or (end is not None and made > end)
    outcome = Outcome.MISMATCHED if outside else Outcome.MATCHED
    return _compared(outcome, made.isoformat(), register, reading)


def same_holder(a: ProductFacts, b: ProductFacts) -> bool:
    """Both registrations name the same holder (the register's names, normalised)."""
    first = normalize_company(a.holder_name) if a.holder_name else ""
    second = normalize_company(b.holder_name) if b.holder_name else ""
    return bool(first) and first == second


def _check_barcode(
    product: ProductFacts, extraction: Extraction, owner: GtinOwner | None
) -> FieldCheck:
    read = extraction.gtin
    if read is None:
        return FieldCheck(Outcome.UNREAD, None, None)
    if owner is None:
        return FieldCheck(Outcome.NOT_IN_REGISTER, read.value, None, read.confidence)
    other = owner.product.id != product.id
    register: dict[str, object] = {
        "belongs_to": owner.product.nafdac_number,
        "mapping_source": owner.source,
    }
    if other:
        register["holder"] = owner.product.holder_name
        register["same_holder"] = same_holder(product, owner.product)
    number = extraction.nafdac_number
    guard = None
    if read.source is not FieldSource.BARCODE:
        guard = GTIN_NOT_FROM_BARCODE
    elif (
        number is None
        or number.value != product.nafdac_number
        or number.confidence < MIN_NUMBER_CONFIDENCE
    ):
        guard = NUMBER_NOT_READ_SURELY
    outcome = Outcome.MISMATCHED if other else Outcome.MATCHED
    return FieldCheck(outcome, read.value, register, read.confidence, guard)


def check_register(
    product: ProductFacts,
    extraction: Extraction,
    batch: BatchFacts | None = None,
    gtin_owner: GtinOwner | None = None,
) -> RegisterCheck:
    return RegisterCheck(
        product_name=_check_name(product, extraction),
        active_ingredients=_check_ingredients(product, extraction),
        strength=_check_strength(product, extraction),
        dosage_form=_check_dosage_form(product, extraction),
        holder=_check_holder(product, extraction),
        registration_window=_check_window(product, extraction, batch),
        barcode=_check_barcode(product, extraction, gtin_owner),
    )


def register_signals(check: RegisterCheck) -> dict[SignalName, Signal]:
    """The check as scoring signals: 1 mismatched, 0 matched, unavailable when unread, not in
    the register or mismatched below the confidence guard (an unread field lowers confidence
    and never raises risk)."""

    def signal(item: FieldCheck) -> Signal:
        if item.outcome is Outcome.MISMATCHED and item.counted:
            return Signal(1.0)
        if item.outcome is Outcome.MATCHED:
            return Signal(0.0)
        return UNAVAILABLE

    return {
        SignalName.INGREDIENT_MISMATCH: signal(check.active_ingredients),
        SignalName.STRENGTH_MISMATCH: signal(check.strength),
        SignalName.DOSAGE_FORM_MISMATCH: signal(check.dosage_form),
        SignalName.HOLDER_MISMATCH: signal(check.holder),
        SignalName.REGISTERED_AFTER_OR_LAPSED_AT_MANUFACTURE: signal(check.registration_window),
    }


def barcode_signals(check: RegisterCheck) -> dict[SignalName, Signal]:
    """The cloned-barcode check as two signals (scored only with its own switch): another
    holder's product (barcode_product_mismatch) or the same holder's other product, likely a
    re-registration (barcode_same_holder_mismatch). Both unavailable when there is no barcode,
    its GTIN is in no registration, or a guard failed."""
    item = check.barcode
    names = (SignalName.BARCODE_PRODUCT_MISMATCH, SignalName.BARCODE_SAME_HOLDER_MISMATCH)
    if item.outcome not in {Outcome.MATCHED, Outcome.MISMATCHED} or not item.counted:
        return dict.fromkeys(names, UNAVAILABLE)
    if item.outcome is Outcome.MATCHED:
        return dict.fromkeys(names, Signal(0.0))
    register = item.register if isinstance(item.register, dict) else {}
    same = bool(register.get("same_holder"))
    return {
        SignalName.BARCODE_PRODUCT_MISMATCH: Signal(0.0 if same else 1.0),
        SignalName.BARCODE_SAME_HOLDER_MISMATCH: Signal(1.0 if same else 0.0),
    }


# The signal each field of the check feeds (the barcode's depends on the holder).
_FIELD_SIGNALS = {
    "active_ingredients": SignalName.INGREDIENT_MISMATCH,
    "strength": SignalName.STRENGTH_MISMATCH,
    "dosage_form": SignalName.DOSAGE_FORM_MISMATCH,
    "holder": SignalName.HOLDER_MISMATCH,
    "registration_window": SignalName.REGISTERED_AFTER_OR_LAPSED_AT_MANUFACTURE,
}


def held_back(check: RegisterCheck) -> list[tuple[SignalName, str]]:
    """Mismatches a guard kept out of the score: (the signal they would have raised, why). For
    the metric medlens_scan_signals_held_back_total, which shows how often the 0.85 bars bite
    (to recalibrate them after the real-photo OCR benchmark)."""
    found: list[tuple[SignalName, str]] = []
    for name in check.__slots__:
        item: FieldCheck = getattr(check, name)
        if item.outcome is not Outcome.MISMATCHED or item.not_counted is None:
            continue
        if name == "barcode":
            register = item.register if isinstance(item.register, dict) else {}
            same = bool(register.get("same_holder"))
            signal = (
                SignalName.BARCODE_SAME_HOLDER_MISMATCH
                if same
                else SignalName.BARCODE_PRODUCT_MISMATCH
            )
        elif name in _FIELD_SIGNALS:
            signal = _FIELD_SIGNALS[name]
        else:
            continue
        found.append((signal, item.not_counted))
    return found

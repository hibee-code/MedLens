"""Reading pack fields from barcodes and OCR text. Pure functions, no I/O.

Every value comes from untrusted input (a photo), so it passes the same normalisers as user input
(``core/identifiers.py``) and carries the source and a confidence in 0..1.
"""

import calendar
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from enum import StrEnum

from medlens.core.identifiers import (
    normalize_batch_number,
    normalize_nafdac_number,
    valid_gtin,
)
from medlens.modules.products.register import KNOWN_INGREDIENTS, Strength


class FieldSource(StrEnum):
    BARCODE = "barcode"
    OCR = "ocr"
    LLM = "llm"
    CLAIMED = "claimed"  # typed by the person scanning


@dataclass(frozen=True, slots=True)
class ExtractedField[T]:
    value: T
    source: FieldSource
    confidence: float


# Identity fields decide which product a pack is; register fields are compared with that
# product's registration (the register check). Only identity fields ever trigger the vision LLM.
IDENTITY_FIELDS = ("nafdac_number", "product_name", "batch_number", "expiry_date", "gtin")
REGISTER_FIELDS = ("active_ingredients", "strengths", "dosage_form", "companies", "manufactured_on")


@dataclass(frozen=True, slots=True)
class Extraction:
    nafdac_number: ExtractedField[str] | None = None
    product_name: ExtractedField[str] | None = None
    batch_number: ExtractedField[str] | None = None
    expiry_date: ExtractedField[date] | None = None
    gtin: ExtractedField[str] | None = None
    # Register fields (pack_details.py): normalised ingredient names, strengths in the pack's
    # order, the dosage-form family, normalised company names, the manufacture date.
    active_ingredients: ExtractedField[tuple[str, ...]] | None = None
    strengths: ExtractedField[tuple[Strength, ...]] | None = None
    dosage_form: ExtractedField[str] | None = None
    companies: ExtractedField[tuple[str, ...]] | None = None
    manufactured_on: ExtractedField[date] | None = None
    # Full OCR text, for name consistency checks. Untrusted: never shown or sent anywhere.
    text: str = field(default="", repr=False)
    text_confidence: float = 0.0  # the OCR engine's confidence in ``text``
    # The OCR engine's confidence per line of ``text``, when measured (OcrResult.lines), and
    # per word of each line (OcrResult.words).
    line_confidences: tuple[tuple[str, float], ...] = field(default=(), repr=False)
    word_confidences: tuple[tuple[str, tuple[float, ...]], ...] = field(default=(), repr=False)

    def merge(self, other: "Extraction") -> "Extraction":
        """Fields of ``other`` fill gaps or replace less confident values."""
        merged = self
        for name in IDENTITY_FIELDS + REGISTER_FIELDS:
            mine, theirs = getattr(self, name), getattr(other, name)
            if theirs is not None and (mine is None or theirs.confidence > mine.confidence):
                merged = replace(merged, **{name: theirs})
        text = " ".join(part for part in (self.text, other.text) if part)
        confidence = max(self.text_confidence, other.text_confidence)
        return replace(
            merged,
            text=text,
            text_confidence=confidence,
            line_confidences=self.line_confidences + other.line_confidences,
            word_confidences=self.word_confidences + other.word_confidences,
        )

    def missing(self, threshold: float) -> list[str]:
        """Required fields absent or below ``threshold`` confidence."""
        fields = {
            "nafdac_number": self.nafdac_number,
            "batch_number": self.batch_number,
            "expiry_date": self.expiry_date,
        }
        return [
            name for name, value in fields.items() if value is None or value.confidence < threshold
        ]

    def as_json(self) -> dict[str, object]:
        """For ``scans.extracted``: values with source and confidence, never the raw text."""
        result: dict[str, object] = {}
        for name in IDENTITY_FIELDS + REGISTER_FIELDS:
            item = getattr(self, name)
            if item is not None:
                result[name] = {
                    "value": _json_value(item.value),
                    "source": item.source.value,
                    "confidence": round(item.confidence, 3),
                }
        return result


def _json_value(value: object) -> object:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Strength):
        return value.as_json()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


# --------------------------------------------------------------------------- GS1
_GS = "\x1d"  # FNC1 as transmitted by scanners
# Application identifiers with a predefined length (value only); everything else listed in
# _VARIABLE ends at a group separator or the end of the data.
_FIXED = {"00": 18, "01": 14, "02": 14, "11": 6, "12": 6, "13": 6, "15": 6, "16": 6, "17": 6}
_VARIABLE = {"10": 20, "21": 20, "22": 20, "240": 30, "241": 30, "30": 8, "37": 8}
_BRACKETED = re.compile(r"\((\d{2,4})\)([^()]+)")


def parse_gs1(data: str) -> dict[str, str]:
    """Application identifier -> value for a GS1 element string, either as scanned
    (``]d2`` prefix, group separators) or human-readable (``(01)…(17)…(10)…``). Parsing stops
    at the first unknown identifier; the result may be partial."""
    data = data.strip()
    if data.startswith("("):
        return {ai: value.strip() for ai, value in _BRACKETED.findall(data)}
    for prefix in ("]d2", "]C1", "]e0", "]Q3"):  # symbology identifiers
        data = data.removeprefix(prefix)
    result: dict[str, str] = {}
    position = 0
    while position < len(data):
        if data[position] == _GS:
            position += 1
            continue
        ai = next(
            (
                data[position : position + n]
                for n in (2, 3)
                if data[position : position + n] in _FIXED | _VARIABLE
            ),
            None,
        )
        if ai is None:
            break
        position += len(ai)
        if ai in _FIXED:
            result[ai] = data[position : position + _FIXED[ai]]
            position += _FIXED[ai]
        else:
            end = data.find(_GS, position)
            end = len(data) if end == -1 else end
            result[ai] = data[position : min(end, position + _VARIABLE[ai])]
            position = end
    return result


def gs1_date(value: str, *, month_end: bool = True) -> date | None:
    """YYMMDD; day 00 means the month's last day (``month_end``, expiry) or first day
    (production dates). Years 51-99 are 19xx (simplified GS1 rule)."""
    if not (len(value) == 6 and value.isdigit()):
        return None
    year = int(value[:2])
    year += 1900 if year >= 51 else 2000
    month, day = int(value[2:4]), int(value[4:])
    if not 1 <= month <= 12:
        return None
    last = calendar.monthrange(year, month)[1]
    if day == 0:
        day = last if month_end else 1
    return date(year, month, day) if day <= last else None


def from_gs1(data: str, confidence: float = 0.98) -> Extraction:
    """Fields from one barcode. A GTIN with a wrong check digit discards the whole read."""
    elements = parse_gs1(data)
    gtin = elements.get("01")
    if gtin is not None and not valid_gtin(gtin):
        return Extraction()
    batch = _batch(elements.get("10"))
    expiry = gs1_date(elements.get("17", ""))
    made = gs1_date(elements.get("11", ""), month_end=False)
    return Extraction(
        gtin=ExtractedField(gtin, FieldSource.BARCODE, confidence) if gtin else None,
        batch_number=ExtractedField(batch, FieldSource.BARCODE, confidence) if batch else None,
        expiry_date=ExtractedField(expiry, FieldSource.BARCODE, confidence) if expiry else None,
        manufactured_on=ExtractedField(made, FieldSource.BARCODE, confidence) if made else None,
    )


# --------------------------------------------------------------------------- OCR text
_SEP = r"[\s:.#\-]*"
_NAFDAC_LABELLED = re.compile(
    rf"(?:NAFDAC|REG(?:ISTRATION)?){_SEP}(?:REG(?:ISTRATION)?)?{_SEP}(?:NO|NUMBER|N°)?{_SEP}"
    r"([A-Z0-9]{1,3}\s?[-\u2010-\u2015\u2212_./ ]\s?[0-9OIS]{4,6}L?)\b",
    re.IGNORECASE,
)
_NAFDAC_BARE = re.compile(r"\b([A-Z]?\d{1,2}\s?[-\u2010-\u2015\u2212]\s?\d{4,6}L?)\b")
_BATCH = re.compile(
    rf"\b(?:BATCH|LOT|B{_SEP}NO|BN){_SEP}(?:NO|NUMBER)?{_SEP}([A-Z0-9][A-Z0-9\-/]{{1,24}})",
    re.IGNORECASE,
)
_EXPIRY = re.compile(
    rf"\b(?:EXP(?:IRY)?(?:{_SEP}DATE)?|E{_SEP}D|USE{_SEP}BY|BEST{_SEP}BEFORE){_SEP}"
    r"([0-3]?\d[/\-.][01]?\d[/\-.]\d{2,4}|[01]?\d[/\-.]\d{2,4}|[A-Z]{3,9}[\s\-./]*\d{2,4}|\d{4}[/\-.][01]?\d)",
    re.IGNORECASE,
)
_MANUFACTURED = re.compile(
    rf"\b(?:MFG|MFD|MANUF(?:ACTURED|ACTURING)?|DATE{_SEP}OF{_SEP}MANUFACTURE|PROD(?:UCTION)?"
    rf"(?:{_SEP}DATE)?|DOM)(?:{_SEP}(?:DATE|ON))?{_SEP}"
    r"([0-3]?\d[/\-.][01]?\d[/\-.]\d{2,4}|[01]?\d[/\-.]\d{2,4}|[A-Z]{3,9}[\s\-./]*\d{2,4}|\d{4}[/\-.][01]?\d)",
    re.IGNORECASE,
)
_MONTHS = {name.upper(): index for index, name in enumerate(calendar.month_abbr) if name}
_MONTHS.update({name.upper(): index for index, name in enumerate(calendar.month_name) if name})
_MONTHS["SEPT"] = 9


def from_ocr(
    text: str,
    ocr_confidence: float,
    lines: tuple[tuple[str, float], ...] = (),
    words: tuple[tuple[str, tuple[float, ...]], ...] = (),
) -> Extraction:
    """Fields found in OCR text. Confidence is the OCR engine's, lowered for unlabelled matches
    and for numbers that needed OCR corrections. The NAFDAC number takes the lowest confidence
    of its own words (``words``), so a clearly read label cannot lift a doubtful number; the
    manufacture date takes its line's (``lines``). Unmeasured: the line's, else the page's. The
    0.85 guards compare these readings (register_check.py)."""
    upper = text.upper()
    return Extraction(
        nafdac_number=_ocr_nafdac(text, ocr_confidence, dict(lines), dict(words)),
        batch_number=_ocr_batch(upper, ocr_confidence),
        expiry_date=_ocr_expiry(upper, ocr_confidence),
        manufactured_on=_ocr_manufactured(text, ocr_confidence, dict(lines)),
        text=text,
        text_confidence=_clamp(ocr_confidence),
        line_confidences=lines,
        word_confidences=words,
    )


def lines_confidence(lines: Iterable[str], measured: Mapping[str, float], default: float) -> float:
    """The lowest measured confidence of the lines a value was read from; a line the engine did
    not measure (or no line) counts as ``default``, the whole text's confidence."""
    values = [measured.get(line.strip(), default) for line in lines]
    return _clamp(min(values) if values else default)


def _ocr_nafdac(
    text: str,
    confidence: float,
    measured: Mapping[str, float] | None = None,
    words: Mapping[str, tuple[float, ...]] | None = None,
) -> ExtractedField[str] | None:
    """The first labelled (else bare) NAFDAC number. Its confidence is the lowest of its own
    words' when measured, else its line's, else the whole text's; lowered for an unlabelled
    number and for one that needed OCR corrections."""
    upper = text.upper()
    for pattern, factor in ((_NAFDAC_LABELLED, 1.0), (_NAFDAC_BARE, 0.7)):
        for match in pattern.finditer(upper):
            raw = match.group(1)
            try:
                number = normalize_nafdac_number(raw)
            except ValueError:
                continue
            corrected = number.replace("-", "") != re.sub(r"[^A-Z0-9]", "", raw.upper())
            read = _words_confidence(text, match.start(1), match.end(1), words or {})
            if read is None:
                read = lines_confidence(_lines_of(text, match), measured or {}, confidence)
            return ExtractedField(
                number, FieldSource.OCR, _clamp(read * factor * (0.85 if corrected else 1.0))
            )
    return None


def _words_confidence(
    text: str, start: int, end: int, words: Mapping[str, tuple[float, ...]]
) -> float | None:
    """The lowest confidence of the words spanning text[start:end], from the OCR engine's word
    confidences of that line; None when the line or its words are not measured."""
    first = text.count("\n", 0, start)
    if first != text.count("\n", 0, max(start, end - 1)):
        return None  # spans lines
    line_start = text.rfind("\n", 0, start) + 1
    line = text.splitlines()[first]
    measured = words.get(line)
    tokens = list(re.finditer(r"\S+", line))
    if measured is None or len(measured) != len(tokens):
        return None
    begin, finish = start - line_start, end - line_start
    spanned = [
        value
        for token, value in zip(tokens, measured, strict=True)
        if token.end() > begin and token.start() < finish
    ]
    return min(spanned) if spanned else None


def _lines_of(text: str, match: re.Match[str]) -> list[str]:
    """The lines of ``text`` a match (in ``text`` or its upper-case copy) spans."""
    first = text.count("\n", 0, match.start())
    last = text.count("\n", 0, match.end())
    return text.splitlines()[first : last + 1]


def find_nafdac_numbers(text: str, *, bare: bool = False) -> list[str]:
    """Every labelled NAFDAC number in free text, normalised, in order. ``bare`` also accepts
    unlabelled numbers (questions: "any alert for 04-8655?"); documents need the label."""
    patterns = (_NAFDAC_LABELLED, _NAFDAC_BARE) if bare else (_NAFDAC_LABELLED,)
    found: dict[str, None] = {}
    for pattern in patterns:
        for match in pattern.finditer(text.upper()):
            try:
                found[normalize_nafdac_number(match.group(1))] = None
            except ValueError:
                continue
    return list(found)


def find_batch_numbers(text: str) -> list[str]:
    """Every labelled batch or lot number in free text, normalised, in order. Prose ("the
    batch the holder made") also follows the label, so values without a digit are skipped."""
    found: dict[str, None] = {}
    for match in _BATCH.finditer(text.upper()):
        batch = _batch(match.group(1))
        if batch and any(char.isdigit() for char in batch):
            found[batch] = None
    return list(found)


def _ocr_batch(text: str, confidence: float) -> ExtractedField[str] | None:
    match = _BATCH.search(text)
    batch = _batch(match.group(1)) if match else None
    return ExtractedField(batch, FieldSource.OCR, _clamp(confidence)) if batch else None


def _ocr_manufactured(
    text: str, confidence: float, measured: Mapping[str, float]
) -> ExtractedField[date] | None:
    match = _MANUFACTURED.search(text.upper())
    made = parse_manufacture_date(match.group(1)) if match else None
    if match is None or made is None:
        return None
    read = lines_confidence(_lines_of(text, match), measured, confidence)
    return ExtractedField(made, FieldSource.OCR, read)


def parse_manufacture_date(raw: str) -> date | None:
    """Manufacture dates as printed. A month-only date means the first day of that month (the
    earliest it can have been made: the registration-window check gives the pack the benefit)."""
    parsed = parse_expiry(raw)
    parts = [part for part in re.split(r"[\s/\-.]+", raw.strip()) if part]
    if parsed is not None and len(parts) == 2:
        return parsed.replace(day=1)
    return parsed


def _ocr_expiry(text: str, confidence: float) -> ExtractedField[date] | None:
    match = _EXPIRY.search(text)
    expiry = parse_expiry(match.group(1)) if match else None
    return ExtractedField(expiry, FieldSource.OCR, _clamp(confidence)) if expiry else None


def parse_expiry(raw: str) -> date | None:
    """Expiry dates as printed on packs. Month-only dates mean the last day of that month."""
    value = raw.strip().upper()
    parts = [part for part in re.split(r"[\s/\-.]+", value) if part]
    try:
        if len(parts) == 3 and all(part.isdigit() for part in parts):  # DD/MM/YYYY
            day, month, year = int(parts[0]), int(parts[1]), _year(parts[2])
            return date(year, month, day)
        if len(parts) == 2:
            first, second = parts
            if first.isdigit() and len(first) == 4:  # YYYY-MM
                return _month_end(int(first), int(second))
            if first.isdigit():  # MM/YYYY or MM/YY
                return _month_end(_year(second), int(first))
            if first in _MONTHS:  # MAR 2027
                return _month_end(_year(second), _MONTHS[first])
    except ValueError:
        return None
    return None


def _year(value: str) -> int:
    if not value.isdigit() or len(value) not in {2, 4}:
        raise ValueError(value)
    year = int(value)
    return 2000 + year if year < 100 else year


def _month_end(year: int, month: int) -> date:
    if not 1 <= month <= 12:
        raise ValueError(month)
    return date(year, month, calendar.monthrange(year, month)[1])


def _batch(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return normalize_batch_number(raw)
    except ValueError:
        return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


# --------------------------------------------------------------------------- names
_NAME_TOKEN = re.compile(r"[A-Z0-9]+")
_GENERIC_WORDS = frozenset(
    {"TABLET", "TABLETS", "CAPLET", "CAPLETS", "CAPSULE", "CAPSULES", "SYRUP", "SUSPENSION",
     "INJECTION", "CREAM", "GEL", "MG", "ML", "G", "MCG", "IU", "FOR", "AND", "THE", "ORAL",
     "POWDER", "DROPS", "SOLUTION"}
)  # fmt: skip


# Ingredient names are not brand words: "Emzor Paracetamol Syrup" is told apart by EMZOR, and
# PARACETAMOL appears on every paracetamol pack (finding only it is no evidence of the brand).
_INGREDIENT_WORDS = frozenset(
    name.upper() for name in KNOWN_INGREDIENTS if " " not in name and "-" not in name
)


def name_tokens(name: str) -> set[str]:
    return {
        token
        for token in _NAME_TOKEN.findall(name.upper())
        if len(token) >= 3
        and token not in _GENERIC_WORDS
        and token not in _INGREDIENT_WORDS
        and not token.isdigit()
    }


def brand_presence(brand_name: str, generic_name: str, text: str) -> float | None:
    """Share of the brand's own words found in the pack text: its distinctive words without the
    product's generic name (nor dictionary ingredients). None when the brand has no word of its
    own ("Paracetamol Tablets BP"): a generic name alone never identifies a brand."""
    generic = set(_NAME_TOKEN.findall(generic_name.upper()))
    wanted = name_tokens(brand_name) - generic
    if not wanted or not text.strip():
        return None
    found = set(_NAME_TOKEN.findall(text.upper()))
    return len(wanted & found) / len(wanted)


def name_presence(registered_name: str, text: str) -> float | None:
    """Share of the registered name's distinctive words found in the pack text, or None when
    the name has no distinctive words or there is no text to compare."""
    wanted = name_tokens(registered_name)
    if not wanted or not text.strip():
        return None
    found = set(_NAME_TOKEN.findall(text.upper()))
    return len(wanted & found) / len(wanted)

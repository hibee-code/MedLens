"""Reading pack fields from barcodes and OCR text. Pure functions, no I/O.

Every value comes from untrusted input (a photo), so it passes the same normalisers as user input
(``core/identifiers.py``) and carries the source and a confidence in 0..1.
"""

import calendar
import re
from dataclasses import dataclass, field, replace
from datetime import date
from enum import StrEnum

from medlens.core.identifiers import normalize_batch_number, normalize_nafdac_number


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


@dataclass(frozen=True, slots=True)
class Extraction:
    nafdac_number: ExtractedField[str] | None = None
    product_name: ExtractedField[str] | None = None
    batch_number: ExtractedField[str] | None = None
    expiry_date: ExtractedField[date] | None = None
    gtin: ExtractedField[str] | None = None
    # Full OCR text, for name consistency checks. Untrusted: never shown or sent anywhere.
    text: str = field(default="", repr=False)

    def merge(self, other: "Extraction") -> "Extraction":
        """Fields of ``other`` fill gaps or replace less confident values."""
        merged = self
        for name in ("nafdac_number", "product_name", "batch_number", "expiry_date", "gtin"):
            mine, theirs = getattr(self, name), getattr(other, name)
            if theirs is not None and (mine is None or theirs.confidence > mine.confidence):
                merged = replace(merged, **{name: theirs})
        text = " ".join(part for part in (self.text, other.text) if part)
        return replace(merged, text=text)

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
        for name in ("nafdac_number", "product_name", "batch_number", "expiry_date", "gtin"):
            item = getattr(self, name)
            if item is not None:
                value = item.value.isoformat() if isinstance(item.value, date) else item.value
                result[name] = {
                    "value": value,
                    "source": item.source.value,
                    "confidence": round(item.confidence, 3),
                }
        return result


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


def valid_gtin(gtin: str) -> bool:
    """GS1 mod-10 check digit."""
    if not (gtin.isdigit() and len(gtin) in {8, 12, 13, 14}):
        return False
    digits = [int(char) for char in gtin]
    total = sum(
        digit * (3 if index % 2 == 0 else 1) for index, digit in enumerate(reversed(digits[:-1]))
    )
    return (10 - total % 10) % 10 == digits[-1]


def gs1_date(value: str) -> date | None:
    """YYMMDD; day 00 means the month's last day. Years 51-99 are 19xx (simplified GS1 rule)."""
    if not (len(value) == 6 and value.isdigit()):
        return None
    year = int(value[:2])
    year += 1900 if year >= 51 else 2000
    month, day = int(value[2:4]), int(value[4:])
    if not 1 <= month <= 12:
        return None
    last = calendar.monthrange(year, month)[1]
    if day == 0:
        day = last
    return date(year, month, day) if day <= last else None


def from_gs1(data: str, confidence: float = 0.98) -> Extraction:
    """Fields from one barcode. A GTIN with a wrong check digit discards the whole read."""
    elements = parse_gs1(data)
    gtin = elements.get("01")
    if gtin is not None and not valid_gtin(gtin):
        return Extraction()
    batch = _batch(elements.get("10"))
    expiry = gs1_date(elements.get("17", ""))
    return Extraction(
        gtin=ExtractedField(gtin, FieldSource.BARCODE, confidence) if gtin else None,
        batch_number=ExtractedField(batch, FieldSource.BARCODE, confidence) if batch else None,
        expiry_date=ExtractedField(expiry, FieldSource.BARCODE, confidence) if expiry else None,
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
_MONTHS = {name.upper(): index for index, name in enumerate(calendar.month_abbr) if name}
_MONTHS.update({name.upper(): index for index, name in enumerate(calendar.month_name) if name})
_MONTHS["SEPT"] = 9


def from_ocr(text: str, ocr_confidence: float) -> Extraction:
    """Fields found in OCR text. Confidence is the OCR engine's, lowered for unlabelled matches
    and for numbers that needed OCR corrections."""
    upper = text.upper()
    return Extraction(
        nafdac_number=_ocr_nafdac(upper, ocr_confidence),
        batch_number=_ocr_batch(upper, ocr_confidence),
        expiry_date=_ocr_expiry(upper, ocr_confidence),
        text=text,
    )


def _ocr_nafdac(text: str, confidence: float) -> ExtractedField[str] | None:
    for pattern, factor in ((_NAFDAC_LABELLED, 1.0), (_NAFDAC_BARE, 0.7)):
        for match in pattern.finditer(text):
            raw = match.group(1)
            try:
                number = normalize_nafdac_number(raw)
            except ValueError:
                continue
            corrected = number.replace("-", "") != re.sub(r"[^A-Z0-9]", "", raw.upper())
            return ExtractedField(
                number, FieldSource.OCR, _clamp(confidence * factor * (0.85 if corrected else 1.0))
            )
    return None


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


def name_tokens(name: str) -> set[str]:
    return {
        token
        for token in _NAME_TOKEN.findall(name.upper())
        if len(token) >= 3 and token not in _GENERIC_WORDS and not token.isdigit()
    }


def name_presence(registered_name: str, text: str) -> float | None:
    """Share of the registered name's distinctive words found in the pack text, or None when
    the name has no distinctive words or there is no text to compare."""
    wanted = name_tokens(registered_name)
    if not wanted or not text.strip():
        return None
    found = set(_NAME_TOKEN.findall(text.upper()))
    return len(wanted & found) / len(wanted)

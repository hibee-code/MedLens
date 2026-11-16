"""Reading the register fields printed on a pack (active ingredients, strengths, dosage form,
company names) from OCR text, with regular expressions and a dictionary. Pure functions, no
I/O; the text is untrusted input.

Built to rather leave a field unread than misread it, because an unread field only lowers
confidence while a misread one could raise risk (CLAUDE.md decision log, register check):

* ingredients are found only by dictionary (KNOWN_INGREDIENTS plus the names the caller adds,
  such as the registration's own), never guessed from unknown words;
* strengths are taken only from lines that name an ingredient and do not give dosing
  instructions ("take 2 tablets", "max 4000 mg a day"); "Each 5 ml contains" turns the
  amounts after it into concentrations;
* a dosage form is reported only when one family is named clearly more often than any other;
* company names only after a label ("Manufactured by", "Marketed by", "Registration holder").
"""

import re
from collections.abc import Callable, Iterable

from medlens.modules.products.register import (
    KNOWN_INGREDIENTS,
    Strength,
    canonical_text,
    dosage_form_family,
    dosage_form_mentions,
    normalize_company,
    normalize_ingredient,
    parse_strengths,
)
from medlens.modules.verification.extraction import (
    ExtractedField,
    Extraction,
    FieldSource,
    lines_confidence,
)

_DOSING = re.compile(
    r"\b(?:dose|dosage|daily|every|hours?|hourly|maximum|max|exceed|take|adults?|children|"
    r"years?|times|weight|kg)\b",
    re.IGNORECASE,
)
_STRENGTH_TOKEN = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|iu|i\.u\.|%)"
    r"(?:\s*/\s*(?:\d+(?:[.,]\d+)?\s*)?(?:ml|l|dose|metered-dose)\b)?"
    r"|\d+(?:[.,]\d+)?(?:\s*(?:mg|mcg|g))?(?:\s*/\s*\d+(?:[.,]\d+)?\s*(?:mg|mcg|g)?)+",
    re.IGNORECASE,
)
_EACH_VOLUME = re.compile(r"\b(?:each|per|every)\s+(\d+(?:[.,]\d+)?)\s*ml\b", re.IGNORECASE)
_COMPANY_LABEL = re.compile(
    r"\b(?:manufactured|mfd|made|produced|packed|marketed|distributed|imported)\s*(?:by|for)"
    r"|\b(?:marketing\s+authori[sz]ation|registration|licen[cs]e|product\s+licen[cs]e)\s+holder"
    r"|\b(?:manufacturer|mah)\b",
    re.IGNORECASE,
)
# Address words that end a company name written on one line with its address.
_ADDRESS = re.compile(
    r",|\b(?:km|plot|road|rd|street|st|avenue|ave|estate|industrial|lagos|abuja|nigeria\s+limited"
    r"|p\.?\s*o\.?\s*box)\b",
    re.IGNORECASE,
)


def _mentions(name: str, canonical: str) -> re.Match[str] | None:
    return re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", canonical)


def find_ingredients(text: str, names: Iterable[str] = KNOWN_INGREDIENTS) -> list[str]:
    """Dictionary ingredients named in ``text``, in order of first mention."""
    canonical = canonical_text(text)
    positions = {name: match.start() for name in names if (match := _mentions(name, canonical))}
    return sorted(positions, key=positions.__getitem__)


def find_strengths(text: str, names: Iterable[str]) -> list[Strength] | None:
    """Strengths on lines naming one of ``names`` (and not dosing instructions), in order.
    None when no such line has a strength."""
    wanted = list(names)
    volume: str | None = None
    found: list[Strength] = []
    for line in text.splitlines():
        each = _EACH_VOLUME.search(line)
        if each:
            volume = each.group(1)
        canonical = canonical_text(line)
        named = any(_mentions(name, canonical) for name in wanted)
        if not named or _DOSING.search(line):
            continue
        for token in _STRENGTH_TOKEN.findall(line):
            parsed = parse_strengths(token)
            if not parsed:
                continue
            for strength in parsed:
                if strength.unit == "mL":
                    continue  # a volume (pack size), not a strength
                found.append(_per_volume(strength, token, volume))
    return found or None


def _per_volume(strength: Strength, token: str, volume: str | None) -> Strength:
    """After "Each 5 ml contains", an amount is a concentration (125 mg -> 125 mg/5 mL)."""
    if volume and strength.per_unit is None and strength.unit != "%":
        concentration = parse_strengths(f"{token} / {volume} ml")
        if concentration and len(concentration) == 1:
            return concentration[0]
    return strength


def find_dosage_form(text: str) -> str | None:
    """The family named most, when it is named at least twice as often as any other."""
    ranked = dosage_form_mentions(text).most_common(2)
    if not ranked:
        return None
    if len(ranked) == 2 and ranked[0][1] < 2 * ranked[1][1]:
        return None  # ambiguous
    return ranked[0][0]


def find_companies(text: str) -> list[str]:
    """Normalised company names after a manufacturer, marketer or holder label, in order."""
    found: dict[str, None] = {}
    for line in text.splitlines():
        label = _COMPANY_LABEL.search(line)
        if not label:
            continue
        rest = line[label.end() :].lstrip(" :.-\t")
        name = _ADDRESS.split(rest, maxsplit=1)[0]
        normalised = normalize_company(name)
        if len(normalised) >= 3 and any(char.isalpha() for char in normalised):
            found[normalised] = None
    return list(found)


def read_pack_details(
    text: str,
    confidence: float,
    extra_names: Iterable[str] = (),
    lines: tuple[tuple[str, float], ...] = (),
) -> Extraction:
    """The register fields found in OCR ``text`` (an Extraction with only those fields set).
    ``extra_names``: ingredient names to look for besides the dictionary (the registration's).

    Each field's confidence is the OCR engine's own for the lines it was read from (``lines``,
    the lowest of them), else ``confidence``, the whole text's: a mismatch counts only when read
    with at least 0.85 (register_check.MIN_MISMATCH_CONFIDENCE), so it must be measured, not
    guessed."""
    if not text.strip():
        return Extraction()
    names = KNOWN_INGREDIENTS | frozenset(extra_names)
    ingredients = find_ingredients(text, names)
    strengths = find_strengths(text, ingredients) if ingredients else None
    form = find_dosage_form(text)
    companies = find_companies(text)
    measured = dict(lines)
    rows = text.splitlines()

    def read_from(found: Callable[[str], bool]) -> float:
        return lines_confidence([row for row in rows if found(row)], measured, confidence)

    return Extraction(
        active_ingredients=(
            ExtractedField(
                tuple(ingredients),
                FieldSource.OCR,
                read_from(lambda row: bool(find_ingredients(row, ingredients))),
            )
            if ingredients
            else None
        ),
        strengths=(
            ExtractedField(
                tuple(strengths),
                FieldSource.OCR,
                read_from(
                    lambda row: (
                        find_strengths(row, ingredients) is not None
                        or _EACH_VOLUME.search(row) is not None
                    )
                ),
            )
            if strengths
            else None
        ),
        dosage_form=(
            ExtractedField(
                form, FieldSource.OCR, read_from(lambda row: form in dosage_form_mentions(row))
            )
            if form
            else None
        ),
        companies=(
            ExtractedField(
                tuple(companies),
                FieldSource.OCR,
                read_from(lambda row: bool(find_companies(row))),
            )
            if companies
            else None
        ),
    )


def details_from_readings(
    *,
    ingredients: tuple[str | None, float],
    strengths: tuple[str | None, float],
    dosage_form: tuple[str | None, float],
    manufacturer: tuple[str | None, float],
    cap: float,
) -> Extraction:
    """The register fields the vision LLM returned (text, confidence) next to the identity
    fields, through the same normalisers as OCR text; confidence capped at ``cap``."""

    def field[T](value: T | None, confidence: float) -> ExtractedField[T] | None:
        if value is None or confidence <= 0:
            return None
        return ExtractedField(value, FieldSource.LLM, min(confidence, cap))

    names_text, names_confidence = ingredients
    names: tuple[str, ...] | None = None
    if names_text:
        parts = re.split(r";|/|\+|,| and ", names_text)
        found = [
            name.strip()
            for part in parts
            for name in normalize_ingredient(part).split(";")
            if name.strip()
        ]
        names = tuple(dict.fromkeys(found)) or None
    strength_text, strength_confidence = strengths
    parsed = parse_strengths(strength_text) if strength_text else None
    form_text, form_confidence = dosage_form
    company_text, company_confidence = manufacturer
    company = normalize_company(_ADDRESS.split(company_text, maxsplit=1)[0]) if company_text else ""
    return Extraction(
        active_ingredients=field(names, names_confidence),
        strengths=field(tuple(parsed) if parsed else None, strength_confidence),
        dosage_form=field(dosage_form_family(form_text) if form_text else None, form_confidence),
        companies=field((company,) if len(company) >= 3 else None, company_confidence),
    )

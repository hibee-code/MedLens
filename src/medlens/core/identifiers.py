"""Normalisation of identifiers that arrive from people, OCR and partner systems.

Pure functions that raise ``ValueError`` on invalid input. ``core/schemas.py`` wraps them in
Pydantic annotated types; the OCR extraction stage (Phase 5) reuses them directly.
"""

import html
import re
import unicodedata
from enum import StrEnum

import nh3

# ---------------------------------------------------------------------------------------------
# NAFDAC registration numbers
#
# THE pattern (keep it here and only here). Canonical form: an optional letter, a one- or
# two-digit class, a hyphen, a four- to six-digit serial, and an optional L suffix:
#
#     A4-1234    04-1234    A4-100459    A11-0234    A7-4942L
#
# Normalisation before matching:
#   * upper-case, trim, and treat any dash, space, dot, slash or underscore between the two
#     parts as the hyphen ("A4 1234", "A4_1234", or a Unicode en/em dash);
#   * in the DIGIT positions only, map the usual OCR confusions O->0, I->1, S->5.
#     The leading letter slot is never remapped, except that a two-character prefix starting
#     with O, I or S is read as two digits ("O4-1234" -> "04-1234"): genuine letter prefixes
#     in those positions are not known to exist, while 0->O is the most common OCR error.
#
# The Greenbook import is the authority. If real numbers fall outside this pattern, widen it
# here and add the example to tests/unit/test_schemas.py. The 2026-11-03 snapshot added the L
# suffix (7 products). Still rejected there: 5 numbers written like "4/1/9086", whose mapping to
# this form is unknown, and empty or placeholder values ("NA", "Not available yet").
# ---------------------------------------------------------------------------------------------
NAFDAC_NUMBER_PATTERN = re.compile(r"^(?:[A-Z])?\d{1,2}-\d{4,6}L?$")
NAFDAC_NUMBER_MAX_LENGTH = 32

_SEPARATORS = re.compile(r"[\s\-\u2010-\u2015\u2212_./]+")  # incl. Unicode dashes
_OCR_DIGIT_CONFUSIONS = str.maketrans({"O": "0", "I": "1", "S": "5"})
_AMBIGUOUS_PREFIX_LETTERS = frozenset("OIS")


def normalize_nafdac_number(raw: str) -> str:
    value = _SEPARATORS.sub("-", raw.strip().upper()).strip("-")
    parts = value.split("-")
    if len(parts) != 2:
        raise ValueError("expected a NAFDAC number like A4-1234 or 04-1234")
    prefix, serial = parts
    letter, digits = _split_prefix(prefix)
    candidate = (
        f"{letter}{digits.translate(_OCR_DIGIT_CONFUSIONS)}-"
        f"{serial.translate(_OCR_DIGIT_CONFUSIONS)}"
    )
    if not NAFDAC_NUMBER_PATTERN.fullmatch(candidate):
        raise ValueError("expected a NAFDAC number like A4-1234 or 04-1234")
    return candidate


def _split_prefix(prefix: str) -> tuple[str, str]:
    """(letter, digits) of the part before the hyphen; see the rules above."""
    has_letter = len(prefix) == 3 or (
        len(prefix) == 2 and prefix[0].isalpha() and prefix[0] not in _AMBIGUOUS_PREFIX_LETTERS
    )
    return (prefix[0], prefix[1:]) if has_letter else ("", prefix)


# ---------------------------------------------------------------------------------------------
# Batch (lot) numbers
#
# Manufacturer-defined: letters, digits and - / . are all legitimate, so OCR confusions cannot
# be corrected safely and are left alone. We only upper-case, trim and drop inner whitespace.
# ---------------------------------------------------------------------------------------------
BATCH_NUMBER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9\-/.]{0,39}$")


def normalize_batch_number(raw: str) -> str:
    value = re.sub(r"\s+", "", raw.strip().upper())
    if not BATCH_NUMBER_PATTERN.fullmatch(value):
        raise ValueError(
            "expected 1-40 characters: letters, digits, '-', '/' or '.', starting with a "
            "letter or digit"
        )
    return value


# ---------------------------------------------------------------------------------------------
# Nigerian mobile numbers -> E.164 (+234XXXXXXXXXX)
#
# Accepts 0803..., 803..., 234803..., +234803..., +234 (0) 803... with spaces, dashes, dots or
# brackets. Only mobile ranges (070, 080, 081, 090, 091) are valid: phone numbers are used for
# SMS OTP, which landlines cannot receive.
# ---------------------------------------------------------------------------------------------
_NG_MOBILE_NATIONAL_NUMBER = re.compile(r"^(?:70|80|81|90|91)\d{8}$")
_PHONE_PUNCTUATION = re.compile(r"[\s\-.()]")


def normalize_ng_phone(raw: str) -> str:
    digits = _PHONE_PUNCTUATION.sub("", raw.strip())
    if digits.startswith("+"):
        digits = digits[1:]
        if not digits.startswith("234"):
            raise ValueError("only Nigerian (+234) numbers are accepted")
    elif digits.startswith("00"):
        digits = digits[2:]
    if not digits.isdigit():
        raise ValueError("expected a Nigerian mobile number, e.g. 0803 123 4567")
    if digits.startswith("234"):
        national = digits[3:].removeprefix("0")
    else:
        national = digits.removeprefix("0")
    if not _NG_MOBILE_NATIONAL_NUMBER.fullmatch(national):
        raise ValueError("expected a Nigerian mobile number, e.g. 0803 123 4567")
    return f"+234{national}"


# ---------------------------------------------------------------------------------------------
# Nigerian states (36 states + FCT), identified by ISO 3166-2:NG codes
# ---------------------------------------------------------------------------------------------
class NigerianState(StrEnum):
    """The 36 states and the FCT. Values are ISO 3166-2:NG codes.

    Input also accepts the state name in any case or spacing ("akwa ibom", "Cross-River"), the
    bare code ("LA") and the aliases "FCT" and "Abuja". Output is always the code.
    """

    ABIA = "NG-AB"
    ADAMAWA = "NG-AD"
    AKWA_IBOM = "NG-AK"
    ANAMBRA = "NG-AN"
    BAUCHI = "NG-BA"
    BAYELSA = "NG-BY"
    BENUE = "NG-BE"
    BORNO = "NG-BO"
    CROSS_RIVER = "NG-CR"
    DELTA = "NG-DE"
    EBONYI = "NG-EB"
    EDO = "NG-ED"
    EKITI = "NG-EK"
    ENUGU = "NG-EN"
    FCT = "NG-FC"
    GOMBE = "NG-GO"
    IMO = "NG-IM"
    JIGAWA = "NG-JI"
    KADUNA = "NG-KD"
    KANO = "NG-KN"
    KATSINA = "NG-KT"
    KEBBI = "NG-KE"
    KOGI = "NG-KO"
    KWARA = "NG-KW"
    LAGOS = "NG-LA"
    NASARAWA = "NG-NA"
    NIGER = "NG-NI"
    OGUN = "NG-OG"
    ONDO = "NG-ON"
    OSUN = "NG-OS"
    OYO = "NG-OY"
    PLATEAU = "NG-PL"
    RIVERS = "NG-RI"
    SOKOTO = "NG-SO"
    TARABA = "NG-TA"
    YOBE = "NG-YO"
    ZAMFARA = "NG-ZA"

    @property
    def display_name(self) -> str:
        if self is NigerianState.FCT:
            return "Federal Capital Territory"
        return self.name.replace("_", " ").title()

    @classmethod
    def _missing_(cls, value: object) -> "NigerianState | None":
        if not isinstance(value, str):
            return None
        return _STATE_LOOKUP.get(re.sub(r"[^a-z]", "", value.lower()))


_STATE_LOOKUP: dict[str, NigerianState] = {
    **{re.sub(r"[^a-z]", "", s.name.lower()): s for s in NigerianState},
    **{s.value.lower().replace("-", ""): s for s in NigerianState},  # "ng-la"
    **{s.value[3:].lower(): s for s in NigerianState},  # "la"
    "abuja": NigerianState.FCT,
    "federalcapitalterritory": NigerianState.FCT,
    "nassarawa": NigerianState.NASARAWA,  # common spelling variant
}


# ---------------------------------------------------------------------------------------------
# Free text
# ---------------------------------------------------------------------------------------------
def sanitize_text(raw: str) -> str:
    """Strip all HTML, returning plain text.

    nh3 removes tags (and the content of script/style) but entity-escapes what remains, which
    would double-escape when clients render the text. We unescape and repeat until stable, so
    entity-encoded markup ("&lt;script&gt;") cannot survive as real tags.
    """
    value = unicodedata.normalize("NFC", raw)
    for _ in range(5):
        cleaned = html.unescape(nh3.clean(value, tags=set()))
        if cleaned == value:
            break
        value = cleaned
    return value.strip()

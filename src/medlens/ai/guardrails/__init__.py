"""Guardrails around model calls (CLAUDE.md §5).

* Untrusted text (pack text, OCR output, documents, questions) is wrapped in tags marked
  untrusted, and cannot close them early.
* Personal data is scrubbed before text reaches a prompt.
* Answers are checked for banned claims, and citations for grounding in what was retrieved.
"""

import html
import re
import unicodedata
from collections.abc import Mapping

_CLOSING_TAG = re.compile(r"</\s*untrusted", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<![\w])(?:\+?234|0)[\s-]?[789][01]\d(?:[\s-]?\d){7}(?![\w])")
_BANNED_PATTERNS = {
    "safety_claim": (
        r"\b(?:is|are|it's|it is)\s+(?:completely\s+|totally\s+)?safe\b",
        r"\bsafe to (?:use|take|consume)\b",
    ),
    "authenticity_claim": (r"\b100\s*%\s*(?:genuine|authentic|original)\b",),
    "dosage_advice": (
        r"\b(?:take|give|use)\s+\d+(?:\.\d+)?\s*"
        r"(?:mg|ml|tablets?|capsules?|caplets?|times)\b",
        r"\b(?:recommended|correct|usual)\s+dos(?:e|age)\s+is\b",
    ),
    "diagnosis": (r"\byou (?:have|probably have|may have|are suffering from)\b",),
}
_BANNED = [
    (kind, re.compile(pattern, re.IGNORECASE))
    for kind, patterns in _BANNED_PATTERNS.items()
    for pattern in patterns
]


def wrap_untrusted(text: str, *, kind: str, attrs: Mapping[str, str] | None = None) -> str:
    """``<untrusted_{kind} k="v">text</untrusted_{kind}>``; the text cannot close the tag."""
    rendered = " ".join(
        f'{key}="{html.escape(str(value), quote=True)}"' for key, value in (attrs or {}).items()
    )
    opening = f"<untrusted_{kind}{' ' + rendered if rendered else ''}>"
    body = _CLOSING_TAG.sub("&lt;/untrusted", text)
    return f"{opening}\n{body}\n</untrusted_{kind}>"


def scrub_pii(text: str) -> str:
    """Replace e-mail addresses and Nigerian phone numbers before text enters a prompt."""
    return _PHONE.sub("[phone]", _EMAIL.sub("[email]", text))


def banned_claims(text: str) -> list[str]:
    """Kinds of claims the assistant must never make, found in ``text``."""
    return sorted({kind for kind, pattern in _BANNED if pattern.search(text)})


def normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = folded.translate({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'})
    return re.sub(r"\s+", " ", folded).strip()


def quote_in_text(quote: str, text: str) -> bool:
    """The quote appears in the text, ignoring case, spacing and typographic quotes."""
    needle = normalise(quote)
    return bool(needle) and needle in normalise(text)

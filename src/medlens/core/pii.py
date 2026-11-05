"""Personal data patterns, removed from text before it leaves the service: prompts (the AI
guardrails), traces and error reports (core.telemetry).

Patterns, not a classifier: e-mail addresses, Nigerian phone numbers in their usual forms,
JSON Web Tokens and bearer credentials. Free text is never logged or traced on purpose; this
catches what slips into an exception message or a URL.
"""

import re

_EMAIL = re.compile(r"\b[\w.+-]+@(?:[\w-]+\.)+[A-Za-z]{2,}\b")  # not "medlens@0.1.0"
_PHONE = re.compile(r"(?<![\w])(?:\+?234|0)[\s-]?[789][01]\d(?:[\s-]?\d){7}(?![\w])")
_JWT = re.compile(r"\beyJ[\w-]+\.[\w-]+\.[\w-]+")
_BEARER = re.compile(r"\b(Bearer|ApiKey)\s+[\w.~+/=-]+", re.IGNORECASE)


def scrub_pii(text: str) -> str:
    """Replace e-mail addresses and Nigerian phone numbers."""
    return _PHONE.sub("[phone]", _EMAIL.sub("[email]", text))


def scrub_secrets_and_pii(text: str) -> str:
    """``scrub_pii`` plus tokens and credentials (for traces and error reports)."""
    return scrub_pii(_BEARER.sub(r"\1 [redacted]", _JWT.sub("[token]", text)))

"""MFA recovery codes: ten one-time codes shown once, for when the authenticator is lost.

Each code is 10 characters from a 31-letter alphabet without look-alikes (0/o, 1/l/i), about
49 bits, printed as ``xxxxx-xxxxx``. High-entropy codes need no slow hash: they are stored as
SHA-256 digests (core.security.token_digest), and guesses are limited per sign-in attempt by
the same guard as authenticator codes.
"""

import secrets

from medlens.core.security import token_digest

ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"  # pragma: allowlist secret (an alphabet)
CODE_COUNT = 10
_LENGTH = 10


def new_codes(count: int = CODE_COUNT) -> list[str]:
    codes = ("".join(secrets.choice(ALPHABET) for _ in range(_LENGTH)) for _ in range(count))
    return [f"{code[:5]}-{code[5:]}" for code in codes]


def digest(code: str) -> str:
    """Case, spaces and hyphens do not matter when a code is typed back."""
    normalised = "".join(code.split()).replace("-", "").lower()
    return token_digest(normalised)

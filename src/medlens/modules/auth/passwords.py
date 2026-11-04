"""Password policy for staff accounts: at least 12 characters, at most 128, not a common
password and not built from the account's e-mail address."""

from functools import lru_cache
from importlib.resources import files

MIN_LENGTH = 12
MAX_LENGTH = 128  # bounds hashing cost


@lru_cache(maxsize=1)
def _common_passwords() -> frozenset[str]:
    text = files("medlens.modules.auth").joinpath("data/common_passwords.txt").read_text()
    return frozenset(line for line in text.splitlines() if line and not line.startswith("# "))


def password_problems(password: str, *, email: str | None = None) -> list[str]:
    """Human-readable reasons the password is not acceptable (empty when it is)."""
    problems: list[str] = []
    if len(password) < MIN_LENGTH:
        problems.append(f"must be at least {MIN_LENGTH} characters long")
    if len(password) > MAX_LENGTH:
        problems.append(f"must be at most {MAX_LENGTH} characters long")
    lowered = password.lower()
    if lowered in _common_passwords():
        problems.append("is too common; choose a less predictable password")
    if email:
        local_part = email.split("@", 1)[0].lower()
        if len(local_part) >= 4 and local_part in lowered:
            problems.append("must not contain your e-mail address")
    return problems

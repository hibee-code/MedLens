"""Files that must never reach Git."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_greenbook_snapshots_are_ignored() -> None:
    """Full register snapshots have no redistribution terms (decision log, "Greenbook data
    handling"): their directory is listed in .gitignore on its own line."""
    rules = {line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()}

    assert "data/greenbook/" in rules

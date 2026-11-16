from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    confidence: float  # mean word confidence, 0..1
    # Each line of ``text`` with its own mean word confidence (0..1), when the engine measures
    # it: the register fields read from a line take that line's confidence.
    lines: tuple[tuple[str, float], ...] = ()
    # Each line of ``text`` with its words' confidences (0..1), in order: the line's words are
    # the line split on spaces. The NAFDAC number takes the lowest of its own words.
    words: tuple[tuple[str, tuple[float, ...]], ...] = ()


class OcrEngine(Protocol):
    """Reads printed text from a pack photo (Tesseract first; CLAUDE.md decision log)."""

    def read(self, image: bytes) -> OcrResult: ...

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    confidence: float  # mean word confidence, 0..1


class OcrEngine(Protocol):
    """Reads printed text from a pack photo (Tesseract first; CLAUDE.md decision log)."""

    def read(self, image: bytes) -> OcrResult: ...

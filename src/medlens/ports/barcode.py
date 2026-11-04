from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class BarcodeRead:
    format: str  # e.g. DataMatrix, QRCode, EAN13
    text: str  # decoded content; GS1 group separators kept as \x1d


class BarcodePort(Protocol):
    """Decodes every barcode in a photo. Deterministic and cheap: always tried before OCR."""

    def decode(self, image: bytes) -> list[BarcodeRead]: ...

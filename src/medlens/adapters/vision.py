"""Real image readers for the worker (the `vision` dependency group): OpenCV quality metrics,
a 64-bit perceptual hash, barcodes with zxing-cpp and printed text with Tesseract. CPU work:
workers only, never the API's event loop."""

import io
from collections import defaultdict
from statistics import fmean

import cv2
import imagehash
import numpy as np
import pytesseract
import zxingcpp
from PIL import Image

from medlens.ports.barcode import BarcodeRead
from medlens.ports.image_analysis import ImageAssessment
from medlens.ports.ocr import OcrResult

_SIGN_BIT = 1 << 63


def _signed64(value: int) -> int:
    """Postgres BIGINT is signed: map the unsigned 64-bit hash onto it."""
    return value - (1 << 64) if value & _SIGN_BIT else value


def _decode(image: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image)) as opened:
        return opened.convert("RGB")


class OpenCvImageAnalysis:
    def assess(self, image: bytes) -> ImageAssessment:
        picture = _decode(image)
        grey = cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2GRAY)
        blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        brightness = float(grey.mean())
        phash = int(str(imagehash.phash(picture)), 16)
        return ImageAssessment(
            width=picture.width,
            height=picture.height,
            blur=blur,
            brightness=brightness,
            phash=_signed64(phash),
        )


class ZxingBarcode:
    """All barcodes in the photo. GS1 content comes back human-readable, ``(01)…(17)…(10)…``,
    which parse_gs1 reads."""

    def decode(self, image: bytes) -> list[BarcodeRead]:
        reads = zxingcpp.read_barcodes(_decode(image), text_mode=zxingcpp.TextMode.HRI)
        return [
            BarcodeRead(format=getattr(read.format, "name", str(read.format)), text=read.text)
            for read in reads
            if read.valid
        ]


def tesseract_available() -> bool:
    """The tesseract binary is installed (the worker image's apt packages)."""
    try:
        pytesseract.get_tesseract_version()
    except (pytesseract.TesseractNotFoundError, OSError):
        return False
    return True


class TesseractOcr:
    """Printed text, one line per printed line (the field extractors match within lines);
    confidence is the mean word confidence. Sparse-text mode suits packs, whose text is
    scattered rather than set in paragraphs."""

    def __init__(self, *, languages: str = "eng", timeout_seconds: float = 10.0) -> None:
        self._languages = languages
        self._timeout = timeout_seconds

    def read(self, image: bytes) -> OcrResult:
        data = pytesseract.image_to_data(
            _decode(image).convert("L"),
            lang=self._languages,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
            timeout=self._timeout,
        )
        lines: defaultdict[tuple[int, int, int], list[str]] = defaultdict(list)
        confidences: list[float] = []
        for index, word in enumerate(data["text"]):
            confidence = float(data["conf"][index])
            if not word.strip() or confidence < 0:
                continue
            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            lines[key].append(word.strip())
            confidences.append(confidence)
        text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
        return OcrResult(text, round(fmean(confidences) / 100, 3) if confidences else 0.0)

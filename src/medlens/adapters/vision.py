"""Real image readers for the worker (the `vision` dependency group): OpenCV quality metrics,
a 64-bit perceptual hash, barcodes with zxing-cpp and printed text with Tesseract. CPU work:
workers only, never the API's event loop."""

import io
from collections import defaultdict
from statistics import fmean
from typing import Literal

import cv2
import imagehash
import numpy as np
import pytesseract
import zxingcpp
from PIL import Image, ImageOps

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


# Tesseract passes read in turn, by OCR mode (MEDLENS_VERIFICATION__OCR_MODE): the page
# segmentation mode (--psm) and whether the image is inverted first.
OcrMode = Literal["sparse_block", "sparse_inverted"]
PASSES: dict[str, tuple[tuple[int, bool], ...]] = {
    # Sparse text, then one uniform block: reads brand names printed light on a dark band.
    "sparse_block": ((11, False), (6, False)),
    # The previous mode (until 2026-11-16): sparse text on the image and on its inverse.
    "sparse_inverted": ((11, False), (11, True)),
}


class TesseractOcr:
    """Printed text, one line per printed line (the field extractors match within lines);
    confidence is the mean word confidence, overall and per line. Two passes over the greyscale
    image, new lines added: sparse-text mode (--psm 11) suits packs, whose text is scattered,
    but skips brand names printed light on a dark band; block mode (--psm 6) reads those. On
    the 20 eval packs block mode added the brand name on 18; the inverted-image pass it
    replaces added only noise, at the same cost (about 0.5 s per pass). ``mode`` "sparse_inverted"
    keeps the previous passes (sparse text on the image and on its inverse).
    CLAUDE.md decision log, "Brand names read by OCR, and unregistered numbers on registered
    brands"."""

    def __init__(
        self,
        *,
        languages: str = "eng",
        timeout_seconds: float = 10.0,
        mode: OcrMode = "sparse_block",
    ) -> None:
        self._languages = languages
        self._timeout = timeout_seconds
        self._passes = PASSES[mode]

    def read(self, image: bytes) -> OcrResult:
        grey = _decode(image).convert("L")
        lines: dict[str, list[float]] = {}
        for mode, inverted in self._passes:
            picture = ImageOps.invert(grey) if inverted else grey
            for line, word_confidences in self._lines(picture, mode):
                lines.setdefault(line, word_confidences)
        confidences = [value for words in lines.values() for value in words]
        return OcrResult(
            "\n".join(lines),
            round(fmean(confidences) / 100, 3) if confidences else 0.0,
            tuple((line, round(fmean(words) / 100, 3)) for line, words in lines.items()),
            tuple(
                (line, tuple(round(value / 100, 3) for value in words))
                for line, words in lines.items()
            ),
        )

    def _lines(self, picture: Image.Image, mode: int) -> list[tuple[str, list[float]]]:
        data = pytesseract.image_to_data(
            picture,
            lang=self._languages,
            config=f"--psm {mode}",
            output_type=pytesseract.Output.DICT,
            timeout=self._timeout,
        )
        words: defaultdict[tuple[int, int, int], list[tuple[str, float]]] = defaultdict(list)
        for index, word in enumerate(data["text"]):
            confidence = float(data["conf"][index])
            if not word.strip() or confidence < 0:
                continue
            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            words[key].append((word.strip(), confidence))
        return [
            (" ".join(word for word, _ in line), [conf for _, conf in line])
            for _, line in sorted(words.items())
        ]

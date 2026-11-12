"""Benchmark the pack readers on labelled photos (the OCR decision in CLAUDE.md §9).

    python scripts/ocr_benchmark.py --photos DIR --labels labels.csv --engines tesseract,llm

labels.csv: file, nafdac_number, brand, batch_number, expiry (MM/YYYY as printed). Each engine
reads every photo; its text goes through the pipeline's own parsers (verification/extraction.py),
so the comparison is the one that matters: fields the pipeline would extract. Reported per
engine: exact-match rate of NAFDAC number, batch and expiry (month), brand name found, the share
of photos with the NAFDAC number read, latency p50/p95 and, for the vision LLM, the cost.

Engines: tesseract (needs the tesseract binary: run in the worker image), paddle (needs the
`ocr-benchmark` dependency group), llm (needs MEDLENS_LLM__ENABLED=true and an API key; spends
money). Unavailable engines are skipped and say why.

The labels are checked first (a photo that is missing, a NAFDAC number, batch or expiry that
cannot be read as one): problems are listed and the run stops, since a wrong label scores a
right reading as a miss. The report says whether the sample is large enough for the decision
(--min-photos, 100) and which engine reads each field best.

The synthetic packs in seed/test_packs work as a smoke test only. The decision needs 100+ real,
labelled photos of Nigerian packs: phone photos in ordinary light, location data stripped, kept
outside Git, one labels.csv row per photo (file,nafdac_number,brand,batch_number,expiry).
"""

import argparse
import csv
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from medlens.core.identifiers import normalize_batch_number, normalize_nafdac_number
from medlens.modules.verification.extraction import (
    Extraction,
    from_ocr,
    name_presence,
    parse_expiry,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Label:
    file: str
    nafdac_number: str
    brand: str
    batch_number: str
    expiry: date | None


@dataclass
class Score:
    engine: str
    photos: int = 0
    nafdac: int = 0
    batch: int = 0
    expiry: int = 0
    brand: int = 0
    latencies: list[float] = field(default_factory=list)
    cost_usd: Decimal = Decimal(0)
    errors: int = 0

    def report(self) -> dict[str, Any]:
        def rate(hits: int) -> float:
            return round(hits / self.photos, 3) if self.photos else 0.0

        ordered = sorted(self.latencies)
        return {
            "engine": self.engine,
            "photos": self.photos,
            "nafdac_number": rate(self.nafdac),
            "batch_number": rate(self.batch),
            "expiry_month": rate(self.expiry),
            "brand_found": rate(self.brand),
            "latency_p50_ms": round(statistics.median(ordered) * 1000) if ordered else None,
            "latency_p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))] * 1000)
            if ordered
            else None,
            "cost_usd": str(self.cost_usd),
            "errors": self.errors,
        }


def _normalise(value: str, normaliser: Callable[[str], str]) -> str | None:
    try:
        return normaliser(value) if value else None
    except ValueError:
        return None


REQUIRED_COLUMNS = ("file", "nafdac_number", "brand", "batch_number", "expiry")


def load_labels(path: Path) -> list[Label]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            Label(
                file=row["file"],
                nafdac_number=_normalise(row["nafdac_number"], normalize_nafdac_number) or "",
                brand=row.get("brand", ""),
                batch_number=_normalise(row.get("batch_number", ""), normalize_batch_number) or "",
                expiry=parse_expiry(row.get("expiry", "")),
            )
            for row in csv.DictReader(handle)
        ]


def label_problems(path: Path, photos: Path) -> list[str]:
    """Rows that would make the benchmark wrong: unknown columns, missing photos, duplicates,
    values that do not parse as what they claim to be. Empty fields are fine (not printed)."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            return [f"missing columns: {', '.join(missing)}"]
        rows = list(reader)
    problems: list[str] = []
    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        name = row["file"].strip()
        if name in seen:
            problems.append(f"line {line}: {name} is labelled twice")
        seen.add(name)
        if not (photos / name).is_file():
            problems.append(f"line {line}: no photo {name}")
        checks = (
            ("nafdac_number", normalize_nafdac_number),
            ("batch_number", normalize_batch_number),
        )
        for column, normaliser in checks:
            value = row[column].strip()
            if value and _normalise(value, normaliser) is None:
                problems.append(f"line {line}: {column} {value!r} does not parse")
        expiry = row["expiry"].strip()
        if expiry and parse_expiry(expiry) is None:
            problems.append(f"line {line}: expiry {expiry!r} does not parse (MM/YYYY)")
    if not rows:
        problems.append("no rows")
    return problems


def best_per_field(engines: list[dict[str, Any]]) -> dict[str, str | None]:
    """The engine with the highest rate per field (ties: the first, so list the cheapest first)."""
    scored = [engine for engine in engines if "skipped" not in engine]
    fields = ("nafdac_number", "batch_number", "expiry_month", "brand_found")
    return {
        field: max(scored, key=lambda engine: engine[field])["engine"] if scored else None
        for field in fields
    }


def grade(score: Score, label: Label, extraction: Extraction, text: str) -> None:
    score.photos += 1
    if extraction.nafdac_number and extraction.nafdac_number.value == label.nafdac_number:
        score.nafdac += 1
    if extraction.batch_number and extraction.batch_number.value == label.batch_number:
        score.batch += 1
    read = extraction.expiry_date.value if extraction.expiry_date else None
    if read and label.expiry and (read.year, read.month) == (label.expiry.year, label.expiry.month):
        score.expiry += 1
    if label.brand and (name_presence(label.brand, text) or 0) >= 0.8:
        score.brand += 1


# --------------------------------------------------------------------------- engines
Reader = Callable[[bytes], tuple[Extraction, str]]


def tesseract() -> Reader:
    from medlens.adapters.vision import TesseractOcr, tesseract_available

    if not tesseract_available():
        raise RuntimeError("tesseract is not installed (run in the worker image)")
    engine = TesseractOcr()

    def read(image: bytes) -> tuple[Extraction, str]:
        result = engine.read(image)
        return from_ocr(result.text, result.confidence), result.text

    return read


def paddle() -> Reader:
    try:
        import numpy as np
        from paddleocr import PaddleOCR  # the ocr-benchmark group
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("paddleocr is not installed (uv sync --group ocr-benchmark)") from exc
    engine = PaddleOCR(lang="en")

    def read(image: bytes) -> tuple[Extraction, str]:
        import io

        picture = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
        lines, confidences = [], []
        for page in engine.predict(picture):
            lines.extend(page["rec_texts"])
            confidences.extend(page["rec_scores"])
        text = "\n".join(lines)
        confidence = statistics.fmean(confidences) if confidences else 0.0
        return from_ocr(text, confidence), text

    return read


def llm(score: Score) -> Reader:
    from medlens.adapters.fakes import MemoryLedger
    from medlens.adapters.llm import GatewayPackReader
    from medlens.ai.gateway.gateway import LLMGateway
    from medlens.ai.gateway.providers import AnthropicProvider
    from medlens.core.config import get_settings
    from medlens.modules.verification.pipeline import PackExtraction

    settings = get_settings().llm
    if not settings.enabled:
        raise RuntimeError("AI is off (MEDLENS_LLM__ENABLED=false)")
    ledger = MemoryLedger()
    reader = GatewayPackReader(LLMGateway(settings, AnthropicProvider(settings), ledger))
    fields = ("nafdac_number", "product_name", "batch_number", "expiry_date")

    def read(image: bytes) -> tuple[Extraction, str]:
        parsed = PackExtraction.model_validate_json(
            reader.extract_pack_fields(image, fields=fields)
        )
        score.cost_usd = sum((call.cost_usd for call in ledger.calls), Decimal(0))
        text = "\n".join(
            f"{label} {reading.text}"
            for label, reading in (
                ("NAFDAC REG NO:", parsed.nafdac_number),
                ("", parsed.product_name),
                ("BATCH NO:", parsed.batch_number),
                ("EXP:", parsed.expiry_date),
            )
            if reading.text
        )
        return from_ocr(text, 1.0), text

    return read


def run(engine: str, labels: list[Label], photos: Path) -> dict[str, Any]:
    score = Score(engine)
    try:
        reader = (
            {"tesseract": tesseract, "paddle": paddle}[engine]() if engine != "llm" else llm(score)
        )
    except (RuntimeError, KeyError) as exc:
        return {"engine": engine, "skipped": str(exc)}
    for label in labels:
        image = (photos / label.file).read_bytes()
        started = time.perf_counter()
        try:
            extraction, text = reader(image)
        except Exception:  # noqa: BLE001 - count it and go on
            score.errors += 1
            score.photos += 1
            continue
        score.latencies.append(time.perf_counter() - started)
        grade(score, label, extraction, text)
    return score.report()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--photos", type=Path, default=ROOT / "seed" / "test_packs")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--engines", default="tesseract,paddle,llm")
    parser.add_argument("--min-photos", type=int, default=100)
    args = parser.parse_args(argv)
    labels_path = args.labels or args.photos / "expected.csv"
    problems = label_problems(labels_path, args.photos)
    if problems:
        sys.stderr.write(f"{labels_path}: fix these labels first\n")
        sys.stderr.writelines(f"  {problem}\n" for problem in problems)
        return 2
    labels = load_labels(labels_path)
    engines = [run(engine.strip(), labels, args.photos) for engine in args.engines.split(",")]
    report = {
        "photos": len(labels),
        "enough_for_the_decision": len(labels) >= args.min_photos,
        "labelled": {
            "nafdac_number": sum(bool(label.nafdac_number) for label in labels),
            "batch_number": sum(bool(label.batch_number) for label in labels),
            "expiry": sum(label.expiry is not None for label in labels),
            "brand": sum(bool(label.brand) for label in labels),
        },
        "engines": engines,
        "best_per_field": best_per_field(engines),
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

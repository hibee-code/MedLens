"""The verification pipeline: a pack photo in, a scored verdict out.

Pure orchestration over ports and the ``Catalog``; loading the scan and saving the result is the
caller's job (``verification/service.py``, run by the worker). Stages, each timed:

    quality   ImageAnalysisPort: size, blur, brightness, perceptual hash. A photo that fails the
              gate ends the pipeline: INCONCLUSIVE with reason codes, so the client can retake.
    barcode   BarcodePort, then GS1 parsing (GTIN, batch, expiry).
    ocr       OcrEngine, when required fields are still missing or unsure.
    llm       LLMPort, only if fields are still missing or unsure and AI is switched on. Its
              output is untrusted: it must validate against ``LlmFields`` and every value passes
              the same normalisers as user input. The LLM never decides the verdict.
    match     exact NAFDAC number, then close numbers (OCR errors), then names on the pack.
    batch     declared batch, pack expiry, expiry mismatch, spread across states, distribution.
    alerts    published NAFDAC alerts and recalls naming the product or its batch.
    visual    embedding against the product's approved reference images.
    score     RiskScorer.

A stage that fails is recorded and its signals become unavailable, which lowers confidence; it
never fails the scan.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from medlens.core.identifiers import (
    normalize_batch_number,
    normalize_nafdac_number,
    sanitize_text,
)
from medlens.modules.products.models import RegistrationStatus
from medlens.modules.verification.catalog import BatchFacts, Catalog, ProductFacts
from medlens.modules.verification.codes import ReasonCode
from medlens.modules.verification.extraction import (
    ExtractedField,
    Extraction,
    FieldSource,
    from_gs1,
    from_ocr,
    name_presence,
    parse_expiry,
)
from medlens.modules.verification.scoring import (
    UNAVAILABLE,
    WEIGHTS_V1,
    RiskScorer,
    Score,
    ScoreInput,
    Signal,
    SignalName,
)
from medlens.ports.barcode import BarcodePort
from medlens.ports.embedding import EmbeddingPort
from medlens.ports.image_analysis import ImageAnalysisPort, ImageAssessment
from medlens.ports.llm import LLMPort, LLMUnavailableError
from medlens.ports.ocr import OcrEngine

StageName = Literal[
    "quality", "barcode", "ocr", "llm", "match", "batch", "alerts", "visual", "score"
]
MatchMethod = Literal["nafdac_number", "similar_number", "name"]

_REGISTRATION_RISK = {
    RegistrationStatus.ACTIVE: 0.0,
    RegistrationStatus.EXPIRED: 0.5,
    RegistrationStatus.SUSPENDED: 1.0,
    RegistrationStatus.REVOKED: 1.0,
}
CLAIMED_CONFIDENCE = 0.5  # a number typed by the person scanning
LLM_MAX_CONFIDENCE = 0.8  # never trusted as much as a barcode or a clear OCR read


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    min_field_confidence: float = 0.6
    # Quality gate
    min_side_pixels: int = 480
    blur_threshold: float = 60.0
    dark_threshold: float = 40.0
    bright_threshold: float = 235.0
    # Matching
    similar_number_threshold: float = 0.6
    name_match_threshold: float = 0.8
    # Batch spread: more distinct states than this within the window is suspicious.
    spread_window_days: int = 30
    spread_state_limit: int = 3
    # Visual match: cosine similarity at or above `visual_match` is a match, at or below
    # `visual_mismatch` a full mismatch, linear in between.
    visual_match: float = 0.85
    visual_mismatch: float = 0.60
    llm_enabled: bool = False
    # Stages that would start later than this are skipped, so every scan finishes.
    deadline_seconds: float = 20.0


@dataclass(frozen=True, slots=True)
class ScanInput:
    scan_id: UUID
    image: bytes
    received_at: datetime
    claimed_nafdac_number: str | None = None
    state: str | None = None  # ISO 3166-2:NG code


@dataclass(frozen=True, slots=True)
class StageResult:
    stage: StageName
    outcome: Literal["ok", "skipped", "failed"]
    duration_ms: float
    error: str | None = None  # exception type only: messages may echo untrusted data

    def as_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "stage": self.stage,
            "outcome": self.outcome,
            "ms": round(self.duration_ms, 1),
        }
        if self.error:
            result["error"] = self.error
        return result


@dataclass(frozen=True, slots=True)
class Match:
    product: ProductFacts
    method: MatchMethod
    score: float  # how sure the identification is, 0..1
    number_name_conflict: bool = False


@dataclass(slots=True)
class PipelineResult:
    score: Score
    extraction: Extraction = field(default_factory=Extraction)
    assessment: ImageAssessment | None = None
    match: Match | None = None
    batch: BatchFacts | None = None
    signals: dict[SignalName, Signal] = field(default_factory=dict)
    stages: list[StageResult] = field(default_factory=list)

    @property
    def llm_used(self) -> bool:
        """A model call was made for this scan (skipped stages made none)."""
        return any(stage.stage == "llm" and stage.outcome != "skipped" for stage in self.stages)

    def signals_json(self) -> dict[str, object]:
        """For ``scans.signals``: the full signal vector, match and stage trace, so the scan can
        be audited and re-scored later."""
        return {
            "signals": {
                name.value: {
                    "value": signal.value,
                    "weight": WEIGHTS_V1[name],
                    **signal.detail,
                }
                for name, signal in self.signals.items()
            },
            "match": (
                {
                    "method": self.match.method,
                    "score": round(self.match.score, 3),
                    "number_name_conflict": self.match.number_name_conflict,
                }
                if self.match
                else None
            ),
            "stages": [stage.as_json() for stage in self.stages],
            "high_priority": self.score.high_priority,
            "llm_used": self.llm_used,
        }


NOT_VISIBLE = "not_visible"


class FieldReading(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    value: str = Field(max_length=200, description='As printed, or "not_visible".')
    confidence: float = Field(ge=0, le=1)

    @property
    def text(self) -> str | None:
        return None if not self.value or self.value.casefold() == NOT_VISIBLE else self.value


class PackExtraction(BaseModel):
    """The only shape accepted from the vision model (also its structured-output schema)."""

    model_config = ConfigDict(extra="forbid")

    nafdac_number: FieldReading
    product_name: FieldReading
    batch_number: FieldReading
    expiry_date: FieldReading


class VerificationPipeline:
    def __init__(
        self,
        *,
        catalog: Catalog,
        scorer: RiskScorer,
        analyzer: ImageAnalysisPort,
        barcode: BarcodePort,
        ocr: OcrEngine | None = None,
        llm: LLMPort | None = None,
        embedder: EmbeddingPort | None = None,
        config: PipelineConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._catalog = catalog
        self._scorer = scorer
        self._analyzer = analyzer
        self._barcode = barcode
        self._ocr = ocr
        self._llm = llm
        self._embedder = embedder
        self._config = config or PipelineConfig()
        self._clock = clock

    def run(self, scan: ScanInput) -> PipelineResult:
        run = _Run(self._clock, self._config.deadline_seconds)
        assessment = run.stage("quality", lambda: self._analyzer.assess(scan.image))
        quality_reasons = self._quality_reasons(assessment)
        if quality_reasons:
            score = self._scorer.score(ScoreInput(quality_reasons=quality_reasons))
            return PipelineResult(score=score, assessment=assessment, stages=run.stages)

        extraction = self._extract(scan, run)
        match = run.stage("match", lambda: self._match(extraction))
        signals: dict[SignalName, Signal] = dict.fromkeys(SignalName, UNAVAILABLE)
        batch: BatchFacts | None = None
        pack_expired = self._expired(extraction.expiry_date, None, scan.received_at)
        if match is not None:
            signals[SignalName.REGISTRATION] = Signal(
                _REGISTRATION_RISK[match.product.registration_status],
                {"status": match.product.registration_status.value},
            )
            signals[SignalName.NAME_MISMATCH] = self._name_signal(match, extraction)
            checked = run.stage("batch", lambda: self._batch(scan, match.product, extraction))
            if checked is not None:
                batch, batch_signals = checked
                signals.update(batch_signals)
                pack_expired = self._expired(extraction.expiry_date, batch, scan.received_at)
            batch_number = extraction.batch_number.value if extraction.batch_number else None
            alert = run.stage(
                "alerts",
                lambda: self._catalog.published_alert(match.product.nafdac_number, batch_number),
            )
            if run.stages[-1].outcome == "ok":
                signals[SignalName.COUNTERFEIT_ALERT] = (
                    Signal(
                        1.0,
                        {
                            "document_id": str(alert.document_id),
                            "kind": alert.kind,
                            "batch_specific": alert.batch_specific,
                        },
                    )
                    if alert is not None
                    else Signal(0.0)
                )
            visual = run.stage("visual", lambda: self._visual(scan, match.product))
            if visual is not None:
                signals[SignalName.VISUAL_MISMATCH] = visual

        facts = ScoreInput(
            product_identified=match is not None,
            nafdac_confidence=(
                extraction.nafdac_number.confidence if extraction.nafdac_number else None
            ),
            number_name_conflict=bool(match and match.number_name_conflict),
            registration_status=match.product.registration_status if match else None,
            pack_expired=pack_expired,
            signals=signals,
            extraction_confidence=self._extraction_confidence(match, extraction),
        )
        return PipelineResult(
            score=run.measure("score", lambda: self._scorer.score(facts)),
            extraction=extraction,
            assessment=assessment,
            match=match,
            batch=batch,
            signals=signals,
            stages=run.stages,
        )

    # ------------------------------------------------------------------ quality
    def _quality_reasons(self, assessment: ImageAssessment | None) -> tuple[ReasonCode, ...]:
        """A failed analyzer is not a bad photo: the gate is skipped, not failed."""
        if assessment is None:
            return ()
        config = self._config
        reasons: list[ReasonCode] = []
        if min(assessment.width, assessment.height) < config.min_side_pixels:
            reasons.append(ReasonCode.TOO_SMALL)
        if assessment.blur < config.blur_threshold:
            reasons.append(ReasonCode.BLURRY)
        if assessment.brightness < config.dark_threshold:
            reasons.append(ReasonCode.TOO_DARK)
        elif assessment.brightness > config.bright_threshold:
            reasons.append(ReasonCode.TOO_BRIGHT)
        return tuple(reasons)

    # ------------------------------------------------------------------ extraction
    def _extract(self, scan: ScanInput, run: "_Run") -> Extraction:
        extraction = Extraction()
        if scan.claimed_nafdac_number:
            try:
                claimed = normalize_nafdac_number(scan.claimed_nafdac_number)
                extraction = Extraction(
                    nafdac_number=ExtractedField(claimed, FieldSource.CLAIMED, CLAIMED_CONFIDENCE)
                )
            except ValueError:
                pass
        reads = run.stage("barcode", lambda: self._barcode.decode(scan.image)) or []
        for read in reads:
            extraction = extraction.merge(from_gs1(read.text))

        threshold = self._config.min_field_confidence
        engine = self._ocr
        if extraction.missing(threshold) and engine is not None:
            ocr = run.stage("ocr", lambda: engine.read(scan.image))
            if ocr is not None:
                extraction = extraction.merge(from_ocr(ocr.text, ocr.confidence))
        else:
            run.skip("ocr")

        missing = extraction.missing(threshold)
        if missing and self._llm is not None and self._config.llm_enabled:
            llm = self._llm
            fields = run.stage("llm", lambda: self._llm_fields(llm, scan, missing))
            if fields is not None:
                extraction = extraction.merge(fields)
        else:
            run.skip("llm")
        return extraction

    @staticmethod
    def _llm_fields(llm: LLMPort, scan: ScanInput, missing: list[str]) -> Extraction:
        """Raises on output that does not validate, so the stage is recorded as failed. Every
        value passes the same normalisers as user input; confidence is capped."""
        raw = llm.extract_pack_fields(scan.image, fields=missing, subject_id=str(scan.scan_id))
        parsed = PackExtraction.model_validate_json(raw)

        def confidence(reading: FieldReading) -> float:
            return min(reading.confidence, LLM_MAX_CONFIDENCE)

        def field_of[T](value: T | None, reading: FieldReading) -> ExtractedField[T] | None:
            if value is None or not reading.confidence:
                return None
            return ExtractedField(value, FieldSource.LLM, confidence(reading))

        def normalised(reading: FieldReading, normalise: Callable[[str], str]) -> str | None:
            if not reading.text:
                return None
            try:
                return normalise(reading.text)
            except ValueError:
                return None

        name = parsed.product_name.text
        return Extraction(
            nafdac_number=field_of(
                normalised(parsed.nafdac_number, normalize_nafdac_number), parsed.nafdac_number
            ),
            product_name=field_of(sanitize_text(name) if name else None, parsed.product_name),
            batch_number=field_of(
                normalised(parsed.batch_number, normalize_batch_number), parsed.batch_number
            ),
            expiry_date=field_of(parse_expiry(parsed.expiry_date.text or ""), parsed.expiry_date),
            text=sanitize_text(name) if name else "",
        )

    # ------------------------------------------------------------------ match
    def _match(self, extraction: Extraction) -> Match | None:
        catalog, config = self._catalog, self._config
        text = extraction.text
        if extraction.product_name:
            text = f"{text} {extraction.product_name.value}"
        number = extraction.nafdac_number
        if number is not None:
            product = catalog.product_by_nafdac(number.value)
            if product is not None:
                return Match(
                    product,
                    "nafdac_number",
                    number.confidence,
                    number_name_conflict=self._names_other_product(product, text),
                )
            for candidate, similarity in catalog.similar_nafdac(number.value):
                presence = name_presence(candidate.brand_name, text) or 0.0
                if similarity >= config.similar_number_threshold and presence >= 0.5:
                    return Match(candidate, "similar_number", number.confidence * similarity)
        if text.strip():
            named = catalog.products_named_in(text)
            if named and named[0][1] >= config.name_match_threshold:
                product, score = named[0]
                # A legible number that is not this product's, and no closer product: the pack
                # carries a number that does not belong to it.
                conflict = number is not None and number.confidence >= config.min_field_confidence
                return Match(product, "name", score, number_name_conflict=conflict)
        return None

    def _names_other_product(self, product: ProductFacts, text: str) -> bool:
        """The number is product A's, but the pack clearly names another product B."""
        if not text.strip():
            return False
        own = name_presence(product.brand_name, text)
        if own is None or own >= 0.34:
            return False
        named = self._catalog.products_named_in(text)
        return bool(
            named
            and named[0][0].id != product.id
            and named[0][1] >= max(self._config.name_match_threshold, 0.9)
        )

    def _name_signal(self, match: Match, extraction: Extraction) -> Signal:
        if match.method == "name":
            return Signal(0.0, {"presence": 1.0})
        text = extraction.text
        if extraction.product_name:
            text = f"{text} {extraction.product_name.value}"
        presence = name_presence(match.product.brand_name, text)
        if presence is None:
            return UNAVAILABLE
        return Signal(round(1.0 - presence, 4), {"presence": round(presence, 3)})

    # ------------------------------------------------------------------ batch
    def _batch(
        self, scan: ScanInput, product: ProductFacts, extraction: Extraction
    ) -> tuple[BatchFacts | None, dict[SignalName, Signal]]:
        catalog, config = self._catalog, self._config
        signals: dict[SignalName, Signal] = {}
        read = extraction.batch_number
        declared = catalog.declared_batch_count(product.id)
        if read is None or declared == 0:
            return None, signals  # nothing to compare against
        batch = catalog.batch(product.id, read.value)
        signals[SignalName.BATCH_NOT_DECLARED] = Signal(
            0.0 if batch else 1.0, {"declared_batches": declared}
        )
        if batch is None:
            return None, signals

        expiry = extraction.expiry_date
        if expiry is not None:
            differs = (expiry.value.year, expiry.value.month) != (
                batch.expires_on.year,
                batch.expires_on.month,
            )
            signals[SignalName.EXPIRY_MISMATCH] = Signal(1.0 if differs else 0.0)

        since = scan.received_at - timedelta(days=config.spread_window_days)
        spread = catalog.batch_spread(batch.id, since, exclude_scan=scan.scan_id)
        states = spread.states | ({scan.state} if scan.state else set())
        limit = config.spread_state_limit
        value = 0.0 if len(states) <= limit else min(1.0, (len(states) - limit) / limit)
        signals[SignalName.BATCH_SPREAD] = Signal(
            round(value, 4),
            {"states": len(states), "geocells": spread.geocells, "days": config.spread_window_days},
        )
        if batch.distribution_states and scan.state:
            outside = scan.state not in batch.distribution_states
            signals[SignalName.OUTSIDE_DISTRIBUTION] = Signal(1.0 if outside else 0.0)
        return batch, signals

    @staticmethod
    def _expired(
        expiry: ExtractedField[date] | None, batch: BatchFacts | None, received_at: datetime
    ) -> bool:
        today = received_at.date()
        if expiry is not None:
            return expiry.value < today
        return batch is not None and batch.expires_on < today

    # ------------------------------------------------------------------ visual
    def _visual(self, scan: ScanInput, product: ProductFacts) -> Signal:
        if self._embedder is None:
            return UNAVAILABLE
        embedding = self._embedder.embed_image(scan.image)
        similarity = self._catalog.reference_similarity(product.id, embedding)
        if similarity is None:
            return UNAVAILABLE  # no approved reference images: lowers confidence only
        high, low = self._config.visual_match, self._config.visual_mismatch
        value = min(1.0, max(0.0, (high - similarity) / (high - low)))
        return Signal(round(value, 4), {"similarity": round(similarity, 4)})

    # ------------------------------------------------------------------ confidence
    @staticmethod
    def _extraction_confidence(match: Match | None, extraction: Extraction) -> float:
        """0.6 x identification + 0.2 x batch number + 0.2 x expiry date (missing fields: 0)."""
        if match is None:
            return 0.0
        batch = extraction.batch_number.confidence if extraction.batch_number else 0.0
        expiry = extraction.expiry_date.confidence if extraction.expiry_date else 0.0
        return 0.6 * match.score + 0.2 * batch + 0.2 * expiry


class _Run:
    """Stage timing. A stage that raises is recorded as failed and yields None; one that would
    start after the deadline, or whose model is unavailable, is skipped."""

    def __init__(self, clock: Callable[[], float], deadline_seconds: float) -> None:
        self._clock = clock
        self._deadline = clock() + deadline_seconds
        self.stages: list[StageResult] = []

    def stage[T](self, name: StageName, work: Callable[[], T]) -> T | None:
        if self._clock() > self._deadline:
            self.stages.append(StageResult(name, "skipped", 0.0, "Deadline"))
            return None
        with self._timed(name) as record:
            try:
                return work()
            except LLMUnavailableError as exc:
                record["skipped"] = exc.reason
            except (ValidationError, ValueError) as exc:
                record["error"] = type(exc).__name__
            except Exception as exc:  # noqa: BLE001 - an adapter failure degrades, never fails
                record["error"] = type(exc).__name__
        return None

    def measure[T](self, name: StageName, work: Callable[[], T]) -> T:
        """Timed, but errors propagate: without a score there is no result to save."""
        started = self._clock()
        result = work()
        self.stages.append(StageResult(name, "ok", (self._clock() - started) * 1000))
        return result

    def skip(self, name: StageName) -> None:
        self.stages.append(StageResult(name, "skipped", 0.0))

    @contextmanager
    def _timed(self, name: StageName) -> Iterator[dict[str, str]]:
        record: dict[str, str] = {}
        started = self._clock()
        try:
            yield record
        finally:
            error, skipped = record.get("error"), record.get("skipped")
            outcome: Literal["ok", "skipped", "failed"] = (
                "failed" if error else "skipped" if skipped else "ok"
            )
            self.stages.append(
                StageResult(name, outcome, (self._clock() - started) * 1000, error or skipped)
            )

"""Processing one scan, in the worker: load it, run the pipeline, save the result.

Safe to run more than once for the same scan (at-least-once delivery): a finished scan is left
alone, and a scan another worker is processing right now is skipped. Transient failures
(object storage or the database unreachable) propagate so the task is retried; anything else
marks the scan ``failed`` with an error code.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal
from uuid import UUID

import structlog
from sqlalchemy import ColumnElement, select
from sqlalchemy.orm import Session, sessionmaker

from medlens.core.config import Settings
from medlens.core.ids import uuid7_time
from medlens.modules.scans.models import Scan, ScanStatus, Verdict
from medlens.modules.scans.repository import id_window
from medlens.modules.verification.catalog import Catalog
from medlens.modules.verification.codes import ReasonCode
from medlens.modules.verification.metrics import record_scan
from medlens.modules.verification.pipeline import (
    PipelineConfig,
    PipelineResult,
    ScanInput,
    VerificationPipeline,
)
from medlens.modules.verification.repository import SqlCatalog
from medlens.modules.verification.scoring import WeightedRuleScorer
from medlens.ports.storage import ObjectNotFoundError, SyncStoragePort
from medlens.queue import outbox

log = structlog.stdlib.get_logger(__name__)

PipelineFactory = Callable[[Catalog], VerificationPipeline]
Outcome = Literal["completed", "rejected", "duplicate", "failed", "skipped"]
_FINISHED = frozenset({ScanStatus.COMPLETED, ScanStatus.REJECTED, ScanStatus.FAILED})
_QUALITY_REASONS = frozenset(
    {ReasonCode.BLURRY, ReasonCode.TOO_DARK, ReasonCode.TOO_BRIGHT, ReasonCode.TOO_SMALL}
)


def pipeline_config(settings: Settings) -> PipelineConfig:
    verification = settings.verification
    return PipelineConfig(
        min_field_confidence=verification.min_field_confidence,
        min_side_pixels=verification.min_side_pixels,
        blur_threshold=verification.blur_threshold,
        dark_threshold=verification.dark_threshold,
        bright_threshold=verification.bright_threshold,
        spread_window_days=verification.spread_window_days,
        spread_state_limit=verification.spread_state_limit,
        llm_enabled=settings.llm.enabled,
        deadline_seconds=verification.pipeline_timeout_seconds,
    )


def scorer_for(settings: Settings) -> WeightedRuleScorer:
    verification = settings.verification
    return WeightedRuleScorer(
        suspicious_threshold=verification.suspicious_threshold,
        high_priority_threshold=verification.high_priority_threshold,
        min_field_confidence=verification.min_field_confidence,
    )


@dataclass(frozen=True, slots=True)
class Processed:
    scan_id: UUID
    outcome: Outcome
    verdict: Verdict | None = None


def needs_case(scan: Scan) -> bool:
    """High-risk scans a regulator should review: SUSPICIOUS with high priority, or UNREGISTERED
    (which the scorer only returns for a clearly read NAFDAC number)."""
    if scan.verdict is Verdict.UNREGISTERED:
        return True
    signals = scan.signals or {}
    return scan.verdict is Verdict.SUSPICIOUS and bool(signals.get("high_priority"))


class VerificationService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        storage: SyncStoragePort,
        pipeline: PipelineFactory,
        *,
        duplicate_window: timedelta = timedelta(hours=24),
    ) -> None:
        self._sessions = sessions
        self._storage = storage
        self._pipeline = pipeline
        self._duplicate_window = duplicate_window

    def process_scan(self, scan_id: UUID) -> Processed:
        started = time.perf_counter()
        if not self._start(scan_id):
            return Processed(scan_id, "skipped")
        with self._sessions() as session, session.begin():
            scan = self._locked(session, scan_id)
            if scan is None or scan.status in _FINISHED:
                return Processed(scan_id, "skipped")  # another worker finished it meanwhile
            try:
                image = self._storage.get(scan.image_s3_key or "")
            except ObjectNotFoundError:
                return self._failed(scan, "image_missing")
            catalog = SqlCatalog(session)
            try:
                result = self._pipeline(catalog).run(
                    ScanInput(
                        scan_id=scan.id,
                        image=image,
                        received_at=scan.created_at,
                        claimed_nafdac_number=(scan.extracted or {}).get("claimed_nafdac_number"),
                        state=scan.state.value if scan.state else None,
                    )
                )
            except Exception:
                log.exception("verification.pipeline_error", scan_id=str(scan.id))
                return self._failed(scan, "pipeline_error")
            earlier = None
            if result.assessment is not None and self._duplicate_window:
                earlier = catalog.near_duplicate(
                    result.assessment.phash,
                    owner=_owner(scan),
                    since=scan.created_at - self._duplicate_window,
                    exclude_scan=scan.id,
                )
            if earlier is not None:
                _copy_result(scan, earlier.scan, result)
                outcome: Outcome = "duplicate"
            else:
                _save_result(scan, result)
                outcome = "rejected" if scan.status is ScanStatus.REJECTED else "completed"
            elapsed = time.perf_counter() - started
            scan.processing_ms = round(elapsed * 1000)
            record_scan(scan.verdict.value if scan.verdict else None, outcome, elapsed, result)
            outbox.emit_sync(
                session,
                outbox.SCAN_COMPLETED,
                aggregate_type="scan",
                aggregate_id=scan.id,
                payload={
                    "created_at": scan.created_at.isoformat(),
                    "verdict": scan.verdict.value if scan.verdict else None,
                    "needs_case": needs_case(scan),
                },
            )
            log.info(
                "scan.scored",
                scan_id=str(scan.id),
                verdict=scan.verdict.value if scan.verdict else None,
                score=scan.risk_score,
                outcome=outcome,
            )
            return Processed(scan.id, outcome, scan.verdict)

    def _start(self, scan_id: UUID) -> bool:
        """Mark the scan processing (visible to clients). False when it is finished or locked
        by another worker."""
        with self._sessions() as session, session.begin():
            scan = self._locked(session, scan_id, skip_locked=True)
            if scan is None or scan.status in _FINISHED:
                return False
            scan.status = ScanStatus.PROCESSING
            return True

    @staticmethod
    def _locked(session: Session, scan_id: UUID, *, skip_locked: bool = False) -> Scan | None:
        uuid7_time(scan_id)  # raises for ids that cannot be ours
        statement = select(Scan).where(id_window(scan_id)).with_for_update(skip_locked=skip_locked)
        return session.scalar(statement)

    @staticmethod
    def _failed(scan: Scan, code: str) -> Processed:
        scan.status = ScanStatus.FAILED
        scan.error_code = code
        log.warning("scan.failed", scan_id=str(scan.id), error_code=code)
        return Processed(scan.id, "failed")


def _owner(scan: Scan) -> ColumnElement[bool]:
    if scan.user_id is not None:
        return Scan.user_id == scan.user_id
    if scan.device_fingerprint_hash is not None:
        return Scan.user_id.is_(None) & (
            Scan.device_fingerprint_hash == scan.device_fingerprint_hash
        )
    if scan.organization_id is not None:
        return Scan.user_id.is_(None) & (Scan.organization_id == scan.organization_id)
    return Scan.id == scan.id  # matches nothing else


def _save_result(scan: Scan, result: PipelineResult) -> None:
    score = result.score
    if result.assessment is not None:
        scan.image_phash = result.assessment.phash
        scan.quality = {**(scan.quality or {}), **result.assessment.as_json()}
    scan.extracted = {**(scan.extracted or {}), **result.extraction.as_json()}
    scan.matched_product_id = result.match.product.id if result.match else None
    scan.matched_batch_id = result.batch.id if result.batch else None
    scan.signals = result.signals_json()
    scan.verdict = score.verdict
    scan.risk_score = score.risk_score
    scan.confidence = score.confidence
    scan.scorer_version = score.scorer_version
    scan.reason_codes = [code.value for code in score.reason_codes]
    scan.advice_codes = [code.value for code in score.advice_codes]
    scan.llm_used = result.llm_used
    quality_failed = bool(set(score.reason_codes) & _QUALITY_REASONS)
    scan.status = ScanStatus.REJECTED if quality_failed else ScanStatus.COMPLETED


def _copy_result(scan: Scan, earlier: Scan, result: PipelineResult) -> None:
    """Same answer as the earlier scan of (nearly) the same photo."""
    if result.assessment is not None:
        scan.image_phash = result.assessment.phash
        scan.quality = {**(scan.quality or {}), **result.assessment.as_json()}
    for name in (
        "extracted",
        "matched_product_id",
        "matched_batch_id",
        "verdict",
        "risk_score",
        "confidence",
        "scorer_version",
        "reason_codes",
        "advice_codes",
        "status",
    ):
        setattr(scan, name, getattr(earlier, name))
    scan.signals = {**(earlier.signals or {}), "duplicate_of": str(earlier.id)}
    scan.llm_used = result.llm_used

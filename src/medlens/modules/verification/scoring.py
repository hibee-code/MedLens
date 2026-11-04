"""Risk scoring: signals in, verdict out. Deterministic and explainable (CLAUDE.md §5); the
formula is documented in docs/scoring.md.

``RiskScorer`` is the interface. ``WeightedRuleScorer`` (v2) is rule-based until labelled lab
results exist; a calibrated LightGBM scorer will implement the same interface and its own
version string. Every scan stores the scorer version and the full signal vector, so scans can
be re-scored.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from medlens.modules.products.models import RegistrationStatus
from medlens.modules.scans.models import Verdict
from medlens.modules.verification.codes import AdviceCode, ReasonCode


class SignalName(StrEnum):
    REGISTRATION = "registration"  # 0 active, 0.5 registration expired, 1 suspended
    NAME_MISMATCH = "name_mismatch"
    BATCH_NOT_DECLARED = "batch_not_declared"
    EXPIRY_MISMATCH = "expiry_mismatch"
    BATCH_SPREAD = "batch_spread"
    OUTSIDE_DISTRIBUTION = "outside_distribution"
    VISUAL_MISMATCH = "visual_mismatch"
    COUNTERFEIT_ALERT = "counterfeit_alert"  # 1: a published alert or recall names the pack


@dataclass(frozen=True, slots=True)
class Signal:
    """A risk indication in 0..1, or None when it could not be computed (that lowers the
    confidence, never raises the risk)."""

    value: float | None
    detail: dict[str, object] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.value is not None


UNAVAILABLE = Signal(None)


@dataclass(frozen=True, slots=True)
class ScoreInput:
    quality_reasons: tuple[ReasonCode, ...] = ()  # non-empty: the photo failed the quality gate
    product_identified: bool = False
    # Confidence of the NAFDAC number read from the pack (None: no number read).
    nafdac_confidence: float | None = None
    # The number read belongs to another product than the one named on the pack.
    number_name_conflict: bool = False
    registration_status: RegistrationStatus | None = None
    pack_expired: bool = False
    signals: dict[SignalName, Signal] = field(default_factory=dict)
    # Mean confidence of the fields the verdict relies on.
    extraction_confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class Score:
    verdict: Verdict
    risk_score: float | None
    confidence: float | None
    high_priority: bool
    reason_codes: tuple[ReasonCode, ...]
    advice_codes: tuple[AdviceCode, ...]
    scorer_version: str


class RiskScorer(Protocol):
    version: str

    def score(self, facts: ScoreInput) -> Score: ...


# Weight of each signal in the risk score; values are 0..1, the weighted sum is clipped to 1.
WEIGHTS_V1: dict[SignalName, float] = {
    SignalName.REGISTRATION: 0.35,
    SignalName.NAME_MISMATCH: 0.30,
    SignalName.BATCH_NOT_DECLARED: 0.30,
    SignalName.EXPIRY_MISMATCH: 0.35,
    SignalName.BATCH_SPREAD: 0.40,
    SignalName.OUTSIDE_DISTRIBUTION: 0.15,
    SignalName.VISUAL_MISMATCH: 0.35,
    SignalName.COUNTERFEIT_ALERT: 0.60,
}
# A signal is reported as a reason once its value reaches this level.
REASON_LEVEL = 0.5
_SIGNAL_REASONS: dict[SignalName, ReasonCode] = {
    SignalName.NAME_MISMATCH: ReasonCode.NAME_MISMATCH,
    SignalName.BATCH_NOT_DECLARED: ReasonCode.BATCH_NOT_DECLARED,
    SignalName.EXPIRY_MISMATCH: ReasonCode.EXPIRY_MISMATCH,
    SignalName.BATCH_SPREAD: ReasonCode.BATCH_SEEN_WIDELY,
    SignalName.OUTSIDE_DISTRIBUTION: ReasonCode.OUTSIDE_DISTRIBUTION,
    SignalName.VISUAL_MISMATCH: ReasonCode.VISUAL_MISMATCH,
}
_REGISTRATION_REASONS = {
    RegistrationStatus.EXPIRED: ReasonCode.REGISTRATION_EXPIRED,
    RegistrationStatus.SUSPENDED: ReasonCode.REGISTRATION_SUSPENDED,
    RegistrationStatus.REVOKED: ReasonCode.REGISTRATION_REVOKED,
}


class WeightedRuleScorer:
    version = "weighted-rules-v2"  # v2: published alerts are a hard rule

    def __init__(
        self,
        *,
        suspicious_threshold: float = 0.3,
        high_priority_threshold: float = 0.6,
        min_field_confidence: float = 0.6,
    ) -> None:
        if not 0 < suspicious_threshold < high_priority_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 < suspicious < high priority <= 1")
        self._suspicious = suspicious_threshold
        self._high = high_priority_threshold
        self._min_confidence = min_field_confidence

    def score(self, facts: ScoreInput) -> Score:
        if facts.quality_reasons:
            return self._result(
                Verdict.INCONCLUSIVE, None, None, facts.quality_reasons, high_priority=False
            )
        if not facts.product_identified:
            return self._unidentified(facts)

        reasons: list[ReasonCode] = []
        floor = 0.0
        if facts.number_name_conflict:
            reasons.append(ReasonCode.NAFDAC_NUMBER_MISMATCH)
            floor = self._high
        if facts.registration_status is RegistrationStatus.REVOKED:
            floor = max(floor, self._high)
        if facts.registration_status in _REGISTRATION_REASONS:
            reasons.append(_REGISTRATION_REASONS[facts.registration_status])
        if facts.pack_expired:
            reasons.append(ReasonCode.PACK_EXPIRED)
            floor = max(floor, self._suspicious)
        alert = facts.signals.get(SignalName.COUNTERFEIT_ALERT, UNAVAILABLE)
        if alert.value is not None and alert.value >= REASON_LEVEL:
            reasons.append(ReasonCode.PUBLISHED_ALERT)
            floor = max(floor, self._high)

        signals = facts.signals
        weighted = sum(
            WEIGHTS_V1[name] * (signal.value or 0.0)
            for name, signal in signals.items()
            if signal.available
        )
        for name, reason in _SIGNAL_REASONS.items():
            signal = signals.get(name, UNAVAILABLE)
            if signal.value is not None and signal.value >= REASON_LEVEL:
                reasons.append(reason)
        risk = round(min(1.0, max(weighted, floor)), 4)
        verdict = Verdict.SUSPICIOUS if risk >= self._suspicious else Verdict.LIKELY_GENUINE
        return self._result(
            verdict,
            risk,
            round(self._confidence(facts), 4),
            tuple(reasons),
            high_priority=risk >= self._high,
            pack_expired=facts.pack_expired,
        )

    def _unidentified(self, facts: ScoreInput) -> Score:
        """No registered product matched. A clearly read number means UNREGISTERED; otherwise
        nothing usable was read and the photo should be retaken."""
        read = facts.nafdac_confidence
        if read is not None and read >= self._min_confidence:
            return self._result(
                Verdict.UNREGISTERED,
                1.0,
                round(read * 0.9, 4),
                (ReasonCode.NOT_IN_REGISTRY,),
                high_priority=True,
            )
        return self._result(
            Verdict.INCONCLUSIVE, None, None, (ReasonCode.UNREADABLE,), high_priority=False
        )

    def _confidence(self, facts: ScoreInput) -> float:
        """Extraction confidence, discounted when signals were unavailable:
        confidence = extraction * (0.4 + 0.6 * coverage), coverage = available weight / total."""
        total = sum(WEIGHTS_V1.values())
        available = sum(
            WEIGHTS_V1[name] for name, signal in facts.signals.items() if signal.available
        )
        coverage = available / total
        return max(0.0, min(1.0, facts.extraction_confidence * (0.4 + 0.6 * coverage)))

    def _result(
        self,
        verdict: Verdict,
        risk: float | None,
        confidence: float | None,
        reasons: tuple[ReasonCode, ...],
        *,
        high_priority: bool,
        pack_expired: bool = False,
    ) -> Score:
        return Score(
            verdict=verdict,
            risk_score=risk,
            confidence=confidence,
            high_priority=high_priority,
            reason_codes=reasons,
            advice_codes=_advice(verdict, high_priority=high_priority, pack_expired=pack_expired),
            scorer_version=self.version,
        )


def _advice(verdict: Verdict, *, high_priority: bool, pack_expired: bool) -> tuple[AdviceCode, ...]:
    if verdict is Verdict.INCONCLUSIVE:
        return (AdviceCode.RETAKE_PHOTO, AdviceCode.ASK_A_PHARMACIST)
    if verdict is Verdict.UNREGISTERED:
        return (AdviceCode.DO_NOT_USE, AdviceCode.KEEP_THE_PACK, AdviceCode.REPORT_TO_NAFDAC)
    if verdict is Verdict.SUSPICIOUS:
        advice = [AdviceCode.DO_NOT_USE] if high_priority or pack_expired else []
        advice.append(AdviceCode.ASK_A_PHARMACIST)
        if high_priority:
            advice += [AdviceCode.KEEP_THE_PACK, AdviceCode.REPORT_TO_NAFDAC]
        return tuple(advice)
    return (AdviceCode.CHECK_EXPIRY_DATE,)

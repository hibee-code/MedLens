"""WeightedRuleScorer v1: a table of signal combinations and the verdicts they must give."""

import pytest

from medlens.modules.products.models import RegistrationStatus
from medlens.modules.scans.models import Verdict
from medlens.modules.verification.codes import AdviceCode, ReasonCode
from medlens.modules.verification.scoring import (
    WEIGHTS_V1,
    ScoreInput,
    Signal,
    SignalName,
    WeightedRuleScorer,
)

SCORER = WeightedRuleScorer()
ACTIVE = RegistrationStatus.ACTIVE
# As the pipeline derives it from the registration status.
REGISTRATION_RISK = {
    RegistrationStatus.ACTIVE: 0.0,
    RegistrationStatus.EXPIRED: 0.5,
    RegistrationStatus.SUSPENDED: 1.0,
    RegistrationStatus.REVOKED: 1.0,
}


def identified(
    *,
    status: RegistrationStatus = ACTIVE,
    pack_expired: bool = False,
    conflict: bool = False,
    extraction: float = 0.9,
    **values: float | None,
) -> ScoreInput:
    signals = {SignalName(name): Signal(value) for name, value in values.items()}
    signals.setdefault(SignalName.REGISTRATION, Signal(REGISTRATION_RISK[status]))
    return ScoreInput(
        product_identified=True,
        nafdac_confidence=0.9,
        number_name_conflict=conflict,
        registration_status=status,
        pack_expired=pack_expired,
        signals=signals,
        extraction_confidence=extraction,
    )


@pytest.mark.parametrize(
    ("facts", "verdict", "high_priority", "reasons"),
    [
        # Genuine-looking packs
        (identified(name_mismatch=0.0, visual_mismatch=0.1), Verdict.LIKELY_GENUINE, False, ()),
        (
            identified(batch_spread=0.5),
            Verdict.LIKELY_GENUINE,
            False,
            (ReasonCode.BATCH_SEEN_WIDELY,),
        ),
        # Registration problems
        (
            identified(status=RegistrationStatus.EXPIRED),
            Verdict.LIKELY_GENUINE,
            False,
            (ReasonCode.REGISTRATION_EXPIRED,),
        ),
        (
            identified(status=RegistrationStatus.SUSPENDED),
            Verdict.SUSPICIOUS,
            False,
            (ReasonCode.REGISTRATION_SUSPENDED,),
        ),
        (
            identified(status=RegistrationStatus.REVOKED),
            Verdict.SUSPICIOUS,
            True,
            (ReasonCode.REGISTRATION_REVOKED,),
        ),
        # Hard rules
        (identified(pack_expired=True), Verdict.SUSPICIOUS, False, (ReasonCode.PACK_EXPIRED,)),
        (identified(conflict=True), Verdict.SUSPICIOUS, True, (ReasonCode.NAFDAC_NUMBER_MISMATCH,)),
        # Weighted signals
        (
            identified(visual_mismatch=1.0),
            Verdict.SUSPICIOUS,
            False,
            (ReasonCode.VISUAL_MISMATCH,),
        ),
        (
            identified(batch_not_declared=1.0, expiry_mismatch=1.0),
            Verdict.SUSPICIOUS,
            True,
            (ReasonCode.BATCH_NOT_DECLARED, ReasonCode.EXPIRY_MISMATCH),
        ),
        (
            identified(batch_spread=1.0, outside_distribution=1.0, name_mismatch=1.0),
            Verdict.SUSPICIOUS,
            True,
            (
                ReasonCode.NAME_MISMATCH,
                ReasonCode.BATCH_SEEN_WIDELY,
                ReasonCode.OUTSIDE_DISTRIBUTION,
            ),
        ),
        (
            identified(outside_distribution=1.0),
            Verdict.LIKELY_GENUINE,
            False,
            (ReasonCode.OUTSIDE_DISTRIBUTION,),
        ),
        # Not identified
        (
            ScoreInput(nafdac_confidence=0.9),
            Verdict.UNREGISTERED,
            True,
            (ReasonCode.NOT_IN_REGISTRY,),
        ),
        (ScoreInput(nafdac_confidence=0.4), Verdict.INCONCLUSIVE, False, (ReasonCode.UNREADABLE,)),
        (ScoreInput(), Verdict.INCONCLUSIVE, False, (ReasonCode.UNREADABLE,)),
        # Bad photo
        (
            ScoreInput(quality_reasons=(ReasonCode.BLURRY, ReasonCode.TOO_DARK)),
            Verdict.INCONCLUSIVE,
            False,
            (ReasonCode.BLURRY, ReasonCode.TOO_DARK),
        ),
    ],
)
def test_verdict_table(
    facts: ScoreInput, verdict: Verdict, high_priority: bool, reasons: tuple[ReasonCode, ...]
) -> None:
    score = SCORER.score(facts)

    assert score.verdict is verdict
    assert score.high_priority is high_priority
    assert set(score.reason_codes) == set(reasons)
    assert score.scorer_version == "weighted-rules-v2"


def test_registration_risk_comes_from_its_signal() -> None:
    score = SCORER.score(identified(status=RegistrationStatus.SUSPENDED))

    assert score.risk_score == pytest.approx(WEIGHTS_V1[SignalName.REGISTRATION])
    assert score.verdict is Verdict.SUSPICIOUS


def test_risk_is_clipped_to_one() -> None:
    score = SCORER.score(identified(**dict.fromkeys(WEIGHTS_V1, 1.0)))

    assert score.risk_score == 1.0


def test_confidence_grows_with_coverage() -> None:
    sparse = SCORER.score(identified())
    full = SCORER.score(identified(**dict.fromkeys(WEIGHTS_V1, 0.0)))

    assert sparse.confidence is not None
    assert full.confidence is not None
    assert sparse.confidence < full.confidence
    assert full.confidence == pytest.approx(0.9)  # extraction x (0.4 + 0.6 x 1)


def test_inconclusive_has_no_score() -> None:
    score = SCORER.score(ScoreInput(quality_reasons=(ReasonCode.TOO_SMALL,)))

    assert (score.risk_score, score.confidence) == (None, None)
    assert score.advice_codes[0] is AdviceCode.RETAKE_PHOTO


def test_advice_follows_the_verdict() -> None:
    expired = SCORER.score(identified(pack_expired=True))
    revoked = SCORER.score(identified(status=RegistrationStatus.REVOKED))
    genuine = SCORER.score(identified())
    unregistered = SCORER.score(ScoreInput(nafdac_confidence=0.95))

    assert AdviceCode.DO_NOT_USE in expired.advice_codes
    assert AdviceCode.REPORT_TO_NAFDAC not in expired.advice_codes
    assert AdviceCode.REPORT_TO_NAFDAC in revoked.advice_codes
    assert genuine.advice_codes == (AdviceCode.CHECK_EXPIRY_DATE,)
    assert unregistered.advice_codes == (
        AdviceCode.DO_NOT_USE,
        AdviceCode.KEEP_THE_PACK,
        AdviceCode.REPORT_TO_NAFDAC,
    )


def test_thresholds_are_validated() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        WeightedRuleScorer(suspicious_threshold=0.7, high_priority_threshold=0.6)


def test_custom_thresholds_apply() -> None:
    strict = WeightedRuleScorer(suspicious_threshold=0.1, high_priority_threshold=0.2)

    assert strict.score(identified(outside_distribution=1.0)).verdict is Verdict.SUSPICIOUS


def test_the_documented_weights_match_the_code() -> None:
    from pathlib import Path

    docs = (Path(__file__).resolve().parents[2] / "docs" / "scoring.md").read_text()
    for name, weight in WEIGHTS_V1.items():
        assert f"| `{name.value}` | {weight:.2f} |" in docs, name


def test_settings_build_the_scorer_and_pipeline(settings_factory) -> None:  # type: ignore[no-untyped-def]
    from medlens.modules.verification.service import pipeline_config, scorer_for

    settings = settings_factory(
        verification={"suspicious_threshold": 0.2, "high_priority_threshold": 0.5},
        llm={"enabled": True},
    )

    config = pipeline_config(settings)
    scorer = scorer_for(settings)

    assert config.llm_enabled is True
    assert config.spread_state_limit == 3
    assert (
        scorer.score(identified(outside_distribution=1.0, batch_spread=0.25)).verdict
        is Verdict.SUSPICIOUS
    )

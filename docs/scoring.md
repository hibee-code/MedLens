# Risk scoring

How a pack photo becomes a verdict. The scorer is **rule-based** (`WeightedRuleScorer`, version
`weighted-rules-v2`) because MedLens has no labelled lab results yet. It is not a trained
model and has not been validated against laboratory results. Its weights are expert
judgement, not fitted values. Once regulators' lab results (stored as labels by case review)
are numerous enough, a calibrated LightGBM scorer will implement the same `RiskScorer`
interface with its own version string. Every scan stores the scorer version and the full signal
vector, so any scan can be re-scored later.

The verdict is deterministic and explainable (CLAUDE.md §5). A vision LLM may help *read* a
hard photo; its output is schema-validated, normalised like user input, capped at 0.8
confidence, and never decides the verdict.

Code: `src/medlens/modules/verification/` (`pipeline.py`, `scoring.py`, `extraction.py`).

## Verdicts

| Verdict | Meaning | Advice codes |
|---|---|---|
| `LIKELY_GENUINE` | Risk score below the suspicious threshold. Never "safe" or "genuine". | `CHECK_EXPIRY_DATE` |
| `SUSPICIOUS` | A hard rule fired, or the risk score reached the suspicious threshold. High priority at the high-priority threshold. | `ASK_A_PHARMACIST`; with high priority or an expired pack also `DO_NOT_USE`; with high priority also `KEEP_THE_PACK`, `REPORT_TO_NAFDAC` |
| `UNREGISTERED` | A NAFDAC number was read clearly (confidence ≥ `min_field_confidence`), and no registered product matches it, a close number or the names on the pack. | `DO_NOT_USE`, `KEEP_THE_PACK`, `REPORT_TO_NAFDAC` |
| `INCONCLUSIVE` | The photo failed the quality gate, or nothing identifying could be read. No risk score. | `RETAKE_PHOTO`, `ASK_A_PHARMACIST` |

Reason and advice codes are stable identifiers (`verification/codes.py`); clients translate
them.

## Order of evaluation

1. **Quality gate.** A photo whose shorter side is under `min_side_pixels` (`TOO_SMALL`), whose
   Laplacian variance is under `blur_threshold` (`BLURRY`), or whose mean grey level is under
   `dark_threshold` (`TOO_DARK`) or over `bright_threshold` (`TOO_BRIGHT`) is INCONCLUSIVE, and
   the scan is `rejected`.
2. **Identification.** Exact NAFDAC number, then a close number (trigram similarity ≥ 0.6
   **and** the product's brand name on the pack, to absorb OCR errors), then a brand name found
   in the pack text (word similarity ≥ 0.8). Unidentified: UNREGISTERED or INCONCLUSIVE, as
   above.
3. **Hard rules.** They set a floor under the risk score:
   * the number on the pack belongs to a different product than the one named on it
     (`NAFDAC_NUMBER_MISMATCH`): floor at the high-priority threshold;
   * registration revoked (`REGISTRATION_REVOKED`): floor at the high-priority threshold;
   * the pack's expiry date (read from the pack, else the declared batch's) has passed
     (`PACK_EXPIRED`): floor at the suspicious threshold;
   * a published NAFDAC alert or recall in the knowledge base names the product, either for
     every batch or for the batch on the pack (`PUBLISHED_ALERT`): floor at the high-priority
     threshold (added in v2).
4. **Weighted signals.** Each available signal has a value in 0..1:

   `risk_score = min(1, max(floor, Σ weightᵢ × valueᵢ))` over available signals.

5. **Thresholds** (`MEDLENS_VERIFICATION__*`): `risk < 0.3` LIKELY_GENUINE; `0.3 ≤ risk < 0.6`
   SUSPICIOUS; `risk ≥ 0.6` SUSPICIOUS with high priority, which opens a review case.

## Signals (v2)

| Signal | Weight | Value | Reason code (value ≥ 0.5) |
|---|---|---|---|
| `registration` | 0.35 | active 0, registration expired 0.5, suspended or revoked 1 | `REGISTRATION_*` (always listed when not active) |
| `name_mismatch` | 0.30 | 1 − share of the brand name's distinctive words found on the pack | `NAME_MISMATCH` |
| `batch_not_declared` | 0.30 | 1 when the manufacturer declares batches and this batch is not one of them | `BATCH_NOT_DECLARED` |
| `expiry_mismatch` | 0.35 | 1 when the printed expiry month differs from the declared one | `EXPIRY_MISMATCH` |
| `batch_spread` | 0.40 | 0 up to `spread_state_limit` states within `spread_window_days`, then rising to 1 at twice the limit | `BATCH_SEEN_WIDELY` |
| `outside_distribution` | 0.15 | 1 when the scan's state is not among the batch's declared distribution states | `OUTSIDE_DISTRIBUTION` |
| `visual_mismatch` | 0.35 | 0 at cosine similarity ≥ 0.85 to the best **approved** reference image, 1 at ≤ 0.60, linear between | `VISUAL_MISMATCH` |
| `counterfeit_alert` | 0.60 | 1 when an ingested alert or recall names the product (all batches) or its batch, else 0 | `PUBLISHED_ALERT` (also a hard rule) |

A signal that cannot be computed is **unavailable**: it adds nothing to the risk and lowers
confidence. For example, there are no reference images yet, the manufacturer declares no
batches, or the batch number is unreadable. A registration that has merely lapsed is a weighted
signal, not a hard rule: about a third of the Greenbook's registrations are past their expiry
date while products stay on sale during renewal.

## Confidence

    extraction = 0.6 × identification + 0.2 × batch-number confidence + 0.2 × expiry confidence
    coverage   = Σ weights of available signals / Σ all weights
    confidence = extraction × (0.4 + 0.6 × coverage)

`identification` is the NAFDAC number's read confidence (scaled by the number similarity for
close matches), or the name match score. A missing field counts as 0. For UNREGISTERED,
confidence is 0.9 × the number's read confidence. INCONCLUSIVE has no confidence.

Field confidences: barcode reads 0.98; OCR uses the engine's mean word confidence, × 0.7 for a
number without a "NAFDAC/REG NO" label and × 0.85 when digits needed OCR corrections; a number
typed by the person scanning 0.5; vision-LLM fields at most 0.8.

## Duplicates and probing

The same photo (SHA-256) from the same person or device within `duplicate_window_hours` returns
the earlier scan. A near-identical photo (perceptual hash within 6 bits) gets the earlier
answer too, so probing the checks with slightly edited photos teaches nothing.

## Published alerts

Alerts and recalls uploaded to the knowledge base (`POST /api/v1/knowledge/documents`, kind
`alert` or `recall`) are chunked by the worker; every labelled NAFDAC number in a chunk becomes a
published alert, paired with the labelled batch numbers in the same chunk (a chunk naming no
batch flags every batch of the product). Batch values without a digit are ignored, so prose
after the word "batch" is not mistaken for a batch number.

## Cases and overrides

A SUSPICIOUS scan with high priority, or an UNREGISTERED scan, opens one review case. It goes to
the active regulator whose jurisdiction covers the scan's state, preferring the most local one;
without a state it goes to a national regulator. Lab results recorded on cases become labels:
the training data for the next scorer.

The pipeline's verdict is stored once and never changed. A reviewer recording a lab result may
also set `override_verdict`: a counterfeit result allows SUSPICIOUS or UNREGISTERED, a genuine
result allows LIKELY_GENUINE (never "safe"), an inconclusive one allows nothing. The override is
what clients see as `verdict` (with `verdict_overridden: true`); regulators also see the
pipeline's verdict (`analysis.pipeline_verdict`) and the deciding case. Every override is
audited (`scan.verdict_overridden`). Hotspots and statistics count the verdict shown. Nothing
else, including the assistant, can change a verdict.

## Changes

* **v2** (2026-11-03): published alerts are a hard rule (`PUBLISHED_ALERT`); the
  `counterfeit_alert` signal is computed for every identified pack.
* **v1**: initial weights and rules.

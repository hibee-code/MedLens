from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import ConfigDict, Field

from medlens.core.identifiers import NigerianState
from medlens.core.schemas import ReadModel
from medlens.modules.scans.models import ScanStatus, Verdict

_ID = "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e80"

DISCLAIMER_CODE = "AUTOMATED_ESTIMATE_NOT_A_LAB_TEST"
DISCLAIMER = (
    "This is an automated risk estimate from a photo, not a laboratory test. It cannot prove "
    "that a medicine is genuine or safe. If in doubt, ask a pharmacist or report the pack to "
    "NAFDAC."
)


class ScanAccepted(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": _ID,
                    "status": "received",
                    "status_url": f"/api/v1/scans/{_ID}",
                    "estimated_wait_seconds": 10,
                    "duplicate": False,
                }
            ]
        }
    )

    id: UUID
    status: ScanStatus
    status_url: str
    estimated_wait_seconds: int = Field(description="Rough time until the result is ready.")
    duplicate: bool = Field(
        description="The same photo was submitted recently: this is the earlier scan."
    )
    result: "ScanRead | None" = Field(
        default=None,
        description=(
            "The finished scan, when the sync fast path is on and the verdict was ready within "
            "the wait (the response is then 200)."
        ),
    )


class ProductSummary(ReadModel):
    nafdac_number: str
    brand_name: str
    generic_name: str


class ScanAnalysis(ReadModel):
    """Detailed evidence, for regulators and platform admins only (scan:read_signals)."""

    risk_score: float | None
    scorer_version: str | None
    signals: dict[str, Any] | None
    extracted: dict[str, Any] | None
    quality: dict[str, Any] | None
    state: NigerianState | None
    geocell: str | None
    processing_ms: int | None
    error_code: str | None
    llm_used: bool = Field(description="The vision LLM helped read the pack.")
    pipeline_verdict: Verdict | None = Field(
        description="The verdict the pipeline gave, before any review override."
    )
    override_case_id: UUID | None = Field(description="The case whose decision overrode it.")


class ScanRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": _ID,
                    "status": "completed",
                    "verdict": "SUSPICIOUS",
                    "confidence": 0.71,
                    "reason_codes": ["BATCH_NOT_DECLARED", "EXPIRY_MISMATCH"],
                    "advice_codes": ["DO_NOT_USE", "ASK_A_PHARMACIST"],
                    "verdict_overridden": False,
                    "product": {
                        "nafdac_number": "04-0005",
                        "brand_name": "Panadol Extra Caplet",
                        "generic_name": "Paracetamol, Caffeine",
                    },
                    "created_at": "2026-11-03T09:00:00Z",
                    "updated_at": "2026-11-03T09:00:04Z",
                    "analysis": None,
                    "disclaimer_code": DISCLAIMER_CODE,
                    "disclaimer": DISCLAIMER,
                }
            ]
        }
    )

    id: UUID
    status: ScanStatus
    verdict: Verdict | None = Field(
        description=(
            "LIKELY_GENUINE, SUSPICIOUS, UNREGISTERED or INCONCLUSIVE; null until done. A "
            "reviewer's decision after a lab result replaces the pipeline's verdict."
        )
    )
    verdict_overridden: bool = Field(
        default=False, description="A reviewer changed the verdict after a lab result."
    )
    confidence: float | None = Field(description="0-1: how much evidence the verdict rests on.")
    reason_codes: list[str] = Field(description="Why; stable codes the client translates.")
    advice_codes: list[str] = Field(description="What to do next; stable codes.")
    product: ProductSummary | None = Field(description="The registered product it matched.")
    created_at: datetime
    updated_at: datetime
    analysis: ScanAnalysis | None = Field(
        default=None, description="Present only for callers allowed to see the evidence."
    )
    disclaimer_code: str = Field(
        default=DISCLAIMER_CODE, description="Stable code for the disclaimer; clients translate."
    )
    disclaimer: str = Field(default=DISCLAIMER, description="The disclaimer, in English.")


ScanAccepted.model_rebuild()

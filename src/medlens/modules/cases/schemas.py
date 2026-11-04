from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import ConfigDict, Field

from medlens.core.identifiers import NigerianState
from medlens.core.schemas import InputModel, ReadModel, SafeText, UpdateModel
from medlens.modules.cases.models import CasePriority, CaseStatus
from medlens.modules.cases.service import ReviewStatus
from medlens.modules.hitl.models import LabelOutcome
from medlens.modules.scans.models import Verdict


class CaseScan(ReadModel):
    id: UUID
    verdict: Verdict | None = Field(description="The pipeline's verdict.")
    verdict_override: Verdict | None = Field(
        default=None, description="The verdict set by this or another case decision, if any."
    )
    reason_codes: list[str]
    signals: dict[str, Any] | None = Field(default=None, description="The scorer's evidence.")
    extracted: dict[str, Any] | None = Field(default=None, description="Fields read off the pack.")
    has_image: bool = Field(
        default=False, description="The photo is still kept: GET /cases/{id}/image."
    )
    created_at: datetime


class CaseRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e90",
                    "status": "open",
                    "priority": "high",
                    "state": "NG-KN",
                    "assigned_org_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e12",
                    "assigned_to_user_id": None,
                    "notes": None,
                    "lab_result": None,
                    "scan": {
                        "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e80",
                        "verdict": "UNREGISTERED",
                        "reason_codes": ["NOT_IN_REGISTRY"],
                        "created_at": "2026-11-03T09:00:00Z",
                    },
                    "created_at": "2026-11-03T09:00:05Z",
                    "updated_at": "2026-11-03T09:00:05Z",
                    "resolved_at": None,
                }
            ]
        }
    )

    id: UUID
    status: CaseStatus
    priority: CasePriority
    state: NigerianState | None
    assigned_org_id: UUID | None
    assigned_to_user_id: UUID | None
    notes: str | None
    lab_result: dict[str, Any] | None
    scan: CaseScan
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None


class CaseUpdate(UpdateModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"status": "closed", "notes": "Duplicate of an open case."}]
        }
    )

    status: ReviewStatus | None = Field(
        default=None,
        description="in_review, inconclusive or closed. Confirmed outcomes come from lab results.",
    )
    notes: SafeText | None = None


class CaseAssignment(InputModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"user_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e31"}]}
    )

    user_id: UUID


class LabResult(InputModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "outcome": "counterfeit",
                    "lab_reference": "NAFDAC-LAB/KN/2026/0412",
                    "notes": "No active ingredient detected.",
                    "override_verdict": "SUSPICIOUS",
                }
            ]
        }
    )

    outcome: LabelOutcome
    lab_reference: str | None = Field(default=None, max_length=200)
    notes: SafeText | None = None
    override_verdict: Verdict | None = Field(
        default=None,
        description=(
            "Replace the verdict shown for the scan (audited; the pipeline's verdict is kept). "
            "counterfeit allows SUSPICIOUS or UNREGISTERED, genuine allows LIKELY_GENUINE, "
            "inconclusive allows none."
        ),
    )

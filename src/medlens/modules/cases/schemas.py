from datetime import datetime
from typing import Any, Self
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator

from medlens.core.identifiers import NigerianState
from medlens.core.schemas import InputModel, ReadModel, SafeText, UpdateModel
from medlens.modules.cases.models import CasePriority, CaseStatus
from medlens.modules.cases.service import ReviewStatus
from medlens.modules.hitl.models import LabelOutcome
from medlens.modules.scans.models import Verdict


class CaseScan(ReadModel):
    id: UUID
    pipeline_verdict: Verdict | None = Field(description="The scorer's verdict (never changes).")
    final_verdict: Verdict | None = Field(description="What clients see.")
    reviewed: bool = Field(default=False, description="A case decision set the final verdict.")
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
                        "pipeline_verdict": "UNREGISTERED",
                        "final_verdict": "UNREGISTERED",
                        "reviewed": False,
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
                    "override_note": "Lab found no artemether; counterfeit confirmed.",
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
            "Set the scan's final verdict (audited, reversible; the pipeline's verdict is "
            "kept). counterfeit allows SUSPICIOUS or UNREGISTERED, genuine allows "
            "LIKELY_GENUINE, inconclusive allows none. Requires override_note."
        ),
    )
    override_note: SafeText | None = Field(default=None, min_length=5, max_length=2000)

    @model_validator(mode="after")
    def _note_with_override(self) -> Self:
        if self.override_verdict is not None and not self.override_note:
            raise ValueError("override_note is required with override_verdict")
        return self


class VerdictOverride(InputModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"verdict": "SUSPICIOUS", "note": "Second lab confirmed the counterfeit."}]
        }
    )

    verdict: Verdict
    note: SafeText = Field(min_length=5, max_length=2000, description="Why (required).")


class OverrideRevert(InputModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"note": "Lab sample was mislabelled; decision withdrawn."}]
        }
    )

    note: SafeText = Field(min_length=5, max_length=2000, description="Why (required).")

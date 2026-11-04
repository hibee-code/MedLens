from datetime import date, datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, JsonValue, model_validator

from medlens.core.identifiers import NigerianState
from medlens.core.schemas import InputModel, NafdacNumber, ReadModel, SafeText
from medlens.modules.products.models import ImageSide, RegistrationStatus, ReviewStatus

_PRODUCT_EXAMPLE: dict[str, JsonValue] = {
    "nafdac_number": "04-0005",
    "brand_name": "Panadol Extra Caplet",
    "generic_name": "Paracetamol, Caffeine",
    "strength": "500 mg/65 mg",
    "dosage_form": "Tablet",
    "pack_size": "1 x 10",
    "manufacturer_name": "GlaxoSmithKline Consumer Nigeria PLC (GSK)",
    "registration_status": "active",
    "registration_expires_on": "2029-01-31",
    "manufacturer_onboarded": True,
    "match_score": None,
}


class ProductRead(ReadModel):
    model_config = ConfigDict(json_schema_extra={"examples": [_PRODUCT_EXAMPLE]})

    nafdac_number: str
    brand_name: str
    generic_name: str
    strength: str | None
    dosage_form: str | None
    pack_size: str | None
    manufacturer_name: str | None = Field(description="Registration holder, as in the register.")
    registration_status: RegistrationStatus
    registration_expires_on: date | None
    manufacturer_onboarded: bool = Field(
        description="The registration holder uses MedLens and declares batches for this product."
    )
    match_score: float | None = Field(
        default=None,
        description="For searches: name similarity 0-1 (1 for an exact NAFDAC number).",
    )


# ------------------------------------------------------------------ batches
_BATCH_FIELDS: dict[str, JsonValue] = {
    "manufactured_on": "2025-01-15",
    "expires_on": "2027-12-31",
    "quantity": 120000,
    "distribution_states": ["NG-LA", "NG-OG"],
}


class BatchWrite(InputModel):
    model_config = ConfigDict(json_schema_extra={"examples": [dict(_BATCH_FIELDS)]})

    manufactured_on: date | None = None
    expires_on: date
    quantity: Annotated[int, Field(ge=0)] | None = None
    distribution_states: list[NigerianState] = Field(
        default_factory=list,
        max_length=37,
        description="States the batch was shipped to (ISO codes or names). Empty: unknown.",
    )

    @model_validator(mode="after")
    def _dates(self) -> "BatchWrite":
        if self.manufactured_on and self.expires_on <= self.manufactured_on:
            raise ValueError("expires_on must be after manufactured_on")
        return self


class BatchRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"batch_number": "B2309-14", **_BATCH_FIELDS, "created_at": "2026-11-03T09:00:00Z"}
            ]
        }
    )

    batch_number: str
    manufactured_on: date | None
    expires_on: date
    quantity: int | None
    distribution_states: list[str]
    created_at: datetime


class BatchImportResult(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"rows": 250, "created": 240, "updated": 10}]}
    )

    rows: int
    created: int
    updated: int


# ------------------------------------------------------------------ reference images
class ReferenceImageRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e40",
                    "nafdac_number": "04-0005",
                    "side": "front",
                    "status": "pending",
                    "created_at": "2026-11-03T09:00:00Z",
                    "reviewed_at": None,
                }
            ]
        }
    )

    id: UUID
    nafdac_number: str
    side: ImageSide
    status: ReviewStatus
    created_at: datetime
    reviewed_at: datetime | None


class ReviewDecision(InputModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"decision": "approve", "note": "Matches the dossier."}]}
    )

    decision: Literal["approve", "reject"]
    note: SafeText | None = None

    @property
    def status(self) -> ReviewStatus:
        return ReviewStatus.APPROVED if self.decision == "approve" else ReviewStatus.REJECTED


# ------------------------------------------------------------------ product claims
class ClaimCreate(InputModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "nafdac_number": "04-0005",
                    "note": "We hold this registration; certificate no. 12345.",
                }
            ]
        }
    )

    nafdac_number: NafdacNumber
    note: SafeText | None = None


class ClaimRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e50",
                    "organization_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e12",
                    "nafdac_number": "04-0005",
                    "status": "pending",
                    "note": "We hold this registration; certificate no. 12345.",
                    "review_note": None,
                    "created_at": "2026-11-03T09:00:00Z",
                    "reviewed_at": None,
                }
            ]
        }
    )

    id: UUID
    organization_id: UUID
    nafdac_number: str
    status: ReviewStatus
    note: str | None
    review_note: str | None
    created_at: datetime
    reviewed_at: datetime | None

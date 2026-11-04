from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import ConfigDict, Field

from medlens.core.identifiers import NigerianState
from medlens.core.schemas import ReadModel


class StateCount(ReadModel):
    state: NigerianState | None = Field(description="null: the client sent no state.")
    total: int
    flagged: int = Field(description="SUSPICIOUS or UNREGISTERED.")


class DayCount(ReadModel):
    day: date
    total: int
    flagged: int


class LLMUsage(ReadModel):
    scans_needing_llm: int = Field(description="Completed scans the vision LLM helped read.")
    share_of_scans: float = Field(description="scans_needing_llm / total_scans (0 when none).")
    calls: int = Field(description="All metered LLM calls in the window (every feature).")
    cost_usd: Decimal


class StatsSummary(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "since": "2026-10-04T09:00:00Z",
                    "until": "2026-11-03T09:00:00Z",
                    "total_scans": 1240,
                    "by_verdict": {
                        "LIKELY_GENUINE": 1010,
                        "SUSPICIOUS": 96,
                        "UNREGISTERED": 41,
                        "INCONCLUSIVE": 93,
                    },
                    "by_state": [{"state": "NG-LA", "total": 702, "flagged": 80}],
                    "by_day": [{"day": "2026-11-02", "total": 51, "flagged": 6}],
                    "llm": {
                        "scans_needing_llm": 310,
                        "share_of_scans": 0.25,
                        "calls": 322,
                        "cost_usd": "0.091200",
                    },
                }
            ]
        }
    )

    since: datetime
    until: datetime
    total_scans: int = Field(description="Completed scans you may see in the window.")
    by_verdict: dict[str, int] = Field(description="The verdict shown, after case decisions.")
    by_state: list[StateCount]
    by_day: list[DayCount] = Field(description="UTC days.")
    llm: LLMUsage


class ProductWeekCell(ReadModel):
    product_id: UUID
    nafdac_number: str
    brand_name: str
    state: NigerianState | None
    iso_week: str = Field(description="ISO 8601 week, e.g. 2026-W44.")
    flagged_scans: int


class ManufacturerReportRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "organization_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e12",
                    "since": "2026-08-11T09:00:00Z",
                    "until": "2026-11-03T09:00:00Z",
                    "min_cell_size": 5,
                    "suppressed_cells": 3,
                    "cells": [
                        {
                            "product_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e11",
                            "nafdac_number": "04-0289",
                            "brand_name": "Emzor Paracetamol Syrup",
                            "state": "NG-KN",
                            "iso_week": "2026-W44",
                            "flagged_scans": 7,
                        }
                    ],
                }
            ]
        }
    )

    organization_id: UUID
    since: datetime
    until: datetime
    min_cell_size: int = Field(description="Cells with fewer flagged scans are left out.")
    suppressed_cells: int = Field(description="How many cells were left out.")
    cells: list[ProductWeekCell]

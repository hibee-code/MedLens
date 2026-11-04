from datetime import date, datetime
from decimal import Decimal

from pydantic import ConfigDict, Field

from medlens.core.schemas import ReadModel


class UsageDay(ReadModel):
    day: date
    route: str = Field(description="pack_extraction or assistant_answer.")
    calls: int
    failed: int = Field(description="Refused, invalid output or provider errors.")
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: Decimal


class Budget(ReadModel):
    daily_usd: Decimal
    spent_today_usd: Decimal
    remaining_usd: Decimal
    share_used: float = Field(description="0-1; calls stop at 1 until 00:00 UTC.")


class AIUsage(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ai_enabled": True,
                    "assistant_enabled": True,
                    "models": {"fast": "claude-haiku-5-5", "balanced": "claude-sonnet-5-5"},
                    "budget": {
                        "daily_usd": "5.00",
                        "spent_today_usd": "0.412300",
                        "remaining_usd": "4.587700",
                        "share_used": 0.0825,
                    },
                    "since": "2026-10-28T00:00:00Z",
                    "until": "2026-11-04T00:00:00Z",
                    "days": [
                        {
                            "day": "2026-11-03",
                            "route": "pack_extraction",
                            "calls": 310,
                            "failed": 2,
                            "input_tokens": 520000,
                            "output_tokens": 61000,
                            "cache_read_tokens": 280000,
                            "cost_usd": "0.085300",
                        }
                    ],
                }
            ]
        }
    )

    ai_enabled: bool = Field(description="MEDLENS_LLM__ENABLED, the global kill switch.")
    assistant_enabled: bool = Field(description="MEDLENS_KNOWLEDGE__ASSISTANT_ENABLED.")
    models: dict[str, str] = Field(description="The model behind each tier.")
    budget: Budget
    since: datetime
    until: datetime
    days: list[UsageDay]

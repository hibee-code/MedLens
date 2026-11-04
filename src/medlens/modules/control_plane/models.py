from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from medlens.db.base import Base, CreatedAt, UUIDv7PrimaryKey
from medlens.db.types import str_enum


class LLMCallOutcome(StrEnum):
    OK = "ok"
    INVALID_OUTPUT = "invalid_output"  # still invalid after the repair attempt
    REFUSED = "refused"
    ERROR = "error"


class LLMCall(UUIDv7PrimaryKey, CreatedAt, Base):
    """One metered model call (CLAUDE.md §5: every AI output is traceable and costed).
    Written by the LLM gateway only; the daily budget sums ``cost_usd``."""

    __tablename__ = "llm_calls"
    __table_args__ = (
        Index("ix_llm_calls_created_at", "created_at"),
        Index("ix_llm_calls_route_created_at", "route", "created_at"),
    )
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012

    route: Mapped[str] = mapped_column(String(64))
    tier: Mapped[str] = mapped_column(String(16))
    requested_model: Mapped[str] = mapped_column(String(100))
    model: Mapped[str] = mapped_column(String(100))  # the model that served (refusal fallback)
    prompt_version: Mapped[str] = mapped_column(String(64))
    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    cache_read_tokens: Mapped[int]
    cache_write_tokens: Mapped[int]
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    latency_ms: Mapped[int]
    attempts: Mapped[int]  # 2 when the output needed a repair
    outcome: Mapped[LLMCallOutcome] = mapped_column(str_enum(LLMCallOutcome, "llm_call_outcome"))
    error_type: Mapped[str | None] = mapped_column(String(100))
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    # What the call was for, e.g. ("scan", <id>) or ("question", <hash>).
    subject_type: Mapped[str | None] = mapped_column(String(32))
    subject_id: Mapped[str | None] = mapped_column(String(64))

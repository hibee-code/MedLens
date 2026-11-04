from datetime import date
from uuid import UUID

from pydantic import ConfigDict, Field

from medlens.core.schemas import InputModel, ReadModel, SafeText
from medlens.modules.knowledge.schemas import SearchResultItem
from medlens.modules.rag.service import Outcome


class Question(InputModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"question": "Has NAFDAC issued an alert for Coartem batch F2034?"}]
        }
    )

    question: SafeText = Field(min_length=3, max_length=1000)


class CitationRead(ReadModel):
    chunk_id: UUID
    document_id: UUID
    title: str
    page: int | None
    published_on: date | None
    source_url: str | None
    quote: str = Field(description="Copied word for word from the document (checked).")


class AnswerRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "outcome": "answered",
                    "answer": "Yes. NAFDAC Public Alert No. 041/2026 (14 October 2026) reports "
                    "counterfeit Coartem 20/120, batch F2034, in Kano and Lagos.",
                    "citations": [
                        {
                            "chunk_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0f02",
                            "document_id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0f01",
                            "title": "Public Alert No. 041/2026: counterfeit Coartem 20/120",
                            "page": 1,
                            "published_on": "2026-10-14",
                            "source_url": "https://nafdac.gov.ng/public-alert-041-2026/",
                            "quote": "Batch No: F2034",
                        }
                    ],
                    "results": [],
                    "reason": None,
                    "vector_search": True,
                }
            ]
        }
    )

    outcome: Outcome = Field(
        description="answered; insufficient_evidence (nothing relevant, or the answer could "
        "not be grounded); degraded (AI unavailable: see `results`); blocked (the answer made "
        "a claim the assistant must not make)."
    )
    answer: str | None
    citations: list[CitationRead]
    results: list[SearchResultItem] = Field(
        description="Search results instead of an answer, when degraded."
    )
    reason: str | None = Field(description="Why there is no answer (a stable code).")
    vector_search: bool

from datetime import date, datetime
from uuid import UUID

from pydantic import ConfigDict, Field

from medlens.core.schemas import ReadModel, UpdateModel
from medlens.modules.knowledge.models import DocumentKind, DocumentStatus, Visibility

_DOC = "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0f01"
_CHUNK = "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0f02"


class DocumentRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": _DOC,
                    "title": "Public Alert No. 041/2026: counterfeit Coartem 20/120",
                    "kind": "alert",
                    "visibility": "professional",
                    "source": "NAFDAC public alerts",
                    "source_url": "https://nafdac.gov.ng/public-alert-041-2026/",
                    "published_on": "2026-10-14",
                    "official": True,
                    "active": True,
                    "media_type": "application/pdf",
                    "size_bytes": 182344,
                    "status": "indexed",
                    "duplicate_of_id": None,
                    "page_count": 2,
                    "chunk_count": 5,
                    "error_code": None,
                    "created_at": "2026-11-03T09:00:00Z",
                    "indexed_at": "2026-11-03T09:00:06Z",
                }
            ]
        }
    )

    id: UUID
    title: str
    kind: DocumentKind
    visibility: Visibility
    source: str | None
    source_url: str | None
    published_on: date | None
    official: bool = Field(description="A regulator vouches for the source.")
    active: bool = Field(description="false: withdrawn or superseded.")
    media_type: str
    size_bytes: int
    status: DocumentStatus = Field(
        description="pending until the worker has indexed it; duplicate when the same text "
        "is already indexed; failed when no text could be extracted."
    )
    duplicate_of_id: UUID | None
    page_count: int | None
    chunk_count: int
    error_code: str | None
    created_at: datetime
    indexed_at: datetime | None


class DocumentUpdate(UpdateModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"active": False}]})

    official: bool | None = Field(
        default=None, description="The source is official (e.g. a NAFDAC publication)."
    )
    active: bool | None = Field(default=None, description="false withdraws the document.")


class SearchResultItem(ReadModel):
    chunk_id: UUID
    document_id: UUID
    title: str
    kind: DocumentKind
    published_on: date | None
    source_url: str | None
    page: int | None = Field(description="1-based page in the PDF.")
    section: str | None = Field(description="The nearest heading.")
    text: str
    score: float = Field(description="Reciprocal Rank Fusion score (higher is better).")
    matched_by: list[str] = Field(description="vector, keyword and/or nafdac_number.")
    similarity: float | None = Field(description="Cosine similarity, if found by vector.")
    rerank_score: float | None = Field(
        default=None, description="The cross-encoder's relevance (0-1), when reranked."
    )


class SearchResults(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "Is there an alert for Coartem batch F2034?",
                    "vector_search": True,
                    "results": [
                        {
                            "chunk_id": _CHUNK,
                            "document_id": _DOC,
                            "title": "Public Alert No. 041/2026: counterfeit Coartem 20/120",
                            "kind": "alert",
                            "published_on": "2026-10-14",
                            "source_url": "https://nafdac.gov.ng/public-alert-041-2026/",
                            "page": 1,
                            "section": "Details of the counterfeit product",
                            "text": "Batch No: F2034 ... Manufacturing date: 03/2025 ...",
                            "score": 0.0325,
                            "matched_by": ["keyword", "vector"],
                            "similarity": 0.87,
                            "rerank_score": 0.94,
                        }
                    ],
                    "reranked": True,
                }
            ]
        }
    )

    query: str
    vector_search: bool = Field(
        description="false: the question could not be embedded in time; keywords and NAFDAC "
        "numbers only."
    )
    reranked: bool = Field(
        default=False,
        description="The cross-encoder ordered the results (false: fused order).",
    )
    results: list[SearchResultItem]

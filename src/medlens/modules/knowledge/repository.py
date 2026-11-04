"""Knowledge queries: documents, and the three retrievers fused by the search service (vector
similarity, Postgres full-text search, exact NAFDAC numbers). Every retriever sees indexed
documents the caller may read only."""

from dataclasses import dataclass
from datetime import date
from uuid import UUID

from sqlalchemy import ColumnElement, Select, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from medlens.core.pagination import PageParams
from medlens.modules.knowledge.models import (
    DocumentKind,
    DocumentStatus,
    KnowledgeChunk,
    KnowledgeDocument,
    Visibility,
)

FTS_CONFIG = "english"


@dataclass(frozen=True, slots=True)
class ChunkHit:
    chunk_id: UUID
    score: float  # cosine similarity, text-search rank, or 1.0 for an exact number


@dataclass(frozen=True, slots=True)
class ChunkDetail:
    chunk_id: UUID
    document_id: UUID
    title: str
    kind: DocumentKind
    published_on: date | None
    source_url: str | None
    page: int | None
    section: str | None
    text: str


def visible(include_regulator_only: bool) -> ColumnElement[bool]:
    condition = KnowledgeDocument.status == DocumentStatus.INDEXED
    if not include_regulator_only:
        condition = condition & (KnowledgeDocument.visibility == Visibility.PROFESSIONAL)
    return condition


def _chunks() -> Select[UUID]:
    return select(KnowledgeChunk.id).join(
        KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id
    )


class KnowledgeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def by_vector(
        self, embedding: list[float], scope: ColumnElement[bool], limit: int
    ) -> list[ChunkHit]:
        distance = KnowledgeChunk.embedding.cosine_distance(embedding)
        rows = await self._session.execute(
            _chunks()
            .add_columns(1 - distance)
            .where(scope, KnowledgeChunk.embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )
        return [ChunkHit(chunk_id, float(similarity)) for chunk_id, similarity in rows]

    async def by_text(self, query: str, scope: ColumnElement[bool], limit: int) -> list[ChunkHit]:
        tsquery = func.websearch_to_tsquery(FTS_CONFIG, query)
        rank = func.ts_rank_cd(KnowledgeChunk.tsv, tsquery)
        rows = await self._session.execute(
            _chunks()
            .add_columns(rank)
            .where(scope, KnowledgeChunk.tsv.bool_op("@@")(tsquery))
            .order_by(rank.desc(), KnowledgeChunk.id)
            .limit(limit)
        )
        return [ChunkHit(chunk_id, float(score)) for chunk_id, score in rows]

    async def by_nafdac_numbers(
        self, numbers: list[str], scope: ColumnElement[bool], limit: int
    ) -> list[ChunkHit]:
        if not numbers:
            return []
        rows = await self._session.execute(
            _chunks()
            .where(scope, KnowledgeChunk.nafdac_numbers.overlap(numbers))
            .order_by(KnowledgeDocument.published_on.desc().nulls_last(), KnowledgeChunk.ordinal)
            .limit(limit)
        )
        return [ChunkHit(chunk_id, 1.0) for chunk_id in rows.scalars()]

    async def details(self, chunk_ids: list[UUID]) -> dict[UUID, ChunkDetail]:
        if not chunk_ids:
            return {}
        rows = await self._session.execute(
            select(KnowledgeChunk, KnowledgeDocument)
            .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
            .where(KnowledgeChunk.id.in_(chunk_ids))
        )
        return {
            chunk.id: ChunkDetail(
                chunk_id=chunk.id,
                document_id=document.id,
                title=document.title,
                kind=document.kind,
                published_on=document.published_on,
                source_url=document.source_url,
                page=chunk.page,
                section=chunk.section,
                text=chunk.text,
            )
            for chunk, document in rows
        }

    async def document(
        self, document_id: UUID, scope: ColumnElement[bool] | None = None
    ) -> KnowledgeDocument | None:
        statement = select(KnowledgeDocument).where(KnowledgeDocument.id == document_id)
        if scope is not None:
            statement = statement.where(scope)
        return await self._session.scalar(statement)

    async def documents(
        self, params: PageParams, *filters: ColumnElement[bool]
    ) -> list[KnowledgeDocument]:
        statement = select(KnowledgeDocument).where(*filters)
        if params.after is not None:
            position = tuple_(params.after.created_at, params.after.id)
            statement = statement.where(
                tuple_(KnowledgeDocument.created_at, KnowledgeDocument.id) < position
            )
        statement = statement.order_by(
            KnowledgeDocument.created_at.desc(), KnowledgeDocument.id.desc()
        ).limit(params.limit + 1)
        return list(await self._session.scalars(statement))

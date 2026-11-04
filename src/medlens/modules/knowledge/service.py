"""Knowledge base for professional users: document upload (indexed by the worker) and hybrid
search.

Search runs three retrievers over the chunks the caller may read: vector similarity with the
question's embedding ("query: " prefix, computed by a worker), Postgres full-text search
(websearch_to_tsquery), and exact NAFDAC numbers named in the question. Their rankings are
fused with Reciprocal Rank Fusion (score = sum of 1 / (k + rank)). A result counts as evidence
only if it names the NAFDAC number asked about, matches the words of the question, or is
semantically close (cosine at least MEDLENS_KNOWLEDGE__MIN_SIMILARITY); the best ``top_k`` are
kept. Without the embedding (worker busy or no model) search runs on the other two and says so.
No model is called here.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import date
from uuid import UUID

import structlog
from sqlalchemy import ColumnElement
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.config import KnowledgeSettings
from medlens.core.errors import NotFound
from medlens.core.ids import uuid7, uuid7_time
from medlens.core.pagination import PageParams
from medlens.core.request_meta import RequestMeta
from medlens.core.uploads import DocumentUpload
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.knowledge.models import (
    DocumentKind,
    KnowledgeDocument,
    Visibility,
)
from medlens.modules.knowledge.repository import ChunkDetail, ChunkHit, KnowledgeRepository, visible
from medlens.modules.organizations.models import OrganizationType
from medlens.modules.rbac.access import Actor
from medlens.modules.verification.extraction import find_nafdac_numbers
from medlens.ports.embedding import QueryEmbeddingPort
from medlens.ports.storage import StoragePort
from medlens.queue import outbox

log = structlog.stdlib.get_logger(__name__)
_EXTENSIONS = {"application/pdf": "pdf", "text/plain": "txt"}


@dataclass(frozen=True, slots=True)
class NewDocument:
    title: str
    kind: DocumentKind
    visibility: Visibility = Visibility.PROFESSIONAL
    source: str | None = None
    source_url: str | None = None
    published_on: date | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: ChunkDetail
    score: float  # fused (RRF)
    matched_by: tuple[str, ...]  # vector, keyword, nafdac_number
    similarity: float | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: list[SearchHit]
    vector_search: bool  # False: the question's embedding was unavailable
    nafdac_numbers: list[str] = field(default_factory=list)


def can_read_regulator_only(actor: Actor) -> bool:
    organization = actor.organization
    return actor.is_platform_admin or (
        organization is not None and organization.type is OrganizationType.REGULATOR
    )


def fuse(rankings: dict[str, list[ChunkHit]], k: int) -> list[tuple[UUID, float]]:
    """Reciprocal Rank Fusion over the retrievers' rankings, best first."""
    scores: dict[UUID, float] = {}
    for hits in rankings.values():
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))


class KnowledgeService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        settings: KnowledgeSettings,
        *,
        storage: StoragePort | None = None,
        query_embedder: QueryEmbeddingPort | None = None,
    ) -> None:
        self._sessions = sessions
        self._settings = settings
        self._storage = storage
        self._query_embedder = query_embedder

    # ------------------------------------------------------------------ documents
    async def upload(
        self, actor: Actor, upload: DocumentUpload, new: NewDocument, meta: RequestMeta
    ) -> KnowledgeDocument:
        if self._storage is None:
            raise RuntimeError("document storage is not configured")
        document_id = uuid7()
        created = uuid7_time(document_id)
        extension = _EXTENSIONS.get(upload.media_type, "bin")
        key = f"documents/{created:%Y/%m}/{document_id}.{extension}"
        await self._storage.put(key, upload.data, content_type=upload.media_type)
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            document = KnowledgeDocument(
                id=document_id,
                title=new.title,
                kind=new.kind,
                visibility=new.visibility,
                source=new.source,
                source_url=new.source_url,
                published_on=new.published_on,
                media_type=upload.media_type,
                s3_key=key,
                size_bytes=len(upload.data),
                uploaded_by=actor.user_id,
            )
            session.add(document)
            await outbox.emit(
                session,
                outbox.KNOWLEDGE_DOCUMENT_UPLOADED,
                aggregate_type="knowledge_document",
                aggregate_id=document_id,
            )
            await AuditService(session).record(
                "knowledge.document_uploaded",
                actor_user_id=actor.user_id,
                actor_org_id=actor.organization.id if actor.organization else None,
                resource_type="knowledge_document",
                resource_id=document_id,
                meta=meta,
                details={"kind": new.kind.value, "sha256": hashlib.sha256(upload.data).hexdigest()},
            )
            await session.flush()
            await uow.commit()
        return document

    async def document(self, actor: Actor, document_id: UUID) -> KnowledgeDocument:
        scope = None if can_read_regulator_only(actor) else _professional()
        async with UnitOfWork(self._sessions) as uow:
            document = await KnowledgeRepository(uow.session).document(document_id, scope)
        if document is None:
            raise NotFound("Document not found.")
        return document

    async def documents(
        self, actor: Actor, params: PageParams, *filters: ColumnElement[bool]
    ) -> list[KnowledgeDocument]:
        scope = [] if can_read_regulator_only(actor) else [_professional()]
        async with UnitOfWork(self._sessions) as uow:
            return await KnowledgeRepository(uow.session).documents(params, *scope, *filters)

    # ------------------------------------------------------------------ search
    async def search(self, actor: Actor, question: str) -> SearchResult:
        settings = self._settings
        scope = visible(can_read_regulator_only(actor))
        numbers = find_nafdac_numbers(question, bare=True)
        embedding = (
            await self._query_embedder.embed_query(question) if self._query_embedder else None
        )
        async with UnitOfWork(self._sessions) as uow:
            repository = KnowledgeRepository(uow.session)
            rankings: dict[str, list[ChunkHit]] = {
                "keyword": await repository.by_text(question, scope, settings.candidates),
                "nafdac_number": await repository.by_nafdac_numbers(
                    numbers, scope, settings.candidates
                ),
            }
            if embedding is not None:
                rankings["vector"] = await repository.by_vector(
                    embedding, scope, settings.candidates
                )
            similarity = {hit.chunk_id: hit.score for hit in rankings.get("vector", [])}
            evidence = {
                hit.chunk_id
                for name, hits in rankings.items()
                for hit in hits
                if name != "vector" or hit.score >= settings.min_similarity
            }
            fused = [
                (chunk_id, score)
                for chunk_id, score in fuse(rankings, settings.rrf_k)
                if chunk_id in evidence
            ][: settings.top_k]
            details = await repository.details([chunk_id for chunk_id, _ in fused])
        hits = [
            SearchHit(
                chunk=details[chunk_id],
                score=round(score, 6),
                matched_by=tuple(
                    name
                    for name, ranked in rankings.items()
                    if any(hit.chunk_id == chunk_id for hit in ranked)
                ),
                similarity=round(similarity[chunk_id], 4) if chunk_id in similarity else None,
            )
            for chunk_id, score in fused
            if chunk_id in details
        ]
        log.info(
            "knowledge.searched",
            results=len(hits),
            vector_search=embedding is not None,
            nafdac_numbers=len(numbers),
        )
        return SearchResult(hits, embedding is not None, numbers)


def _professional() -> ColumnElement[bool]:
    return KnowledgeDocument.visibility == Visibility.PROFESSIONAL

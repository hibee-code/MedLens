from datetime import date
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Form, Query, Request, status

from medlens.core.config import Settings
from medlens.core.openapi import problem_responses
from medlens.core.pagination import Cursor, Page, PageParams, build_page, page_params
from medlens.core.routing import api_v1_router
from medlens.core.schemas import SafeText
from medlens.core.uploads import DocumentUpload, read_document_upload
from medlens.modules.auth.deps import Meta as RequestMetaDep
from medlens.modules.knowledge.models import (
    DocumentKind,
    DocumentStatus,
    KnowledgeDocument,
    Visibility,
)
from medlens.modules.knowledge.schemas import (
    DocumentRead,
    DocumentUpdate,
    SearchResultItem,
    SearchResults,
)
from medlens.modules.knowledge.service import KnowledgeService, NewDocument, SearchResult
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.ports.embedding import QueryEmbeddingPort
from medlens.ports.rerank import RerankingPort
from medlens.ports.storage import StoragePort
from medlens.resources import Resources, get_app_settings, get_resources, get_storage

router = api_v1_router("/knowledge", tags=["knowledge"])
Reader = Annotated[Actor, Depends(require_permissions(Permission.KNOWLEDGE_READ))]
Manager = Annotated[
    Actor, Depends(require_permissions(Permission.KNOWLEDGE_READ, Permission.KNOWLEDGE_MANAGE))
]


def get_query_embedder(request: Request) -> QueryEmbeddingPort:
    """A dependency so tests can swap the worker round trip for an in-process model."""
    return get_resources(request).query_embedder


def get_reranker(request: Request) -> RerankingPort:
    """The worker's cross-encoder (a dependency so tests can swap it)."""
    return get_resources(request).reranker


def get_knowledge_service(
    resources: Annotated[Resources, Depends(get_resources)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    storage: Annotated[StoragePort, Depends(get_storage)],
    query_embedder: Annotated[QueryEmbeddingPort, Depends(get_query_embedder)],
    reranker: Annotated[RerankingPort, Depends(get_reranker)],
) -> KnowledgeService:
    return KnowledgeService(
        resources.session_factory,
        settings.knowledge,
        storage=storage,
        query_embedder=query_embedder,
        reranker=reranker,
    )


ServiceDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]


def document_read(document: KnowledgeDocument) -> DocumentRead:
    return DocumentRead.model_validate(document, from_attributes=True)


def search_results(query: str, result: SearchResult) -> SearchResults:
    return SearchResults(
        query=query,
        vector_search=result.vector_search,
        reranked=result.reranked,
        results=[
            SearchResultItem(
                chunk_id=hit.chunk.chunk_id,
                document_id=hit.chunk.document_id,
                title=hit.chunk.title,
                kind=hit.chunk.kind,
                published_on=hit.chunk.published_on,
                source_url=hit.chunk.source_url,
                page=hit.chunk.page,
                section=hit.chunk.section,
                text=hit.chunk.text,
                score=hit.score,
                matched_by=list(hit.matched_by),
                similarity=hit.similarity,
                rerank_score=hit.rerank_score,
            )
            for hit in result.hits
        ],
    )


@router.post(
    "/documents",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentRead,
    summary="Add a document to the knowledge base",
    description=(
        "Uploads a PDF, plain-text or Markdown document (up to MEDLENS_KNOWLEDGE__MAX_DOCUMENT_"
        "BYTES; the real type is checked by content, and PDFs with scripts, attachments or "
        "forms are refused). A worker extracts the text, skips it if the same text is already "
        "indexed, removes personal data, chunks it, embeds the chunks and, for alerts and "
        "recalls, records the NAFDAC numbers and batches they name. Only `official`, `active` "
        "alerts and recalls flag scans. Poll GET /knowledge/documents/{id} "
        "for the status."
    ),
    responses=problem_responses(401, 403, 413, 415, 422),
)
async def upload_document(
    actor: Manager,
    service: ServiceDep,
    meta: RequestMetaDep,
    upload: Annotated[DocumentUpload, Depends(read_document_upload)],
    title: Annotated[SafeText, Form(min_length=3, max_length=300)],
    kind: Annotated[DocumentKind, Form()],
    visibility: Annotated[Visibility, Form()] = Visibility.PROFESSIONAL,
    source: Annotated[SafeText | None, Form(max_length=200)] = None,
    source_url: Annotated[str | None, Form(max_length=2000, pattern=r"^https://")] = None,
    published_on: Annotated[date | None, Form()] = None,
    official: Annotated[
        bool, Form(description="The source is official, e.g. a NAFDAC publication.")
    ] = False,
    active: Annotated[bool, Form(description="false: already withdrawn or superseded.")] = True,
) -> DocumentRead:
    document = await service.upload(
        actor,
        upload,
        NewDocument(title, kind, visibility, source, source_url, published_on, official, active),
        meta,
    )
    return document_read(document)


@router.patch(
    "/documents/{document_id}",
    response_model=DocumentRead,
    summary="Mark a document official or withdraw it",
    description=(
        "Sets `official` (a regulator vouches for the source) and/or `active` (false withdraws "
        "it). Official, active alerts and recalls flag scans of the products they name, so "
        "every change is audited."
    ),
    responses=problem_responses(401, 403, 404, 422),
)
async def update_document(
    document_id: UUID,
    body: DocumentUpdate,
    actor: Manager,
    service: ServiceDep,
    meta: RequestMetaDep,
) -> DocumentRead:
    return document_read(
        await service.update_flags(
            actor, document_id, official=body.official, active=body.active, meta=meta
        )
    )


@router.get(
    "/documents",
    response_model=Page[DocumentRead],
    summary="List knowledge documents",
    description="Documents you may read, newest first, with their indexing status.",
    responses=problem_responses(401, 403, 422),
)
async def list_documents(
    actor: Reader,
    service: ServiceDep,
    params: Annotated[PageParams, Depends(page_params)],
    document_status: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    kind: DocumentKind | None = None,
) -> Page[DocumentRead]:
    filters = []
    if document_status is not None:
        filters.append(KnowledgeDocument.status == document_status)
    if kind is not None:
        filters.append(KnowledgeDocument.kind == kind)
    documents = await service.documents(actor, params, *filters)
    return build_page(
        documents,
        limit=params.limit,
        cursor_of=lambda document: Cursor(created_at=document.created_at, id=document.id),
        to_item=document_read,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentRead,
    summary="Get a knowledge document",
    description="Its metadata and indexing status. Documents you may not read answer 404.",
    responses=problem_responses(401, 403, 404),
)
async def get_document(document_id: UUID, actor: Reader, service: ServiceDep) -> DocumentRead:
    return document_read(await service.document(actor, document_id))


@router.get(
    "/search",
    response_model=SearchResults,
    summary="Search the knowledge base",
    description=(
        "Hybrid search without any generative model: vector similarity, full-text search and "
        "exact NAFDAC numbers, fused with Reciprocal Rank Fusion, then re-ordered by a "
        "cross-encoder. Results must clear an evidence bar (a matching NAFDAC number, the "
        "question's words, or close semantic similarity); an empty list means there is no "
        "evidence."
    ),
    responses=problem_responses(401, 403, 422),
)
async def search(
    actor: Reader,
    service: ServiceDep,
    q: Annotated[SafeText, Query(min_length=3, max_length=500, description="The question.")],
) -> SearchResults:
    return search_results(q, await service.search(actor, q))

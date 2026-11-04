from typing import Annotated

from fastapi import Depends, Request

from medlens.ai.gateway.gateway import AsyncLLMGateway
from medlens.core.config import Settings
from medlens.core.openapi import problem_responses
from medlens.core.routing import api_v1_router
from medlens.modules.knowledge.router import ServiceDep as KnowledgeDep
from medlens.modules.knowledge.router import search_results
from medlens.modules.knowledge.service import SearchResult
from medlens.modules.rag.metrics import ASSISTANT_REQUESTS
from medlens.modules.rag.schemas import AnswerRead, CitationRead, Question
from medlens.modules.rag.service import AssistantService
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.ratelimit.deps import limit_assistant
from medlens.resources import get_app_settings, get_resources

router = api_v1_router("/assistant", tags=["assistant"])


def get_llm_gateway(request: Request) -> AsyncLLMGateway:
    """A dependency so tests can swap the model provider."""
    return get_resources(request).llm


@router.post(
    "/ask",
    response_model=AnswerRead,
    summary="Ask the assistant",
    description=(
        "Answers from the knowledge base only, citing the documents it used (each quote is "
        "checked against the document). Without relevant documents the answer is "
        "`insufficient_evidence` and no model is called; with AI switched off or over budget "
        "it is `degraded` and returns the search results. For pharmacists, manufacturers, "
        "regulators and platform admins. The assistant does not give dosage or diagnostic "
        "advice and never judges whether a pack is genuine: scan the pack for that."
    ),
    responses=problem_responses(401, 403, 422, 429),
    dependencies=[Depends(limit_assistant)],
)
async def ask(
    body: Question,
    actor: Annotated[Actor, Depends(require_permissions(Permission.KNOWLEDGE_READ))],
    knowledge: KnowledgeDep,
    gateway: Annotated[AsyncLLMGateway, Depends(get_llm_gateway)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> AnswerRead:
    reply = await AssistantService(knowledge, gateway, settings.knowledge).ask(actor, body.question)
    ASSISTANT_REQUESTS.labels(outcome=reply.outcome.value).inc()
    results = search_results(body.question, SearchResult(reply.results, reply.vector_search))
    return AnswerRead(
        outcome=reply.outcome,
        answer=reply.answer,
        citations=[
            CitationRead(
                chunk_id=source.hit.chunk.chunk_id,
                document_id=source.hit.chunk.document_id,
                title=source.hit.chunk.title,
                page=source.hit.chunk.page,
                published_on=source.hit.chunk.published_on,
                source_url=source.hit.chunk.source_url,
                quote=source.quote,
            )
            for source in reply.citations
        ],
        results=results.results,
        reason=reply.reason,
        vector_search=reply.vector_search,
    )

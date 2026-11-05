from typing import Annotated

from fastapi import Depends, Request
from fastapi.responses import StreamingResponse

from medlens.ai.gateway.gateway import AsyncLLMGateway
from medlens.core.config import Settings
from medlens.core.errors import NotFound
from medlens.core.openapi import problem_responses
from medlens.core.routing import api_v1_router
from medlens.modules.knowledge.router import ServiceDep as KnowledgeDep
from medlens.modules.knowledge.router import search_results
from medlens.modules.knowledge.service import SearchResult
from medlens.modules.rag.metrics import ASSISTANT_REQUESTS
from medlens.modules.rag.schemas import AnswerRead, CitationRead, Question
from medlens.modules.rag.service import AssistantReply, AssistantService
from medlens.modules.rag.streaming import stream
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
    return answer_read(body.question, reply)


def answer_read(question: str, reply: AssistantReply) -> AnswerRead:
    ASSISTANT_REQUESTS.labels(outcome=reply.outcome.value).inc()
    results = search_results(
        question, SearchResult(reply.results, reply.vector_search, reranked=reply.reranked)
    )
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
        reranked=reply.reranked,
    )


@router.post(
    "/ask/stream",
    response_class=StreamingResponse,
    summary="Ask the assistant, streaming progress",
    description=(
        "The same answer as POST /assistant/ask, as server-sent events (text/event-stream): "
        "`status` events as each stage starts (searching, retrieved, generating, "
        "checking_citations, retrying), comment lines every 10 s while the model works, then "
        "one `answer` event with the same body as /ask, or an `error` event. The answer is "
        "sent only after its citations are checked, never token by token: when the check "
        "fails, the `answer` event says `insufficient_evidence` and no unchecked text was "
        "ever sent. Answers 404 while MEDLENS_KNOWLEDGE__ASSISTANT_STREAMING is off."
    ),
    responses={
        200: {
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
            "description": "Server-sent events.",
        },
        **problem_responses(401, 403, 404, 422, 429),
    },
    dependencies=[Depends(limit_assistant)],
)
async def ask_stream(
    body: Question,
    actor: Annotated[Actor, Depends(require_permissions(Permission.KNOWLEDGE_READ))],
    knowledge: KnowledgeDep,
    gateway: Annotated[AsyncLLMGateway, Depends(get_llm_gateway)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> StreamingResponse:
    if not settings.knowledge.assistant_streaming:
        raise NotFound("Streaming is switched off: use POST /api/v1/assistant/ask.")
    service = AssistantService(knowledge, gateway, settings.knowledge)
    events = stream(
        lambda progress: service.ask(actor, body.question, progress),
        lambda reply: answer_read(body.question, reply).model_dump(mode="json"),
    )
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )

"""The assistant: answers professional users' questions from the knowledge base, with citations
(CLAUDE.md §5: grounded or silent).

1. The question is scrubbed of personal data and searched (``KnowledgeService.search``).
2. No evidence: ``insufficient_evidence``, without calling a model.
3. AI off (MEDLENS_LLM__ENABLED, MEDLENS_KNOWLEDGE__ASSISTANT_ENABLED), over budget or the model
   failing: ``degraded``, with the search results instead of an answer.
4. Otherwise the gateway (balanced tier) answers from the retrieved chunks, which reach the
   model wrapped as untrusted documents, with the question last.
5. Guardrails: every citation must name a retrieved chunk and quote it exactly; an answer must
   cite at least once. A failing answer is retried once with the problems listed, then
   replaced by ``insufficient_evidence``. Answers making banned claims (safety, authenticity,
   dosage, diagnosis) are ``blocked``.

The assistant never sees or changes a scan's verdict.
"""

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from medlens.ai.gateway.gateway import AsyncLLMGateway, Part, Subject, TextPart, UntrustedPart
from medlens.ai.guardrails import banned_claims, quote_in_text, scrub_pii
from medlens.core.config import KnowledgeSettings
from medlens.modules.knowledge.service import KnowledgeService, SearchHit, SearchResult
from medlens.modules.rbac.access import Actor
from medlens.ports.llm import LLMCallFailedError, LLMUnavailableError

log = structlog.stdlib.get_logger(__name__)
ROUTE = "assistant_answer"


class Outcome(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DEGRADED = "degraded"  # no answer: AI off or failing; search results instead
    BLOCKED = "blocked"  # the answer made a claim the assistant must never make


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(max_length=64, description="The id of a document supplied.")
    quote: str = Field(min_length=1, max_length=500, description="Copied word for word.")


class ModelAnswer(BaseModel):
    """The model's structured output."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(max_length=4000)
    citations: list[Citation] = Field(max_length=12)
    insufficient_evidence: bool


@dataclass(frozen=True, slots=True)
class CitedSource:
    hit: SearchHit
    quote: str


@dataclass(frozen=True, slots=True)
class AssistantReply:
    outcome: Outcome
    answer: str | None = None
    citations: list[CitedSource] = field(default_factory=list)
    results: list[SearchHit] = field(default_factory=list)  # degraded: what search found
    reason: str | None = None  # why degraded, blocked or without evidence
    vector_search: bool = True


def check_grounding(answer: ModelAnswer, hits: dict[str, SearchHit]) -> list[str]:
    """What is wrong with the answer's citations (empty: grounded)."""
    if answer.insufficient_evidence:
        return []
    problems = []
    if not answer.citations:
        problems.append("the answer cites no document")
    for citation in answer.citations:
        hit = hits.get(citation.chunk_id)
        if hit is None:
            problems.append(f"{citation.chunk_id!r} is not one of the documents supplied")
        elif not quote_in_text(citation.quote, hit.chunk.text):
            problems.append(f"the quote from {citation.chunk_id} is not in that document")
    return problems


class AssistantService:
    def __init__(
        self,
        knowledge: KnowledgeService,
        gateway: AsyncLLMGateway | None,
        settings: KnowledgeSettings,
    ) -> None:
        self._knowledge = knowledge
        self._gateway = gateway
        self._settings = settings

    async def ask(self, actor: Actor, question: str) -> AssistantReply:
        question = scrub_pii(question)
        found = await self._knowledge.search(actor, question)
        if not found.hits:
            return AssistantReply(
                Outcome.INSUFFICIENT_EVIDENCE,
                reason="no_matching_documents",
                vector_search=found.vector_search,
            )
        if self._gateway is None or not self._settings.assistant_enabled:
            return self._degraded(found, "disabled")
        hits = {str(hit.chunk.chunk_id): hit for hit in found.hits}
        subject = Subject("question", hashlib.sha256(question.encode()).hexdigest()[:32])
        parts = self._parts(found.hits, question)
        try:
            answer = await self._gateway.run(
                ROUTE, parts, ModelAnswer, user_id=actor.user_id, subject=subject
            )
            problems = check_grounding(answer, hits)
            if problems:
                log.info("assistant.ungrounded", problems=len(problems), attempt=1)
                retry = [*parts, TextPart(_retry_note(problems))]
                answer = await self._gateway.run(
                    ROUTE, retry, ModelAnswer, user_id=actor.user_id, subject=subject
                )
                problems = check_grounding(answer, hits)
        except LLMUnavailableError as exc:
            return self._degraded(found, exc.reason)
        except LLMCallFailedError as exc:
            return self._degraded(found, f"model_{exc.reason}")
        if problems:
            log.info("assistant.ungrounded", problems=len(problems), attempt=2)
            return AssistantReply(
                Outcome.INSUFFICIENT_EVIDENCE,
                reason="citations_not_grounded",
                vector_search=found.vector_search,
            )
        if answer.insufficient_evidence:
            return AssistantReply(
                Outcome.INSUFFICIENT_EVIDENCE,
                answer=answer.answer or None,
                reason="model_found_no_answer",
                vector_search=found.vector_search,
            )
        claims = banned_claims(answer.answer)
        if claims:
            log.warning("assistant.blocked", claims=claims)
            return AssistantReply(
                Outcome.BLOCKED, reason=",".join(claims), vector_search=found.vector_search
            )
        return AssistantReply(
            Outcome.ANSWERED,
            answer=answer.answer,
            citations=[CitedSource(hits[c.chunk_id], c.quote) for c in answer.citations],
            vector_search=found.vector_search,
        )

    @staticmethod
    def _parts(hits: list[SearchHit], question: str) -> list[Part]:
        documents: list[Part] = [
            UntrustedPart("document", hit.chunk.text, _attributes(hit)) for hit in hits
        ]
        return [
            TextPart("Documents retrieved for this question:"),
            *documents,
            UntrustedPart("question", question),
        ]

    @staticmethod
    def _degraded(found: SearchResult, reason: str) -> AssistantReply:
        log.info("assistant.degraded", reason=reason)
        return AssistantReply(
            Outcome.DEGRADED, results=found.hits, reason=reason, vector_search=found.vector_search
        )


def _attributes(hit: SearchHit) -> dict[str, str]:
    chunk = hit.chunk
    attributes: dict[str, Any] = {"id": str(chunk.chunk_id), "title": chunk.title}
    if chunk.page is not None:
        attributes["page"] = chunk.page
    if chunk.published_on is not None:
        attributes["published"] = chunk.published_on.isoformat()
    return {key: str(value) for key, value in attributes.items()}


def _retry_note(problems: list[str]) -> str:
    listed = "; ".join(problems[:5])
    return (
        f"Your previous answer was rejected: {listed}. Answer again. Cite only the document ids "
        "supplied above and copy each quote exactly from that document. If the documents do "
        "not answer the question, set insufficient_evidence to true."
    )

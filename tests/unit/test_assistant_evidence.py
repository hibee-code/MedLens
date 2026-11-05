"""The assistant's evidence bar on reranked passages (MEDLENS_KNOWLEDGE__MIN_RERANK_SCORE):
weak passages never reach the model, and with none left no model is called."""

from typing import Any
from uuid import uuid4

import pytest

from medlens.core.config import KnowledgeSettings
from medlens.modules.knowledge.models import DocumentKind
from medlens.modules.knowledge.repository import ChunkDetail
from medlens.modules.knowledge.service import SearchHit, SearchResult
from medlens.modules.rag.service import AssistantService, Outcome


def hit(
    rerank: float | None, matched_by: tuple[str, ...] = ("vector",), text: str = "passage"
) -> SearchHit:
    chunk = ChunkDetail(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title="Alert",
        kind=DocumentKind.ALERT,
        published_on=None,
        source_url=None,
        page=None,
        section=None,
        text=text,
    )
    return SearchHit(chunk, 0.016, matched_by, 0.85, rerank)


class Knowledge:
    def __init__(self, result: SearchResult) -> None:
        self.result = result

    async def search(self, actor: Any, question: str) -> SearchResult:
        return self.result


class NoModel:
    """A gateway that must not be called."""

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the model was called")


def assistant(result: SearchResult, **settings: Any) -> AssistantService:
    return AssistantService(
        Knowledge(result),  # type: ignore[arg-type]
        NoModel(),  # type: ignore[arg-type]
        KnowledgeSettings(**settings),
    )


async def test_weak_reranked_evidence_means_no_model_call() -> None:
    found = SearchResult([hit(0.02), hit(0.001)], vector_search=True, reranked=True)

    reply = await assistant(found).ask(None, "What is the price of Coartem in Kano?")  # type: ignore[arg-type]

    assert reply.outcome is Outcome.INSUFFICIENT_EVIDENCE
    assert reply.reason == "weak_evidence"
    assert reply.reranked is True


def evidence(found: SearchResult, **settings: Any) -> list[SearchHit]:
    return assistant(found, **settings)._evidence(found)


def test_only_passages_above_the_bar_are_evidence() -> None:
    strong, weak, unscored = hit(0.96), hit(0.05), hit(None)
    found = SearchResult([strong, weak, unscored], vector_search=True, reranked=True)

    assert evidence(found) == [strong]
    assert evidence(found, min_rerank_score=0.01) == [strong, weak]


def test_a_passage_naming_the_asked_nafdac_number_always_counts() -> None:
    named = hit(0.001, ("nafdac_number", "vector"))
    found = SearchResult([named, hit(0.001)], vector_search=True, reranked=True)

    assert evidence(found) == [named]


@pytest.mark.parametrize("vector_search", [True, False])
def test_without_reranking_the_search_floor_is_the_only_bar(vector_search: bool) -> None:
    hits = [hit(None), hit(None)]
    found = SearchResult(hits, vector_search=vector_search, reranked=False)

    assert evidence(found) == hits

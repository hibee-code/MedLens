"""Retrieval quality of the assistant's search, per reranker, and the evidence bar's calibration.

    python scripts/retrieval_eval.py [--questions FILE] [--rerankers NAME,NAME] [--onnx-dir DIR]

Needs the seeded knowledge base (`make up`, `make seed`, `make embeddings`), the text model
(`make install-ml`; `make onnx` for the ONNX exports) and, for each reranker named, its
sentence-transformers original (`ml-export` group). "onnx" names the deployed ONNX reranker.

For every answerable question, the search's top candidates (MEDLENS_KNOWLEDGE__RERANK_CANDIDATES,
fused vectors + keywords + NAFDAC numbers, as in production) are ranked by each reranker, and
by the fused order alone. A passage is relevant when it comes from the question's expected
document. Reported, as percentages: recall@1, recall@3, recall@top_k (the passages the model
sees) and MRR (mean reciprocal rank of the first relevant passage, 0 when none is retrieved).

The evidence bar: each question's best reranker score, answerable and unanswerable apart. The
assistant answers only from passages scoring at least MEDLENS_KNOWLEDGE__MIN_RERANK_SCORE; a
good bar lies above every unanswerable question's best score and below every answerable one's.

Questions CSV (docs/assistant-eval.md): question, expected_document (a file in
seed/documents/index.csv, or a document title), answerable (yes/no).
"""

import argparse
import asyncio
import csv
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RERANKERS = "cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base"
Scorer = Callable[[str, Sequence[str]], list[float]]


@dataclass(frozen=True)
class Question:
    text: str
    expected_title: str | None
    answerable: bool


@dataclass(frozen=True)
class EvalActor:
    """Reads every document (this measures ranking, not access control)."""

    user_id: None = None
    organization: None = None
    is_platform_admin: bool = True


def load_questions(path: Path) -> list[Question]:
    index = ROOT / "seed" / "documents" / "index.csv"
    titles = {row["file"]: row["title"] for row in csv.DictReader(index.open(encoding="utf-8"))}
    questions = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        expected = row.get("expected_document", "").strip()
        questions.append(
            Question(
                text=row["question"].strip(),
                expected_title=titles.get(expected, expected) or None,
                answerable=row["answerable"].strip().lower() in {"yes", "true", "1"},
            )
        )
    return questions


# --------------------------------------------------------------------------- metrics
def first_relevant(titles: list[str], expected: str) -> int | None:
    return next((rank for rank, title in enumerate(titles, 1) if title == expected), None)


def ranking_metrics(ranks: list[int | None], top_k: int) -> dict[str, float]:
    """Percentages over the questions, from each question's first relevant rank."""
    count = len(ranks) or 1

    def recall(k: int) -> float:
        return round(100 * sum(1 for rank in ranks if rank is not None and rank <= k) / count, 1)

    mrr = sum(1 / rank for rank in ranks if rank is not None) / count
    return {"recall@1": recall(1), "recall@3": recall(3), f"recall@{top_k}": recall(top_k),
            "mrr": round(100 * mrr, 1)}  # fmt: skip


def bar_range(answerable: list[float], unanswerable: list[float]) -> dict[str, Any]:
    """The evidence bars that keep every answerable question and drop every unanswerable one."""
    lowest_answerable = min(answerable, default=None)
    highest_unanswerable = max(unanswerable, default=None)
    separable = (
        lowest_answerable is not None
        and highest_unanswerable is not None
        and highest_unanswerable < lowest_answerable
    )
    return {
        "lowest_answerable_best": lowest_answerable,
        "highest_unanswerable_best": highest_unanswerable,
        "separable": separable,
        "bar_between": [highest_unanswerable, lowest_answerable] if separable else None,
    }


def at_bar(answerable: list[float], unanswerable: list[float], bar: float) -> dict[str, str]:
    return {
        "answerable_kept": f"{sum(score >= bar for score in answerable)}/{len(answerable)}",
        "unanswerable_dropped": f"{sum(score < bar for score in unanswerable)}/{len(unanswerable)}",
    }


# --------------------------------------------------------------------------- models
def cross_encoder(name: str) -> Scorer:
    import numpy as np
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(name, device="cpu")

    def score(query: str, passages: Sequence[str]) -> list[float]:
        logits = np.asarray(model.predict([(query, passage) for passage in passages]))
        logits = logits.reshape(len(passages), -1)[:, 0]  # one relevance logit per pair
        return [float(value) for value in 1 / (1 + np.exp(-logits))]

    return score


def onnx_reranker(directory: Path) -> Scorer:
    from medlens.adapters.onnx_models import OnnxReranker

    model = OnnxReranker(directory)
    return lambda query, passages: model.rerank(query, list(passages))


class LocalQueryEmbedder:
    def __init__(self, model: Any) -> None:
        self._model = model

    async def embed_query(self, text: str) -> Any:
        from medlens.ports.embedding import QueryEmbedding

        return QueryEmbedding(self._model.model_version, self._model.embed_query(text))


# --------------------------------------------------------------------------- run
async def candidates(questions: list[Question], onnx_dir: Path | None) -> dict[str, Any]:
    """Each question's fused candidates (title, text), as the search hands them to a reranker."""
    from medlens.core.config import KnowledgeSettings, get_settings
    from medlens.db.session import create_db_engine, create_session_factory
    from medlens.modules.knowledge.service import KnowledgeService
    from medlens.workers.runtime import build_models

    settings = get_settings()
    if onnx_dir is not None:
        embeddings = settings.embeddings.model_copy(update={"onnx_dir": str(onnx_dir)})
        settings = settings.model_copy(update={"embeddings": embeddings})
    text_model = build_models(settings).text
    if text_model is None:
        raise SystemExit("no text model: `make install-ml` and `make onnx`")
    knowledge_settings = KnowledgeSettings()
    wide = knowledge_settings.model_copy(
        update={"top_k": knowledge_settings.rerank_candidates, "rerank": False}
    )
    engine = create_db_engine(settings.database, application_name="medlens-retrieval-eval")
    service = KnowledgeService(
        create_session_factory(engine), wide, query_embedder=LocalQueryEmbedder(text_model)
    )
    found: dict[str, Any] = {"text_model": text_model.model_version, "questions": []}
    try:
        for question in questions:
            result = await service.search(EvalActor(), question.text)  # type: ignore[arg-type]
            found["questions"].append([(hit.chunk.title, hit.chunk.text) for hit in result.hits])
    finally:
        await engine.dispose()
    found["top_k"] = knowledge_settings.top_k
    found["bar"] = knowledge_settings.min_rerank_score
    return found


def evaluate(
    questions: list[Question], pools: list[list[tuple[str, str]]], scorers: dict[str, Scorer],
    top_k: int, bar: float,
) -> dict[str, Any]:  # fmt: skip
    report: dict[str, Any] = {}
    orders: dict[str, Callable[[Question, list[tuple[str, str]]], tuple[list[str], float | None]]]
    orders = {"fused (no reranker)": lambda _, pool: ([title for title, _ in pool], None)}
    for name, scorer in scorers.items():

        def order(
            question: Question, pool: list[tuple[str, str]], scorer: Scorer = scorer
        ) -> tuple[list[str], float | None]:
            if not pool:
                return [], None
            scores = scorer(question.text, [text for _, text in pool])
            ranked = sorted(zip(scores, (title for title, _ in pool), strict=True), reverse=True)
            return [title for _, title in ranked], ranked[0][0]

        orders[name] = order
    for name, order in orders.items():
        ranks: list[int | None] = []
        best: dict[bool, list[float]] = {True: [], False: []}
        for question, pool in zip(questions, pools, strict=True):
            titles, top = order(question, pool)
            if question.answerable and question.expected_title:
                ranks.append(first_relevant(titles, question.expected_title))
            if top is not None or name != "fused (no reranker)":
                best[question.answerable].append(round(top or 0.0, 6))
        entry: dict[str, Any] = ranking_metrics(ranks, top_k)
        if name != "fused (no reranker)":
            entry["evidence_bar"] = bar_range(best[True], best[False])
            entry[f"at_bar_{bar}"] = at_bar(best[True], best[False], bar)
        report[name] = entry
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, default=ROOT / "seed" / "eval_questions.csv")
    parser.add_argument("--rerankers", default=DEFAULT_RERANKERS)
    parser.add_argument("--onnx-dir", type=Path, default=ROOT / "models" / "onnx")
    args = parser.parse_args(argv)
    questions = load_questions(args.questions)
    found = asyncio.run(candidates(questions, args.onnx_dir))
    scorers: dict[str, Scorer] = {}
    for name in filter(None, (item.strip() for item in args.rerankers.split(","))):
        scorers[name] = (
            onnx_reranker(args.onnx_dir / "reranker") if name == "onnx" else cross_encoder(name)
        )
    report = {
        "questions": len(questions),
        "answerable": sum(question.answerable for question in questions),
        "candidates_per_question": max((len(pool) for pool in found["questions"]), default=0),
        "text_model": found["text_model"],
        "rankings": evaluate(questions, found["questions"], scorers, found["top_k"], found["bar"]),
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

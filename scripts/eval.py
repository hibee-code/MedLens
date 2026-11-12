"""Evaluate the scan pipeline and the assistant (``make eval``).

Scans: every pack in seed/test_packs/expected.csv runs through the real pipeline in-process,
with the seed register (seed/products.csv), the seed batches (seed/batches.csv) and the alerts
in seed/documents as the catalog; OpenCV and zxing-cpp read the photos, Tesseract too when
installed. The vision LLM is Claude through the gateway when ANTHROPIC_API_KEY is set, otherwise
a stand-in that reads each pack's printed fields from the CSV (so without a key the eval
measures the readers and the scorer, not the model).

Assistant: the questions in seed/eval_questions.csv run against the knowledge base in the
configured database (``make up`` then ``make seed``), skipped if there is none. With
ANTHROPIC_API_KEY the balanced model answers; otherwise a stand-in cites the top result, which
measures retrieval and abstention only.

Prints accuracy per expected verdict, the share of scans that needed the LLM, mean latency,
citation correctness, abstain accuracy and the total model cost. Exit status 1 if scan accuracy
is below --min-accuracy.
"""

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from medlens.adapters.fakes import (
    AsyncMemoryLedger,
    FakeLLM,
    MemoryLedger,
    image_key,
)
from medlens.adapters.llm import GatewayPackReader
from medlens.ai.gateway.gateway import AsyncLLMGateway, LLMGateway
from medlens.ai.gateway.pricing import TokenUsage
from medlens.ai.gateway.providers import (
    AnthropicProvider,
    AsyncAnthropicProvider,
    ProviderRequest,
    ProviderResponse,
)
from medlens.core.config import KnowledgeSettings, LLMSettings
from medlens.core.identifiers import normalize_nafdac_number
from medlens.modules.knowledge.text import chunk, extract_pages
from medlens.modules.products.models import RegistrationStatus
from medlens.modules.verification.catalog import AlertFacts, BatchFacts, BatchSpread, ProductFacts
from medlens.modules.verification.extraction import (
    find_batch_numbers,
    find_nafdac_numbers,
    name_presence,
)
from medlens.modules.verification.pipeline import PipelineConfig, ScanInput, VerificationPipeline
from medlens.modules.verification.scoring import WeightedRuleScorer

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "seed"


# --------------------------------------------------------------------------- the seed catalog
@dataclass
class SeedCatalog:
    """The pipeline's Catalog over the seed files (no database)."""

    products: dict[str, ProductFacts] = field(default_factory=dict)
    batches: dict[UUID, list[BatchFacts]] = field(default_factory=dict)
    alerts: dict[tuple[str, str], AlertFacts] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "SeedCatalog":
        catalog = cls()
        with (SEED / "products.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                number = normalize_nafdac_number(row["nafdac_number"])
                catalog.products[number] = ProductFacts(
                    uuid4(),
                    number,
                    row["brand_name"],
                    row["generic_name"],
                    RegistrationStatus(row["registration_status"]),
                )
        with (SEED / "batches.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                product = catalog.products[row["nafdac_number"]]
                catalog.batches.setdefault(product.id, []).append(
                    BatchFacts(
                        uuid4(),
                        row["batch_number"],
                        date.fromisoformat(row["expires_on"]),
                        frozenset(row["distribution_states"].split(";")),
                    )
                )
        catalog._load_alerts()
        return catalog

    def _load_alerts(self) -> None:
        with (SEED / "documents" / "index.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                official = row.get("official") == "true" and row.get("active") == "true"
                if row["kind"] not in {"alert", "recall"} or not official:
                    continue  # only official, active notices flag scans
                pages = extract_pages((SEED / "documents" / row["file"]).read_bytes(), "text/plain")
                for piece in chunk(pages):
                    for number in find_nafdac_numbers(piece.text):
                        for batch in find_batch_numbers(piece.text) or [""]:
                            self.alerts[(number, batch)] = AlertFacts(
                                uuid4(), row["kind"], bool(batch), None
                            )

    def product_by_nafdac(self, nafdac_number: str) -> ProductFacts | None:
        return self.products.get(nafdac_number)

    def similar_nafdac(self, nafdac_number: str) -> list[tuple[ProductFacts, float]]:
        del nafdac_number  # exact numbers only: the stand-in reads them correctly
        return []

    def products_named_in(self, text: str) -> list[tuple[ProductFacts, float]]:
        scored = [(p, name_presence(p.brand_name, text) or 0.0) for p in self.products.values()]
        return sorted((item for item in scored if item[1] > 0), key=lambda item: -item[1])[:5]

    def declared_batch_count(self, product_id: UUID) -> int:
        return len(self.batches.get(product_id, []))

    def batch(self, product_id: UUID, batch_number: str) -> BatchFacts | None:
        found = (b for b in self.batches.get(product_id, []) if b.batch_number == batch_number)
        return next(found, None)

    def batch_spread(self, batch_id: UUID, since: datetime, *, exclude_scan: UUID) -> BatchSpread:
        del batch_id, since, exclude_scan  # no scan history in the eval
        return BatchSpread(geocells=0, states=frozenset())

    def published_alerts(self, nafdac_number: str, batch_number: str | None) -> list[AlertFacts]:
        keys = {(nafdac_number, ""), (nafdac_number, batch_number or "")}
        return [self.alerts[key] for key in keys if key in self.alerts]

    def reference_similarity(
        self, product_id: UUID, embedding: list[float], *, model_version: str
    ) -> float | None:
        del product_id, embedding, model_version  # no reference images in the eval
        return None


# --------------------------------------------------------------------------- scans
def readers() -> tuple[Any, Any, Any]:
    from medlens.adapters.vision import (
        OpenCvImageAnalysis,
        TesseractOcr,
        ZxingBarcode,
        tesseract_available,
    )

    return OpenCvImageAnalysis(), ZxingBarcode(), TesseractOcr() if tesseract_available() else None


def stand_in_reading(row: dict[str, str]) -> str:
    return FakeLLM.reading(
        confidence=0.9,
        nafdac_number=row["nafdac_number"],
        product_name=row["brand"],
        batch_number=row["batch_number"],
        expiry_date=row["expiry"],
    )


def evaluate_scans(live: bool) -> tuple[dict[str, Any], Decimal]:
    rows = list(csv.DictReader((SEED / "test_packs" / "expected.csv").open(encoding="utf-8")))
    images = {row["file"]: (SEED / "test_packs" / row["file"]).read_bytes() for row in rows}
    ledger = MemoryLedger()
    if live:
        settings = LLMSettings(enabled=True)
        llm: Any = GatewayPackReader(LLMGateway(settings, AnthropicProvider(settings), ledger))
    else:
        llm = FakeLLM({image_key(images[row["file"]]): stand_in_reading(row) for row in rows})
    analyzer, barcode, ocr = readers()
    pipeline = VerificationPipeline(
        catalog=SeedCatalog.load(),
        scorer=WeightedRuleScorer(),
        analyzer=analyzer,
        barcode=barcode,
        ocr=ocr,
        llm=llm,
        config=PipelineConfig(llm_enabled=True),
    )
    per_verdict: dict[str, Counter[str]] = defaultdict(Counter)
    latencies, llm_scans, mistakes = [], 0, []
    for row in rows:
        started = time.perf_counter()
        result = pipeline.run(ScanInput(uuid4(), images[row["file"]], datetime.now(UTC)))
        latencies.append(time.perf_counter() - started)
        got, expected = result.score.verdict.value, row["expected_verdict"]
        per_verdict[expected]["right" if got == expected else "wrong"] += 1
        llm_scans += result.llm_used
        if got != expected:
            mistakes.append(f"{row['file']}: expected {expected}, got {got}")
    right = sum(counts["right"] for counts in per_verdict.values())
    report = {
        "packs": len(rows),
        "accuracy": round(right / len(rows), 3),
        "accuracy_per_verdict": {
            verdict: f"{counts['right']}/{counts['right'] + counts['wrong']}"
            for verdict, counts in sorted(per_verdict.items())
        },
        "share_needing_llm": round(llm_scans / len(rows), 3),
        "mean_latency_ms": round(statistics.fmean(latencies) * 1000, 1),
        "ocr": ocr is not None,
        "mistakes": mistakes,
    }
    return report, sum((call.cost_usd for call in ledger.calls), Decimal(0))


# --------------------------------------------------------------------------- the assistant
class CiteTopResult:
    """Stand-in model without an API key: cites the first sentence of the top document."""

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        blocks = request.messages[0]["content"]
        documents = [b["text"] for b in blocks if b["text"].startswith("<untrusted_document")]
        first = documents[0]
        chunk_id = first.split('id="', 1)[1].split('"', 1)[0]
        body = first.split(">", 1)[1].strip().splitlines()[0][:200]
        answer = {
            "answer": "See the cited document.",
            "citations": [{"chunk_id": chunk_id, "quote": body}],
            "insufficient_evidence": False,
        }
        return ProviderResponse(json.dumps(answer), request.model, TokenUsage(), "end_turn")


class LocalQueryEmbedder:
    """The worker's text model, in this process (no Celery round trip)."""

    def __init__(self, model: Any) -> None:
        self._model = model

    async def embed_query(self, text: str) -> Any:
        from medlens.ports.embedding import QueryEmbedding

        return QueryEmbedding(self._model.model_version, self._model.embed_query(text))


class LocalReranker:
    def __init__(self, model: Any) -> None:
        self._model = model

    async def rerank(self, query: str, passages: list[str]) -> list[float] | None:
        scores: list[float] = self._model.rerank(query, passages)
        return scores


def local_retrieval_models(settings: Any) -> tuple[Any, Any]:
    """The text model and reranker when installed (the worker image, or `make install-ml`):
    then retrieval runs as in production, with vectors and reranking; else keywords only."""
    from medlens.workers.runtime import build_models

    models = build_models(settings)
    return (
        LocalQueryEmbedder(models.text) if models.text else None,
        LocalReranker(models.reranker) if models.reranker else None,
    )


@dataclass(frozen=True)
class EvalActor:
    """A reader of every document (the eval is not about access control)."""

    user_id: None = None
    organization: None = None
    is_platform_admin: bool = True


async def evaluate_assistant(
    live: bool, questions: Path | None = None
) -> tuple[dict[str, Any], Decimal]:
    from sqlalchemy import select

    from medlens.core.config import get_settings
    from medlens.db.session import create_db_engine, create_session_factory
    from medlens.modules.knowledge.models import KnowledgeChunk, KnowledgeDocument
    from medlens.modules.knowledge.service import KnowledgeService
    from medlens.modules.rag.service import AssistantService, Outcome

    settings = get_settings()
    engine = create_db_engine(settings.database, application_name="medlens-eval")
    sessions = create_session_factory(engine)
    ledger = AsyncMemoryLedger()
    llm_settings = LLMSettings(enabled=True)
    provider: Any = AsyncAnthropicProvider(llm_settings) if live else CiteTopResult()
    gateway = AsyncLLMGateway(llm_settings, provider, ledger)
    query_embedder, reranker = local_retrieval_models(settings)
    knowledge = KnowledgeService(
        sessions, KnowledgeSettings(), query_embedder=query_embedder, reranker=reranker
    )
    assistant = AssistantService(knowledge, gateway, KnowledgeSettings())
    source = questions or SEED / "eval_questions.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    titles = {
        row["file"]: row["title"]
        for row in csv.DictReader((SEED / "documents" / "index.csv").open(encoding="utf-8"))
    }
    try:
        async with sessions() as session:
            if not await session.scalar(select(KnowledgeChunk.id).limit(1)):
                return {"skipped": "no knowledge base: run `make seed` first"}, Decimal(0)
            known = set(await session.scalars(select(KnowledgeDocument.title)))
        cited_right = cited = abstained_right = unanswerable = answered_right = answerable = 0
        for row in rows:
            reply = await assistant.ask(EvalActor(), row["question"])  # type: ignore[arg-type]
            if row["answerable"] == "yes":
                answerable += 1
                expected = titles.get(row["expected_document"], row["expected_document"])
                titles_cited = {source.hit.chunk.title for source in reply.citations}
                answered_right += reply.outcome is Outcome.ANSWERED and expected in titles_cited
                cited += len(reply.citations)
                cited_right += sum(source.hit.chunk.title in known for source in reply.citations)
            else:
                unanswerable += 1
                abstained_right += reply.outcome is not Outcome.ANSWERED
    except OSError as exc:
        return {"skipped": f"database unavailable ({type(exc).__name__})"}, Decimal(0)
    finally:
        await engine.dispose()
    report = {
        "retrieval": "vectors + keywords + reranker"
        if query_embedder and reranker
        else "keywords and NAFDAC numbers only (no local models)",
        "questions": len(rows),
        "answered_from_the_expected_document": f"{answered_right}/{answerable}",
        "citation_correctness": round(cited_right / cited, 3) if cited else None,
        "abstain_accuracy": f"{abstained_right}/{unanswerable}",
    }
    return report, sum((call.cost_usd for call in ledger.calls), Decimal(0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument("--skip-assistant", action="store_true")
    parser.add_argument("--skip-scans", action="store_true")
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="CSV: question, expected_document, answerable (default seed/eval_questions.csv)",
    )
    args = parser.parse_args()
    live = bool(os.environ.get("ANTHROPIC_API_KEY"))
    scans, scan_cost = (
        ({"skipped": "--skip-scans"}, Decimal(0)) if args.skip_scans else evaluate_scans(live)
    )
    assistant, assistant_cost = (
        ({"skipped": "--skip-assistant"}, Decimal(0))
        if args.skip_assistant
        else asyncio.run(evaluate_assistant(live, args.questions))
    )
    report = {
        "models": "live (ANTHROPIC_API_KEY)" if live else "stand-ins (no API key)",
        "scans": scans,
        "assistant": assistant,
        "total_cost_usd": str(scan_cost + assistant_cost),
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if args.skip_scans or scans["accuracy"] >= args.min_accuracy else 1


if __name__ == "__main__":
    raise SystemExit(main())

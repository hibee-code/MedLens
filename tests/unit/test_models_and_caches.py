"""The worker's ONNX models (tiny fixture graphs, no downloads), how the runtime picks them, the
caches (question embeddings, product lookups, model answers), reranking over the worker RPC and
the assistant's event stream."""

import asyncio
import io
import json
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from PIL import Image
from pydantic import BaseModel

from medlens.adapters.fakes import AsyncFakeLLMProvider, AsyncMemoryLedger, FakeQueryEmbedder
from medlens.adapters.query_embedding import CachedQueryEmbedder, CeleryReranker
from medlens.ai.gateway.cache import AnswerCache
from medlens.ai.gateway.gateway import AsyncLLMGateway, TextPart
from medlens.core.config import LLMSettings, Settings
from medlens.modules.products.models import RegistrationStatus
from medlens.modules.rag.service import AssistantReply, Outcome
from medlens.modules.rag.streaming import stream
from medlens.modules.verification.catalog import ProductFacts
from medlens.modules.verification.product_cache import ProductCache
from medlens.modules.verification.repository import SqlCatalog
from medlens.ports.llm import LLMUnavailableError
from medlens.workers import runtime
from tests.support import build_settings

onnx_models = pytest.importorskip("medlens.adapters.onnx_models")  # the `ml` group
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "onnx"


# ------------------------------------------------------------------------- ONNX adapters
def test_the_text_model_prefixes_and_checks_its_dimension() -> None:
    model = onnx_models.OnnxTextEmbedder(FIXTURES / "text", 4)

    assert model.model_version == "fixture/text@onnx-int8"
    assert model.embed_passages(["alert", "batch recall"]) == [[0.5] * 4, [0.5] * 4]
    assert model.embed_query("recall") == [0.5] * 4
    assert model.probe_dimension() == 4
    with pytest.raises(ValueError, match="wrong dimension"):
        onnx_models.OnnxTextEmbedder(FIXTURES / "text", 5).embed_query("q")


def test_the_reranker_scores_each_passage() -> None:
    model = onnx_models.OnnxReranker(FIXTURES / "reranker")

    short, long = model.rerank("alert", ["batch", "batch recall alert"])  # logit = tokens - 4

    assert model.model_version == "fixture/reranker@onnx"
    assert short == pytest.approx(0.119203)
    assert long == pytest.approx(0.5)
    assert model.rerank("alert", []) == []


def test_the_image_model_uses_clip_preprocessing() -> None:
    model = onnx_models.OnnxImageEmbedder(FIXTURES / "image", 3)
    picture = io.BytesIO()
    Image.new("RGB", (400, 300), (255, 0, 128)).save(picture, "PNG")

    vector = model.embed_image(picture.getvalue())  # channel means, normalised by 0.5/0.5

    assert vector == pytest.approx([1.0, -1.0, 0.0039], abs=1e-3)
    assert model.pixels(Image.new("RGB", (640, 200))).shape == (1, 3, 224, 224)
    assert model.probe_dimension() == 3
    with pytest.raises(ValueError, match="3 dimensions"):
        onnx_models.OnnxImageEmbedder(FIXTURES / "image", 512).embed_image(picture.getvalue())


def test_exports_are_found_only_when_complete(tmp_path: Path) -> None:
    assert onnx_models.available(FIXTURES / "text")
    assert not onnx_models.available(tmp_path)


# ------------------------------------------------------------------------- choosing models
def onnx_settings(settings_factory: Any, **embeddings: Any) -> Settings:
    settings: Settings = settings_factory(
        embeddings={"enabled": True, "backend": "onnx", "onnx_dir": str(FIXTURES), **embeddings}
    )
    return settings


def test_the_runtime_prefers_the_onnx_exports(settings_factory: Any) -> None:
    models = runtime.build_models(
        onnx_settings(settings_factory, image_dimension=3, text_dimension=4)
    )

    assert isinstance(models.image, onnx_models.OnnxImageEmbedder)
    assert isinstance(models.text, onnx_models.OnnxTextEmbedder)
    assert isinstance(models.reranker, onnx_models.OnnxReranker)


@pytest.mark.parametrize("arena", [True, False])
def test_the_memory_arena_is_a_setting(settings_factory: Any, arena: bool) -> None:
    models = runtime.build_models(
        onnx_settings(
            settings_factory, image_dimension=3, text_dimension=4, onnx_memory_arena=arena
        )
    )

    for model in (models.image, models.text, models.reranker):
        assert model._session.get_session_options().enable_cpu_mem_arena is arena  # type: ignore[union-attr]


def test_models_that_do_not_fit_the_columns_are_refused(settings_factory: Any) -> None:
    models = runtime.build_models(onnx_settings(settings_factory))  # 512 and 384 configured

    assert models.image is None
    assert models.text is None
    assert models.reranker is not None  # no vector column to fit


def test_missing_exports_fall_back_to_torch(
    settings_factory: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fallback = runtime.Models()
    monkeypatch.setattr(runtime, "_torch_models", lambda settings: fallback)

    models = runtime.build_models(onnx_settings(settings_factory, onnx_dir=str(tmp_path)))

    assert models == fallback
    assert runtime.build_models(settings_factory(embeddings={"enabled": False})) == fallback


# ------------------------------------------------------------------------- caches
class AsyncDictRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def mget(self, keys: list[str]) -> list[str | None]:
        return [self.data.get(key) for key in keys]

    async def set(self, key: str, value: str, ex: int) -> None:
        self.data[key] = value


class DictRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def set(self, key: str, value: str, ex: int) -> None:
        self.data[key] = value


async def test_question_embeddings_are_cached_per_model() -> None:
    from medlens.adapters.fakes import FakeTextEmbedder

    inner = FakeQueryEmbedder(FakeTextEmbedder())
    redis = AsyncDictRedis()
    cached = CachedQueryEmbedder(inner, redis, model="e5:onnx", ttl_seconds=60)  # type: ignore[arg-type]
    other_model = CachedQueryEmbedder(inner, redis, model="e5:torch", ttl_seconds=60)  # type: ignore[arg-type]

    first = await cached.embed_query("any alert for Coartem?")
    second = await cached.embed_query("any alert for Coartem?")
    await other_model.embed_query("any alert for Coartem?")
    off = CachedQueryEmbedder(inner, redis, model="e5", ttl_seconds=0)  # type: ignore[arg-type]
    await off.embed_query("any alert for Coartem?")

    assert first == second
    assert first is not None
    assert first.model_version == "fake-text-embedder-1"
    assert len(inner.calls) == 3  # once per model, and once with the cache off
    assert all("Coartem" not in key for key in redis.data)  # questions are hashed


async def test_a_new_model_version_invalidates_cached_question_embeddings() -> None:
    from medlens.adapters.fakes import FakeTextEmbedder
    from medlens.cache.keys import TEXT_MODEL_VERSION_KEY

    inner = FakeQueryEmbedder(FakeTextEmbedder())
    redis = AsyncDictRedis()
    cached = CachedQueryEmbedder(inner, redis, model="e5", ttl_seconds=60)  # type: ignore[arg-type]
    redis.data[TEXT_MODEL_VERSION_KEY] = "fake-text-embedder-1"  # what the workers run

    await cached.embed_query("recall of Adalat?")
    await cached.embed_query("recall of Adalat?")
    assert len(inner.calls) == 1

    redis.data[TEXT_MODEL_VERSION_KEY] = "e5@onnx-int8-per-channel"  # workers now run another
    await cached.embed_query("recall of Adalat?")
    assert len(inner.calls) == 2  # the old entry is ignored


class AsyncDownRedis:
    def __getattr__(self, name: str) -> Any:
        from redis.exceptions import ConnectionError as RedisConnectionError

        async def fail(*args: Any, **kwargs: Any) -> Any:
            raise RedisConnectionError("down")

        return fail


async def test_question_embeddings_work_while_redis_is_down() -> None:
    from medlens.adapters.fakes import FakeTextEmbedder

    inner = FakeQueryEmbedder(FakeTextEmbedder())
    cached = CachedQueryEmbedder(inner, AsyncDownRedis(), model="e5", ttl_seconds=60)  # type: ignore[arg-type]

    assert await cached.embed_query("q") is not None


async def test_unavailable_embeddings_are_not_cached() -> None:
    redis = AsyncDictRedis()
    cached = CachedQueryEmbedder(FakeQueryEmbedder(), redis, model="e5", ttl_seconds=60)  # type: ignore[arg-type]

    assert await cached.embed_query("q") is None
    assert redis.data == {}


def test_products_are_cached_including_unregistered_numbers() -> None:
    product = ProductFacts(uuid4(), "A4-1234", "Panadol", "Paracetamol", RegistrationStatus.ACTIVE)

    class Session:
        def __init__(self) -> None:
            self.queries = 0

        def scalar(self, statement: Any) -> Any:
            self.queries += 1
            return None

    cache = ProductCache(DictRedis(), ttl_seconds=60)  # type: ignore[arg-type]
    cache.put("A4-1234", product)
    session = Session()
    catalog = SqlCatalog(session, cache)  # type: ignore[arg-type]

    assert catalog.product_by_nafdac("A4-1234") == product
    assert catalog.product_by_nafdac("Z9-000001") is None  # a database miss...
    assert catalog.product_by_nafdac("Z9-000001") is None  # ...cached as a miss
    assert session.queries == 1
    assert cache.get("A4-1234") == (True, product)
    assert cache.get("A4-9999") == (False, None)


class Answer(BaseModel):
    answer: str


async def test_cacheable_answers_are_reused_without_a_call() -> None:
    provider = AsyncFakeLLMProvider(script=['{"answer": "first"}', '{"answer": "second"}'])
    ledger = AsyncMemoryLedger()
    cache = AnswerCache(AsyncDictRedis(), ttl_seconds=60)  # type: ignore[arg-type]
    gateway = AsyncLLMGateway(LLMSettings(enabled=True), provider, ledger, cache=cache)

    first = await gateway.run("assistant_answer", [TextPart("same question")], Answer)
    again = await gateway.run("assistant_answer", [TextPart("same question")], Answer)
    other = await gateway.run("assistant_answer", [TextPart("another question")], Answer)

    assert (first.answer, again.answer, other.answer) == ("first", "first", "second")
    assert len(provider.requests) == 2
    assert len(ledger.calls) == 2  # a cache hit is not a metered call


async def test_the_kill_switch_also_stops_cached_answers() -> None:
    redis = AsyncDictRedis()
    cache = AnswerCache(redis, ttl_seconds=60)  # type: ignore[arg-type]
    on = AsyncLLMGateway(
        LLMSettings(enabled=True),
        AsyncFakeLLMProvider(script=['{"answer": "a"}']),
        AsyncMemoryLedger(),
        cache=cache,
    )
    await on.run("assistant_answer", [TextPart("q")], Answer)
    off = AsyncLLMGateway(
        LLMSettings(enabled=False), AsyncFakeLLMProvider(), AsyncMemoryLedger(), cache=cache
    )

    with pytest.raises(LLMUnavailableError):
        await off.run("assistant_answer", [TextPart("q")], Answer)


async def test_answers_are_not_cached_when_turned_off_or_not_cacheable() -> None:
    provider = AsyncFakeLLMProvider(script=['{"answer": "a"}', '{"answer": "b"}'])
    disabled = AnswerCache(AsyncDictRedis(), ttl_seconds=0)  # type: ignore[arg-type]
    gateway = AsyncLLMGateway(
        LLMSettings(enabled=True), provider, AsyncMemoryLedger(spent=Decimal(0)), cache=disabled
    )

    await gateway.run("assistant_answer", [TextPart("q")], Answer)
    await gateway.run("assistant_answer", [TextPart("q")], Answer)

    assert len(provider.requests) == 2


# ------------------------------------------------------------------------- reranking RPC
class FakeResult:
    def __init__(self, value: object = None, error: Exception | None = None) -> None:
        self.value, self.error = value, error

    def get(self, timeout: float) -> object:
        if self.error:
            raise self.error
        return self.value


class FakeCelery:
    def __init__(self, result: FakeResult) -> None:
        self.result = result
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def send_task(self, name: str, **options: Any) -> FakeResult:
        self.sent.append((name, options))
        return self.result


async def test_reranking_runs_in_a_worker_or_not_at_all() -> None:
    worker = FakeCelery(FakeResult([0.9, 0.1]))
    reranker = CeleryReranker(worker, timeout_seconds=2)  # type: ignore[arg-type]

    assert await reranker.rerank("q", ["a", "b"]) == [0.9, 0.1]
    name, options = worker.sent[0]
    assert name == "medlens.knowledge.rerank"
    assert options["args"] == ["q", ["a", "b"]]
    for broken in (FakeResult(error=TimeoutError()), FakeResult(None), FakeResult([0.5])):
        failing = CeleryReranker(FakeCelery(broken), timeout_seconds=1)  # type: ignore[arg-type]
        assert await failing.rerank("q", ["a", "b"]) is None


# ------------------------------------------------------------------------- event stream
def parse(chunks: list[str]) -> list[tuple[str, Any]]:
    events = []
    for chunk in chunks:
        if chunk.startswith(":"):
            events.append(("comment", chunk.strip()))
            continue
        name, data = chunk.strip().split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


async def test_the_stream_reports_stages_then_the_answer() -> None:
    async def run(progress: Any) -> AssistantReply:
        await progress("searching", {})
        await asyncio.sleep(0.05)
        await progress("retrieved", {"documents": 2})
        await asyncio.sleep(0.05)
        return AssistantReply(Outcome.ANSWERED, answer="ok")

    chunks = [
        chunk
        async for chunk in stream(
            run, lambda r: {"outcome": r.outcome.value}, keepalive_seconds=0.02
        )
    ]

    events = [event for event in parse(chunks) if event[0] != "comment"]
    assert events == [
        ("status", {"stage": "searching"}),
        ("status", {"stage": "retrieved", "documents": 2}),
        ("answer", {"outcome": "answered"}),
    ]
    assert ("comment", ": keepalive") in parse(chunks)


async def test_a_failure_ends_the_stream_with_an_error_event() -> None:
    async def run(progress: Any) -> AssistantReply:
        await progress("searching", {})
        raise RuntimeError("boom")

    chunks = [chunk async for chunk in stream(run, lambda r: {})]

    name, data = parse(chunks)[-1]
    assert name == "error"
    assert data["type"] == "urn:medlens:problem:internal-error"
    assert data["status"] == 500


# ------------------------------------------------------------------------- invalidation, Redis down
def test_cached_answers_are_keyed_by_what_the_model_sees() -> None:
    """Content-addressed: a changed or withdrawn document, a new prompt version or a caller who
    sees other documents (regulator-only ones) gives another key. Nothing about the asker is in
    the key or the value: the question is scrubbed and answers depend only on the documents."""
    from medlens.ai.gateway.cache import answer_key
    from medlens.ai.gateway.providers import ProviderRequest

    def request(*documents: str) -> ProviderRequest:
        content = [{"type": "text", "text": text} for text in (*documents, "the question")]
        return ProviderRequest(
            model="m", system="s", messages=({"role": "user", "content": content},),
            schema={}, max_tokens=100, effort="low",
        )  # fmt: skip

    professional = request("public alert v1")
    assert answer_key(professional, "v1") == answer_key(request("public alert v1"), "v1")
    assert answer_key(professional, "v1") != answer_key(request("public alert v2"), "v1")
    assert answer_key(professional, "v1") != answer_key(professional, "v2")
    assert answer_key(professional, "v1") != answer_key(
        request("public alert v1", "regulator-only memo"), "v1"
    )


async def test_answers_work_while_redis_is_down() -> None:
    cache = AnswerCache(AsyncDownRedis(), ttl_seconds=60)  # type: ignore[arg-type]

    assert await cache.get("key") is None
    await cache.put("key", "text")  # no error


class SyncDownRedis:
    def __getattr__(self, name: str) -> Any:
        from redis.exceptions import ConnectionError as RedisConnectionError

        def fail(*args: Any, **kwargs: Any) -> Any:
            raise RedisConnectionError("down")

        return fail


def test_products_work_while_redis_is_down() -> None:
    product = ProductFacts(uuid4(), "A4-1234", "Panadol", "Paracetamol", RegistrationStatus.ACTIVE)
    cache = ProductCache(SyncDownRedis(), ttl_seconds=60)  # type: ignore[arg-type]

    cache.put("A4-1234", product)  # no error
    assert cache.get("A4-1234") == (False, None)  # a miss: the database answers


def test_a_register_import_clears_cached_products(monkeypatch: Any) -> None:
    from medlens.modules.products.cache import clear_product_facts, product_facts_key
    from medlens.seed import greenbook

    class ScanRedis(DictRedis):
        def scan_iter(self, match: str, count: int) -> Any:
            prefix = match.rstrip("*")
            return iter([key for key in list(self.data) if key.startswith(prefix)])

        def unlink(self, *keys: str) -> int:
            for key in keys:
                self.data.pop(key, None)
            return len(keys)

        def close(self) -> None:
            pass

    redis = ScanRedis()
    cache = ProductCache(redis, ttl_seconds=60)  # type: ignore[arg-type]
    for number in ("A4-0001", "A4-0002", "Z9-000003"):
        cache.put(number, None)
    redis.data["medlens:v1:other"] = "kept"

    assert clear_product_facts(redis) == 3  # type: ignore[arg-type]
    assert redis.data == {"medlens:v1:other": "kept"}
    assert product_facts_key("A4-0001").startswith("medlens:v1:catalog:product:")
    assert clear_product_facts(SyncDownRedis()) == 0  # type: ignore[arg-type]

    cache.put("A4-0001", None)
    monkeypatch.setattr("medlens.cache.redis.create_sync_redis_client", lambda settings, db: redis)
    greenbook.clear_cached_products(build_settings())
    assert redis.data == {"medlens:v1:other": "kept"}


async def test_hotspot_caching_survives_redis_being_down() -> None:
    from medlens.cache.aside import cached_json
    from medlens.modules.hotspots.service import _generation, invalidate_hotspots

    down = AsyncDownRedis()

    await invalidate_hotspots(down)  # type: ignore[arg-type]  # no error: entries expire anyway
    assert await _generation(down) == "unavailable"  # type: ignore[arg-type]

    async def compute() -> dict[str, str]:
        return {"clusters": "fresh"}

    assert await cached_json(down, "key", 60, compute) == ({"clusters": "fresh"}, False)  # type: ignore[arg-type]

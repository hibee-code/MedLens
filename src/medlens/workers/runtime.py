"""Per-process worker resources: database engine, object storage, Redis, the LLM gateway, and
the pipeline's readers and models. Created once per worker process after the fork
(``worker_process_init``), never per task (CLAUDE.md §7), and disposed at shutdown. Tests
install their own with ``set_runtime``.

Readers: OpenCV, zxing-cpp and Tesseract (MEDLENS_VERIFICATION__READERS=real, the `vision`
group), the vision LLM through the gateway, and sentence-transformers embeddings (the `ml`
group). A reader or model that is not installed leaves its signal unavailable, which lowers
confidence instead of failing scans: without Tesseract, barcodes and the vision LLM read the
pack.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import structlog
from redis import Redis
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from medlens.adapters.fakes import FakeBarcode, FakeImageAnalysis
from medlens.adapters.llm import GatewayPackReader
from medlens.adapters.s3 import create_sync_s3_client
from medlens.adapters.storage import SyncS3Storage
from medlens.ai.gateway.gateway import LLMGateway
from medlens.ai.gateway.ledger import SyncLedger
from medlens.ai.gateway.providers import AnthropicProvider
from medlens.core.config import Settings
from medlens.db.session import create_sync_engine, create_sync_session_factory
from medlens.modules.scans.service import scan_done_channel
from medlens.modules.verification.catalog import Catalog
from medlens.modules.verification.pipeline import VerificationPipeline
from medlens.modules.verification.service import (
    VerificationService,
    pipeline_config,
    scorer_for,
)
from medlens.ports.barcode import BarcodePort
from medlens.ports.embedding import EmbeddingPort, TextEmbeddingPort
from medlens.ports.image_analysis import ImageAnalysisPort
from medlens.ports.ocr import OcrEngine
from medlens.ports.storage import SyncStoragePort

log = structlog.stdlib.get_logger(__name__)


@dataclass
class WorkerRuntime:
    settings: Settings
    sessions: sessionmaker[Session]
    storage: SyncStoragePort
    redis: Redis
    analyzer: ImageAnalysisPort
    embedder: EmbeddingPort | None
    verification: VerificationService
    text_embedder: TextEmbeddingPort | None = None
    gateway: LLMGateway | None = None
    engine: Engine | None = None

    def announce_scan_done(self, scan_id: str) -> None:
        """Best effort: a waiting API request falls back to polling if this is lost."""
        try:
            self.redis.publish(scan_done_channel(scan_id), "done")
        except Exception as exc:  # noqa: BLE001 - never fail a finished scan over a notification
            log.warning("scan.done_publish_failed", error=type(exc).__name__)

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
        self.redis.close()


def build_readers(settings: Settings) -> tuple[ImageAnalysisPort, BarcodePort, OcrEngine | None]:
    if settings.verification.readers == "fake":
        return FakeImageAnalysis(), FakeBarcode(), None
    from medlens.adapters.vision import (
        OpenCvImageAnalysis,
        TesseractOcr,
        ZxingBarcode,
        tesseract_available,
    )

    ocr = TesseractOcr() if tesseract_available() else None
    if ocr is None:
        log.warning("ocr.unavailable", reason="tesseract is not installed")
    return OpenCvImageAnalysis(), ZxingBarcode(), ocr


def build_embedders(settings: Settings) -> tuple[EmbeddingPort | None, TextEmbeddingPort | None]:
    embeddings = settings.embeddings
    if not embeddings.enabled:
        return None, None
    try:
        from medlens.adapters.embeddings import (
            SentenceTransformerImageEmbedder,
            SentenceTransformerTextEmbedder,
        )

        return (
            SentenceTransformerImageEmbedder(embeddings.image_model, embeddings.image_dimension),
            SentenceTransformerTextEmbedder(embeddings.text_model, embeddings.text_dimension),
        )
    except Exception as exc:  # noqa: BLE001 - no models: their signals are unavailable
        log.warning("embeddings.unavailable", error=type(exc).__name__)
        return None, None


def build_runtime(settings: Settings) -> WorkerRuntime:
    engine = create_sync_engine(
        settings.database, application_name=f"{settings.observability.service_name}-worker"
    )
    sessions = create_sync_session_factory(engine)
    storage = SyncS3Storage(create_sync_s3_client(settings.s3), settings.s3.bucket)
    redis = Redis.from_url(
        settings.redis.url(settings.redis.cache_db),
        socket_timeout=settings.redis.socket_timeout_seconds,
        socket_connect_timeout=settings.redis.socket_connect_timeout_seconds,
    )
    analyzer, barcode, ocr = build_readers(settings)
    embedder, text_embedder = build_embedders(settings)
    gateway = LLMGateway(settings.llm, AnthropicProvider(settings.llm), SyncLedger(sessions, redis))
    reader = GatewayPackReader(gateway)
    config, scorer = pipeline_config(settings), scorer_for(settings)

    def pipeline(catalog: Catalog) -> VerificationPipeline:
        return VerificationPipeline(
            catalog=catalog,
            scorer=scorer,
            analyzer=analyzer,
            barcode=barcode,
            ocr=ocr,
            llm=reader,
            embedder=embedder,
            config=config,
        )

    verification = VerificationService(
        sessions,
        storage,
        pipeline,
        duplicate_window=timedelta(hours=settings.verification.duplicate_window_hours),
    )
    return WorkerRuntime(
        settings=settings,
        sessions=sessions,
        storage=storage,
        redis=redis,
        analyzer=analyzer,
        embedder=embedder,
        text_embedder=text_embedder,
        verification=verification,
        gateway=gateway,
        engine=engine,
    )


_runtime: WorkerRuntime | None = None


def get_runtime() -> WorkerRuntime:
    """The process's runtime; built on first use where no worker_process_init ran (Beat,
    eager tasks)."""
    global _runtime  # noqa: PLW0603 - one runtime per process, by design
    if _runtime is None:
        from medlens.core.config import get_settings

        _runtime = build_runtime(get_settings())
    return _runtime


def set_runtime(runtime: WorkerRuntime | None) -> WorkerRuntime | None:
    """Install a runtime (tests); returns the previous one."""
    global _runtime
    previous, _runtime = _runtime, runtime
    return previous


def init_runtime(**_: Any) -> None:
    """``worker_process_init``: connections and models after the fork, once per process."""
    get_runtime()


def shutdown_runtime(**_: Any) -> None:
    """``worker_process_shutdown``."""
    runtime = set_runtime(None)
    if runtime is not None:
        runtime.close()

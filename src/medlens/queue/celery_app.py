"""Celery application: queues, routing and limits.

Configuration is registered as a lazy default, so importing this module never reads settings;
they load when Celery first needs its configuration.

    celery -A medlens.queue.celery_app:celery_app worker -Q scans.pipeline,events,...
    celery -A medlens.queue.celery_app:celery_app beat

Queues (one worker pool per queue in production, so a backlog in one cannot starve another):

    scans.pipeline  scan processing, highest priority. Runs the whole verification pipeline in
                    one task on ML-capable workers (the image is fetched once; no hand-offs).
    ml.ocr          standalone OCR jobs (benchmarks, re-reads); CPU-heavy, concurrency = cores.
    ml.embed        reference-image and re-embedding jobs; CPU-heavy.
    llm             model calls, rate limited by Celery and the gateway's circuit breaker.
    events          outbox relay and follow-ups (case opening, revocation publishing).
    maintenance     partitions, retention, hotspot recomputation.

Results are not used: business results live in Postgres (``task_ignore_result``). Messages are
JSON only, acknowledged after the task finishes, and re-delivered if a worker dies.
"""

from collections.abc import Callable
from typing import Any

from celery import Celery
from celery.signals import setup_logging

from medlens.core.bootstrap import configure_process
from medlens.core.config import Settings, get_settings
from medlens.queue.backpressure import PRIORITY_SEP, PRIORITY_STEPS
from medlens.queue.beat_schedule import beat_schedule

QUEUES = ("scans.pipeline", "ml.ocr", "ml.embed", "llm", "events", "maintenance")
TASK_ROUTES = {
    "medlens.scans.*": {"queue": "scans.pipeline"},
    "medlens.ocr.*": {"queue": "ml.ocr"},
    "medlens.reference_images.*": {"queue": "ml.embed"},
    "medlens.knowledge.*": {"queue": "ml.embed"},
    "medlens.llm.*": {"queue": "llm"},
    "medlens.outbox.*": {"queue": "events"},
    "medlens.cases.*": {"queue": "events"},
    "medlens.auth.*": {"queue": "events"},
    "medlens.hotspots.*": {"queue": "maintenance"},
    "medlens.maintenance.*": {"queue": "maintenance"},
}
TASK_MODULES = ("medlens.workers.tasks", "medlens.workers.failures")


def build_celery_config(settings: Settings) -> dict[str, Any]:
    redis, queue = settings.redis, settings.queue
    return {
        "broker_url": redis.url(redis.broker_db),
        "result_backend": redis.url(redis.result_db),
        # JSON only: pickle would let anyone who can write to the broker run code on workers.
        "task_serializer": "json",
        "result_serializer": "json",
        "accept_content": ["json"],
        "timezone": "UTC",
        "enable_utc": True,
        "task_ignore_result": True,  # except query embeddings, which the API waits for
        "result_expires": 300,
        "task_acks_late": True,
        "task_reject_on_worker_lost": True,
        "worker_prefetch_multiplier": 1,  # long CPU-bound tasks: never hoard messages
        "task_soft_time_limit": queue.task_soft_time_limit_seconds,
        "task_time_limit": queue.task_time_limit_seconds,
        "task_default_queue": "events",
        "task_default_priority": 5,
        "task_routes": TASK_ROUTES,  # queues are created on first use (task_create_missing_queues)
        "broker_transport_options": {
            # Must exceed the longest task, or a running task is delivered twice.
            "visibility_timeout": queue.visibility_timeout_seconds,
            "priority_steps": list(PRIORITY_STEPS),
            "sep": PRIORITY_SEP,
            "queue_order_strategy": "priority",
        },
        "broker_connection_retry_on_startup": True,
        "worker_hijack_root_logger": False,
        "imports": TASK_MODULES,
        "beat_schedule": beat_schedule(settings),
    }


def _configure_worker_process(**_: Any) -> None:
    configure_process(get_settings())


celery_app = Celery("medlens")
celery_app.add_defaults(lambda: build_celery_config(get_settings()))
# Connecting to setup_logging stops Celery from installing its own logging handlers; we use the
# hook to run the same process setup as the API (logging, field encryption, image limits).
setup_logging.connect(_configure_worker_process)


def task(**options: Any) -> Callable[[Callable[..., Any]], Any]:
    """``celery_app.task`` with a type mypy accepts (Celery ships no type hints)."""
    decorator: Callable[[Callable[..., Any]], Any] = celery_app.task(**options)
    return decorator

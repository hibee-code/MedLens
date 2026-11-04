"""Prometheus metrics for Celery workers.

Prefork children each count in their own memory, so in the worker image
PROMETHEUS_MULTIPROC_DIR points prometheus_client at a shared directory: children write their
samples there, and the main process serves the sum on
MEDLENS_OBSERVABILITY__WORKER_METRICS_PORT (0: off). Without the variable (development,
``--pool=solo``) the process's own registry is served.
"""

import os
import shutil
from pathlib import Path
from typing import Any

import structlog
from prometheus_client import REGISTRY, CollectorRegistry, multiprocess, start_http_server

from medlens.core.config import get_settings

log = structlog.stdlib.get_logger(__name__)
MULTIPROC_ENV = "PROMETHEUS_MULTIPROC_DIR"


def registry() -> CollectorRegistry:
    directory = os.environ.get(MULTIPROC_ENV)
    if not directory:
        return REGISTRY
    collected = CollectorRegistry()
    multiprocess.MultiProcessCollector(collected, path=directory)  # type: ignore[no-untyped-call]
    return collected


def reset_multiproc_dir() -> None:
    """Samples from a previous run would be summed into this one."""
    directory = os.environ.get(MULTIPROC_ENV)
    if directory:
        shutil.rmtree(directory, ignore_errors=True)
        Path(directory).mkdir(parents=True, exist_ok=True)


def start_metrics_server(**_: Any) -> None:
    """``worker_init`` (main process, before the pool forks)."""
    port = get_settings().observability.worker_metrics_port
    if not port:
        return
    reset_multiproc_dir()
    start_http_server(port, registry=registry())
    log.info("worker.metrics_serving", port=port)


def mark_process_dead(**_: Any) -> None:
    """``worker_process_shutdown``: a dead child's live gauges stop counting."""
    if os.environ.get(MULTIPROC_ENV):
        multiprocess.mark_process_dead(os.getpid())  # type: ignore[no-untyped-call]

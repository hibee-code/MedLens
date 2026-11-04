"""Scan metrics, recorded by the worker that scores the scan (see workers/metrics.py for how
worker processes expose them)."""

from prometheus_client import Counter, Histogram

from medlens.modules.verification.pipeline import PipelineResult

SCANS = Counter("medlens_scans", "Scans processed, by verdict and outcome.", ["verdict", "outcome"])
SCAN_DURATION = Histogram(
    "medlens_scan_duration_seconds",
    "Worker time to verify one scan, from pick-up to saved result.",
    buckets=(0.25, 0.5, 1, 2, 4, 8, 15, 30, 60),
)
STAGE_DURATION = Histogram(
    "medlens_scan_stage_duration_seconds",
    "Time spent in each pipeline stage that ran.",
    ["stage"],
    buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)


def record_scan(verdict: str | None, outcome: str, seconds: float, result: PipelineResult) -> None:
    SCANS.labels(verdict=verdict or "none", outcome=outcome).inc()
    SCAN_DURATION.observe(seconds)
    for stage in result.stages:
        if stage.outcome != "skipped":
            STAGE_DURATION.labels(stage=stage.stage).observe(stage.duration_ms / 1000)

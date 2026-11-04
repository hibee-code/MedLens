"""Health metrics. Alert rules: docker/observability/alerts.yml."""

from prometheus_client import Gauge

DEPENDENCY_UP = Gauge(
    "medlens_dependency_up",
    "1 when the latest probe of a dependency succeeded, 0 when it failed or timed out.",
    ["dependency"],
)

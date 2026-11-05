"""docker/observability/alerts.yml only uses metrics the code defines."""

import re
from pathlib import Path

import yaml
from prometheus_client import REGISTRY

import medlens.ai.gateway.metrics
import medlens.modules.auth.metrics
import medlens.modules.health.metrics
import medlens.modules.verification.metrics
import medlens.queue.backpressure
import medlens.queue.relay
import medlens.workers.failures  # noqa: F401 - registers the metrics
from medlens.main import _http_metrics

_http_metrics()  # the HTTP request metrics (prometheus-fastapi-instrumentator)

ROOT = Path(__file__).resolve().parents[2]
RULES = yaml.safe_load((ROOT / "docker/observability/alerts.yml").read_text())
SCRAPE = yaml.safe_load((ROOT / "docker/observability/prometheus.yml").read_text())


def defined_samples() -> set[str]:
    names: set[str] = set()
    suffixes = {"counter": ("_total",), "histogram": ("_bucket", "_sum", "_count")}
    for family in REGISTRY.collect():
        names.add(family.name)
        # Labelled metrics have no samples until first used: add the series they will have.
        names.update(family.name + suffix for suffix in suffixes.get(family.type, ()))
        names.update(sample.name for sample in family.samples)
    return names


def test_alert_expressions_use_defined_metrics() -> None:
    defined = defined_samples()
    for group in RULES["groups"]:
        for rule in group["rules"]:
            used = set(re.findall(r"\b(?:medlens|http)_[a-z_]+", rule["expr"]))
            assert used, rule["alert"]
            assert used <= defined, f"{rule['alert']}: {used - defined}"


def test_redis_down_is_alerted() -> None:
    alerts = {rule["alert"]: rule for group in RULES["groups"] for rule in group["rules"]}

    assert 'dependency="redis"' in alerts["MedLensRedisDown"]["expr"]
    assert alerts["MedLensRedisDown"]["labels"]["severity"] == "critical"
    assert "MedLensRevocationChecksSkipped" in alerts


def test_prometheus_loads_the_rules_and_scrapes_the_api() -> None:
    jobs = {job["job_name"]: job for job in SCRAPE["scrape_configs"]}

    assert SCRAPE["rule_files"] == ["/etc/prometheus/alerts.yml"]
    assert jobs["api"]["metrics_path"] == "/metrics"


def test_operations_are_alerted() -> None:
    alerts = {rule["alert"]: rule for group in RULES["groups"] for rule in group["rules"]}

    assert {
        "MedLensScanQueueBacklog",
        "MedLensScanQueueFull",
        "MedLensScansSlow",
        "MedLensScansFailing",
        "MedLensOutboxDeadLetters",
        "MedLensTasksFailing",
        "MedLensLLMBudgetNearlySpent",
        "MedLensLLMBudgetSpent",
        "MedLensLLMErrors",
        "MedLensHTTPErrors",
    } <= set(alerts)
    assert all(rule["labels"]["severity"] in {"warning", "critical"} for rule in alerts.values())


def test_the_dashboard_queries_defined_metrics() -> None:
    import json

    board = json.loads(
        (ROOT / "docker/observability/grafana/provisioning/dashboards/medlens.json").read_text()
    )
    defined = defined_samples()
    queries = [target["expr"] for panel in board["panels"] for target in panel.get("targets", [])]
    assert len(queries) >= 16
    for expr in queries:
        used = set(re.findall(r"\b(?:medlens|http)_[a-z_]+", expr))
        assert used, expr
        assert used <= defined, f"{expr}: {used - defined}"

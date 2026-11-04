"""docker/observability/alerts.yml only uses metrics the code defines."""

import re
from pathlib import Path

import yaml
from prometheus_client import REGISTRY

import medlens.modules.auth.metrics
import medlens.modules.health.metrics  # noqa: F401 - registers the metrics

ROOT = Path(__file__).resolve().parents[2]
RULES = yaml.safe_load((ROOT / "docker/observability/alerts.yml").read_text())
SCRAPE = yaml.safe_load((ROOT / "docker/observability/prometheus.yml").read_text())


def defined_samples() -> set[str]:
    names: set[str] = set()
    for family in REGISTRY.collect():
        names.add(family.name)
        names.update(sample.name for sample in family.samples)
    return names


def test_alert_expressions_use_defined_metrics() -> None:
    defined = defined_samples()
    for group in RULES["groups"]:
        for rule in group["rules"]:
            used = set(re.findall(r"\bmedlens_[a-z_]+", rule["expr"]))
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

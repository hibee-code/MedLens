"""Write the Grafana dashboard (docker/observability/grafana/provisioning/dashboards/medlens.json).

Panels are listed here, not edited in Grafana: the file is provisioned read-only and a test
checks that every query uses metrics the code defines. Run after changing the panels.
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docker/observability/grafana/provisioning/dashboards"
DATASOURCE = {"type": "prometheus", "uid": "prometheus"}

P95 = "histogram_quantile(0.95, sum by ({by}) (rate({metric}_bucket[10m])))"


def panel(title: str, expr: str, legend: str, unit: str = "short") -> tuple[str, str, str, str]:
    return title, expr, legend, unit


ROWS: list[tuple[str, list[tuple[str, str, str, str]]]] = [
    (
        "API",
        [
            panel(
                "Requests per second",
                "sum by (handler) (rate(http_requests_total[5m]))",
                "{{handler}}",
                "reqps",
            ),
            panel(
                "Latency p95",
                P95.format(by="le, handler", metric="http_request_duration_seconds"),
                "{{handler}}",
                "s",
            ),
            panel(
                "5xx share",
                'sum(rate(http_requests_total{status=~"5.."}[5m]))'
                " / sum(rate(http_requests_total[5m]))",
                "5xx",
                "percentunit",
            ),
            panel(
                "Dependencies up", "min by (dependency) (medlens_dependency_up)", "{{dependency}}"
            ),
        ],
    ),
    (
        "Scans",
        [
            panel(
                "Scans per minute by verdict",
                "sum by (verdict) (rate(medlens_scans_total[5m])) * 60",
                "{{verdict}}",
            ),
            panel(
                "Scan time p95",
                P95.format(by="le", metric="medlens_scan_duration_seconds"),
                "p95",
                "s",
            ),
            panel(
                "Stage time p95",
                P95.format(by="le, stage", metric="medlens_scan_stage_duration_seconds"),
                "{{stage}}",
                "s",
            ),
            panel("Queue depth", "max by (queue) (medlens_queue_depth)", "{{queue}}"),
        ],
    ),
    (
        "AI",
        [
            panel(
                "Model calls per minute",
                "sum by (route, outcome) (rate(medlens_llm_calls_total[5m])) * 60",
                "{{route}} {{outcome}}",
            ),
            panel(
                "Model cost per hour",
                "sum by (route) (rate(medlens_llm_cost_usd_total[1h])) * 3600",
                "{{route}}",
                "currencyUSD",
            ),
            panel(
                "Budget used today",
                "max(medlens_llm_spent_today_usd) / max(medlens_llm_daily_budget_usd)",
                "used",
                "percentunit",
            ),
            panel(
                "Assistant outcomes per minute",
                "sum by (outcome) (rate(medlens_assistant_requests_total[15m])) * 60",
                "{{outcome}}",
            ),
        ],
    ),
    (
        "Safety and failures",
        [
            panel(
                "Rate-limit rejections per minute",
                "sum by (bucket) (rate(medlens_ratelimit_rejections_total[5m])) * 60",
                "{{bucket}}",
            ),
            panel(
                "Revocation checks skipped per minute",
                "sum(rate(medlens_auth_revocation_checks_skipped_total[5m])) * 60",
                "skipped",
            ),
            panel(
                "Tasks failed for good (1h)",
                "sum by (task) (increase(medlens_tasks_failed_total[1h]))",
                "{{task}}",
            ),
            panel(
                "Outbox dead letters (1h)",
                "sum by (event_type) (increase(medlens_outbox_dead_lettered_total[1h]))",
                "{{event_type}}",
            ),
        ],
    ),
]


def dashboard() -> dict[str, object]:
    panels: list[dict[str, object]] = []
    y = 0
    for row_id, (title, row) in enumerate(ROWS):
        panels.append(
            {
                "type": "row",
                "title": title,
                "id": 100 + row_id,
                "collapsed": False,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
            }
        )
        y += 1
        for index, (name, expr, legend, unit) in enumerate(row):
            panels.append(
                {
                    "id": 100 * (row_id + 2) + index,
                    "type": "timeseries",
                    "title": name,
                    "datasource": DATASOURCE,
                    "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
                    "gridPos": {"h": 8, "w": 6, "x": 6 * index, "y": y},
                    "targets": [
                        {
                            "datasource": DATASOURCE,
                            "expr": expr,
                            "legendFormat": legend,
                            "refId": "A",
                        }
                    ],
                }
            )
        y += 8
    return {
        "uid": "medlens-overview",
        "title": "MedLens overview",
        "tags": ["medlens"],
        "timezone": "utc",
        "schemaVersion": 39,
        "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "panels": panels,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "medlens.json").write_text(json.dumps(dashboard(), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

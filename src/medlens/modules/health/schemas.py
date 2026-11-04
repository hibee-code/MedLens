from typing import Literal

from pydantic import ConfigDict, Field

from medlens.core.schemas import ReadModel

CheckStatus = Literal["ok", "error", "timeout"]
ReadinessStatus = Literal["ready", "degraded", "not_ready"]


class LivenessResponse(ReadModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"status": "ok"}]})

    status: Literal["ok"] = "ok"


class DependencyCheck(ReadModel):
    status: CheckStatus = Field(description="ok, error (probe failed) or timeout.")
    latency_ms: float = Field(description="Time the probe took, in milliseconds.")
    error: str | None = Field(
        default=None, description="Exception type when the probe failed; never a message."
    )


class ReadinessResponse(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ready",
                    "checks": {
                        "postgres": {"status": "ok", "latency_ms": 1.9},
                        "redis": {"status": "ok", "latency_ms": 0.6},
                        "s3": {"status": "ok", "latency_ms": 4.2},
                    },
                },
                {
                    "status": "degraded",
                    "checks": {
                        "postgres": {"status": "ok", "latency_ms": 2.1},
                        "redis": {"status": "timeout", "latency_ms": 2000.4},
                        "s3": {"status": "ok", "latency_ms": 3.8},
                    },
                },
                {
                    "status": "not_ready",
                    "checks": {
                        "postgres": {"status": "ok", "latency_ms": 2.3},
                        "redis": {"status": "ok", "latency_ms": 0.7},
                        "s3": {"status": "error", "latency_ms": 3.0, "error": "ClientError"},
                    },
                },
            ]
        }
    )

    status: ReadinessStatus = Field(
        description=(
            "ready: every dependency answered. degraded: only Redis failed; the instance keeps "
            "serving on its fallbacks. not_ready: Postgres or object storage failed."
        )
    )
    checks: dict[str, DependencyCheck]

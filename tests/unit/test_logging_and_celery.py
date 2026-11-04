import contextvars
import json
import os
import subprocess
import sys

import pytest
import structlog

from medlens.core.context import request_scope
from medlens.core.logging import configure_logging
from medlens.queue.celery_app import build_celery_config


def test_json_logs_carry_request_context(settings_factory, capsys: pytest.CaptureFixture[str]):
    configure_logging(settings_factory(observability={"log_format": "json"}))
    log = structlog.stdlib.get_logger("medlens.test")

    def emit() -> None:
        with request_scope("0190a8d4-0000-7000-8000-000000000000"):
            log.info("scan.scored", verdict="SUSPICIOUS")

    contextvars.copy_context().run(emit)

    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["event"] == "scan.scored"
    assert line["verdict"] == "SUSPICIOUS"
    assert line["request_id"] == "0190a8d4-0000-7000-8000-000000000000"
    assert line["level"] == "info"
    assert line["timestamp"].endswith("Z")


def test_stdlib_logs_are_rendered_by_structlog(settings_factory, capsys) -> None:
    import logging

    configure_logging(settings_factory(observability={"log_format": "json"}))
    logging.getLogger("uvicorn.error").warning("server notice")

    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line == line | {"event": "server notice", "logger": "uvicorn.error", "level": "warning"}


def test_celery_uses_json_and_separate_redis_databases(settings_factory) -> None:
    config = build_celery_config(settings_factory())

    assert config["accept_content"] == ["json"]
    assert config["task_serializer"] == config["result_serializer"] == "json"
    assert config["broker_url"].endswith("/1")
    assert config["result_backend"].endswith("/2")
    assert config["task_acks_late"] is True


def test_importing_celery_app_does_not_load_settings() -> None:
    """In a fresh interpreter: other tests configure the shared app."""
    code = (
        "import medlens.queue.celery_app as m, medlens.workers.tasks; "
        "assert m.celery_app.configured is False"
    )
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("MEDLENS_")
    }

    result = subprocess.run(  # noqa: S603 - fixed arguments
        [sys.executable, "-c", code], env=environment, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr.decode()

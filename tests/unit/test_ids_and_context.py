import contextvars
import time
import uuid

from medlens.core.context import org_id_var, request_scope, snapshot, user_id_var
from medlens.core.ids import uuid7


def test_uuid7_has_version_variant_and_timestamp() -> None:
    before_ms = time.time_ns() // 1_000_000
    value = uuid7()
    after_ms = time.time_ns() // 1_000_000

    assert value.version == 7
    assert value.variant == uuid.RFC_4122
    assert before_ms <= value.int >> 80 <= after_ms


def test_uuid7_sorts_by_creation_time() -> None:
    first = uuid7()
    time.sleep(0.002)
    second = uuid7()

    assert first < second


def test_request_scope_binds_and_restores_context() -> None:
    def scenario() -> None:
        user_id = uuid.uuid4()
        user_id_var.set(user_id)  # leftover from an outer scope

        with request_scope("req-1", trace_id="trace-1"):
            assert snapshot() == {"request_id": "req-1", "trace_id": "trace-1"}
            org_id = uuid.uuid4()
            org_id_var.set(org_id)
            assert snapshot()["org_id"] == str(org_id)

        assert snapshot() == {"user_id": str(user_id)}

    contextvars.copy_context().run(scenario)  # keep the leftover out of other tests

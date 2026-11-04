"""M6: rate limits (429 problem+json with Retry-After), security headers, metrics and the AI
usage report."""

from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import pytest
from redis.asyncio import Redis

from medlens.ai.gateway.ledger import AsyncLedger, CallRecord
from medlens.ai.gateway.pricing import TokenUsage
from medlens.core.config import Settings
from medlens.modules.control_plane.models import LLMCallOutcome
from medlens.modules.rbac.models import RoleName
from medlens.ratelimit.limiter import SlidingWindowLimiter
from tests.integration.api_support import Api, open_api, photo
from tests.integration.support import make_settings

pytestmark = pytest.mark.integration

LIMITS = {
    "enabled": True,
    "guest_scans_per_minute": 2,
    "user_scans_per_minute": 3,
    "logins_per_minute": 2,
    "assistant_per_minute": 1,
}


@pytest.fixture
async def limited(
    migrated_settings: Settings, database_groups: dict[str, Any], redis_groups: dict[str, Any]
) -> AsyncIterator[Api]:
    settings = make_settings(database=database_groups, redis=redis_groups, ratelimit=LIMITS)
    async with open_api(settings) as api:
        yield api


async def scan(api: Api, headers: dict[str, str]) -> Any:
    return await api.client.post(
        "/api/v1/scans", files={"image": ("p.jpg", photo(), "image/jpeg")}, headers=headers
    )


# ------------------------------------------------------------------------- rate limits
async def test_guest_scans_are_limited_per_client(limited: Api) -> None:
    device = {"X-Device-Id": f"device-{uuid4().hex}"}

    statuses = [(await scan(limited, device)).status_code for _ in range(3)]
    refused = await scan(limited, {"X-Device-Id": f"device-{uuid4().hex}"})  # same address

    assert statuses == [202, 202, 429]
    assert refused.status_code == 429
    assert refused.headers["content-type"] == "application/problem+json"
    assert refused.json()["type"] == "urn:medlens:problem:rate-limited"
    assert 1 <= int(refused.headers["retry-after"]) <= 60


async def test_signed_in_users_have_their_own_allowance(limited: Api) -> None:
    for _ in range(2):
        await scan(limited, {"X-Device-Id": f"device-{uuid4().hex}"})  # guests use theirs up
    user = await limited.create_user()
    headers = limited.bearer(limited.token(user))

    statuses = [(await scan(limited, headers)).status_code for _ in range(4)]

    assert statuses == [202, 202, 202, 429]


async def test_sign_in_attempts_are_limited(limited: Api) -> None:
    statuses = [
        (
            await limited.client.post(
                "/api/v1/auth/login",
                json={"email": f"{uuid4().hex}@example.org", "password": "wrong-password-x"},
            )
        ).status_code
        for _ in range(3)
    ]
    otp = await limited.client.post("/api/v1/auth/otp/request", json={"phone": "+2348031112222"})

    assert statuses == [401, 401, 429]
    assert otp.status_code == 429  # one sign-in bucket per address


async def test_assistant_questions_are_limited(limited: Api) -> None:
    user, organization = await limited.user_with_role(RoleName.PHARMACY_STAFF)
    headers = limited.bearer(limited.token(user, organization=organization))
    limited.use_llm()

    first = await limited.client.post(
        "/api/v1/assistant/ask", json={"question": "anything new xyzzy?"}, headers=headers
    )
    second = await limited.client.post(
        "/api/v1/assistant/ask", json={"question": "anything new xyzzy?"}, headers=headers
    )

    assert first.status_code == 200
    assert second.status_code == 429


async def test_limits_are_skipped_while_redis_is_down() -> None:
    down = Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2)
    limiter = SlidingWindowLimiter(down)

    assert await limiter.retry_after("scans_guest", "x", limit=1) is None
    await down.aclose()


# ------------------------------------------------------------------------- headers and metrics
async def test_security_headers_are_on_every_response(api: Api) -> None:
    ok = await api.client.get("/health/live")
    error = await api.client.get("/api/v1/scans/not-a-uuid", headers={"X-Device-Id": "d-1"})

    for response in (ok, error):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["strict-transport-security"].startswith("max-age=")
    assert error.headers["cache-control"] == "no-store"


async def test_metrics_include_http_and_scan_metrics(api: Api) -> None:
    product = await api.create_product()
    response = await scan(api, {"X-Device-Id": f"device-{uuid4().hex}"})
    api.worker(text=f"NAFDAC REG NO: {product.nafdac_number}").process_scan(
        UUID(response.json()["id"])
    )

    metrics = (await api.client.get("/metrics")).text

    assert "http_requests_total" in metrics
    assert 'handler="/api/v1/scans"' in metrics
    assert "medlens_scans_total" in metrics
    assert "medlens_scan_duration_seconds_bucket" in metrics
    assert 'medlens_scan_stage_duration_seconds_count{stage="quality"}' in metrics
    assert "medlens_llm_calls_total" in metrics or "medlens_llm" not in metrics


# ------------------------------------------------------------------------- AI usage
async def test_platform_admins_see_ai_usage_and_budget(api: Api, app_settings: Settings) -> None:
    ledger = AsyncLedger(api.resources.session_factory, api.resources.redis)
    await ledger.record(
        CallRecord(
            route="assistant_answer",
            tier="balanced",
            requested_model="claude-sonnet-5-5",
            model="claude-sonnet-5-5",
            prompt_version="assistant_answer@v1",
            usage=TokenUsage(input=3000, output=400, cache_read=2000),
            cost_usd=Decimal("0.010400"),
            latency_ms=900,
            attempts=1,
            outcome=LLMCallOutcome.OK,
        )
    )
    admin = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)
    regulator, organization = await api.user_with_role(RoleName.REGULATOR_ADMIN)

    usage = await api.client.get("/api/v1/admin/ai/usage", headers=api.bearer(api.token(admin)))
    refused = await api.client.get(
        "/api/v1/admin/ai/usage",
        headers=api.bearer(api.token(regulator, organization=organization)),
    )

    body = usage.json()
    assert usage.status_code == 200, usage.text
    assert body["ai_enabled"] is app_settings.llm.enabled
    assert body["models"]["balanced"] == app_settings.llm.model_balanced
    today = next(row for row in body["days"] if row["route"] == "assistant_answer")
    assert today["calls"] >= 1
    assert Decimal(today["cost_usd"]) >= Decimal("0.0104")
    assert Decimal(body["budget"]["spent_today_usd"]) >= Decimal("0.0104")
    assert 0 < body["budget"]["share_used"] <= 1
    assert refused.status_code == 403

"""Session revocation while Redis is down: consumer actions fail open, privileged sessions and
sensitive account routes fall back to users.tokens_valid_after in Postgres."""

import time
from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import REGISTRY
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import select

from medlens.db.session import UnitOfWork
from medlens.modules.rbac.models import RoleName
from medlens.modules.users.models import User
from tests.integration.api_support import PASSWORD, Api, new_email

pytestmark = pytest.mark.integration


def two_minutes_ago() -> int:
    """Within the access-token lifetime, before end_all_sessions."""
    return int(time.time()) - 120


def metric(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
def redis_down(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    """The revocation lookup's Redis call fails; the rest of Redis keeps working (the permission
    cache has its own fallback, tested elsewhere)."""

    async def unavailable(*keys: str) -> int:
        raise RedisConnectionError("redis down")

    monkeypatch.setattr(api.resources.redis, "exists", unavailable)


async def end_all_sessions(api: Api, user: User) -> None:
    """What log-out-everywhere, a password change or erasure leaves behind in Postgres."""
    async with UnitOfWork(api.resources.session_factory) as uow:
        stored = await uow.session.get(User, user.id)
        assert stored is not None
        stored.tokens_valid_after = datetime.now(UTC) - timedelta(minutes=1)
        await uow.commit()


async def test_consumer_actions_fail_open_and_are_counted(api: Api, redis_down: None) -> None:
    consumer = await api.create_user()
    await end_all_sessions(api, consumer)
    skipped = metric("medlens_auth_revocation_checks_skipped_total")

    response = await api.client.get(
        "/api/v1/_probe/scan:create",
        headers=api.bearer(api.token(consumer, issued_at=two_minutes_ago())),
    )

    assert response.status_code == 200
    assert metric("medlens_auth_revocation_checks_skipped_total") == skipped + 1


async def test_privileged_sessions_fall_back_to_postgres(api: Api, redis_down: None) -> None:
    user, organization = await api.user_with_role(RoleName.PHARMACY_STAFF)
    await end_all_sessions(api, user)
    revoked = metric("medlens_auth_revocation_fallbacks_total", result="revoked")
    active = metric("medlens_auth_revocation_fallbacks_total", result="active")

    old = await api.client.get(
        "/api/v1/_probe/scan:create",  # a consumer permission, but a privileged session
        headers=api.bearer(api.token(user, organization=organization, issued_at=two_minutes_ago())),
    )
    new = await api.client.get(
        "/api/v1/_probe/scan:read_org",
        headers=api.bearer(api.token(user, organization=organization)),
    )

    assert old.status_code == 401
    assert "session has ended" in old.json()["detail"]
    assert new.status_code == 200
    assert metric("medlens_auth_revocation_fallbacks_total", result="revoked") == revoked + 1
    assert metric("medlens_auth_revocation_fallbacks_total", result="active") == active + 1


async def test_sensitive_account_routes_fall_back_for_consumers_too(
    api: Api, redis_down: None
) -> None:
    consumer = await api.create_user()
    await end_all_sessions(api, consumer)
    headers = api.bearer(api.token(consumer, issued_at=two_minutes_ago()))

    profile = await api.client.get("/api/v1/users/me", headers=headers)
    erase = await api.client.delete("/api/v1/users/me", headers=headers)

    assert profile.status_code == 200  # fail open
    assert erase.status_code == 401  # checked in Postgres


async def test_the_check_is_free_while_redis_answers(api: Api) -> None:
    user, organization = await api.user_with_role(RoleName.PHARMACY_STAFF)
    await end_all_sessions(api, user)
    fallbacks = metric("medlens_auth_revocation_fallbacks_total", result="revoked")

    response = await api.client.get(
        "/api/v1/_probe/scan:read_org",
        headers=api.bearer(api.token(user, organization=organization, issued_at=two_minutes_ago())),
    )

    assert response.status_code == 200  # Redis is the source of truth when it answers
    assert metric("medlens_auth_revocation_fallbacks_total", result="revoked") == fallbacks


async def test_logout_everywhere_stamps_tokens_valid_after(api: Api) -> None:
    email = new_email()
    await api.create_user(email=email)
    login = (await api.password_login(email, PASSWORD)).json()
    before = datetime.now(UTC).replace(microsecond=0)

    response = await api.client.post(
        "/api/v1/auth/logout-all", headers=api.bearer(login["access_token"])
    )

    async with UnitOfWork(api.resources.session_factory) as uow:
        stamped = await uow.session.scalar(
            select(User.tokens_valid_after).where(User.email == email)
        )
    assert response.status_code == 204
    assert stamped is not None
    assert stamped >= before


async def test_switching_organization_needs_a_live_session(api: Api) -> None:
    """A revoked session cannot mint a fresh access token, whatever Redis says."""
    user, organization = await api.user_with_role(RoleName.PHARMACY_STAFF)

    response = await api.client.post(
        "/api/v1/auth/switch-organization",
        json={"organization_id": str(organization.id)},
        headers=api.bearer(api.token(user, organization=organization)),  # no refresh token
    )

    assert response.status_code == 401

"""Auth under failure: SMS outages, Redis outages, and checks that only matter when things
go wrong."""

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from medlens.modules.auth.deps import get_auth_service
from medlens.modules.auth.service import AuthService
from medlens.modules.auth.stores import OtpStore
from medlens.modules.rbac.access import PermissionVersions
from medlens.modules.rbac.models import RoleName
from medlens.ports.sms import SmsDeliveryError, SmsUnavailableError
from tests.integration.api_support import Api, new_phone

pytestmark = pytest.mark.integration


class BrokenSms:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def send(self, to_e164: str, message: str) -> None:
        raise self.error

    async def aclose(self) -> None:
        return None


def use_sms(api: Api, sms: BrokenSms) -> None:
    resources, settings = api.resources, api.app.state.settings
    api.app.dependency_overrides[get_auth_service] = lambda: AuthService(
        sessions=resources.session_factory,
        keyring=resources.keyring,
        sms=sms,
        redis=resources.redis,
        settings=settings.auth,
    )


async def test_sms_outage_is_a_503_with_retry_after(api: Api) -> None:
    use_sms(api, BrokenSms(SmsUnavailableError("down")))

    response = await api.client.post("/api/v1/auth/otp/request", json={"phone": new_phone()})

    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    assert "sms" not in response.text.lower().replace("could not send", "")  # topology not leaked


async def test_refused_numbers_get_the_same_answer(api: Api) -> None:
    use_sms(api, BrokenSms(SmsDeliveryError("bad number")))

    response = await api.client.post("/api/v1/auth/otp/request", json={"phone": new_phone()})

    assert response.status_code == 202  # no oracle on which numbers can receive SMS


async def test_otp_fails_closed_without_redis(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable(*args: object, **kwargs: object) -> str:
        raise RedisConnectionError("redis down")

    monkeypatch.setattr(OtpStore, "issue", unavailable)
    monkeypatch.setattr(OtpStore, "check", unavailable)

    request = await api.client.post("/api/v1/auth/otp/request", json={"phone": new_phone()})
    verify = await api.client.post(
        "/api/v1/auth/otp/verify", json={"phone": new_phone(), "code": "123456"}
    )

    assert request.status_code == verify.status_code == 503


async def test_permissions_fall_back_to_the_database_when_redis_fails(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    user, organization = await api.user_with_role(RoleName.PHARMACY_STAFF)

    async def unavailable(*args: object, **kwargs: object) -> None:
        raise RedisConnectionError("redis down")

    monkeypatch.setattr(PermissionVersions, "cached", unavailable)

    response = await api.client.get(
        "/api/v1/_probe/scan:read_org",
        headers=api.bearer(api.token(user, organization=organization)),
    )

    assert response.status_code == 200


async def test_sensitive_routes_require_an_mfa_session(api: Api) -> None:
    user, organization = await api.user_with_role(RoleName.REGULATOR_ADMIN)

    without = await api.client.get(
        "/api/v1/_probe/mfa", headers=api.bearer(api.token(user, organization=organization))
    )
    with_mfa = await api.client.get(
        "/api/v1/_probe/mfa",
        headers=api.bearer(api.token(user, organization=organization, auth_methods=("pwd", "mfa"))),
    )

    assert without.status_code == 403
    assert with_mfa.status_code == 200


async def test_suspended_users_lose_access_with_a_valid_token(api: Api) -> None:
    from medlens.db.session import UnitOfWork
    from medlens.modules.users.models import User, UserStatus

    user, organization = await api.user_with_role(RoleName.PHARMACY_STAFF)
    headers = api.bearer(api.token(user, organization=organization))
    async with UnitOfWork(api.resources.session_factory) as uow:
        stored = await uow.session.get(User, user.id)
        assert stored is not None
        stored.status = UserStatus.SUSPENDED
        stored.perm_ver += 1
        await uow.commit()

    response = await api.client.get("/api/v1/_probe/scan:create", headers=headers)

    assert response.status_code == 401


async def test_audit_log_filters_and_pages(api: Api) -> None:
    platform, _ = await api.user_with_role(RoleName.PLATFORM_ADMIN)
    headers = api.bearer(api.token(platform))
    for _ in range(3):
        await api.client.post("/api/v1/auth/otp/request", json={"phone": new_phone()})

    first = await api.client.get(
        "/api/v1/admin/audit-logs",
        headers=headers,
        params={"action": "auth.otp_requested", "limit": 2},
    )
    body = first.json()
    second = await api.client.get(
        "/api/v1/admin/audit-logs",
        headers=headers,
        params={"action": "auth.otp_requested", "limit": 2, "cursor": body["next_cursor"]},
    )

    assert first.status_code == 200
    assert {item["action"] for item in body["items"]} == {"auth.otp_requested"}
    assert body["has_more"] is True
    assert "phone_hash" in body["items"][0]["metadata"]  # never the phone itself
    assert second.json()["items"][0]["id"] not in {item["id"] for item in body["items"]}

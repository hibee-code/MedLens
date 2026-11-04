"""Organization API keys: creation, scopes, use, revocation and expiry."""

from datetime import UTC, datetime, timedelta

import pytest

from medlens.db.session import UnitOfWork
from medlens.modules.auth.models import ApiKey
from medlens.modules.rbac.models import RoleName
from tests.integration.api_support import Api

pytestmark = pytest.mark.integration


async def manufacturer_admin(api: Api) -> tuple[dict[str, str], str]:
    user, organization = await api.user_with_role(RoleName.MANUFACTURER_ADMIN)
    return api.bearer(api.token(user, organization=organization)), str(user.id)


async def create_key(api: Api, headers: dict[str, str], scopes: list[str]) -> dict[str, object]:
    response = await api.client.post(
        "/api/v1/api-keys", headers=headers, json={"name": "ERP sync", "scopes": scopes}
    )
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


async def test_key_is_shown_once_and_works_within_its_scopes(api: Api) -> None:
    headers, _ = await manufacturer_admin(api)

    created = await create_key(api, headers, ["product:read", "batch:write"])
    key = str(created["key"])
    listed = (await api.client.get("/api/v1/api-keys", headers=headers)).json()

    assert key.startswith(f"ml_live_{created['prefix']}_")
    assert listed[0]["prefix"] == created["prefix"]
    assert "key" not in listed[0]
    machine = {"X-API-Key": key}
    assert (await api.client.get("/api/v1/_probe/product:read", headers=machine)).status_code == 200
    assert (await api.client.get("/api/v1/_probe/batch:write", headers=machine)).status_code == 200
    assert (
        await api.client.get("/api/v1/_probe/product:write", headers=machine)
    ).status_code == 403
    assert (await api.client.get("/api/v1/users/me", headers=machine)).status_code == 403
    assert "apikey.created" in await api.audit_actions()
    async with UnitOfWork(api.resources.session_factory) as uow:
        stored = await uow.session.get(ApiKey, created["id"])
    assert stored is not None
    assert key not in (stored.key_hash, stored.prefix)  # only a hash is kept
    assert stored.last_used_at is not None


@pytest.mark.parametrize(
    ("scopes", "status"),
    [
        (["member:manage"], 422),  # administrative: never delegable to keys
        (["apikey:manage"], 422),
        (["scan:read_all"], 403),  # the creator does not hold it
        (["not:a_permission"], 422),
    ],
)
async def test_scopes_are_restricted(api: Api, scopes: list[str], status: int) -> None:
    headers, _ = await manufacturer_admin(api)

    response = await api.client.post(
        "/api/v1/api-keys", headers=headers, json={"name": "x", "scopes": scopes}
    )

    assert response.status_code == status


async def test_revoked_and_expired_keys_stop_working(api: Api) -> None:
    headers, _ = await manufacturer_admin(api)
    revoked = await create_key(api, headers, ["product:read"])
    expired = await create_key(api, headers, ["product:read"])
    async with UnitOfWork(api.resources.session_factory) as uow:
        row = await uow.session.get(ApiKey, expired["id"])
        assert row is not None
        row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
        await uow.commit()

    assert (
        await api.client.delete(f"/api/v1/api-keys/{revoked['id']}", headers=headers)
    ).status_code == 204
    assert (
        await api.client.delete(f"/api/v1/api-keys/{revoked['id']}", headers=headers)
    ).status_code == 204

    for key in (revoked["key"], expired["key"]):
        response = await api.client.get(
            "/api/v1/_probe/product:read", headers={"X-API-Key": str(key)}
        )
        assert response.status_code == 401
    assert "apikey.revoked" in await api.audit_actions()


async def test_keys_are_private_to_their_organization(api: Api) -> None:
    owner_headers, _ = await manufacturer_admin(api)
    other_headers, _ = await manufacturer_admin(api)
    key = await create_key(api, owner_headers, ["product:read"])

    listed = (await api.client.get("/api/v1/api-keys", headers=other_headers)).json()
    revoke = await api.client.delete(f"/api/v1/api-keys/{key['id']}", headers=other_headers)

    assert listed == []
    assert revoke.status_code == 404


async def test_managing_keys_needs_the_permission_and_an_organization(api: Api) -> None:
    staff, organization = await api.user_with_role(RoleName.MANUFACTURER_STAFF)
    admin, admin_org = await api.user_with_role(RoleName.MANUFACTURER_ADMIN)

    no_permission = await api.client.get(
        "/api/v1/api-keys", headers=api.bearer(api.token(staff, organization=organization))
    )
    no_organization = await api.client.get("/api/v1/api-keys", headers=api.bearer(api.token(admin)))

    assert no_permission.status_code == 403
    assert no_organization.status_code == 403  # without an active org there is no admin role
    assert admin_org is not None

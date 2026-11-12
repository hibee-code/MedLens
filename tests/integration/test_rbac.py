"""RBAC end to end: the role matrix, tenant isolation, escalation, last admin, perm_ver."""

from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from medlens.db.session import UnitOfWork
from medlens.modules.organizations.models import OrganizationStatus, OrganizationType
from medlens.modules.rbac.access import PermissionVersions
from medlens.modules.rbac.models import RoleName, RolePermission
from medlens.modules.rbac.permissions import ROLE_PERMISSIONS, Permission
from tests.integration.api_support import Api

pytestmark = pytest.mark.integration


async def headers_for(api: Api, role: RoleName) -> dict[str, str]:
    user, organization = await api.user_with_role(role)
    return api.bearer(api.token(user, organization=organization))


# ------------------------------------------------------------------ the matrix
@pytest.mark.parametrize("role", list(RoleName))
async def test_every_role_gets_exactly_its_permissions(api: Api, role: RoleName) -> None:
    headers = await headers_for(api, role)

    granted = set()
    for permission in Permission:
        response = await api.client.get(f"/api/v1/_probe/{permission.value}", headers=headers)
        assert response.status_code in {200, 403}, response.text
        if response.status_code == 200:
            granted.add(permission)

    assert granted == set(ROLE_PERMISSIONS[role])


async def test_database_matrix_matches_code_and_docs(api: Api) -> None:
    async with UnitOfWork(api.resources.session_factory) as uow:
        rows = await uow.session.execute(
            select(RolePermission.role_name, RolePermission.permission_name)
        )
        stored: dict[str, set[str]] = {}
        for role, permission in rows:
            stored.setdefault(role, set()).add(permission)

    expected = {role.value: {p.value for p in perms} for role, perms in ROLE_PERMISSIONS.items()}
    assert stored == expected  # every role has at least one permission


# ------------------------------------------------------------------ representative endpoints
@pytest.mark.parametrize(
    ("role", "allowed"),
    [
        (RoleName.CONSUMER, set()),
        (RoleName.PHARMACY_STAFF, {"members"}),
        (RoleName.PHARMACY_ADMIN, {"members", "roles", "api_keys"}),
        (RoleName.MANUFACTURER_ADMIN, {"members", "roles", "api_keys"}),
        (RoleName.REGULATOR_ANALYST, {"members"}),
        (RoleName.REGULATOR_ADMIN, {"members", "roles", "api_keys"}),
        (RoleName.PLATFORM_ADMIN, {"roles", "audit"}),
    ],
)
async def test_real_endpoints_by_role(api: Api, role: RoleName, allowed: set[str]) -> None:
    user, organization = await api.user_with_role(role)
    headers = api.bearer(api.token(user, organization=organization))
    own_org = organization.id if organization else uuid4()
    endpoints = {
        "members": f"/api/v1/organizations/{own_org}/members",
        "roles": "/api/v1/admin/roles",
        "audit": "/api/v1/admin/audit-logs",
        "api_keys": "/api/v1/api-keys",
    }

    for name, url in endpoints.items():
        status = (await api.client.get(url, headers=headers)).status_code
        if name in allowed:
            assert status == 200, (name, status)
        elif role is RoleName.PLATFORM_ADMIN and name == "api_keys":
            assert status == 422, status  # allowed, but keys belong to an active organization
        else:
            assert status in {403, 404}, (name, status)


# ------------------------------------------------------------------ tenant isolation
async def test_other_organizations_look_like_they_do_not_exist(api: Api) -> None:
    admin, own = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    stranger, other = await api.user_with_role(RoleName.PHARMACY_STAFF)
    assert own is not None
    assert other is not None
    headers = api.bearer(api.token(admin, organization=own))
    base = f"/api/v1/organizations/{other.id}/members"

    listed = await api.client.get(base, headers=headers)
    assigned = await api.client.put(
        f"{base}/{stranger.id}", headers=headers, json={"role": "pharmacy_admin"}
    )
    removed = await api.client.delete(f"{base}/{stranger.id}", headers=headers)
    missing = await api.client.get(f"/api/v1/organizations/{uuid4()}/members", headers=headers)

    assert listed.status_code == assigned.status_code == removed.status_code == 404
    assert missing.status_code == 404  # same answer as a real organization of someone else


async def test_org_admin_manages_own_members(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    assert organization is not None
    member = await api.create_user()
    await api.add_member(member, organization, RoleName.PHARMACY_STAFF)
    headers = api.bearer(api.token(admin, organization=organization))
    url = f"/api/v1/organizations/{organization.id}/members/{member.id}"

    promoted = await api.client.put(url, headers=headers, json={"role": "pharmacy_admin"})
    listed = (
        await api.client.get(f"/api/v1/organizations/{organization.id}/members", headers=headers)
    ).json()
    removed = await api.client.delete(url, headers=headers)

    assert promoted.status_code == 200
    assert promoted.json()["role"] == "pharmacy_admin"
    assert {item["user_id"] for item in listed["items"]} == {str(admin.id), str(member.id)}
    assert removed.status_code == 204
    actions = await api.audit_actions(admin.id)
    assert {"member.role_assigned", "member.removed"} <= set(actions)


async def test_only_platform_admins_add_non_members(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    platform, _ = await api.user_with_role(RoleName.PLATFORM_ADMIN)
    assert organization is not None
    outsider = await api.create_user()
    url = f"/api/v1/organizations/{organization.id}/members/{outsider.id}"

    by_org_admin = await api.client.put(
        url,
        headers=api.bearer(api.token(admin, organization=organization)),
        json={"role": "pharmacy_staff"},
    )
    by_platform = await api.client.put(
        url, headers=api.bearer(api.token(platform)), json={"role": "pharmacy_staff"}
    )

    assert by_org_admin.status_code == 404
    assert by_platform.status_code == 200


# ------------------------------------------------------------------ escalation
async def test_roles_must_suit_the_organization_type(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    assert organization is not None
    member = await api.create_user()
    await api.add_member(member, organization, RoleName.PHARMACY_STAFF)

    response = await api.client.put(
        f"/api/v1/organizations/{organization.id}/members/{member.id}",
        headers=api.bearer(api.token(admin, organization=organization)),
        json={"role": "regulator_admin"},
    )

    assert response.status_code == 422


async def test_nobody_grants_permissions_they_lack(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    assert organization is not None
    member = await api.create_user()
    await api.add_member(member, organization, RoleName.PHARMACY_STAFF)
    headers = api.bearer(api.token(admin, organization=organization))
    # An operator widens pharmacy_staff with a permission pharmacy admins do not hold.
    async with UnitOfWork(api.resources.session_factory) as uow:
        uow.session.add(RolePermission(role_name="pharmacy_staff", permission_name="audit:read"))
        await uow.commit()
    await PermissionVersions(api.resources.redis).bump_rbac_version()
    try:
        response = await api.client.put(
            f"/api/v1/organizations/{organization.id}/members/{member.id}",
            headers=headers,
            json={"role": "pharmacy_staff"},
        )
        global_role = await api.client.put(
            f"/api/v1/admin/users/{member.id}/global-role",
            headers=headers,
            json={"role": "platform_admin"},
        )
    finally:
        async with UnitOfWork(api.resources.session_factory) as uow:
            await uow.session.execute(
                delete(RolePermission).where(
                    RolePermission.role_name == "pharmacy_staff",
                    RolePermission.permission_name == "audit:read",
                )
            )
            await uow.commit()
        await PermissionVersions(api.resources.redis).bump_rbac_version()

    assert response.status_code == 403
    assert "audit:read" in response.json()["detail"]
    assert global_role.status_code == 403


async def test_platform_admin_sets_global_roles(api: Api) -> None:
    platform, _ = await api.user_with_role(RoleName.PLATFORM_ADMIN)
    user = await api.create_user()
    headers = api.bearer(api.token(platform))

    made_health_worker = await api.client.put(
        f"/api/v1/admin/users/{user.id}/global-role",
        headers=headers,
        json={"role": "health_worker"},
    )
    organization_role = await api.client.put(
        f"/api/v1/admin/users/{user.id}/global-role",
        headers=headers,
        json={"role": "pharmacy_staff"},
    )

    assert made_health_worker.status_code == 200
    assert made_health_worker.json()["global_role"] == "health_worker"
    assert organization_role.status_code == 422


# ------------------------------------------------------------------ last admin
async def test_the_last_admin_stays(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.MANUFACTURER_ADMIN)
    assert organization is not None
    headers = api.bearer(api.token(admin, organization=organization))
    url = f"/api/v1/organizations/{organization.id}/members/{admin.id}"

    demote = await api.client.put(url, headers=headers, json={"role": "manufacturer_staff"})
    remove = await api.client.delete(url, headers=headers)
    second_admin = await api.create_user()
    await api.add_member(second_admin, organization, RoleName.MANUFACTURER_ADMIN)
    demote_now = await api.client.put(url, headers=headers, json={"role": "manufacturer_staff"})

    assert demote.status_code == remove.status_code == 409
    assert demote_now.status_code == 200


# ------------------------------------------------------------------ perm_ver
async def test_role_changes_apply_on_the_next_request(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.PHARMACY_ADMIN)
    platform, _ = await api.user_with_role(RoleName.PLATFORM_ADMIN)
    assert organization is not None
    await api.add_member(await api.create_user(), organization, RoleName.PHARMACY_ADMIN)
    old_token = api.bearer(api.token(admin, organization=organization))
    assert (await api.client.get("/api/v1/admin/roles", headers=old_token)).status_code == 200

    demoted = await api.client.put(
        f"/api/v1/organizations/{organization.id}/members/{admin.id}",
        headers=api.bearer(api.token(platform)),
        json={"role": "pharmacy_staff"},
    )
    assert demoted.status_code == 200

    # Same (old) access token: the demotion already applies...
    assert (await api.client.get("/api/v1/admin/roles", headers=old_token)).status_code == 403
    # ...and sensitive endpoints refuse the outdated token outright.
    stale = await api.client.get("/api/v1/api-keys", headers=old_token)
    assert stale.status_code == 401
    assert "Refresh your access token" in stale.json()["detail"]
    reloaded = await api.reload(type(admin), admin.id)
    assert reloaded.perm_ver == admin.perm_ver + 1


async def test_memberships_in_inactive_organizations_grant_nothing(api: Api) -> None:
    organization = await api.create_org(
        OrganizationType.PHARMACY, status=OrganizationStatus.SUSPENDED
    )
    user = await api.create_user()
    await api.add_member(user, organization, RoleName.PHARMACY_ADMIN)
    headers = api.bearer(api.token(user, organization=organization))

    assert (await api.client.get("/api/v1/_probe/org:read", headers=headers)).status_code == 403
    assert (await api.client.get("/api/v1/_probe/scan:create", headers=headers)).status_code == 200


async def test_permissions_survive_a_redis_outage(api: Api) -> None:
    user, organization = await api.user_with_role(RoleName.REGULATOR_ANALYST)
    headers = api.bearer(api.token(user, organization=organization))
    await api.resources.redis.flushdb()

    response = await api.client.get("/api/v1/_probe/scan:read_all", headers=headers)

    assert response.status_code == 200

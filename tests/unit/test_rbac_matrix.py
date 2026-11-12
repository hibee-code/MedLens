"""The code matrix (rbac/permissions.py) and the role definitions agree."""

from medlens.modules.organizations.models import OrganizationType
from medlens.modules.rbac.models import ROLE_DEFINITIONS, RoleName, RoleScope
from medlens.modules.rbac.permissions import (
    CONSUMER_PERMISSIONS,
    GLOBAL_ROLES,
    NON_DELEGABLE_TO_API_KEYS,
    ORGANIZATION_ADMIN_ROLES,
    PERMISSION_DESCRIPTIONS,
    ROLE_ORGANIZATION_TYPES,
    ROLE_PERMISSIONS,
    Permission,
)


def test_member_management_is_never_delegated_to_api_keys() -> None:
    assert Permission.MEMBER_MANAGE in NON_DELEGABLE_TO_API_KEYS
    assert Permission.PRODUCT_WRITE not in NON_DELEGABLE_TO_API_KEYS


def test_every_role_and_permission_is_defined() -> None:
    assert set(ROLE_PERMISSIONS) == set(RoleName) == set(ROLE_DEFINITIONS)
    assert set(PERMISSION_DESCRIPTIONS) == set(Permission)
    assert ROLE_PERMISSIONS[RoleName.PLATFORM_ADMIN] == frozenset(Permission)


def test_role_scopes_are_consistent() -> None:
    for role, (scope, _) in ROLE_DEFINITIONS.items():
        assert (scope is RoleScope.GLOBAL) == (role in GLOBAL_ROLES)
        assert (scope is RoleScope.ORGANIZATION) == (role in ROLE_ORGANIZATION_TYPES)
    assert set(ROLE_ORGANIZATION_TYPES) >= ORGANIZATION_ADMIN_ROLES


def test_admin_roles_extend_their_staff_roles() -> None:
    pairs = [
        (RoleName.PHARMACY_STAFF, RoleName.PHARMACY_ADMIN),
        (RoleName.MANUFACTURER_STAFF, RoleName.MANUFACTURER_ADMIN),
        (RoleName.REGULATOR_ANALYST, RoleName.REGULATOR_ADMIN),
    ]
    for staff, admin in pairs:
        assert ROLE_PERMISSIONS[staff] < ROLE_PERMISSIONS[admin]


def test_signals_stay_with_regulators_and_platform_admins() -> None:
    holders = {
        role for role, perms in ROLE_PERMISSIONS.items() if Permission.SCAN_READ_SIGNALS in perms
    }

    assert holders == {
        RoleName.REGULATOR_ANALYST,
        RoleName.REGULATOR_ADMIN,
        RoleName.PLATFORM_ADMIN,
    }


def test_api_keys_cannot_administer() -> None:
    assert Permission.MEMBER_MANAGE in NON_DELEGABLE_TO_API_KEYS
    assert Permission.PRODUCT_WRITE not in NON_DELEGABLE_TO_API_KEYS


def test_manufacturers_never_read_raw_scans() -> None:
    raw = {Permission.SCAN_READ_ORG, Permission.SCAN_READ_ALL, Permission.SCAN_READ_SIGNALS}
    for role in (RoleName.MANUFACTURER_STAFF, RoleName.MANUFACTURER_ADMIN):
        assert not ROLE_PERMISSIONS[role] & raw, role


def test_the_audit_log_is_for_platform_admins_only() -> None:
    holders = {role for role, perms in ROLE_PERMISSIONS.items() if Permission.AUDIT_READ in perms}

    assert holders == {RoleName.PLATFORM_ADMIN}


def test_distributors_use_the_pharmacy_roles() -> None:
    assert not any("distributor" in role.value for role in RoleName)
    for role in (RoleName.PHARMACY_STAFF, RoleName.PHARMACY_ADMIN):
        assert OrganizationType.DISTRIBUTOR in ROLE_ORGANIZATION_TYPES[role]


def test_consumer_permissions_are_what_consumers_hold() -> None:
    """Sessions holding only these skip the revocation check while Redis is down."""
    assert ROLE_PERMISSIONS[RoleName.CONSUMER] == CONSUMER_PERMISSIONS
    assert ROLE_PERMISSIONS[RoleName.HEALTH_WORKER] == CONSUMER_PERMISSIONS

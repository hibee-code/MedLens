"""docs/rbac.md, the code matrix and the role definitions agree."""

import re
from pathlib import Path

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

DOCS = Path(__file__).resolve().parents[2] / "docs" / "rbac.md"


def documented_matrix() -> dict[str, set[str]]:
    """Parse the role -> permission table of docs/rbac.md."""
    lines = DOCS.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Permission | `"))
    header, table = lines[start], []
    for line in lines[start + 2 :]:
        if not line.startswith("| "):
            break
        table.append(line)
    roles = re.findall(r"`([a-z_]+)`", header)
    matrix: dict[str, set[str]] = {role: set() for role in roles}
    for line in table:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        permission = cells[0].strip("`")
        if ":" not in permission:
            continue
        for role, cell in zip(roles, cells[1:], strict=True):
            if cell == "✓":
                matrix[role].add(permission)
    return matrix


def test_docs_match_the_code_matrix() -> None:
    expected = {role.value: {p.value for p in perms} for role, perms in ROLE_PERMISSIONS.items()}

    assert documented_matrix() == expected


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


def documented_organization_types() -> dict[str, set[str]]:
    """Parse the role -> organization types table of docs/rbac.md."""
    found: dict[str, set[str]] = {}
    for line in DOCS.read_text().splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 2 and cells[0].startswith("`") and "_" in cells[0]:
            for role in re.findall(r"`([a-z_]+)`", cells[0]):
                found[role] = {kind.strip() for kind in cells[1].split(",")}
    return found


def test_docs_match_the_role_organization_types() -> None:
    expected = {
        role.value: {kind.value for kind in kinds}
        for role, kinds in ROLE_ORGANIZATION_TYPES.items()
    }

    assert documented_organization_types() == expected


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

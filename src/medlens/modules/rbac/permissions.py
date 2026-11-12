"""Permissions and the role -> permission matrix.

At runtime permissions are read from the ``roles`` / ``role_permissions`` tables, so the
matrix can change without a deploy. ``ROLE_PERMISSIONS`` is the seed (migration 0008) and the
documented baseline; tests/unit/test_rbac_matrix.py checks its consistency.

Resource-level limits (own products, jurisdiction, own scans) are not permissions; they live in
``policies.py``.
"""

from enum import StrEnum

from medlens.modules.organizations.models import OrganizationType
from medlens.modules.rbac.models import RoleName


class Permission(StrEnum):
    SCAN_CREATE = "scan:create"
    SCAN_READ_OWN = "scan:read_own"
    SCAN_READ_ORG = "scan:read_org"
    SCAN_READ_ALL = "scan:read_all"
    SCAN_READ_SIGNALS = "scan:read_signals"
    PRODUCT_READ = "product:read"
    PRODUCT_WRITE = "product:write"
    PRODUCT_REVIEW = "product:review"
    BATCH_WRITE = "batch:write"
    REFERENCE_IMAGE_WRITE = "reference_image:write"
    CASE_READ = "case:read"
    CASE_REVIEW = "case:review"
    CASE_ASSIGN = "case:assign"
    HOTSPOT_READ = "hotspot:read"
    HOTSPOT_EXPORT = "hotspot:export"
    ORG_READ = "org:read"
    ORG_MANAGE = "org:manage"
    MEMBER_INVITE = "member:invite"
    MEMBER_MANAGE = "member:manage"
    APIKEY_MANAGE = "apikey:manage"  # pragma: allowlist secret
    AUDIT_READ = "audit:read"
    REPORTS_READ_OWN = "reports:read_own"
    KNOWLEDGE_READ = "knowledge:read"
    KNOWLEDGE_MANAGE = "knowledge:manage"
    USER_MANAGE = "user:manage"
    PLATFORM_ADMIN = "platform:admin"


PERMISSION_DESCRIPTIONS: dict[Permission, str] = {
    Permission.SCAN_CREATE: "Submit a pack photo for verification.",
    Permission.SCAN_READ_OWN: "Read scans one submitted.",
    Permission.SCAN_READ_ORG: "Read scans submitted under one's organization.",
    Permission.SCAN_READ_ALL: "Read every scan in one's jurisdiction.",
    Permission.SCAN_READ_SIGNALS: "Read the detailed risk signals behind a verdict.",
    Permission.PRODUCT_READ: "Read registered products and batches.",
    Permission.PRODUCT_WRITE: "Update one's organization's products.",
    Permission.PRODUCT_REVIEW: "Review manufacturers' product claims and reference images.",
    Permission.BATCH_WRITE: "Declare batches of one's organization's products.",
    Permission.REFERENCE_IMAGE_WRITE: "Upload reference pack images for one's products.",
    Permission.CASE_READ: "Read review cases in one's jurisdiction.",
    Permission.CASE_REVIEW: "Review cases and record outcomes.",
    Permission.CASE_ASSIGN: "Assign cases to reviewers.",
    Permission.HOTSPOT_READ: "Read hotspots in one's jurisdiction.",
    Permission.HOTSPOT_EXPORT: "Export hotspot data.",
    Permission.ORG_READ: "Read one's organization and its members.",
    Permission.ORG_MANAGE: "Update one's organization and branches.",
    Permission.MEMBER_INVITE: "Invite people to one's organization.",
    Permission.MEMBER_MANAGE: "Change members' roles and remove members.",
    Permission.APIKEY_MANAGE: "Create and revoke the organization's API keys.",
    Permission.AUDIT_READ: "Read the security audit log.",
    Permission.REPORTS_READ_OWN: (
        "Read aggregated reports about one's own products (no individual scans)."
    ),
    Permission.KNOWLEDGE_READ: "Search the knowledge base and ask the assistant.",
    Permission.KNOWLEDGE_MANAGE: "Add documents to the knowledge base.",
    Permission.USER_MANAGE: "Manage any user account and global roles.",
    Permission.PLATFORM_ADMIN: "Operate the platform.",
}

# What consumers can do. When Redis is down, sessions holding only these skip the revocation
# check (fail open); a session holding anything more is checked in Postgres.
CONSUMER_PERMISSIONS = frozenset(
    {Permission.SCAN_CREATE, Permission.SCAN_READ_OWN, Permission.PRODUCT_READ}
)
_SCANNER = set(CONSUMER_PERMISSIONS)
_PHARMACY_STAFF = _SCANNER | {
    Permission.SCAN_READ_ORG,
    Permission.ORG_READ,
    Permission.KNOWLEDGE_READ,
}
_MANUFACTURER_STAFF = _SCANNER | {
    Permission.REPORTS_READ_OWN,
    Permission.KNOWLEDGE_READ,
    Permission.PRODUCT_WRITE,
    Permission.BATCH_WRITE,
    Permission.REFERENCE_IMAGE_WRITE,
    Permission.ORG_READ,
}
_REGULATOR_ANALYST = _SCANNER | {
    Permission.SCAN_READ_ALL,
    Permission.SCAN_READ_SIGNALS,
    Permission.CASE_READ,
    Permission.CASE_REVIEW,
    Permission.HOTSPOT_READ,
    Permission.ORG_READ,
    Permission.KNOWLEDGE_READ,
    Permission.KNOWLEDGE_MANAGE,
}
_ORG_ADMIN = {
    Permission.ORG_MANAGE,
    Permission.MEMBER_INVITE,
    Permission.MEMBER_MANAGE,
    Permission.APIKEY_MANAGE,
}

ROLE_PERMISSIONS: dict[RoleName, frozenset[Permission]] = {
    RoleName.CONSUMER: frozenset(_SCANNER),
    RoleName.HEALTH_WORKER: frozenset(_SCANNER),
    RoleName.PHARMACY_STAFF: frozenset(_PHARMACY_STAFF),
    RoleName.PHARMACY_ADMIN: frozenset(_PHARMACY_STAFF | _ORG_ADMIN),
    RoleName.MANUFACTURER_STAFF: frozenset(_MANUFACTURER_STAFF),
    RoleName.MANUFACTURER_ADMIN: frozenset(_MANUFACTURER_STAFF | _ORG_ADMIN),
    RoleName.REGULATOR_ANALYST: frozenset(_REGULATOR_ANALYST),
    RoleName.REGULATOR_ADMIN: frozenset(
        _REGULATOR_ANALYST
        | _ORG_ADMIN
        | {Permission.CASE_ASSIGN, Permission.HOTSPOT_EXPORT, Permission.PRODUCT_REVIEW}
    ),
    RoleName.PLATFORM_ADMIN: frozenset(Permission),
}

# Distributors hold the pharmacy roles until bulk stock verification gives them their own
# (CLAUDE.md decision log).
_PHARMACY_TYPES = frozenset(
    {
        OrganizationType.PHARMACY,
        OrganizationType.PPMV,
        OrganizationType.CLINIC,
        OrganizationType.DISTRIBUTOR,
    }
)
# Which organizations each organization-scoped role can be held in.
ROLE_ORGANIZATION_TYPES: dict[RoleName, frozenset[OrganizationType]] = {
    RoleName.PHARMACY_STAFF: _PHARMACY_TYPES,
    RoleName.PHARMACY_ADMIN: _PHARMACY_TYPES,
    RoleName.MANUFACTURER_STAFF: frozenset({OrganizationType.MANUFACTURER}),
    RoleName.MANUFACTURER_ADMIN: frozenset({OrganizationType.MANUFACTURER}),
    RoleName.REGULATOR_ANALYST: frozenset({OrganizationType.REGULATOR}),
    RoleName.REGULATOR_ADMIN: frozenset({OrganizationType.REGULATOR}),
}
GLOBAL_ROLES = frozenset({RoleName.CONSUMER, RoleName.HEALTH_WORKER, RoleName.PLATFORM_ADMIN})
# An organization must always keep at least one active member in one of these roles.
ORGANIZATION_ADMIN_ROLES = frozenset(
    {RoleName.PHARMACY_ADMIN, RoleName.MANUFACTURER_ADMIN, RoleName.REGULATOR_ADMIN}
)
# Machine clients never administer or approve: API key scopes may not include these.
NON_DELEGABLE_TO_API_KEYS = frozenset(
    {
        Permission.PRODUCT_REVIEW,
        Permission.ORG_MANAGE,
        Permission.MEMBER_INVITE,
        Permission.MEMBER_MANAGE,
        Permission.APIKEY_MANAGE,
        Permission.AUDIT_READ,
        Permission.USER_MANAGE,
        Permission.PLATFORM_ADMIN,
    }
)

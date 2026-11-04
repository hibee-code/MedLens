"""RBAC seed: the 22 permissions and the role -> permission matrix.

A snapshot of modules/rbac/permissions.py at the time of writing. Later changes to the matrix
are new migrations; tests check the migrated tables match the code and docs/rbac.md.

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

PERMISSIONS = [
    ("scan:create", "Submit a pack photo for verification."),
    ("scan:read_own", "Read scans one submitted."),
    ("scan:read_org", "Read scans submitted under one's organization."),
    ("scan:read_all", "Read every scan in one's jurisdiction."),
    ("scan:read_signals", "Read the detailed risk signals behind a verdict."),
    ("product:read", "Read registered products and batches."),
    ("product:write", "Update one's organization's products."),
    ("batch:write", "Declare batches of one's organization's products."),
    ("reference_image:write", "Upload reference pack images for one's products."),
    ("case:read", "Read review cases in one's jurisdiction."),
    ("case:review", "Review cases and record outcomes."),
    ("case:assign", "Assign cases to reviewers."),
    ("hotspot:read", "Read hotspots in one's jurisdiction."),
    ("hotspot:export", "Export hotspot data."),
    ("org:read", "Read one's organization and its members."),
    ("org:manage", "Update one's organization and branches."),
    ("member:invite", "Invite people to one's organization."),
    ("member:manage", "Change members' roles and remove members."),
    ("apikey:manage", "Create and revoke the organization's API keys."),
    ("audit:read", "Read the security audit log."),
    ("user:manage", "Manage any user account and global roles."),
    ("platform:admin", "Operate the platform."),
]

ROLE_PERMISSIONS = {
    "consumer": [
        "product:read",
        "scan:create",
        "scan:read_own",
    ],
    "health_worker": [
        "product:read",
        "scan:create",
        "scan:read_own",
    ],
    "pharmacy_staff": [
        "org:read",
        "product:read",
        "scan:create",
        "scan:read_org",
        "scan:read_own",
    ],
    "pharmacy_admin": [
        "apikey:manage",
        "member:invite",
        "member:manage",
        "org:manage",
        "org:read",
        "product:read",
        "scan:create",
        "scan:read_org",
        "scan:read_own",
    ],
    "manufacturer_staff": [
        "batch:write",
        "org:read",
        "product:read",
        "product:write",
        "reference_image:write",
        "scan:create",
        "scan:read_own",
    ],
    "manufacturer_admin": [
        "apikey:manage",
        "batch:write",
        "member:invite",
        "member:manage",
        "org:manage",
        "org:read",
        "product:read",
        "product:write",
        "reference_image:write",
        "scan:create",
        "scan:read_own",
    ],
    "regulator_analyst": [
        "case:read",
        "case:review",
        "hotspot:read",
        "org:read",
        "product:read",
        "scan:create",
        "scan:read_all",
        "scan:read_own",
        "scan:read_signals",
    ],
    "regulator_admin": [
        "apikey:manage",
        "case:assign",
        "case:read",
        "case:review",
        "hotspot:export",
        "hotspot:read",
        "member:invite",
        "member:manage",
        "org:manage",
        "org:read",
        "product:read",
        "scan:create",
        "scan:read_all",
        "scan:read_own",
        "scan:read_signals",
    ],
    "platform_admin": [
        "apikey:manage",
        "audit:read",
        "batch:write",
        "case:assign",
        "case:read",
        "case:review",
        "hotspot:export",
        "hotspot:read",
        "member:invite",
        "member:manage",
        "org:manage",
        "org:read",
        "platform:admin",
        "product:read",
        "product:write",
        "reference_image:write",
        "scan:create",
        "scan:read_all",
        "scan:read_org",
        "scan:read_own",
        "scan:read_signals",
        "user:manage",
    ],
}


def upgrade() -> None:
    permissions = sa.table("permissions", sa.column("name"), sa.column("description"))
    role_permissions = sa.table(
        "role_permissions", sa.column("role_name"), sa.column("permission_name")
    )
    op.bulk_insert(permissions, [{"name": name, "description": text} for name, text in PERMISSIONS])
    op.bulk_insert(
        role_permissions,
        [
            {"role_name": role, "permission_name": permission}
            for role, granted in ROLE_PERMISSIONS.items()
            for permission in granted
        ],
    )


def downgrade() -> None:
    op.execute("DELETE FROM role_permissions")
    op.execute("DELETE FROM permissions")

"""Who a database query is being run for.

Every repository method that touches tenant data takes a ``TenantContext`` and filters by it
(CLAUDE.md §3). Auth (Phase 3) builds it from the principal; RBAC (Phase 4) adds roles,
permissions and regulator jurisdiction.
"""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class TenantContext:
    user_id: UUID | None
    organization_id: UUID | None
    # Sees every tenant. Only platform admins and system jobs, set by trusted code, never input.
    unrestricted: bool = False

    @classmethod
    def system(cls) -> "TenantContext":
        """For workers and maintenance jobs acting on behalf of no particular tenant."""
        return cls(user_id=None, organization_id=None, unrestricted=True)

"""Imports every model module so ``Base.metadata`` is complete (Alembic, tests)."""

from medlens.db import models as infrastructure
from medlens.db.base import Base
from medlens.modules.audit import models as audit
from medlens.modules.auth import models as auth
from medlens.modules.cases import models as cases
from medlens.modules.control_plane import models as control_plane
from medlens.modules.hitl import models as hitl
from medlens.modules.hotspots import models as hotspots
from medlens.modules.knowledge import models as knowledge
from medlens.modules.organizations import models as organizations
from medlens.modules.products import models as products
from medlens.modules.rbac import models as rbac
from medlens.modules.scans import models as scans
from medlens.modules.users import models as users

__all__ = [
    "Base",
    "audit",
    "auth",
    "cases",
    "control_plane",
    "hitl",
    "hotspots",
    "infrastructure",
    "knowledge",
    "organizations",
    "products",
    "rbac",
    "scans",
    "users",
]

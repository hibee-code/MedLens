"""Who is calling: a signed-in user (access token) or an organization's machine client (API key).

A ``Principal`` only says who the caller is. What they may do is resolved separately, and
lazily, by RBAC (``modules/rbac/access.py``), so endpoints that need no permissions pay nothing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import UUID

from medlens.core.errors import Forbidden

MFA_METHOD = "mfa"


@dataclass(frozen=True, slots=True)
class Principal:
    kind: Literal["user", "api_key"]
    user_id: UUID | None
    organization_id: UUID | None  # the active organization (token "org" claim / key owner)
    session_id: UUID | None = None
    perm_ver: int = 0
    auth_methods: tuple[str, ...] = ()
    api_key_id: UUID | None = None
    scopes: frozenset[str] = field(default_factory=frozenset)
    issued_at: datetime | None = None  # access token "iat"
    # False when Redis was unavailable and the session-revocation check was skipped.
    revocation_checked: bool = True

    @property
    def is_user(self) -> bool:
        return self.kind == "user"

    @property
    def mfa_verified(self) -> bool:
        return self.kind == "user" and MFA_METHOD in self.auth_methods

    def require_user_id(self) -> UUID:
        if self.user_id is None:
            raise Forbidden("This endpoint is for signed-in users, not API keys.")
        return self.user_id

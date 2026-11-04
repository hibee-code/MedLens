from datetime import datetime
from enum import StrEnum

from sqlalchemy import CheckConstraint, ForeignKey, String, Text, false, text
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.orm import Mapped, mapped_column

from medlens.core.crypto import phone_lookup_hash
from medlens.db.base import Entity
from medlens.db.types import EncryptedText, str_enum
from medlens.modules.rbac.models import RoleName


class UserStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"  # erased under the NDPA: PII removed, row kept so scans stay linked


class User(Entity):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "status = 'deleted' OR phone_hash IS NOT NULL OR email IS NOT NULL", name="identity"
        ),
        CheckConstraint("(phone_e164 IS NULL) = (phone_hash IS NULL)", name="phone_pair"),
    )

    # Encrypted at rest; looked up through phone_hash (HMAC blind index). Use set_phone().
    phone_e164: Mapped[str | None] = mapped_column(EncryptedText("users.phone_e164"))
    phone_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    email: Mapped[str | None] = mapped_column(CITEXT, unique=True)
    password_hash: Mapped[str | None] = mapped_column(Text)
    full_name: Mapped[str | None] = mapped_column(Text)
    status: Mapped[UserStatus] = mapped_column(
        str_enum(UserStatus, "user_status"), default=UserStatus.ACTIVE, server_default="active"
    )
    global_role: Mapped[str] = mapped_column(
        ForeignKey("roles.name", onupdate="CASCADE"),
        default=RoleName.CONSUMER,
        server_default=RoleName.CONSUMER.value,
    )
    mfa_secret: Mapped[str | None] = mapped_column(EncryptedText("users.mfa_secret"))
    mfa_enabled: Mapped[bool] = mapped_column(default=False, server_default=false())
    last_login_at: Mapped[datetime | None]
    # Incremented whenever the user's roles or memberships change; access tokens carry the
    # version they were issued under, so changed permissions apply on the next request.
    perm_ver: Mapped[int] = mapped_column(default=1, server_default=text("1"))
    # Access tokens issued (iat) before this second are revoked. Set whenever all of a user's
    # sessions end; read only when Redis, which normally holds revocations, is unavailable.
    tokens_valid_after: Mapped[datetime | None]

    def set_phone(self, phone_e164: str | None) -> None:
        """Set the phone and its lookup hash together (a CHECK constraint enforces the pair)."""
        self.phone_e164 = phone_e164
        self.phone_hash = phone_lookup_hash(phone_e164) if phone_e164 else None

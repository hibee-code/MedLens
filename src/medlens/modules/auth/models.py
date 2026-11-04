from datetime import datetime
from uuid import UUID

from sqlalchemy import ARRAY, ForeignKey, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from medlens.db.base import Base, CreatedAt, Entity, UUIDv7PrimaryKey


class RefreshToken(UUIDv7PrimaryKey, CreatedAt, Base):
    """Opaque refresh token, stored hashed. Tokens rotate on use; all tokens descended from one
    login share a family_id, so reuse of an old token can revoke the whole family."""

    __tablename__ = "refresh_tokens"
    __table_args__ = (
        Index("ix_refresh_tokens_user_id", "user_id"),
        Index("ix_refresh_tokens_family_id", "family_id"),
    )
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    family_id: Mapped[UUID]
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime]
    revoked_at: Mapped[datetime | None]
    replaced_by_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("refresh_tokens.id", ondelete="SET NULL")
    )
    user_agent: Mapped[str | None] = mapped_column(Text)
    ip_hash: Mapped[str | None] = mapped_column(String(64))
    # Session state carried across rotations: the active organization and how the user
    # authenticated (RFC 8176 amr values: pwd, otp, mfa).
    organization_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL")
    )
    amr: Mapped[list[str]] = mapped_column(
        ARRAY(String(16)), default=list, server_default=text("'{}'")
    )


class ApiKey(Entity):
    """Machine-client key ``ml_live_<prefix>_<secret>``: only the prefix and a hash are stored."""

    __tablename__ = "api_keys"
    __table_args__ = (Index("ix_api_keys_organization_id", "organization_id"),)

    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(128))
    prefix: Mapped[str] = mapped_column(String(16), unique=True)
    key_hash: Mapped[str] = mapped_column(Text)
    scopes: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), default=list, server_default=text("'{}'")
    )
    last_used_at: Mapped[datetime | None]
    expires_at: Mapped[datetime | None]
    revoked_at: Mapped[datetime | None]

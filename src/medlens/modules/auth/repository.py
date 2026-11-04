"""SQL for authentication: credentials, sessions (refresh tokens) and API keys."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import exists, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from medlens.modules.auth.models import ApiKey, RefreshToken
from medlens.modules.organizations.models import (
    Membership,
    MembershipStatus,
    Organization,
    OrganizationStatus,
    OrganizationType,
)
from medlens.modules.users.models import User


class AuthRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ------------------------------------------------------------------ users
    async def user_by_id(self, user_id: UUID) -> User | None:
        return await self.session.get(User, user_id)

    async def user_by_email(self, email: str) -> User | None:
        return (await self.session.scalars(select(User).where(User.email == email))).first()

    async def user_by_phone_hash(self, phone_hash: str) -> User | None:
        statement = select(User).where(User.phone_hash == phone_hash)
        return (await self.session.scalars(statement)).first()

    async def active_organization_ids(self, user_id: UUID) -> list[UUID]:
        """Organizations where the user's membership and the organization are both active."""
        statement = (
            select(Membership.organization_id)
            .join(Organization, Organization.id == Membership.organization_id)
            .where(
                Membership.user_id == user_id,
                Membership.status == MembershipStatus.ACTIVE,
                Organization.status == OrganizationStatus.ACTIVE,
                Organization.deleted_at.is_(None),
            )
        )
        return list(await self.session.scalars(statement))

    async def belongs_to_regulator(self, user_id: UUID) -> bool:
        statement = select(
            exists().where(
                Membership.user_id == user_id,
                Membership.status == MembershipStatus.ACTIVE,
                Membership.organization_id == Organization.id,
                Organization.type == OrganizationType.REGULATOR,
            )
        )
        return bool(await self.session.scalar(statement))

    # ------------------------------------------------------------------ sessions
    async def refresh_token_for_update(self, token_hash: str) -> RefreshToken | None:
        """Locks the row so two concurrent refreshes with one token cannot both succeed."""
        statement = (
            select(RefreshToken).where(RefreshToken.token_hash == token_hash).with_for_update()
        )
        return (await self.session.scalars(statement)).first()

    async def current_refresh_token(self, family_id: UUID) -> RefreshToken | None:
        statement = select(RefreshToken).where(
            RefreshToken.family_id == family_id,
            RefreshToken.revoked_at.is_(None),
            RefreshToken.replaced_by_id.is_(None),
        )
        return (await self.session.scalars(statement)).first()

    async def tokens_valid_after(self, user_id: UUID) -> datetime | None:
        statement = select(User.tokens_valid_after).where(User.id == user_id)
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def invalidate_access_tokens(self, user_id: UUID, now: datetime) -> None:
        await self.session.execute(
            update(User).where(User.id == user_id).values(tokens_valid_after=now)
        )

    async def revoke_families(self, family_ids: list[UUID], now: datetime) -> None:
        await self.session.execute(
            update(RefreshToken)
            .where(RefreshToken.family_id.in_(family_ids), RefreshToken.revoked_at.is_(None))
            .values(revoked_at=now)
        )

    async def active_family_ids(self, user_id: UUID) -> list[UUID]:
        statement = (
            select(RefreshToken.family_id)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
            .distinct()
        )
        return list(await self.session.scalars(statement))

    # ------------------------------------------------------------------ API keys
    async def api_key_by_prefix(self, prefix: str) -> tuple[ApiKey, Organization] | None:
        statement = (
            select(ApiKey, Organization)
            .join(Organization, Organization.id == ApiKey.organization_id)
            .where(ApiKey.prefix == prefix)
        )
        row = (await self.session.execute(statement)).first()
        return (row[0], row[1]) if row else None

"""Organization API keys for machine clients (manufacturer integrations).

A key belongs to the caller's active organization. Its scopes must be permissions the creator
holds and never administrative ones. The full key is returned once; only its prefix and hash
are stored.
"""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.errors import NotFound, ValidationFailed
from medlens.core.request_meta import RequestMeta
from medlens.core.security import new_api_key
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.auth.models import ApiKey
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.permissions import NON_DELEGABLE_TO_API_KEYS, Permission
from medlens.modules.rbac.service import ensure_no_escalation


def _organization_id(actor: Actor) -> UUID:
    if actor.organization is None:
        message = "Switch to the organization first (POST /api/v1/auth/switch-organization)."
        raise ValidationFailed(message)
    return actor.organization.id


class ApiKeyService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(
        self,
        actor: Actor,
        *,
        name: str,
        scopes: list[Permission],
        expires_at: datetime | None,
        meta: RequestMeta,
    ) -> tuple[ApiKey, str]:
        organization_id = _organization_id(actor)
        requested = frozenset(scopes)
        administrative = requested & NON_DELEGABLE_TO_API_KEYS
        if administrative:
            message = "API keys cannot hold administrative permissions: " + ", ".join(
                sorted(administrative)
            )
            raise ValidationFailed(
                message, errors=[{"loc": ["body", "scopes"], "msg": message, "type": "scopes"}]
            )
        ensure_no_escalation(actor, requested)
        credential = new_api_key()
        async with UnitOfWork(self._sessions) as uow:
            api_key = ApiKey(
                organization_id=organization_id,
                name=name,
                prefix=credential.prefix,
                key_hash=credential.digest,
                scopes=sorted(requested),
                expires_at=expires_at,
            )
            uow.session.add(api_key)
            await uow.session.flush()
            await AuditService(uow.session).record(
                "apikey.created",
                actor_user_id=actor.user_id,
                actor_org_id=organization_id,
                resource_type="api_key",
                resource_id=api_key.id,
                meta=meta,
                details={"scopes": api_key.scopes, "prefix": api_key.prefix},
            )
            await uow.commit()
        return api_key, credential.key

    async def list(self, actor: Actor) -> list[ApiKey]:
        organization_id = _organization_id(actor)
        async with UnitOfWork(self._sessions) as uow:
            statement = (
                select(ApiKey)
                .where(ApiKey.organization_id == organization_id)
                .order_by(ApiKey.created_at.desc())
            )
            return list(await uow.session.scalars(statement))

    async def revoke(self, actor: Actor, key_id: UUID, meta: RequestMeta) -> None:
        organization_id = _organization_id(actor)
        async with UnitOfWork(self._sessions) as uow:
            api_key = await uow.session.get(ApiKey, key_id, with_for_update=True)
            if api_key is None or api_key.organization_id != organization_id:
                raise NotFound("API key not found.")
            if api_key.revoked_at is None:
                api_key.revoked_at = datetime.now(UTC)
                await AuditService(uow.session).record(
                    "apikey.revoked",
                    actor_user_id=actor.user_id,
                    actor_org_id=organization_id,
                    resource_type="api_key",
                    resource_id=api_key.id,
                    meta=meta,
                )
            await uow.commit()

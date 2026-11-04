"""Authentication flows: phone OTP, e-mail + password, MFA (TOTP), session refresh and logout.

Every flow runs in its own unit of work. Failed attempts are audited and committed *before*
the error is raised, so a rejected login still leaves its trace; the same goes for revoking a
session family when a used refresh token is replayed.

Sessions: a login starts a session (a refresh-token family, whose id is the access token's
``sid``). Refresh tokens rotate on every use; presenting an already-rotated token is treated as
theft and ends the whole session.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.core.config import AuthSettings
from medlens.core.crypto import ip_hash, phone_lookup_hash
from medlens.core.errors import (
    Conflict,
    DependencyUnavailable,
    Forbidden,
    NotFound,
    RateLimited,
    Unauthorized,
    ValidationFailed,
)
from medlens.core.ids import uuid7
from medlens.core.request_meta import RequestMeta
from medlens.core.security import (
    InvalidTokenError,
    JwtKeyRing,
    TokenClaims,
    constant_time_equal,
    hash_password,
    new_opaque_token,
    parse_api_key_prefix,
    token_digest,
    verify_password,
)
from medlens.db.session import UnitOfWork
from medlens.modules.audit.service import AuditService
from medlens.modules.auth import totp
from medlens.modules.auth.metrics import REVOCATION_FALLBACKS
from medlens.modules.auth.models import RefreshToken
from medlens.modules.auth.passwords import password_problems
from medlens.modules.auth.principal import MFA_METHOD, Principal
from medlens.modules.auth.repository import AuthRepository
from medlens.modules.auth.stores import (
    LoginLockout,
    MfaGuard,
    OtpResult,
    OtpStore,
    SessionRevocations,
)
from medlens.modules.organizations.models import OrganizationStatus
from medlens.modules.rbac.models import RoleName
from medlens.modules.users.models import User, UserStatus
from medlens.ports.sms import SmsDeliveryError, SMSPort, SmsUnavailableError
from medlens.queue import outbox

log = structlog.stdlib.get_logger(__name__)

INVALID_CODE = "The code is invalid or has expired."
INVALID_CREDENTIALS = "Invalid e-mail address or password."
INVALID_REFRESH = "The refresh token is invalid, expired or already used. Sign in again."
OTP_MESSAGE = "Your MedLens code is {code}. It expires in {minutes} minutes. Never share it."
_API_KEY_TOUCH_INTERVAL = timedelta(minutes=1)


@dataclass(frozen=True, slots=True)
class SessionTokens:
    access_token: str
    refresh_token: str
    expires_in: int
    refresh_expires_in: int
    organization_id: UUID | None


@dataclass(frozen=True, slots=True)
class MfaChallenge:
    status: Literal["mfa_required", "mfa_enrolment_required"]
    mfa_token: str
    expires_in: int


@dataclass(frozen=True, slots=True)
class AccessToken:
    access_token: str
    expires_in: int
    organization_id: UUID | None


@dataclass(frozen=True, slots=True)
class MfaSetup:
    secret: str
    otpauth_uri: str


SESSION_ENDED = "This session has ended. Sign in again."


def _now() -> datetime:
    return datetime.now(UTC)


def _redis_unavailable() -> DependencyUnavailable:
    return DependencyUnavailable(
        "Sign-in is temporarily unavailable. Try again shortly.",
        dependency="redis",
        retry_after_seconds=10,
    )


class AuthService:
    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        keyring: JwtKeyRing,
        sms: SMSPort,
        redis: Redis,
        settings: AuthSettings,
    ) -> None:
        self._sessions = sessions
        self._keyring = keyring
        self._sms = sms
        self._settings = settings
        self._otp = OtpStore(redis, settings)
        self._lockout = LoginLockout(redis, settings)
        self._mfa = MfaGuard(redis, challenge_ttl_seconds=settings.mfa_token_ttl_seconds)
        self.revocations = SessionRevocations(
            redis, access_ttl_seconds=settings.access_token_ttl_seconds
        )

    # ------------------------------------------------------------------ phone OTP
    async def request_otp(self, phone: str, meta: RequestMeta) -> int:
        """Sends a code whether or not an account exists (a new one is created on verify), so
        the response never reveals which numbers are registered."""
        try:
            code = await self._otp.issue(phone)
        except RedisError:
            raise _redis_unavailable() from None
        minutes = self._settings.otp_ttl_seconds // 60
        try:
            await self._sms.send(phone, OTP_MESSAGE.format(code=code, minutes=minutes))
        except SmsUnavailableError:
            raise DependencyUnavailable(
                "We could not send the code right now. Try again shortly.",
                dependency="sms",
                retry_after_seconds=30,
            ) from None
        except SmsDeliveryError:
            log.warning("auth.otp_not_delivered")  # same response: no oracle on numbers
        async with UnitOfWork(self._sessions) as uow:
            await AuditService(uow.session).record(
                "auth.otp_requested", meta=meta, details={"phone_hash": phone_lookup_hash(phone)}
            )
            await uow.commit()
        return self._settings.otp_ttl_seconds

    async def verify_otp(self, phone: str, code: str, meta: RequestMeta) -> SessionTokens:
        try:
            result = await self._otp.check(phone, code)
        except RedisError:
            raise _redis_unavailable() from None
        async with UnitOfWork(self._sessions) as uow:
            repo, audit = AuthRepository(uow.session), AuditService(uow.session)
            if result is not OtpResult.OK:
                await audit.record(
                    "auth.login_failed",
                    meta=meta,
                    details={"method": "otp", "reason": result.value},
                )
                await uow.commit()
                raise Unauthorized(INVALID_CODE)
            user = await repo.user_by_phone_hash(phone_lookup_hash(phone))
            created = user is None
            if user is None:
                user = User()
                user.set_phone(phone)
                uow.session.add(user)
                await uow.session.flush()
            await self._require_active(uow, user, meta)
            tokens = await self._start_session(uow.session, user, ("otp",), meta)
            await audit.record(
                "auth.login_succeeded",
                actor_user_id=user.id,
                meta=meta,
                details={"method": "otp", "new_account": created},
            )
            await uow.commit()
        return tokens

    # ------------------------------------------------------------------ e-mail + password
    async def login(
        self, email: str, password: str, meta: RequestMeta
    ) -> SessionTokens | MfaChallenge:
        try:
            retry_after = await self._lockout.retry_after(email, meta.ip)
        except RedisError:
            raise _redis_unavailable() from None
        if retry_after:
            raise RateLimited(
                "Too many failed sign-in attempts. Try again later.",
                retry_after_seconds=retry_after,
            )
        async with UnitOfWork(self._sessions) as uow:
            repo, audit = AuthRepository(uow.session), AuditService(uow.session)
            user = await repo.user_by_email(email)
            valid, upgraded_hash = verify_password(password, user.password_hash if user else None)
            if user is None or not valid:
                try:
                    await self._lockout.record_failure(email, meta.ip)
                except RedisError:
                    raise _redis_unavailable() from None
                await audit.record(
                    "auth.login_failed",
                    actor_user_id=user.id if user else None,
                    meta=meta,
                    details={"method": "password"},
                )
                await uow.commit()
                raise Unauthorized(INVALID_CREDENTIALS)
            await self._lockout.record_success(email)
            if upgraded_hash:
                user.password_hash = upgraded_hash  # argon2 parameters changed since it was set
            await self._require_active(uow, user, meta)

            if user.mfa_enabled or await self._mfa_mandatory(repo, user):
                purpose = "verify" if user.mfa_enabled else "enrol"
                token = self._keyring.issue(
                    "mfa", subject=user.id, auth_methods=("pwd",), purpose=purpose
                )
                await audit.record(
                    "auth.mfa_challenged",
                    actor_user_id=user.id,
                    meta=meta,
                    details={"purpose": purpose},
                )
                await uow.commit()
                status: Literal["mfa_required", "mfa_enrolment_required"] = (
                    "mfa_required" if user.mfa_enabled else "mfa_enrolment_required"
                )
                return MfaChallenge(status, token, self._keyring.mfa_ttl_seconds)

            tokens = await self._start_session(uow.session, user, ("pwd",), meta)
            await audit.record(
                "auth.login_succeeded",
                actor_user_id=user.id,
                meta=meta,
                details={"method": "password"},
            )
            await uow.commit()
        return tokens

    # ------------------------------------------------------------------ MFA
    def mfa_claims(self, token: str, *, purpose: str) -> TokenClaims:
        try:
            claims = self._keyring.verify(token, "mfa")
        except InvalidTokenError:
            raise Unauthorized("This sign-in attempt has expired. Sign in again.") from None
        if claims.purpose != purpose:
            raise Unauthorized("This token cannot be used here.")
        return claims

    async def verify_mfa(self, mfa_token: str, code: str, meta: RequestMeta) -> SessionTokens:
        claims = self.mfa_claims(mfa_token, purpose="verify")
        if not await self._mfa.register_attempt(claims.token_id):
            raise Unauthorized("Too many wrong codes. Sign in again.")
        async with UnitOfWork(self._sessions) as uow:
            repo, audit = AuthRepository(uow.session), AuditService(uow.session)
            user = await repo.user_by_id(claims.subject)
            if user is None or not user.mfa_enabled or user.mfa_secret is None:
                raise Unauthorized("This sign-in attempt has expired. Sign in again.")
            if not await self._totp_ok(user, code):
                await audit.record("auth.mfa_failed", actor_user_id=user.id, meta=meta)
                await uow.commit()
                raise Unauthorized(INVALID_CODE)
            await self._require_active(uow, user, meta)
            tokens = await self._start_session(uow.session, user, ("pwd", MFA_METHOD), meta)
            await audit.record(
                "auth.login_succeeded",
                actor_user_id=user.id,
                meta=meta,
                details={"method": "password+mfa"},
            )
            await uow.commit()
        return tokens

    async def start_mfa_enrolment(self, user_id: UUID, meta: RequestMeta) -> MfaSetup:
        async with UnitOfWork(self._sessions) as uow:
            user = await AuthRepository(uow.session).user_by_id(user_id)
            if user is None or user.status is not UserStatus.ACTIVE:
                raise Unauthorized("Sign in again.")
            if user.mfa_enabled:
                raise Conflict("Multi-factor authentication is already enabled.")
            secret = totp.new_secret()
            user.mfa_secret = secret
            await AuditService(uow.session).record(
                "auth.mfa_enrolment_started", actor_user_id=user.id, meta=meta
            )
            await uow.commit()
        account = user.email or str(user.id)
        uri = totp.provisioning_uri(secret, account=account, issuer=self._settings.mfa_issuer)
        return MfaSetup(secret=secret, otpauth_uri=uri)

    async def confirm_mfa(
        self, user_id: UUID, code: str, prior_methods: tuple[str, ...], meta: RequestMeta
    ) -> SessionTokens:
        async with UnitOfWork(self._sessions) as uow:
            audit = AuditService(uow.session)
            user = await AuthRepository(uow.session).user_by_id(user_id)
            if user is None or user.status is not UserStatus.ACTIVE:
                raise Unauthorized("Sign in again.")
            if user.mfa_enabled:
                raise Conflict("Multi-factor authentication is already enabled.")
            if user.mfa_secret is None:
                raise Conflict("Start enrolment first.")
            if not await self._totp_ok(user, code):
                await audit.record("auth.mfa_failed", actor_user_id=user.id, meta=meta)
                await uow.commit()
                raise ValidationFailed(
                    INVALID_CODE,
                    errors=[{"loc": ["body", "code"], "msg": INVALID_CODE, "type": "mfa_code"}],
                )
            user.mfa_enabled = True
            await audit.record("auth.mfa_enabled", actor_user_id=user.id, meta=meta)
            methods = tuple(dict.fromkeys((*prior_methods, MFA_METHOD)))
            tokens = await self._start_session(uow.session, user, methods, meta)
            await uow.commit()
        return tokens

    # ------------------------------------------------------------------ sessions
    async def refresh(self, refresh_token: str, meta: RequestMeta) -> SessionTokens:
        now = _now()
        async with UnitOfWork(self._sessions) as uow:
            repo = AuthRepository(uow.session)
            current = await repo.refresh_token_for_update(token_digest(refresh_token))
            if current is None:
                raise Unauthorized(INVALID_REFRESH)
            if current.replaced_by_id is not None:
                # Already rotated: someone is replaying a stolen token. End the session.
                await self._revoke(uow.session, current.user_id, [current.family_id], now)
                await AuditService(uow.session).record(
                    "auth.refresh_token_reuse",
                    actor_user_id=current.user_id,
                    resource_type="session",
                    resource_id=current.family_id,
                    meta=meta,
                )
                await uow.commit()
                await self._publish_revocations([current.family_id])
                raise Unauthorized(INVALID_REFRESH)
            if current.revoked_at is not None or current.expires_at <= now:
                raise Unauthorized(INVALID_REFRESH)
            user = await repo.user_by_id(current.user_id)
            if user is None or user.status is not UserStatus.ACTIVE:
                await self._revoke(uow.session, current.user_id, [current.family_id], now)
                await uow.commit()
                raise Unauthorized(INVALID_REFRESH)

            secret, successor = self._new_refresh_token(
                user.id, current.family_id, current.organization_id, tuple(current.amr), meta
            )
            uow.session.add(successor)
            await uow.session.flush()
            current.replaced_by_id, current.revoked_at = successor.id, now
            tokens = self._tokens(user, successor, secret)
            await uow.commit()
        return tokens

    async def logout(self, principal: Principal, meta: RequestMeta) -> None:
        if principal.session_id is None or principal.user_id is None:
            raise Forbidden("Only signed-in users can log out.")
        async with UnitOfWork(self._sessions) as uow:
            await self._revoke(uow.session, principal.user_id, [principal.session_id], _now())
            await AuditService(uow.session).record(
                "auth.logout", actor_user_id=principal.user_id, meta=meta
            )
            await uow.commit()
        await self._publish_revocations([principal.session_id])

    async def end_sessions(
        self,
        session: AsyncSession,
        user_id: UUID,
        *,
        keep: UUID | None = None,
    ) -> list[UUID]:
        """Revoke a user's sessions inside the caller's transaction; returns their ids. Call
        ``publish_revocations`` after committing.

        Also stamps ``users.tokens_valid_after``, which revokes every access token issued
        before now (including the kept session's) whenever the check has to fall back to
        Postgres; the kept session gets a new access token on its next refresh."""
        repo = AuthRepository(session)
        now = _now()
        await repo.invalidate_access_tokens(user_id, now)
        families = [family for family in await repo.active_family_ids(user_id) if family != keep]
        if families:
            await self._revoke(session, user_id, families, now)
        return families

    async def _revoke(
        self, session: AsyncSession, user_id: UUID, family_ids: list[UUID], now: datetime
    ) -> None:
        """Revoke sessions in the caller's transaction. The outbox event is the durable copy of
        the Redis publish that follows the commit: if Redis is down then, the revocation
        reaches it once it is back."""
        await AuthRepository(session).revoke_families(family_ids, now)
        await outbox.emit(
            session,
            outbox.SESSIONS_REVOKED,
            aggregate_type="user",
            aggregate_id=user_id,
            payload={
                "session_ids": [str(family) for family in family_ids],
                "ttl_seconds": self.revocations.ttl_seconds,
            },
        )

    async def ensure_not_revoked(self, principal: Principal) -> None:
        """Postgres fallback when the Redis revocation check was skipped (a no-op otherwise):
        rejects access tokens issued before the user's sessions last ended.

        It sees user-wide events only (log out everywhere, password change, erasure); a
        single-session logout is recorded in Redis alone. Tokens issued in the same second as
        the event still pass, because "iat" has whole-second precision."""
        if principal.revocation_checked or principal.user_id is None:
            return
        async with UnitOfWork(self._sessions) as uow:
            valid_after = await AuthRepository(uow.session).tokens_valid_after(principal.user_id)
        revoked = valid_after is not None and (
            principal.issued_at is None or principal.issued_at < valid_after.replace(microsecond=0)
        )
        REVOCATION_FALLBACKS.labels(result="revoked" if revoked else "active").inc()
        if revoked:
            log.info("auth.revoked_token_rejected", source="postgres")
            raise Unauthorized(SESSION_ENDED)

    async def logout_all(self, principal: Principal, meta: RequestMeta) -> None:
        if principal.user_id is None:
            raise Forbidden("Only signed-in users can log out.")
        async with UnitOfWork(self._sessions) as uow:
            families = await self.end_sessions(uow.session, principal.user_id)
            await AuditService(uow.session).record(
                "auth.logout_all",
                actor_user_id=principal.user_id,
                meta=meta,
                details={"sessions": len(families)},
            )
            await uow.commit()
        await self._publish_revocations(families)

    async def publish_revocations(self, session_ids: list[UUID]) -> None:
        await self._publish_revocations(session_ids)

    async def switch_organization(
        self, principal: Principal, organization_id: UUID | None, meta: RequestMeta
    ) -> AccessToken:
        if principal.user_id is None or principal.session_id is None:
            raise Forbidden("Only signed-in users can switch organization.")
        async with UnitOfWork(self._sessions) as uow:
            repo = AuthRepository(uow.session)
            user = await repo.user_by_id(principal.user_id)
            if user is None or user.status is not UserStatus.ACTIVE:
                raise Unauthorized("Sign in again.")
            if organization_id is not None and organization_id not in set(
                await repo.active_organization_ids(user.id)
            ):
                raise NotFound("Organization not found.")
            # Minting a token needs a live session, checked here rather than trusted from
            # Redis: otherwise a revoked token could mint a fresh one while Redis is down.
            current = await repo.current_refresh_token(principal.session_id)
            if current is None:
                raise Unauthorized(SESSION_ENDED)
            current.organization_id = organization_id  # kept across refreshes
            await AuditService(uow.session).record(
                "auth.organization_switched",
                actor_user_id=user.id,
                actor_org_id=organization_id,
                meta=meta,
            )
            access = self._keyring.issue(
                "access",
                subject=user.id,
                session_id=principal.session_id,
                organization_id=organization_id,
                perm_ver=user.perm_ver,
                auth_methods=principal.auth_methods,
            )
            await uow.commit()
        return AccessToken(access, self._keyring.access_ttl_seconds, organization_id)

    # ------------------------------------------------------------------ password
    async def change_password(
        self, principal: Principal, current_password: str, new_password: str, meta: RequestMeta
    ) -> None:
        if principal.user_id is None:
            raise Forbidden("Only signed-in users can change their password.")
        async with UnitOfWork(self._sessions) as uow:
            audit = AuditService(uow.session)
            user = await AuthRepository(uow.session).user_by_id(principal.user_id)
            if user is None:
                raise Unauthorized("Sign in again.")
            if user.password_hash is None:
                raise Conflict("This account signs in with a phone code and has no password.")
            if not verify_password(current_password, user.password_hash)[0]:
                await audit.record("auth.password_change_failed", actor_user_id=user.id, meta=meta)
                await uow.commit()
                raise _field_error("current_password", "The current password is incorrect.")
            problems = password_problems(new_password, email=user.email)
            if problems:
                raise ValidationFailed(
                    "The new password is not acceptable.",
                    errors=[
                        {"loc": ["body", "new_password"], "msg": problem, "type": "password"}
                        for problem in problems
                    ],
                )
            user.password_hash = hash_password(new_password)
            # Other sessions may belong to whoever knew the old password.
            ended = await self.end_sessions(uow.session, user.id, keep=principal.session_id)
            await audit.record(
                "auth.password_changed",
                actor_user_id=user.id,
                meta=meta,
                details={"other_sessions_ended": len(ended)},
            )
            await uow.commit()
        await self._publish_revocations(ended)

    # ------------------------------------------------------------------ API keys
    async def authenticate_api_key(self, key: str) -> Principal | None:
        prefix = parse_api_key_prefix(key)
        if prefix is None:
            return None
        async with UnitOfWork(self._sessions) as uow:
            found = await AuthRepository(uow.session).api_key_by_prefix(prefix)
            if found is None:
                return None
            api_key, organization = found
            now = _now()
            usable = (
                constant_time_equal(api_key.key_hash, token_digest(key))
                and api_key.revoked_at is None
                and (api_key.expires_at is None or api_key.expires_at > now)
                and organization.status is OrganizationStatus.ACTIVE
                and organization.deleted_at is None
            )
            if not usable:
                return None
            if api_key.last_used_at is None or now - api_key.last_used_at > _API_KEY_TOUCH_INTERVAL:
                api_key.last_used_at = now
                await uow.commit()
        return Principal(
            kind="api_key",
            user_id=None,
            organization_id=organization.id,
            api_key_id=api_key.id,
            scopes=frozenset(api_key.scopes),
        )

    # ------------------------------------------------------------------ helpers
    async def _require_active(self, uow: UnitOfWork, user: User, meta: RequestMeta) -> None:
        if user.status is UserStatus.ACTIVE:
            return
        await AuditService(uow.session).record(
            "auth.login_blocked",
            actor_user_id=user.id,
            meta=meta,
            details={"status": user.status.value},
        )
        await uow.commit()
        raise Forbidden("This account is suspended.")

    @staticmethod
    async def _mfa_mandatory(repo: AuthRepository, user: User) -> bool:
        if user.global_role == RoleName.PLATFORM_ADMIN:
            return True
        return await repo.belongs_to_regulator(user.id)

    async def _totp_ok(self, user: User, code: str) -> bool:
        if user.mfa_secret is None:
            return False
        step = totp.matching_timestep(user.mfa_secret, code)
        return step is not None and await self._mfa.claim_timestep(user.id, step)

    async def _start_session(
        self,
        session: AsyncSession,
        user: User,
        auth_methods: tuple[str, ...],
        meta: RequestMeta,
    ) -> SessionTokens:
        organizations = await AuthRepository(session).active_organization_ids(user.id)
        organization_id = organizations[0] if len(organizations) == 1 else None
        secret, row = self._new_refresh_token(user.id, uuid7(), organization_id, auth_methods, meta)
        session.add(row)
        user.last_login_at = _now()
        await session.flush()
        return self._tokens(user, row, secret)

    def _new_refresh_token(
        self,
        user_id: UUID,
        family_id: UUID,
        organization_id: UUID | None,
        auth_methods: tuple[str, ...],
        meta: RequestMeta,
    ) -> tuple[str, RefreshToken]:
        secret = new_opaque_token()
        row = RefreshToken(
            id=uuid7(),
            user_id=user_id,
            family_id=family_id,
            token_hash=token_digest(secret),
            expires_at=_now() + timedelta(days=self._settings.refresh_token_ttl_days),
            organization_id=organization_id,
            amr=list(auth_methods),
            user_agent=meta.user_agent,
            ip_hash=ip_hash(meta.ip) if meta.ip else None,
        )
        return secret, row

    def _tokens(self, user: User, row: RefreshToken, secret: str) -> SessionTokens:
        access = self._keyring.issue(
            "access",
            subject=user.id,
            session_id=row.family_id,
            organization_id=row.organization_id,
            perm_ver=user.perm_ver,
            auth_methods=tuple(row.amr),
        )
        return SessionTokens(
            access_token=access,
            refresh_token=secret,
            expires_in=self._keyring.access_ttl_seconds,
            refresh_expires_in=self._settings.refresh_token_ttl_days * 86400,
            organization_id=row.organization_id,
        )

    async def _publish_revocations(self, session_ids: list[UUID]) -> None:
        """After commit: tell API processes to reject these sessions' access tokens now. If
        Redis is down they still expire within the access-token lifetime."""
        try:
            await self.revocations.revoke(*session_ids)
        except RedisError as exc:
            log.warning("auth.revocation_publish_failed", error=type(exc).__name__)


def _field_error(field: str, message: str) -> ValidationFailed:
    return ValidationFailed(
        message, errors=[{"loc": ["body", field], "msg": message, "type": field}]
    )

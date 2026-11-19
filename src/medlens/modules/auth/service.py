"""Authentication flows: phone OTP, e-mail + password (with e-mail verification and password
reset by a code sent by e-mail), MFA (TOTP), session refresh and logout.

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
from medlens.modules.auth import recovery, totp
from medlens.modules.auth.metrics import REVOCATION_FALLBACKS
from medlens.modules.auth.models import RefreshToken
from medlens.modules.auth.passwords import password_problems
from medlens.modules.auth.principal import MFA_METHOD, Principal
from medlens.modules.auth.repository import AuthRepository
from medlens.modules.auth.stores import (
    EmailCodeStore,
    LoginLockout,
    MfaGuard,
    OtpResult,
    OtpStore,
    SessionRevocations,
)
from medlens.modules.organizations.models import OrganizationStatus
from medlens.modules.users.models import User, UserStatus
from medlens.ports.email import EmailDeliveryError, EmailPort, EmailUnavailableError
from medlens.ports.sms import SmsDeliveryError, SMSPort, SmsUnavailableError
from medlens.queue import outbox

log = structlog.stdlib.get_logger(__name__)

INVALID_CODE = "The code is invalid or has expired."
INVALID_CREDENTIALS = "Invalid e-mail address or password."
INVALID_REFRESH = "The refresh token is invalid, expired or already used. Sign in again."
OTP_MESSAGE = "Your MedLens code is {code}. It expires in {minutes} minutes. Never share it."
EMAIL_NOT_VERIFIED = "Verify your e-mail address first: enter the code we sent to it."
STAFF_SIGN_IN = (
    "Staff accounts sign in with their e-mail address, password and authenticator app, "
    "not with a code."
)
# Code purposes (EmailCodeStore): each code works only for its own flow.
VERIFY_EMAIL, RESET_PASSWORD, ADD_EMAIL = "verify_email", "reset_password", "add_email"
CONSUMER_SIGN_IN = "consumer_sign_in"
EMAIL_TEXT = {
    CONSUMER_SIGN_IN: (
        "Your MedLens sign-in code",
        "Your MedLens code is {code}. It expires in {minutes} minutes. Never share it. If you "
        "did not try to sign in to MedLens, ignore this message.",
    ),
    VERIFY_EMAIL: (
        "Verify your e-mail address for MedLens",
        "Your MedLens code is {code}. It expires in {minutes} minutes. Enter it to verify this "
        "e-mail address and finish registering your organization. If you did not register "
        "with MedLens, ignore this message.",
    ),
    RESET_PASSWORD: (
        "Reset your MedLens password",
        "Your MedLens password-reset code is {code}. It expires in {minutes} minutes. If you did "
        "not ask to reset your password, ignore this message: your password is unchanged.",
    ),
    ADD_EMAIL: (
        "Confirm your e-mail address for MedLens",
        "Your MedLens code is {code}. It expires in {minutes} minutes. Enter it to add this "
        "e-mail address to your account. If you did not ask for this, ignore this message.",
    ),
}
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


@dataclass(frozen=True, slots=True)
class MfaEnabled:
    tokens: SessionTokens
    recovery_codes: list[str]  # shown once


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
        email: EmailPort | None = None,
    ) -> None:
        self._sessions = sessions
        self._keyring = keyring
        self._sms = sms
        self._email = email
        self._settings = settings
        self._otp = OtpStore(redis, settings)
        self._email_codes = EmailCodeStore(redis, settings)
        self._lockout = LoginLockout(redis, settings)
        self._mfa = MfaGuard(redis, challenge_ttl_seconds=settings.mfa_token_ttl_seconds)
        self.revocations = SessionRevocations(
            redis, access_ttl_seconds=settings.access_token_ttl_seconds
        )

    # ------------------------------------------------------------------ phone OTP
    async def request_otp(self, phone: str, meta: RequestMeta) -> int:
        """Sends a code whether or not an account exists (a new one is created on verify), so
        the response never reveals which numbers are registered. Staff accounts sign in with a
        password and MFA: their numbers get no code (and the same answer)."""
        async with UnitOfWork(self._sessions) as uow:
            repo = AuthRepository(uow.session)
            existing = await repo.user_by_phone_hash(phone_lookup_hash(phone))
            staff = existing is not None and await repo.is_staff(existing)
        if staff:
            log.info("auth.otp_skipped_staff")
            return self._settings.otp_ttl_seconds
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
            if user is not None and await repo.is_staff(user):
                await audit.record(
                    "auth.login_failed",
                    actor_user_id=user.id,
                    meta=meta,
                    details={"method": "otp", "reason": "staff_account"},
                )
                await uow.commit()
                raise Forbidden(STAFF_SIGN_IN)
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
            if user.email_verified_at is None:
                await uow.commit()
                raise Forbidden(EMAIL_NOT_VERIFIED)

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

    # ------------------------------------------------------------------ e-mail codes
    async def _send_code(self, purpose: str, email: str, *, key: str | None = None) -> int:
        """Issue a code for ``purpose`` and e-mail it to ``email``; returns how long it is valid.
        ``key`` scopes the code (default: the address). A refused address gets the same answer
        as any other (no oracle on addresses)."""
        if self._email is None:
            raise DependencyUnavailable(
                "E-mail is not configured.", dependency="email", retry_after_seconds=60
            )
        try:
            code = await self._email_codes.issue(purpose, key or email)
        except RedisError:
            raise _redis_unavailable() from None
        subject, text = EMAIL_TEXT[purpose]
        minutes = self._settings.email_code_ttl_seconds // 60
        try:
            await self._email.send(email, subject, text.format(code=code, minutes=minutes))
        except EmailUnavailableError:
            raise DependencyUnavailable(
                "We could not send the e-mail right now. Try again shortly.",
                dependency="email",
                retry_after_seconds=30,
            ) from None
        except EmailDeliveryError:
            log.warning("auth.email_not_delivered", purpose=purpose)
        return self._settings.email_code_ttl_seconds

    async def _check_code(self, purpose: str, email: str, code: str) -> OtpResult:
        try:
            return await self._email_codes.check(purpose, email, code)
        except RedisError:
            raise _redis_unavailable() from None

    @staticmethod
    def _password_ok(password: str, email: str, field: str) -> None:
        problems = password_problems(password, email=email)
        if problems:
            raise ValidationFailed(
                "The password is not acceptable.",
                errors=[
                    {"loc": ["body", field], "msg": problem, "type": "password"}
                    for problem in problems
                ],
            )

    async def request_email_otp(self, email: str, meta: RequestMeta) -> int:
        """Consumers who prefer e-mail to SMS: a sign-in code by e-mail; the account is created
        on first sign-in, as with phone codes. The same answer for every address; staff
        accounts get no code (they sign in with a password and MFA)."""
        async with UnitOfWork(self._sessions) as uow:
            repo = AuthRepository(uow.session)
            existing = await repo.user_by_email(email)
            staff = existing is not None and await repo.is_staff(existing)
            await AuditService(uow.session).record(
                "auth.email_otp_requested",
                actor_user_id=existing.id if existing else None,
                meta=meta,
                details={"staff_account": staff},
            )
            await uow.commit()
        if staff:
            return self._settings.email_code_ttl_seconds
        return await self._send_code(CONSUMER_SIGN_IN, email)

    async def verify_email_otp(self, email: str, code: str, meta: RequestMeta) -> SessionTokens:
        """Exchanges the e-mailed code for a consumer session, creating the account (with a
        verified address) on first sign-in."""
        result = await self._check_code(CONSUMER_SIGN_IN, email, code)
        async with UnitOfWork(self._sessions) as uow:
            repo, audit = AuthRepository(uow.session), AuditService(uow.session)
            user = await repo.user_by_email(email)
            if result is not OtpResult.OK:
                await audit.record(
                    "auth.login_failed",
                    meta=meta,
                    details={"method": "email_otp", "reason": result.value},
                )
                await uow.commit()
                raise Unauthorized(INVALID_CODE)
            if user is not None and await repo.is_staff(user):
                await audit.record(
                    "auth.login_failed",
                    actor_user_id=user.id,
                    meta=meta,
                    details={"method": "email_otp", "reason": "staff_account"},
                )
                await uow.commit()
                raise Forbidden(STAFF_SIGN_IN)
            created = user is None
            if user is None:
                user = User(email=email, email_verified_at=_now())
                uow.session.add(user)
                await uow.session.flush()
            elif user.email_verified_at is None:
                user.email_verified_at = _now()  # the code proves the address
            await self._require_active(uow, user, meta)
            tokens = await self._start_session(uow.session, user, ("email_otp",), meta)
            await audit.record(
                "auth.login_succeeded",
                actor_user_id=user.id,
                meta=meta,
                details={"method": "email_otp", "new_account": created},
            )
            await uow.commit()
        return tokens

    async def send_email_verification(self, email: str) -> int:
        """E-mails a code proving the address (organization founders, at registration)."""
        return await self._send_code(VERIFY_EMAIL, email)

    async def verify_email(
        self, email: str, code: str, meta: RequestMeta
    ) -> SessionTokens | MfaChallenge:
        """Proves the address with its sign-up code and signs the account in."""
        result = await self._check_code(VERIFY_EMAIL, email, code)
        async with UnitOfWork(self._sessions) as uow:
            repo, audit = AuthRepository(uow.session), AuditService(uow.session)
            user = await repo.user_by_email(email)
            if result is not OtpResult.OK or user is None:
                await audit.record(
                    "auth.email_verification_failed", meta=meta, details={"reason": result.value}
                )
                await uow.commit()
                raise Unauthorized(INVALID_CODE)
            await self._require_active(uow, user, meta)
            if user.email_verified_at is None:
                user.email_verified_at = _now()
            await audit.record("auth.email_verified", actor_user_id=user.id, meta=meta)
            return await self._session_or_challenge(uow, user, meta)

    async def sign_in_invited(
        self, user_id: UUID, meta: RequestMeta
    ) -> SessionTokens | MfaChallenge:
        """Right after an account was created from an invitation (which proved its address and
        set its password): the next step, an MFA enrolment challenge for staff."""
        async with UnitOfWork(self._sessions) as uow:
            user = await AuthRepository(uow.session).user_by_id(user_id)
            if user is None:
                raise Unauthorized("Sign in again.")
            await self._require_active(uow, user, meta)
            return await self._session_or_challenge(uow, user, meta)

    async def _session_or_challenge(
        self, uow: UnitOfWork, user: User, meta: RequestMeta
    ) -> SessionTokens | MfaChallenge:
        """After a proven password or e-mail code: a session, or an MFA challenge when the
        account needs a second factor. Commits."""
        repo = AuthRepository(uow.session)
        if user.mfa_enabled or await self._mfa_mandatory(repo, user):
            purpose = "verify" if user.mfa_enabled else "enrol"
            token = self._keyring.issue(
                "mfa", subject=user.id, auth_methods=("pwd",), purpose=purpose
            )
            await uow.commit()
            status: Literal["mfa_required", "mfa_enrolment_required"] = (
                "mfa_required" if user.mfa_enabled else "mfa_enrolment_required"
            )
            return MfaChallenge(status, token, self._keyring.mfa_ttl_seconds)
        tokens = await self._start_session(uow.session, user, ("pwd",), meta)
        await AuditService(uow.session).record(
            "auth.login_succeeded",
            actor_user_id=user.id,
            meta=meta,
            details={"method": "email_code"},
        )
        await uow.commit()
        return tokens

    async def forgot_password(self, email: str, meta: RequestMeta) -> int:
        """E-mails a reset code to an active account with this address; the answer is the same
        for any address."""
        async with UnitOfWork(self._sessions) as uow:
            user = await AuthRepository(uow.session).user_by_email(email)
            active = user is not None and user.status is UserStatus.ACTIVE
            await AuditService(uow.session).record(
                "auth.password_reset_requested",
                actor_user_id=user.id if user else None,
                meta=meta,
                details={"account_found": active},
            )
            await uow.commit()
        if not active:
            return self._settings.email_code_ttl_seconds
        return await self._send_code(RESET_PASSWORD, email)

    async def reset_password(
        self, email: str, code: str, new_password: str, meta: RequestMeta
    ) -> None:
        """Sets a new password with the e-mailed code and ends every session. The code also
        proves the address, so an unverified account becomes verified."""
        self._password_ok(new_password, email, "new_password")
        result = await self._check_code(RESET_PASSWORD, email, code)
        async with UnitOfWork(self._sessions) as uow:
            repo, audit = AuthRepository(uow.session), AuditService(uow.session)
            user = await repo.user_by_email(email)
            if result is not OtpResult.OK or user is None:
                await audit.record(
                    "auth.password_reset_failed", meta=meta, details={"reason": result.value}
                )
                await uow.commit()
                raise Unauthorized(INVALID_CODE)
            await self._require_active(uow, user, meta)
            user.password_hash = hash_password(new_password)
            if user.email_verified_at is None:
                user.email_verified_at = _now()
            ended = await self.end_sessions(uow.session, user.id)
            await audit.record(
                "auth.password_reset",
                actor_user_id=user.id,
                meta=meta,
                details={"sessions_ended": len(ended)},
            )
            await uow.commit()
        await self._publish_revocations(ended)
        try:
            await self._lockout.record_success(email)  # a reset clears a lockout
        except RedisError:
            log.warning("auth.lockout_not_cleared")

    async def request_email_change(
        self, principal: Principal, email: str, meta: RequestMeta
    ) -> int:
        """Signed in (e.g. by phone): e-mails a code to add this address to the account. An
        address another account uses gets no code, and the same answer."""
        if principal.user_id is None:
            raise Forbidden("Only signed-in users can add an e-mail address.")
        async with UnitOfWork(self._sessions) as uow:
            owner = await AuthRepository(uow.session).user_by_email(email)
            taken = owner is not None and owner.id != principal.user_id
            await AuditService(uow.session).record(
                "auth.email_change_requested",
                actor_user_id=principal.user_id,
                meta=meta,
                details={"address_taken": taken},
            )
            await uow.commit()
        if taken:
            return self._settings.email_code_ttl_seconds
        return await self._send_code(ADD_EMAIL, email, key=_scoped(principal.user_id, email))

    async def confirm_email_change(
        self, principal: Principal, email: str, code: str, meta: RequestMeta
    ) -> User:
        """Sets the account's (verified) e-mail address with the code. To sign in with it, set a
        password with the password-reset flow."""
        if principal.user_id is None:
            raise Forbidden("Only signed-in users can add an e-mail address.")
        result = await self._check_code(ADD_EMAIL, _scoped(principal.user_id, email), code)
        async with UnitOfWork(self._sessions) as uow:
            repo, audit = AuthRepository(uow.session), AuditService(uow.session)
            if result is not OtpResult.OK:
                await audit.record(
                    "auth.email_change_failed",
                    actor_user_id=principal.user_id,
                    meta=meta,
                    details={"reason": result.value},
                )
                await uow.commit()
                raise Unauthorized(INVALID_CODE)
            owner = await repo.user_by_email(email)
            if owner is not None and owner.id != principal.user_id:
                raise Conflict("This e-mail address belongs to another account.")
            user = await repo.user_by_id(principal.user_id)
            if user is None:
                raise Unauthorized("Sign in again.")
            user.email = email
            user.email_verified_at = _now()
            await audit.record("auth.email_changed", actor_user_id=user.id, meta=meta)
            await uow.commit()
        return user

    # ------------------------------------------------------------------ MFA
    def mfa_claims(self, token: str, *, purpose: str) -> TokenClaims:
        try:
            claims = self._keyring.verify(token, "mfa")
        except InvalidTokenError:
            raise Unauthorized("This sign-in attempt has expired. Sign in again.") from None
        if claims.purpose != purpose:
            raise Unauthorized("This token cannot be used here.")
        return claims

    async def verify_mfa(
        self,
        mfa_token: str,
        code: str | None,
        meta: RequestMeta,
        *,
        recovery_code: str | None = None,
    ) -> SessionTokens:
        """The second step of a sign-in: an authenticator code, or one of the account's unused
        recovery codes (each works once). Both count against the challenge's attempts."""
        claims = self.mfa_claims(mfa_token, purpose="verify")
        if not await self._mfa.register_attempt(claims.token_id):
            raise Unauthorized("Too many wrong codes. Sign in again.")
        async with UnitOfWork(self._sessions) as uow:
            repo, audit = AuthRepository(uow.session), AuditService(uow.session)
            user = await repo.user_by_id(claims.subject)
            if user is None or not user.mfa_enabled or user.mfa_secret is None:
                raise Unauthorized("This sign-in attempt has expired. Sign in again.")
            method = "password+mfa"
            if recovery_code is not None:
                used = await repo.unused_recovery_code(user.id, recovery.digest(recovery_code))
                if used is not None:
                    used.used_at = _now()
                    method = "password+recovery_code"
                verified = used is not None
            else:
                verified = code is not None and await self._totp_ok(user, code)
            if not verified:
                await audit.record("auth.mfa_failed", actor_user_id=user.id, meta=meta)
                await uow.commit()
                raise Unauthorized(INVALID_CODE)
            await self._require_active(uow, user, meta)
            tokens = await self._start_session(uow.session, user, ("pwd", MFA_METHOD), meta)
            details: dict[str, object] = {"method": method}
            if recovery_code is not None:
                await uow.session.flush()
                details["recovery_codes_left"] = await repo.recovery_codes_left(user.id)
            await audit.record(
                "auth.login_succeeded", actor_user_id=user.id, meta=meta, details=details
            )
            await uow.commit()
        return tokens

    async def regenerate_recovery_codes(
        self, principal: Principal, code: str, meta: RequestMeta
    ) -> list[str]:
        """New recovery codes for an MFA-verified session that proves the authenticator again;
        the earlier codes stop working."""
        user_id = principal.require_user_id()
        async with UnitOfWork(self._sessions) as uow:
            repo, audit = AuthRepository(uow.session), AuditService(uow.session)
            user = await repo.user_by_id(user_id)
            if user is None or user.status is not UserStatus.ACTIVE:
                raise Unauthorized("Sign in again.")
            if not user.mfa_enabled:
                raise Conflict("Multi-factor authentication is not enabled.")
            if not await self._totp_ok(user, code):
                await audit.record("auth.mfa_failed", actor_user_id=user.id, meta=meta)
                await uow.commit()
                raise ValidationFailed(
                    INVALID_CODE,
                    errors=[{"loc": ["body", "code"], "msg": INVALID_CODE, "type": "mfa_code"}],
                )
            codes = await self._issue_recovery_codes(repo, user.id)
            await audit.record(
                "auth.mfa_recovery_codes_regenerated", actor_user_id=user.id, meta=meta
            )
            await uow.commit()
        return codes

    async def recovery_codes_left(self, user_id: UUID) -> int:
        async with UnitOfWork(self._sessions) as uow:
            return await AuthRepository(uow.session).recovery_codes_left(user_id)

    @staticmethod
    async def _issue_recovery_codes(repo: AuthRepository, user_id: UUID) -> list[str]:
        codes = recovery.new_codes()
        await repo.replace_recovery_codes(user_id, [recovery.digest(code) for code in codes])
        return codes

    async def reset_mfa(self, session: AsyncSession, user: User) -> list[UUID]:
        """An administrator's reset for a lost authenticator, in the caller's transaction:
        removes the secret and the recovery codes and ends every session (returned: publish
        them after committing). Staff set MFA up again at their next sign-in."""
        user.mfa_enabled = False
        user.mfa_secret = None
        await AuthRepository(session).delete_recovery_codes(user.id)
        return await self.end_sessions(session, user.id)

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
    ) -> MfaEnabled:
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
            codes = await self._issue_recovery_codes(AuthRepository(uow.session), user.id)
            await audit.record("auth.mfa_enabled", actor_user_id=user.id, meta=meta)
            methods = tuple(dict.fromkeys((*prior_methods, MFA_METHOD)))
            tokens = await self._start_session(uow.session, user, methods, meta)
            await uow.commit()
        return MfaEnabled(tokens, codes)

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
        """Every staff account needs a second factor (decision log, "Accounts: consumers sign
        themselves up, staff are invited")."""
        return await repo.is_staff(user)

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


def _scoped(user_id: UUID, email: str) -> str:
    """An add-address code belongs to one account and one address."""
    return f"{user_id}:{email}"


def _field_error(field: str, message: str) -> ValidationFailed:
    return ValidationFailed(
        message, errors=[{"loc": ["body", field], "msg": message, "type": field}]
    )

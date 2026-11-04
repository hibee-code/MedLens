"""Short-lived authentication state in Redis.

Redis errors here are deliberately not swallowed: OTP checks and login lockouts fail closed
(the caller turns the error into a 503). Only the session-revocation lookup tolerates an outage,
see ``SessionRevocations.is_revoked``.
"""

import secrets
from enum import StrEnum
from uuid import UUID

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from medlens.cache.keys import redis_key
from medlens.core.config import AuthSettings
from medlens.core.crypto import ip_hash, keyed_digest, phone_lookup_hash
from medlens.core.security import constant_time_equal
from medlens.modules.auth.metrics import REVOCATION_CHECKS_SKIPPED

log = structlog.stdlib.get_logger(__name__)


class OtpResult(StrEnum):
    OK = "ok"
    INVALID = "invalid"
    LOCKED = "locked"  # too many wrong attempts; a new code must be requested
    MISSING = "missing"  # expired or never issued


class OtpStore:
    """One pending code per phone, stored only as a keyed hash, with an attempt counter."""

    def __init__(self, redis: Redis, settings: AuthSettings) -> None:
        self._redis = redis
        self._ttl = settings.otp_ttl_seconds
        self._max_attempts = settings.otp_max_attempts

    @staticmethod
    def _key(phone: str) -> str:
        return redis_key("otp", phone_lookup_hash(phone))

    @staticmethod
    def _digest(phone: str, code: str) -> str:
        return keyed_digest(f"{phone}:{code}", purpose="otp")

    async def issue(self, phone: str) -> str:
        """A new 6-digit code; replaces any pending one (and its attempt count)."""
        code = f"{secrets.randbelow(1_000_000):06d}"
        key = self._key(phone)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            pipe.hset(key, mapping={"digest": self._digest(phone, code), "attempts": 0})
            pipe.expire(key, self._ttl)
            await pipe.execute()
        return code

    async def check(self, phone: str, code: str) -> OtpResult:
        key = self._key(phone)
        stored = await self._redis.hget(key, "digest")
        if stored is None:
            return OtpResult.MISSING
        # Count the attempt before comparing, so parallel guesses cannot exceed the limit.
        attempts = await self._redis.hincrby(key, "attempts", 1)
        if attempts > self._max_attempts:
            return OtpResult.LOCKED
        stored_digest = stored.decode() if isinstance(stored, bytes) else stored
        if constant_time_equal(stored_digest, self._digest(phone, code)):
            await self._redis.delete(key)
            return OtpResult.OK
        return OtpResult.INVALID


class LoginLockout:
    """Progressive lockout per account and per client IP.

    ``login_max_failures`` failures within the window lock the subject for the base duration;
    each further lockout doubles it, up to the maximum. Unknown accounts are counted exactly
    like real ones, so lockouts reveal nothing about which e-mail addresses exist.
    """

    _LEVEL_TTL_SECONDS = 24 * 3600

    def __init__(self, redis: Redis, settings: AuthSettings) -> None:
        self._redis = redis
        self._settings = settings

    @staticmethod
    def _subjects(email: str, ip: str | None) -> list[str]:
        subjects = [f"account:{keyed_digest(email.strip().lower(), purpose='login')}"]
        if ip:
            subjects.append(f"ip:{ip_hash(ip)}")
        return subjects

    async def retry_after(self, email: str, ip: str | None) -> int | None:
        """Seconds until the account or IP is unlocked, or None when neither is locked."""
        remaining = [
            await self._redis.ttl(redis_key("login", "lock", subject))
            for subject in self._subjects(email, ip)
        ]
        locked = [seconds for seconds in remaining if seconds > 0]
        return max(locked) if locked else None

    async def record_failure(self, email: str, ip: str | None) -> None:
        settings = self._settings
        for subject in self._subjects(email, ip):
            failures_key = redis_key("login", "failures", subject)
            failures = await self._redis.incr(failures_key)
            if failures == 1:
                await self._redis.expire(failures_key, settings.login_failure_window_seconds)
            if failures < settings.login_max_failures:
                continue
            level_key = redis_key("login", "level", subject)
            level = await self._redis.incr(level_key)
            await self._redis.expire(level_key, self._LEVEL_TTL_SECONDS)
            duration = min(
                settings.login_lockout_base_seconds * 2 ** (level - 1),
                settings.login_lockout_max_seconds,
            )
            await self._redis.set(redis_key("login", "lock", subject), 1, ex=duration)
            await self._redis.delete(failures_key)
            log.warning("auth.lockout", subject=subject.split(":")[0], seconds=duration)

    async def record_success(self, email: str) -> None:
        subject = self._subjects(email, None)[0]
        await self._redis.delete(
            redis_key("login", "failures", subject), redis_key("login", "level", subject)
        )


class SessionRevocations:
    """Revoked session ids, kept as long as an access token of that session can still be valid."""

    def __init__(self, redis: Redis, *, access_ttl_seconds: int) -> None:
        self._redis = redis
        self.ttl_seconds = access_ttl_seconds + 60  # covers clock leeway

    async def revoke(self, *session_ids: UUID) -> None:
        if not session_ids:
            return
        async with self._redis.pipeline(transaction=False) as pipe:
            for session_id in session_ids:
                pipe.set(redis_key("revoked_session", session_id), 1, ex=self.ttl_seconds)
            await pipe.execute()

    async def is_revoked(self, session_id: UUID) -> bool | None:
        """None when Redis is unavailable: the check was skipped, and is counted. Callers then
        let consumer actions through (a revoked token works until it expires, at most the
        access-token lifetime) and check privileged ones in Postgres
        (``AuthService.ensure_not_revoked``), so an outage does not fail every request."""
        try:
            return bool(await self._redis.exists(redis_key("revoked_session", session_id)))
        except RedisError as exc:
            REVOCATION_CHECKS_SKIPPED.inc()
            log.warning("auth.revocation_check_skipped", error=type(exc).__name__)
            return None


class MfaGuard:
    """Limits guesses per MFA challenge and rejects reuse of a TOTP code within its window."""

    _MAX_ATTEMPTS = 5

    def __init__(self, redis: Redis, *, challenge_ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl = challenge_ttl_seconds

    async def register_attempt(self, challenge_id: str) -> bool:
        """False once the challenge has been used up."""
        key = redis_key("mfa", "attempts", challenge_id)
        attempts = await self._redis.incr(key)
        if attempts == 1:
            await self._redis.expire(key, self._ttl)
        return attempts <= self._MAX_ATTEMPTS

    async def claim_timestep(self, user_id: UUID, timestep: int) -> bool:
        """False if this user already used the code of this 30-second step."""
        claimed = await self._redis.set(
            redis_key("mfa", "used", user_id, timestep), 1, nx=True, ex=120
        )
        return bool(claimed)

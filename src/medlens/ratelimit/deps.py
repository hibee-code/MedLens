"""Rate-limit dependencies. Declared in the route decorator (``dependencies=[...]``), so they
run before the request's other dependencies, such as decoding an uploaded image.

    guest scans      MEDLENS_RATELIMIT__GUEST_SCANS_PER_MINUTE per client IP
    signed-in scans  MEDLENS_RATELIMIT__USER_SCANS_PER_MINUTE per user or API key
    sign-in          MEDLENS_RATELIMIT__LOGINS_PER_MINUTE per client IP (codes, passwords, MFA)
    assistant        MEDLENS_RATELIMIT__ASSISTANT_PER_MINUTE per user

Client IPs are kept only as keyed hashes, even in Redis.
"""

from typing import Annotated

from fastapi import Depends, Request

from medlens.core.config import Settings
from medlens.core.crypto import ip_hash
from medlens.core.errors import RateLimited
from medlens.modules.auth.deps import OptionalPrincipal
from medlens.modules.auth.principal import Principal
from medlens.resources import get_app_settings, get_resources

SettingsDep = Annotated[Settings, Depends(get_app_settings)]


def _client(request: Request) -> str:
    return ip_hash(request.client.host if request.client else "unknown")


def _principal_key(principal: Principal) -> str:
    if principal.api_key_id is not None:
        return f"apikey:{principal.api_key_id}"
    return f"user:{principal.user_id}"


async def _enforce(request: Request, bucket: str, subject: str, limit: int) -> None:
    settings: Settings = get_app_settings(request)
    if not settings.ratelimit.enabled:
        return
    retry = await get_resources(request).rate_limiter.retry_after(bucket, subject, limit=limit)
    if retry is not None:
        raise RateLimited("Too many requests. Try again in a moment.", retry_after_seconds=retry)


async def limit_scans(
    request: Request, principal: OptionalPrincipal, settings: SettingsDep
) -> None:
    limits = settings.ratelimit
    if principal is None:
        await _enforce(request, "scans_guest", _client(request), limits.guest_scans_per_minute)
    else:
        await _enforce(
            request, "scans_user", _principal_key(principal), limits.user_scans_per_minute
        )


async def limit_logins(request: Request, settings: SettingsDep) -> None:
    await _enforce(request, "login", _client(request), settings.ratelimit.logins_per_minute)


async def limit_assistant(
    request: Request, principal: OptionalPrincipal, settings: SettingsDep
) -> None:
    subject = _principal_key(principal) if principal is not None else _client(request)
    await _enforce(request, "assistant", subject, settings.ratelimit.assistant_per_minute)

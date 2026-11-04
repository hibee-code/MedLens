"""Authentication dependencies.

``CurrentPrincipal``   a signed-in user (Bearer access token) or machine client (X-API-Key).
``OptionalPrincipal``  the same, or None for guests (e.g. scan submission). Credentials that
                       are present but invalid are still rejected, never downgraded to guest.
``UserPrincipal``      a signed-in user only.
``SessionCheckedUser`` a signed-in user whose session is confirmed not revoked even while Redis
                       is down (Postgres fallback): for account changes and minting tokens.
``MfaVerifiedUser``    a session-checked user whose session passed a second factor.

Session revocation lives in Redis. When Redis is down the check is skipped (and counted):
consumer actions fail open, while privileged ones fall back to Postgres, either here
(``SessionCheckedUser``) or when RBAC resolves an actor holding a privileged permission
(``modules/rbac/deps.py``).
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from medlens.core.config import Settings
from medlens.core.context import org_id_var, user_id_var
from medlens.core.errors import Forbidden, Unauthorized
from medlens.core.request_meta import RequestMeta, get_request_meta
from medlens.core.security import InvalidTokenError
from medlens.modules.auth.principal import Principal
from medlens.modules.auth.service import SESSION_ENDED, AuthService
from medlens.resources import Resources, get_app_settings, get_resources

_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="bearerAuth",
    description="Access token from the auth endpoints.",
)
_api_key = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    scheme_name="apiKeyAuth",
    description="Organization API key.",
)


def get_auth_service(
    resources: Annotated[Resources, Depends(get_resources)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> AuthService:
    return AuthService(
        sessions=resources.session_factory,
        keyring=resources.keyring,
        sms=resources.sms,
        redis=resources.redis,
        settings=settings.auth,
    )


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
Meta = Annotated[RequestMeta, Depends(get_request_meta)]


async def _authenticate(
    resources: Resources,
    auth: AuthService,
    credentials: HTTPAuthorizationCredentials | None,
    api_key: str | None,
) -> Principal | None:
    if credentials is not None:
        try:
            claims = resources.keyring.verify(credentials.credentials, "access")
        except InvalidTokenError:
            raise Unauthorized("The access token is invalid or has expired.") from None
        revoked = (
            await auth.revocations.is_revoked(claims.session_id)
            if claims.session_id is not None
            else False
        )
        if revoked:
            raise Unauthorized(SESSION_ENDED)
        principal = Principal(
            kind="user",
            user_id=claims.subject,
            organization_id=claims.organization_id,
            session_id=claims.session_id,
            perm_ver=claims.perm_ver,
            auth_methods=claims.auth_methods,
            issued_at=datetime.fromtimestamp(claims.issued_at, UTC),
            revocation_checked=revoked is not None,
        )
    elif api_key is not None:
        found = await auth.authenticate_api_key(api_key)
        if found is None:
            raise Unauthorized("The API key is invalid, expired or revoked.")
        principal = found
    else:
        return None
    user_id_var.set(principal.user_id)
    org_id_var.set(principal.organization_id)
    return principal


async def get_optional_principal(
    request: Request,
    auth: AuthServiceDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    api_key: Annotated[str | None, Depends(_api_key)],
) -> Principal | None:
    return await _authenticate(get_resources(request), auth, credentials, api_key)


async def get_current_principal(
    principal: Annotated[Principal | None, Depends(get_optional_principal)],
) -> Principal:
    if principal is None:
        raise Unauthorized("Sign in, or send an API key.")
    return principal


async def get_user_principal(
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> Principal:
    if not principal.is_user:
        raise Forbidden("This endpoint is for signed-in users, not API keys.")
    return principal


async def get_session_checked_user(
    principal: Annotated[Principal, Depends(get_user_principal)],
    auth: AuthServiceDep,
) -> Principal:
    await auth.ensure_not_revoked(principal)
    return principal


async def require_mfa_verified(
    principal: Annotated[Principal, Depends(get_session_checked_user)],
) -> Principal:
    if not principal.mfa_verified:
        raise Forbidden("This action requires a session verified with multi-factor authentication.")
    return principal


async def get_mfa_enrollee(
    request: Request,
    auth: AuthServiceDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """For MFA enrolment: a signed-in user, or someone holding the enrolment token returned by
    a login that requires MFA before any session is issued."""
    if credentials is None:
        raise Unauthorized("Sign in first.")
    try:
        claims = auth.mfa_claims(credentials.credentials, purpose="enrol")
    except Unauthorized:  # not an enrolment token: it must be a normal access token
        principal = await _authenticate(get_resources(request), auth, credentials, None)
        if principal is None or not principal.is_user:
            raise Unauthorized("Sign in first.") from None
        await auth.ensure_not_revoked(principal)
        return principal
    user_id_var.set(claims.subject)
    return Principal(
        kind="user",
        user_id=claims.subject,
        organization_id=None,
        auth_methods=claims.auth_methods,
    )


OptionalPrincipal = Annotated[Principal | None, Depends(get_optional_principal)]
CurrentPrincipal = Annotated[Principal, Depends(get_current_principal)]
UserPrincipal = Annotated[Principal, Depends(get_user_principal)]
SessionCheckedUser = Annotated[Principal, Depends(get_session_checked_user)]
MfaVerifiedUser = Annotated[Principal, Depends(require_mfa_verified)]
MfaEnrollee = Annotated[Principal, Depends(get_mfa_enrollee)]

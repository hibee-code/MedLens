from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status

from medlens.core.openapi import problem_responses
from medlens.core.routing import api_v1_router
from medlens.modules.auth.api_keys import ApiKeyService
from medlens.modules.auth.deps import (
    AuthServiceDep,
    MfaEnrollee,
    SessionCheckedUser,
    UserPrincipal,
)
from medlens.modules.auth.deps import Meta as RequestMetaDep
from medlens.modules.auth.schemas import (
    AccessTokenResponse,
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyRead,
    JwksResponse,
    LoginRequest,
    LoginResponse,
    MfaChallengeResponse,
    MfaConfirmRequest,
    MfaEnrolment,
    MfaVerifyRequest,
    OtpRequest,
    OtpRequested,
    OtpVerify,
    RefreshRequest,
    SwitchOrganizationRequest,
    TokenResponse,
)
from medlens.modules.auth.service import MfaChallenge, SessionTokens
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.deps import require_permissions
from medlens.modules.rbac.permissions import Permission
from medlens.ratelimit.deps import limit_logins
from medlens.resources import Resources, get_resources

router = api_v1_router("/auth", tags=["auth"])
api_keys_router = api_v1_router("/api-keys", tags=["auth"])
well_known_router = APIRouter(tags=["auth"])


def _token_response(tokens: SessionTokens) -> TokenResponse:
    return TokenResponse(
        access_token=tokens.access_token,
        expires_in=tokens.expires_in,
        refresh_token=tokens.refresh_token,
        refresh_expires_in=tokens.refresh_expires_in,
        organization_id=tokens.organization_id,
    )


# ------------------------------------------------------------------ phone OTP
@router.post(
    "/otp/request",
    response_model=OtpRequested,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Send a sign-in code by SMS",
    description=(
        "Sends a 6-digit code to a Nigerian mobile number (valid 5 minutes). The response is "
        "identical whether or not an account exists: the account is created on first sign-in."
    ),
    responses=problem_responses(422, 429, 503),
    dependencies=[Depends(limit_logins)],
)
async def request_otp(body: OtpRequest, auth: AuthServiceDep, meta: RequestMetaDep) -> OtpRequested:
    expires_in = await auth.request_otp(body.phone, meta)
    return OtpRequested(
        detail="If the number can receive SMS, a code is on its way.", expires_in=expires_in
    )


@router.post(
    "/otp/verify",
    response_model=TokenResponse,
    summary="Sign in with an SMS code",
    description=(
        "Exchanges the code for a session. Five wrong attempts invalidate the code; request a "
        "new one. Creates the account on first sign-in."
    ),
    responses=problem_responses(401, 403, 422, 429, 503),
    dependencies=[Depends(limit_logins)],
)
async def verify_otp(body: OtpVerify, auth: AuthServiceDep, meta: RequestMetaDep) -> TokenResponse:
    return _token_response(await auth.verify_otp(body.phone, body.code, meta))


# ------------------------------------------------------------------ e-mail + password
@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Sign in with e-mail and password",
    description=(
        "For staff accounts. Returns a session (`status: authenticated`) or, when a second "
        "factor is needed, an MFA challenge: `mfa_required` (call /auth/mfa/verify) or "
        "`mfa_enrolment_required` (regulators and platform admins must set up MFA first). "
        "Repeated failures lock the account and the client IP for a growing period."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "examples": {
                        "authenticated": {
                            "summary": "Signed in",
                            "value": TokenResponse.model_json_schema()["examples"][0],
                        },
                        "mfa_required": {
                            "summary": "Second factor needed",
                            "value": MfaChallengeResponse.model_json_schema()["examples"][0],
                        },
                    }
                }
            }
        },
        **problem_responses(401, 403, 422, 429, 503),
    },
    dependencies=[Depends(limit_logins)],
)
async def login(
    body: LoginRequest, auth: AuthServiceDep, meta: RequestMetaDep
) -> TokenResponse | MfaChallengeResponse:
    result = await auth.login(body.email, body.password.get_secret_value(), meta)
    if isinstance(result, MfaChallenge):
        return MfaChallengeResponse(
            status=result.status, mfa_token=result.mfa_token, expires_in=result.expires_in
        )
    return _token_response(result)


# ------------------------------------------------------------------ MFA
@router.post(
    "/mfa/verify",
    response_model=TokenResponse,
    summary="Complete sign-in with an authenticator code",
    description="Second step of a login that returned `mfa_required`. A code works only once.",
    responses=problem_responses(401, 403, 422, 429, 503),
    dependencies=[Depends(limit_logins)],
)
async def verify_mfa(
    body: MfaVerifyRequest, auth: AuthServiceDep, meta: RequestMetaDep
) -> TokenResponse:
    return _token_response(await auth.verify_mfa(body.mfa_token, body.code, meta))


@router.post(
    "/mfa/enrol",
    response_model=MfaEnrolment,
    summary="Start setting up an authenticator app",
    description=(
        "Returns a TOTP secret and an otpauth:// URI to show as a QR code. Authenticate with an "
        "access token, or with the `mfa_token` of a login that returned `mfa_enrolment_required`."
    ),
    responses=problem_responses(401, 409),
)
async def enrol_mfa(
    enrollee: MfaEnrollee, auth: AuthServiceDep, meta: RequestMetaDep
) -> MfaEnrolment:
    setup = await auth.start_mfa_enrolment(enrollee.require_user_id(), meta)
    return MfaEnrolment(secret=setup.secret, otpauth_uri=setup.otpauth_uri)


@router.post(
    "/mfa/confirm",
    response_model=TokenResponse,
    summary="Turn on MFA with a first authenticator code",
    description=(
        "Confirms the authenticator app and enables MFA. Returns a new session verified with "
        "MFA (same authentication as /auth/mfa/enrol)."
    ),
    responses=problem_responses(401, 409, 422),
)
async def confirm_mfa(
    body: MfaConfirmRequest, enrollee: MfaEnrollee, auth: AuthServiceDep, meta: RequestMetaDep
) -> TokenResponse:
    tokens = await auth.confirm_mfa(
        enrollee.require_user_id(), body.code, enrollee.auth_methods, meta
    )
    return _token_response(tokens)


# ------------------------------------------------------------------ sessions
@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate the refresh token",
    description=(
        "Returns a new access token and a new refresh token; the old refresh token stops "
        "working. Presenting a refresh token that was already used ends the whole session "
        "(it signals a stolen token)."
    ),
    responses=problem_responses(401, 422),
)
async def refresh(
    body: RefreshRequest, auth: AuthServiceDep, meta: RequestMetaDep
) -> TokenResponse:
    return _token_response(await auth.refresh(body.refresh_token, meta))


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="End this session",
    description="Revokes the current session's refresh tokens and access tokens.",
    responses=problem_responses(401, 403),
)
async def logout(principal: UserPrincipal, auth: AuthServiceDep, meta: RequestMetaDep) -> None:
    await auth.logout(principal, meta)


@router.post(
    "/logout-all",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="End every session of this account",
    description="Signs the account out on every device.",
    responses=problem_responses(401, 403),
)
async def logout_all(principal: UserPrincipal, auth: AuthServiceDep, meta: RequestMetaDep) -> None:
    await auth.logout_all(principal, meta)


@router.post(
    "/switch-organization",
    response_model=AccessTokenResponse,
    summary="Act on behalf of another organization",
    description=(
        "Issues an access token whose active organization is one the user belongs to (or none). "
        "The choice is kept when the session is refreshed."
    ),
    responses=problem_responses(401, 403, 404, 422),
)
async def switch_organization(
    body: SwitchOrganizationRequest,
    principal: SessionCheckedUser,
    auth: AuthServiceDep,
    meta: RequestMetaDep,
) -> AccessTokenResponse:
    token = await auth.switch_organization(principal, body.organization_id, meta)
    return AccessTokenResponse(
        access_token=token.access_token,
        expires_in=token.expires_in,
        organization_id=token.organization_id,
    )


# ------------------------------------------------------------------ JWKS
@well_known_router.get(
    "/.well-known/jwks.json",
    response_model=JwksResponse,
    summary="Public keys for verifying access tokens",
    description=(
        "JSON Web Key Set (Ed25519) of every key that may have signed a currently valid "
        "access token. Match the token's `kid` header."
    ),
)
async def jwks(
    response: Response, resources: Annotated[Resources, Depends(get_resources)]
) -> JwksResponse:
    response.headers["Cache-Control"] = "public, max-age=300"
    return JwksResponse.model_validate(resources.keyring.jwks())


# ------------------------------------------------------------------ API keys
def get_api_key_service(resources: Annotated[Resources, Depends(get_resources)]) -> ApiKeyService:
    return ApiKeyService(resources.session_factory)


ApiKeyServiceDep = Annotated[ApiKeyService, Depends(get_api_key_service)]
ApiKeyManager = Annotated[Actor, Depends(require_permissions(Permission.APIKEY_MANAGE, fresh=True))]


def _api_key_read(api_key: object) -> ApiKeyRead:
    return ApiKeyRead.model_validate(api_key)


@api_keys_router.post(
    "",
    response_model=ApiKeyCreated,
    status_code=status.HTTP_201_CREATED,
    summary="Create an API key",
    description=(
        "Creates a key for the active organization. The full key is returned only in this "
        "response. Scopes must be permissions you hold, and cannot be administrative."
    ),
    responses=problem_responses(401, 403, 422),
)
async def create_api_key(
    body: ApiKeyCreate, actor: ApiKeyManager, service: ApiKeyServiceDep, meta: RequestMetaDep
) -> ApiKeyCreated:
    api_key, key = await service.create(
        actor, name=body.name, scopes=body.scopes, expires_at=body.expires_at, meta=meta
    )
    return ApiKeyCreated(**_api_key_read(api_key).model_dump(), key=key)


@api_keys_router.get(
    "",
    response_model=list[ApiKeyRead],
    summary="List the organization's API keys",
    description="Keys of the active organization, newest first. Secrets are never returned.",
    responses=problem_responses(401, 403, 422),
)
async def list_api_keys(actor: ApiKeyManager, service: ApiKeyServiceDep) -> list[ApiKeyRead]:
    return [_api_key_read(api_key) for api_key in await service.list(actor)]


@api_keys_router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Revoke an API key",
    description="The key stops working immediately. Revoking twice is harmless.",
    responses=problem_responses(401, 403, 404, 422),
)
async def revoke_api_key(
    key_id: UUID, actor: ApiKeyManager, service: ApiKeyServiceDep, meta: RequestMetaDep
) -> None:
    await service.revoke(actor, key_id, meta)

"""Who is on the other end of a request, for audit logs and lockouts.

The client address is what Uvicorn reports. With ``--proxy-headers`` it is taken from
X-Forwarded-For only when the immediate peer is in FORWARDED_ALLOW_IPS, so it cannot be spoofed
by clients. Raw IPs are never stored: audit logs keep an HMAC (core.crypto.ip_hash).
"""

from dataclasses import dataclass

from fastapi import Request

_MAX_USER_AGENT = 256


@dataclass(frozen=True, slots=True)
class RequestMeta:
    ip: str | None
    user_agent: str | None


def get_request_meta(request: Request) -> RequestMeta:
    user_agent = request.headers.get("user-agent")
    return RequestMeta(
        ip=request.client.host if request.client else None,
        user_agent=user_agent[:_MAX_USER_AGENT] if user_agent else None,
    )

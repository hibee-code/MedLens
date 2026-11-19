"""Invite the first platform admin (`make bootstrap-admin EMAIL=...`).

    make bootstrap-admin EMAIL=ops@example.ng

MedLens seeds no users: the first platform admin is invited like everyone else. This command
creates a platform-admin invitation for the address and e-mails it; the invitee opens it and
creates the account (`POST /api/v1/invitations/{token}/accept-new`), then sets up an
authenticator app. It refuses once an active platform admin exists: from then on admins invite
admins (`POST /api/v1/admin/invitations`). Running it again before the invitation is accepted
replaces the earlier invitation.

The token is printed only in local and test environments (for the demo and for development
without an e-mail provider). Elsewhere it travels only by e-mail: if sending fails, the command
exits with an error and can be run again.
"""

import argparse
import asyncio
import sys

from pydantic import ValidationError

from medlens.core.config import Environment, Settings
from medlens.core.errors import Conflict
from medlens.core.request_meta import RequestMeta

PRINTS_TOKEN = frozenset({Environment.LOCAL, Environment.TEST})
OPERATOR = RequestMeta(ip=None, user_agent="medlens-bootstrap-admin")


async def bootstrap(settings: Settings, email: str) -> tuple[str, bool, str]:
    """Returns the token, whether it was e-mailed, and the masked address."""
    from medlens.adapters.email import create_email_adapter
    from medlens.cache.redis import create_redis_client
    from medlens.db.session import create_db_engine, create_session_factory
    from medlens.modules.organizations.onboarding import InvitationService

    engine = create_db_engine(settings.database, application_name="medlens-bootstrap-admin")
    redis = create_redis_client(settings.redis, db=settings.redis.cache_db)
    mailer = create_email_adapter(settings.email)
    try:
        service = InvitationService(
            create_session_factory(engine),
            redis,
            mailer,
            frontend_url=settings.app.frontend_url,
        )
        created = await service.bootstrap_platform(email, OPERATOR)
        return created.token, created.emailed, created.recipient
    finally:
        await mailer.aclose()
        await redis.aclose()
        await engine.dispose()


def _address(raw: str) -> str:
    from medlens.modules.organizations.schemas import PlatformInvitationCreate

    return PlatformInvitationCreate.model_validate({"email": raw}).email


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--email", required=True, help="the first platform admin's address")
    args = parser.parse_args(argv)

    from medlens.core.bootstrap import configure_process
    from medlens.core.config import get_settings

    try:
        email = _address(args.email)
    except ValidationError:
        sys.stderr.write("refusing: not a valid e-mail address\n")
        return 2
    settings = get_settings()
    configure_process(settings)
    try:
        token, emailed, recipient = asyncio.run(bootstrap(settings, email))
    except Conflict as exc:
        sys.stderr.write(f"refusing: {exc.detail}\n")
        return 1
    if emailed:
        sys.stdout.write(f"Invitation e-mailed to {recipient}; it expires in 7 days.\n")
    if settings.app.environment in PRINTS_TOKEN:
        sys.stdout.write(f"Invitation token ({settings.app.environment.value} only): {token}\n")
        return 0
    if not emailed:
        sys.stderr.write(
            "The invitation e-mail could not be sent (see the log). Check the e-mail settings "
            "and run the command again: it replaces this invitation.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

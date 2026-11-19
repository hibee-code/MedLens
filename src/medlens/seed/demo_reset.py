"""Clear the local demo's scan data (`make demo-reset`, applied only with CONFIRM=1).

    make demo-reset                       # report what would be deleted
    make demo-reset CONFIRM=1             # delete it
    make demo-reset ACCOUNTS=1 CONFIRM=1  # also erase the demo's accounts

Deletes every scan and what hangs off scans: their review cases and the labels recorded from
lab results, the scan photos in object storage, hotspot snapshots, idempotency keys (only scan
submissions use them), queued events (the outbox, failed tasks and the Celery queues) and the
cached hotspots. Keeps products, batches, organizations, users, sessions, knowledge documents
and the model-call ledger. The audit log is append-only by design (a trigger refuses deletes),
so its entries about the deleted scans stay.

With ACCOUNTS=1 it also erases the demo's accounts (addresses ending in @medlens.test, and the
"Seed ..." accounts earlier seeds created) the way an account erasure does: personal data
removed, memberships deleted, sessions and open invitations revoked, recovery codes deleted, the
row kept with status deleted. The demo's saved sessions and authenticator secrets in .demo/ are
removed with them. Then `make bootstrap-admin` (or the demo) can invite a first platform admin
again.

Refuses unless MEDLENS_APP__ENVIRONMENT is local: never against test, staging or production
data.
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import delete, or_, select, text, update
from sqlalchemy.orm import Session

from medlens.core.config import Environment, Settings

# What is deleted, in this order (foreign keys: labels and cases refer to scans): fixed SQL,
# no input reaches it.
TABLES = {
    "labels": "FROM labels WHERE scan_id IS NOT NULL",
    "cases": "FROM cases",
    "scans": "FROM scans",
    "hotspot_snapshots": "FROM hotspot_snapshots",
    "idempotency_keys": "FROM idempotency_keys",
    "outbox_events": "FROM outbox_events",
    "failed_tasks": "FROM failed_tasks",
}
PHOTO_BATCH = 1000
DEMO_DOMAIN = "@medlens.test"
SEED_NAME_PREFIX = "Seed "  # full names of the accounts earlier seeds created
DEMO_STATE = Path(__file__).resolve().parents[3] / ".demo"


@dataclass(frozen=True, slots=True)
class Inventory:
    rows: dict[str, int]
    photos: list[str]


def inventory(session: Session) -> Inventory:
    rows = {
        table: int(session.scalar(text("SELECT count(*) " + rows_of)) or 0)
        for table, rows_of in TABLES.items()
    }
    photos = list(
        session.scalars(text("SELECT image_s3_key FROM scans WHERE image_s3_key IS NOT NULL"))
    )
    return Inventory(rows, photos)


def reset(session: Session) -> Inventory:
    """Delete the scan data in the caller's transaction; returns what was there."""
    found = inventory(session)
    for rows_of in TABLES.values():
        session.execute(text("DELETE " + rows_of))
    return found


def demo_accounts(session: Session) -> list[UUID]:
    from medlens.modules.users.models import User, UserStatus

    return list(
        session.scalars(
            select(User.id).where(
                User.status != UserStatus.DELETED,
                or_(
                    User.email.ilike("%" + DEMO_DOMAIN),
                    User.full_name.startswith(SEED_NAME_PREFIX),
                ),
            )
        )
    )


def erase_accounts(session: Session) -> int:
    """Erase the demo's accounts in the caller's transaction; returns how many."""
    from medlens.modules.audit.models import AuditLog
    from medlens.modules.auth.models import MfaRecoveryCode, RefreshToken
    from medlens.modules.organizations.models import Invitation, Membership
    from medlens.modules.users.models import User, UserStatus

    ids = demo_accounts(session)
    now = datetime.now(UTC)
    session.execute(
        update(Invitation)
        .where(
            Invitation.email.ilike("%" + DEMO_DOMAIN),
            Invitation.accepted_at.is_(None),
            Invitation.revoked_at.is_(None),
        )
        .values(revoked_at=now)
    )
    if not ids:
        return 0
    session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id.in_(ids), RefreshToken.revoked_at.is_(None))
        .values(revoked_at=now)
    )
    session.execute(delete(Membership).where(Membership.user_id.in_(ids)))
    session.execute(delete(MfaRecoveryCode).where(MfaRecoveryCode.user_id.in_(ids)))
    session.execute(
        update(User)
        .where(User.id.in_(ids))
        .values(
            status=UserStatus.DELETED,
            email=None,
            email_verified_at=None,
            phone_e164=None,
            phone_hash=None,
            password_hash=None,
            full_name=None,
            mfa_secret=None,
            mfa_enabled=False,
            last_login_at=None,
            perm_ver=User.perm_ver + 1,
            tokens_valid_after=now,
        )
    )
    session.add(AuditLog(action="demo.accounts_erased", metadata_={"accounts": len(ids)}))
    return len(ids)


def _forget_demo_sessions(state: Path = DEMO_STATE) -> None:
    """The saved sessions and authenticator secrets belonged to the erased accounts."""
    for path in [*state.glob("sessions/*"), *state.glob("*.totp"), *state.glob("*.recovery")]:
        path.unlink(missing_ok=True)


def render(found: Inventory) -> str:
    lines = [f"  {table}: {count}" for table, count in found.rows.items()]
    lines.append(f"  scan photos in object storage: {len(found.photos)}")
    return "\n".join(lines)


def _delete_photos(settings: Settings, keys: list[str]) -> None:
    from medlens.adapters.s3 import create_sync_s3_client

    client = create_sync_s3_client(settings.s3)
    for start in range(0, len(keys), PHOTO_BATCH):
        batch = keys[start : start + PHOTO_BATCH]
        client.delete_objects(
            Bucket=settings.s3.bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
        )


def _clear_queues_and_caches(settings: Settings) -> int:
    """Purge the Celery queues and bump the hotspot cache generation; returns the messages
    purged."""
    from medlens.cache.redis import create_sync_redis_client
    from medlens.modules.hotspots.service import GENERATION_KEY
    from medlens.queue.celery_app import celery_app

    redis = create_sync_redis_client(settings.redis, db=settings.redis.cache_db)
    try:
        redis.incr(GENERATION_KEY)
    finally:
        redis.close()
    purged: Any = celery_app.control.purge()
    return int(purged or 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--confirm", action="store_true", help="delete (else report only)")
    parser.add_argument(
        "--accounts", action="store_true", help="also erase the demo's @medlens.test accounts"
    )
    args = parser.parse_args(argv)

    from medlens.core.bootstrap import configure_process
    from medlens.core.config import get_settings
    from medlens.db.session import create_sync_engine, create_sync_session_factory

    settings = get_settings()
    if settings.app.environment is not Environment.LOCAL:
        sys.stderr.write(
            f"refusing: demo-reset clears local demo data only; this environment is "
            f"{settings.app.environment.value}\n"
        )
        return 1
    configure_process(settings)
    engine = create_sync_engine(settings.database, application_name="medlens-demo-reset")
    try:
        with create_sync_session_factory(engine)() as session:
            if not args.confirm:
                report = render(inventory(session))
                if args.accounts:
                    report += f"\n  demo accounts to erase: {len(demo_accounts(session))}"
                sys.stdout.write("Would delete:\n" + report + "\n")
                sys.stderr.write("Nothing deleted. Run again with CONFIRM=1 to delete.\n")
                return 0
            found = reset(session)
            erased = erase_accounts(session) if args.accounts else None
            session.commit()
    finally:
        engine.dispose()
    _delete_photos(settings, found.photos)
    purged = _clear_queues_and_caches(settings)
    accounts = ""
    if erased is not None:
        _forget_demo_sessions()
        accounts = f"  demo accounts erased: {erased} (and their saved sessions in .demo/)\n"
    sys.stdout.write(
        "Deleted:\n"
        + render(found)
        + f"\n  queued Celery messages: {purged}\n"
        + accounts
        + "Kept: products, batches, organizations, other users, documents, the model-call "
        "ledger and the audit log (append-only).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

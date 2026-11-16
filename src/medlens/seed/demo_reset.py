"""Clear the local demo's scan data (`make demo-reset`, applied only with CONFIRM=1).

    make demo-reset              # report what would be deleted
    make demo-reset CONFIRM=1    # delete it

Deletes every scan and what hangs off scans: their review cases and the labels recorded from
lab results, the scan photos in object storage, hotspot snapshots, idempotency keys (only scan
submissions use them), queued events (the outbox, failed tasks and the Celery queues) and the
cached hotspots. Keeps products, batches, organizations, users, sessions, knowledge documents
and the model-call ledger. The audit log is append-only by design (a trigger refuses deletes),
so its entries about the deleted scans stay.

Refuses unless MEDLENS_APP__ENVIRONMENT is local: never against test, staging or production
data.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
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
                sys.stdout.write("Would delete:\n" + render(inventory(session)) + "\n")
                sys.stderr.write("Nothing deleted. Run again with CONFIRM=1 to delete.\n")
                return 0
            found = reset(session)
            session.commit()
    finally:
        engine.dispose()
    _delete_photos(settings, found.photos)
    purged = _clear_queues_and_caches(settings)
    sys.stdout.write(
        "Deleted:\n" + render(found) + f"\n  queued Celery messages: {purged}\n"
        "Kept: products, batches, organizations, users, sessions, documents, the model-call "
        "ledger and the audit log (append-only).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

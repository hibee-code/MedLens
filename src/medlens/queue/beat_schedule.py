"""Periodic tasks run by Celery Beat (one Beat process per deployment)."""

from typing import Any

from celery.schedules import crontab

from medlens.core.config import Settings


def beat_schedule(settings: Settings) -> dict[str, dict[str, Any]]:
    return {
        "outbox-relay": {
            "task": "medlens.outbox.relay",
            "schedule": settings.queue.relay_interval_seconds,
            # A relay run older than its own interval is stale: the next one supersedes it.
            "options": {"expires": max(settings.queue.relay_interval_seconds * 5, 5)},
        },
        "hotspots-recompute": {
            "task": "medlens.hotspots.recompute",
            "schedule": crontab(minute="*/15"),
        },
        "reference-embeddings-backfill": {
            "task": "medlens.reference_images.backfill",
            "schedule": crontab(minute="7"),
        },
        "knowledge-embeddings-backfill": {
            "task": "medlens.knowledge.backfill_embeddings",
            "schedule": crontab(minute="37"),
        },
        "scans-partitions": {
            "task": "medlens.maintenance.create_partitions",
            "schedule": crontab(hour="2", minute="0"),
        },
        "purge-expired": {
            "task": "medlens.maintenance.purge_expired",
            "schedule": crontab(hour="3", minute="0"),
        },
    }

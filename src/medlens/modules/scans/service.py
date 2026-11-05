"""Scan submission and reading.

Submission (the API's part; the verdict comes from the worker):

1. The photo was already validated and re-encoded (core/uploads.py); only those bytes go on.
2. Idempotency-Key: a retry replays the first response; the same key with another photo or
   other fields is a 409.
3. The same photo (SHA-256) from the same person or device within the duplicate window returns
   the earlier scan (200) instead of a new one. Near-identical photos are caught in the worker
   by perceptual hash.
4. The photo is stored, then the scan row (status ``received``) and a ``scan.received`` outbox
   event are written in one transaction, and 202 is returned.

Location: raw coordinates never reach the database; they are snapped to a ~1 km cell with a
geohash-6 label (core/geo.py).
"""

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import ColumnElement, false, or_, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from medlens.cache.keys import redis_key
from medlens.core.config import QueueSettings, VerificationSettings
from medlens.core.errors import Forbidden, NotFound, ValidationFailed
from medlens.core.geo import coarsen
from medlens.core.idempotency import claim, request_fingerprint, store
from medlens.core.identifiers import NigerianState
from medlens.core.ids import uuid7, uuid7_time
from medlens.core.pagination import PageParams
from medlens.core.uploads import ValidatedImage
from medlens.db.session import UnitOfWork
from medlens.modules.products.models import Product
from medlens.modules.rbac import policies
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.permissions import Permission
from medlens.modules.scans.models import Scan, ScanStatus, Verdict
from medlens.modules.scans.repository import ScanRepository
from medlens.ports.storage import StoragePort
from medlens.queue import outbox
from medlens.queue.backpressure import BASE_WAIT_SECONDS, Backpressure, QueueDepth

log = structlog.stdlib.get_logger(__name__)
_NOT_FOUND = "Scan not found."


def scan_done_channel(scan_id: object) -> str:
    """Redis pub/sub channel on which the worker announces a scan's result."""
    return redis_key("scan_done", scan_id)


class ScanWaiter(Protocol):
    async def wait(self, scan_id: UUID, within: float) -> bool:
        """True if the worker announced the scan's result within ``within`` seconds."""
        ...


class RedisScanWaiter:
    """The sync fast path. Subscribes after the scan is committed: the relay only picks the
    scan up on its next run, so the announcement cannot come first in practice, and a missed
    one only means the client polls as usual."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def wait(self, scan_id: UUID, within: float) -> bool:
        pubsub = self._redis.pubsub()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + within
        try:
            await pubsub.subscribe(scan_done_channel(scan_id))
            while (remaining := deadline - loop.time()) > 0:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=remaining
                )
                if message is not None:
                    return True
            return False
        except RedisError as exc:
            log.warning("scan.fast_path_unavailable", error=type(exc).__name__)
            return False
        finally:
            await pubsub.reset()


@dataclass(frozen=True, slots=True)
class Submitter:
    """A signed-in user, an organization's API key, or a guest identified by device."""

    actor: Actor | None
    device_hash: str | None

    def __post_init__(self) -> None:
        if self.actor is None and self.device_hash is None:
            raise ValidationFailed(
                "Guests must send an X-Device-Id header to submit and read scans.",
                errors=[{"loc": ["header", "X-Device-Id"], "msg": "required", "type": "missing"}],
            )

    @property
    def scope(self) -> str:
        """Who owns idempotency keys and duplicates."""
        if self.actor is not None:
            principal = self.actor.principal
            if principal.api_key_id is not None:
                return f"apikey:{principal.api_key_id}"
            return f"user:{principal.user_id}"
        return f"device:{self.device_hash}"

    def owns(self) -> ColumnElement[bool]:
        if self.actor is not None:
            principal = self.actor.principal
            if principal.api_key_id is not None and principal.organization_id is not None:
                return (Scan.organization_id == principal.organization_id) & Scan.user_id.is_(None)
            return Scan.user_id == principal.user_id
        return Scan.user_id.is_(None) & (Scan.device_fingerprint_hash == self.device_hash)


@dataclass(frozen=True, slots=True)
class ScanSubmission:
    image: ValidatedImage
    latitude: float | None = None
    longitude: float | None = None
    state: NigerianState | None = None
    claimed_nafdac_number: str | None = None
    idempotency_key: str | None = None

    def fingerprint(self) -> str:
        return request_fingerprint(
            self.image.sha256,
            None if self.latitude is None else repr(self.latitude),
            None if self.longitude is None else repr(self.longitude),
            self.state.value if self.state else None,
            self.claimed_nafdac_number,
        )


@dataclass(frozen=True, slots=True)
class Submitted:
    status_code: int
    body: dict[str, Any]
    finished: bool = False  # the fast path saw the result: the caller may inline it

    @property
    def scan_id(self) -> UUID:
        return UUID(self.body["id"])


def accepted_body(
    scan_id: UUID, status: ScanStatus, *, duplicate: bool, wait: int = BASE_WAIT_SECONDS
) -> dict[str, Any]:
    pending = status in {ScanStatus.RECEIVED, ScanStatus.PROCESSING}
    return {
        "id": str(scan_id),
        "status": status.value,
        "status_url": f"/api/v1/scans/{scan_id}",
        "estimated_wait_seconds": wait if pending else 0,
        "duplicate": duplicate,
    }


class ScanService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        storage: StoragePort,
        settings: VerificationSettings,
        *,
        queue: QueueSettings | None = None,
        queue_depth: QueueDepth | None = None,
        waiter: ScanWaiter | None = None,
    ) -> None:
        self._sessions = sessions
        self._storage = storage
        self._settings = settings
        self._queue = queue or QueueSettings()
        self._queue_depth = queue_depth
        self._waiter = waiter
        self._backpressure = Backpressure(
            soft_limit=self._queue.scans_soft_limit,
            hard_limit=self._queue.scans_hard_limit,
            seconds_per_queued_scan=self._queue.seconds_per_queued_scan,
        )

    async def submit(self, submitter: Submitter, submission: ScanSubmission) -> Submitted:
        actor = submitter.actor
        if actor is not None and not actor.can(Permission.SCAN_CREATE):
            raise Forbidden("You cannot submit scans.")
        scope, key = submitter.scope, submission.idempotency_key
        now = datetime.now(UTC)
        depth = await self._queue_depth.depth() if self._queue_depth else None
        wait = self._backpressure.estimated_wait(depth)  # 503 when the queue is too long
        async with UnitOfWork(self._sessions) as uow:
            session = uow.session
            if key is not None:
                replay = await claim(
                    session, scope=scope, key=key, request_hash=submission.fingerprint()
                )
                if replay is not None:
                    return Submitted(replay.status, replay.body)

            result = await self._duplicate(session, submitter, submission, now)
            if result is None:
                result = await self._create(session, submitter, submission, wait)
            if key is not None:
                await store(
                    session, scope=scope, key=key, status=result.status_code, body=result.body
                )
            await uow.commit()
        return await self._fast_path(result, depth)

    async def _fast_path(self, result: Submitted, depth: int | None) -> Submitted:
        timeout = self._queue.sync_wait_seconds
        quiet = depth is not None and depth <= self._queue.scans_soft_limit
        if result.status_code != 202 or self._waiter is None or not timeout or not quiet:
            return result
        finished = await self._waiter.wait(result.scan_id, timeout)
        return replace(result, finished=finished)

    async def _duplicate(
        self, session: AsyncSession, submitter: Submitter, submission: ScanSubmission, now: datetime
    ) -> Submitted | None:
        hours = self._settings.duplicate_window_hours
        if not hours:
            return None
        earlier = await ScanRepository(session).recent_duplicate(
            sha256=submission.image.sha256,
            owner=submitter.owns(),
            since=now - timedelta(hours=hours),
        )
        if earlier is None:
            return None
        return Submitted(200, accepted_body(earlier.id, earlier.status, duplicate=True))

    async def _create(
        self, session: AsyncSession, submitter: Submitter, submission: ScanSubmission, wait: int
    ) -> Submitted:
        scan_id = uuid7()
        created_at = uuid7_time(scan_id)  # same instant, so lookups by id prune partitions
        key = f"scans/{created_at:%Y/%m}/{scan_id}.jpg"
        await self._storage.put(key, submission.image.jpeg, content_type="image/jpeg")
        actor = submitter.actor
        principal = actor.principal if actor else None
        location = (
            coarsen(submission.latitude, submission.longitude)
            if submission.latitude is not None and submission.longitude is not None
            else None
        )
        scan = Scan(
            id=scan_id,
            created_at=created_at,
            user_id=principal.user_id if principal else None,
            organization_id=actor.organization.id if actor and actor.organization else None,
            device_fingerprint_hash=submitter.device_hash,
            image_s3_key=key,
            image_sha256=submission.image.sha256,
            quality={"width": submission.image.width, "height": submission.image.height},
            extracted=(
                {"claimed_nafdac_number": submission.claimed_nafdac_number}
                if submission.claimed_nafdac_number
                else None
            ),
            location_coarse=location.wkt if location else None,
            geocell=location.geocell if location else None,
            state=submission.state,
            idempotency_key=submission.idempotency_key,
        )
        session.add(scan)
        await outbox.emit(
            session,
            outbox.SCAN_RECEIVED,
            aggregate_type="scan",
            aggregate_id=scan_id,
            payload={"created_at": created_at.isoformat()},
        )
        await session.flush()
        return Submitted(
            202, accepted_body(scan_id, ScanStatus.RECEIVED, duplicate=False, wait=wait)
        )

    async def get(
        self, actor: Actor | None, device_hash: str | None, scan_id: UUID
    ) -> tuple[Scan, Product | None, bool]:
        """The scan, its product, and whether the caller may see the evidence."""
        async with UnitOfWork(self._sessions) as uow:
            found = await ScanRepository(uow.session).get(scan_id, _visibility(actor, device_hash))
        if found is None:
            raise NotFound(_NOT_FOUND)
        scan, product = found
        return scan, product, actor is not None and policies.can_view_scan_signals(actor)

    async def list_page(
        self, actor: Actor, params: PageParams, *filters: ColumnElement[bool]
    ) -> list[tuple[Scan, Product | None]]:
        async with UnitOfWork(self._sessions) as uow:
            return await ScanRepository(uow.session).page(
                policies.scan_visibility(actor), params, *filters
            )


def _visibility(actor: Actor | None, device_hash: str | None) -> ColumnElement[bool]:
    """The actor's scans by policy, plus (for anyone) guest scans made on this device."""
    conditions: list[ColumnElement[bool]] = []
    if actor is not None:
        conditions.append(policies.scan_visibility(actor))
    if device_hash is not None:
        conditions.append(Scan.user_id.is_(None) & (Scan.device_fingerprint_hash == device_hash))
    return or_(*conditions) if conditions else false()


async def detach_user(session: AsyncSession, user_id: UUID) -> None:
    """NDPA erasure: scans stay (anonymised) for analytics, but no longer point at the user."""
    await session.execute(update(Scan).where(Scan.user_id == user_id).values(user_id=None))


@dataclass(frozen=True, slots=True)
class Override:
    verdict: Verdict
    case_id: UUID
    lab_result: dict[str, Any]
    note: str


async def override_verdict(
    session: AsyncSession, scan: Scan, override: Override, *, by: UUID | None
) -> Verdict | None:
    """A regulator's case decision sets the final verdict; the pipeline's verdict never
    changes. Returns the final verdict before. The caller checks permission and audits."""
    before = scan.final_verdict
    scan.final_verdict = override.verdict
    scan.overridden_by = by
    scan.overridden_at = datetime.now(UTC)
    scan.override_case_id = override.case_id
    scan.override_lab_result = override.lab_result
    scan.override_note = override.note
    await session.flush()
    return before


async def revert_override(session: AsyncSession, scan: Scan) -> Verdict | None:
    """Back to the pipeline's verdict. Returns the overridden verdict; the audit log keeps the
    history."""
    before = scan.final_verdict
    scan.final_verdict = scan.pipeline_verdict
    scan.overridden_by = None
    scan.overridden_at = None
    scan.override_case_id = None
    scan.override_lab_result = None
    scan.override_note = None
    await session.flush()
    return before

"""The scan flow end to end: submit (API), process (worker, fake readers), read the verdict."""

from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, text

from medlens.core.config import QueueSettings
from medlens.db.models import OutboxEvent
from medlens.db.session import UnitOfWork
from medlens.modules.organizations.models import OrganizationType
from medlens.modules.rbac.models import RoleName
from medlens.modules.scans.models import Scan, ScanStatus
from medlens.modules.scans.router import get_scan_service
from medlens.modules.scans.service import ScanService
from medlens.ports.image_analysis import ImageAssessment
from medlens.queue.outbox import SCAN_COMPLETED, SCAN_RECEIVED
from tests.integration.api_support import Api, photo

pytestmark = pytest.mark.integration

LAGOS = {"latitude": "6.524379", "longitude": "3.379206", "state": "NG-LA"}


def device() -> dict[str, str]:
    return {"X-Device-Id": f"device-{uuid4().hex}"}


async def submit(
    api: Api,
    image: bytes | None = None,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    response = await api.client.post(
        "/api/v1/scans",
        files={"image": ("pack.jpg", image or photo(), "image/jpeg")},
        data=data or {},
        headers=headers or {},
    )
    return response.status_code, response.json()


async def stored_scan(api: Api, scan_id: object) -> Scan:
    async with UnitOfWork(api.resources.session_factory) as uow:
        scan = await uow.session.scalar(select(Scan).where(Scan.id == UUID(str(scan_id))))
    assert scan is not None
    return scan


# ------------------------------------------------------------------------- submission
async def test_guests_need_a_device_id(api: Api) -> None:
    status, body = await submit(api)

    assert status == 422
    assert body["errors"][0]["loc"] == ["header", "X-Device-Id"]


async def test_a_guest_scan_is_accepted_and_queued(api: Api) -> None:
    status, body = await submit(api, headers=device(), data=LAGOS)

    scan = await stored_scan(api, body["id"])
    async with UnitOfWork(api.resources.session_factory) as uow:
        events = list(
            await uow.session.scalars(
                select(OutboxEvent.event_type).where(OutboxEvent.aggregate_id == str(scan.id))
            )
        )
        raw = await uow.session.scalar(
            text("SELECT ST_AsText(location_coarse) FROM scans WHERE id = :id"), {"id": scan.id}
        )
    assert status == 202
    assert body["status"] == "received"
    assert body["status_url"] == f"/api/v1/scans/{body['id']}"
    assert scan.status is ScanStatus.RECEIVED
    assert scan.image_s3_key in api.storage.objects
    assert events == [SCAN_RECEIVED]
    assert scan.geocell is not None
    assert len(scan.geocell) == 6
    assert "6.524379" not in raw  # never the raw coordinates
    assert "3.379206" not in raw


async def test_the_same_idempotency_key_replays_and_a_different_body_conflicts(api: Api) -> None:
    headers = device() | {"Idempotency-Key": f"key-{uuid4().hex}"}
    image = photo()

    first = await submit(api, image, headers=headers)
    replay = await submit(api, image, headers=headers)
    different = await submit(api, photo(), headers=headers)

    assert first[0] == replay[0] == 202
    assert replay[1] == first[1]
    assert different[0] == 409
    assert different[1]["type"] == "urn:medlens:problem:idempotency-key-reused"


async def test_the_same_photo_returns_the_earlier_scan(api: Api) -> None:
    headers = device()
    image = photo()

    first = await submit(api, image, headers=headers)
    again = await submit(api, image, headers=headers)
    other_device = await submit(api, image, headers=device())

    assert again[0] == 200
    assert again[1]["id"] == first[1]["id"]
    assert again[1]["duplicate"] is True
    assert other_device[0] == 202  # another person's scan of the same photo is their own


async def test_coordinates_come_in_pairs(api: Api) -> None:
    status, _ = await submit(api, headers=device(), data={"latitude": "6.5"})

    assert status == 422


# ------------------------------------------------------------------------- results
async def test_a_genuine_pack_end_to_end(api: Api) -> None:
    product = await api.create_product()
    headers = device()
    _, body = await submit(api, headers=headers, data=LAGOS)
    worker = api.worker(text=f"{product.brand_name} NAFDAC REG NO: {product.nafdac_number}")

    processed = worker.process_scan(UUID(str(body["id"])))
    again = worker.process_scan(UUID(str(body["id"])))
    response = await api.client.get(f"/api/v1/scans/{body['id']}", headers=headers)

    result = response.json()
    assert processed.outcome == "completed"
    assert again.outcome == "skipped"  # at-least-once delivery is harmless
    assert response.status_code == 200
    assert result["status"] == "completed"
    assert result["verdict"] == "LIKELY_GENUINE"
    assert result["product"]["nafdac_number"] == product.nafdac_number
    assert result["advice_codes"] == ["CHECK_EXPIRY_DATE"]
    assert result["analysis"] is None  # consumers never see the evidence
    async with UnitOfWork(api.resources.session_factory) as uow:
        completed = await uow.session.scalar(
            select(func.count())
            .select_from(OutboxEvent)
            .where(
                OutboxEvent.aggregate_id == str(body["id"]),
                OutboxEvent.event_type == SCAN_COMPLETED,
            )
        )
    assert completed == 1


async def test_regulators_see_the_evidence(api: Api) -> None:
    _, body = await submit(api, headers=device(), data=LAGOS)
    api.worker(text="Fakeamol NAFDAC NO: Z9-000001").process_scan(UUID(str(body["id"])))
    regulator, organization = await api.user_with_role(RoleName.REGULATOR_ANALYST)

    response = await api.client.get(
        f"/api/v1/scans/{body['id']}",
        headers=api.bearer(api.token(regulator, organization=organization)),
    )

    result = response.json()
    assert result["verdict"] == "UNREGISTERED"
    assert result["reason_codes"] == ["NOT_IN_REGISTRY"]
    assert result["analysis"]["scorer_version"] == "weighted-rules-v2"
    assert result["analysis"]["extracted"]["nafdac_number"]["value"] == "Z9-000001"
    assert result["analysis"]["state"] == "NG-LA"


async def test_a_bad_photo_is_rejected_with_retake_advice(api: Api) -> None:
    headers = device()
    _, body = await submit(api, headers=headers)
    blurry = ImageAssessment(width=1200, height=1600, blur=5.0, brightness=128.0, phash=7)

    api.worker(assessment=blurry).process_scan(UUID(str(body["id"])))
    result = (await api.client.get(f"/api/v1/scans/{body['id']}", headers=headers)).json()

    assert result["status"] == "rejected"
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["reason_codes"] == ["BLURRY"]
    assert result["advice_codes"][0] == "RETAKE_PHOTO"


async def test_a_near_identical_photo_gets_the_earlier_answer(api: Api) -> None:
    headers = device()
    first = (await submit(api, headers=headers))[1]
    second = (await submit(api, headers=headers))[1]
    same_photo = ImageAssessment(width=1200, height=1600, blur=250.0, brightness=128.0, phash=42)

    api.worker(assessment=same_photo, text="NAFDAC NO: Z9-000002").process_scan(
        UUID(str(first["id"]))
    )
    outcome = api.worker(assessment=same_photo, text="nothing readable").process_scan(
        UUID(str(second["id"]))
    )

    scan = await stored_scan(api, second["id"])
    assert outcome.outcome == "duplicate"
    assert scan.verdict is not None
    assert scan.verdict.value == "UNREGISTERED"
    assert scan.signals is not None
    assert scan.signals["duplicate_of"] == first["id"]


async def test_a_missing_image_fails_the_scan(api: Api) -> None:
    _, body = await submit(api, headers=device())
    scan = await stored_scan(api, body["id"])
    api.storage.objects.clear()

    outcome = api.worker().process_scan(scan.id)

    assert outcome.outcome == "failed"
    assert (await stored_scan(api, body["id"])).error_code == "image_missing"


# ------------------------------------------------------------------------- reading and lists
async def test_other_devices_and_strangers_cannot_read_a_scan(api: Api) -> None:
    _, body = await submit(api, headers=device())
    stranger, _ = await api.user_with_role(RoleName.CONSUMER)

    other_device = await api.client.get(f"/api/v1/scans/{body['id']}", headers=device())
    anonymous = await api.client.get(f"/api/v1/scans/{body['id']}")
    signed_in = await api.client.get(
        f"/api/v1/scans/{body['id']}", headers=api.bearer(api.token(stranger))
    )

    assert other_device.status_code == 404
    assert anonymous.status_code == 401
    assert signed_in.status_code == 404


async def test_lists_are_scoped_to_the_caller(api: Api) -> None:
    consumer = await api.create_user()
    staff, pharmacy = await api.user_with_role(RoleName.PHARMACY_STAFF)
    colleague = await api.create_user()
    assert pharmacy is not None
    await api.add_member(colleague, pharmacy, RoleName.PHARMACY_STAFF)
    national, national_org = await api.user_with_role(RoleName.REGULATOR_ANALYST)
    kano = await api.create_org(OrganizationType.REGULATOR, jurisdiction=["NG-KN"])
    kano_analyst = await api.create_user()
    await api.add_member(kano_analyst, kano, RoleName.REGULATOR_ANALYST)

    mine = (await submit(api, headers=api.bearer(api.token(consumer)), data=LAGOS))[1]["id"]
    org_scan = (
        await submit(
            api, headers=api.bearer(api.token(colleague, organization=pharmacy)), data=LAGOS
        )
    )[1]["id"]

    async def listed(user, organization=None) -> set[str]:  # type: ignore[no-untyped-def]
        response = await api.client.get(
            "/api/v1/scans",
            headers=api.bearer(api.token(user, organization=organization)),
            params={"limit": 100, "state": "NG-LA"},
        )
        assert response.status_code == 200, response.text
        return {item["id"] for item in response.json()["items"]}

    assert mine in await listed(consumer)
    assert org_scan not in await listed(consumer)
    assert org_scan in await listed(staff, pharmacy)
    assert mine not in await listed(staff, pharmacy)
    assert {mine, org_scan} <= await listed(national, national_org)
    assert not {mine, org_scan} & await listed(kano_analyst, kano)


# ------------------------------------------------------------------------- queue awareness
class FixedDepth:
    def __init__(self, depth: int | None) -> None:
        self.value = depth

    async def depth(self) -> int | None:
        return self.value


class ProcessingWaiter:
    """Stands in for the worker announcing the result: processes the scan, then says done."""

    def __init__(self, api: Api, text: str) -> None:
        self.api, self.text = api, text

    async def wait(self, scan_id: UUID, within: float) -> bool:
        self.api.worker(text=self.text).process_scan(scan_id)
        return True


def use_queue(
    api: Api, depth: int | None, *, sync_wait: float = 0.0, waiter: object = None
) -> None:

    settings = api.app.state.settings
    api.app.dependency_overrides[get_scan_service] = lambda: ScanService(
        api.resources.session_factory,
        api.storage,
        settings.verification,
        queue=QueueSettings(sync_wait_seconds=sync_wait),
        queue_depth=FixedDepth(depth),  # type: ignore[arg-type]
        waiter=waiter,  # type: ignore[arg-type]
    )


async def test_the_fast_path_returns_the_verdict_in_one_call(api: Api) -> None:
    product = await api.create_product()
    use_queue(
        api,
        depth=0,
        sync_wait=2.0,
        waiter=ProcessingWaiter(api, f"{product.brand_name} NAFDAC NO: {product.nafdac_number}"),
    )

    status, body = await submit(api, headers=device())

    assert status == 200
    assert body["status"] == "completed"
    assert body["estimated_wait_seconds"] == 0
    assert body["result"]["verdict"] == "LIKELY_GENUINE"


async def test_a_busy_queue_skips_the_fast_path_and_grows_the_wait(api: Api) -> None:
    use_queue(api, depth=600, sync_wait=2.0, waiter=ProcessingWaiter(api, ""))

    status, body = await submit(api, headers=device())

    assert status == 202
    assert body["result"] is None
    assert body["estimated_wait_seconds"] == 10 + 300


async def test_an_overloaded_queue_refuses_new_scans(api: Api) -> None:
    use_queue(api, depth=50_000)
    image = photo()

    response = await api.client.post(
        "/api/v1/scans",
        files={"image": ("pack.jpg", image, "image/jpeg")},
        headers=device(),
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "300"
    assert api.storage.objects == {}  # refused before anything was stored

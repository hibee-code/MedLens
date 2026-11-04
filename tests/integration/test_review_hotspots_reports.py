"""M4: verdict overrides through case decisions, the evidence photo, hotspot clustering with its
cache, the scan summary and the manufacturer report."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, select

from medlens.core.identifiers import NigerianState
from medlens.core.ids import uuid7, uuid7_time
from medlens.modules.audit.models import AuditLog
from medlens.modules.hotspots.models import HotspotSnapshot
from medlens.modules.hotspots.service import record_snapshots, window_end
from medlens.modules.organizations.models import Organization, OrganizationType
from medlens.modules.products.models import Product
from medlens.modules.rbac.models import RoleName
from medlens.modules.scans.models import Scan, ScanStatus, Verdict
from medlens.modules.users.models import User
from tests.integration.api_support import Api
from tests.integration.test_cases import open_case, unregistered_scan

pytestmark = pytest.mark.integration

# Borno and Adamawa are used by no other test, so their clusters are this module's alone.
MAIDUGURI = [(11.83, 13.15), (11.84, 13.15), (11.83, 13.16), (11.84, 13.16), (11.82, 13.15)]
YOLA = [(9.20, 12.48), (9.21, 12.48), (9.20, 12.49)]
MUBI = (10.27, 13.27)  # alone: noise, not a cluster


async def regulator(api: Api, states: list[str]) -> tuple[User, Organization]:
    organization = await api.create_org(OrganizationType.REGULATOR, jurisdiction=states)
    user = await api.create_user()
    await api.add_member(user, organization, RoleName.REGULATOR_ADMIN)
    return user, organization


def auth(api: Api, user: User, organization: Organization | None = None) -> dict[str, str]:
    return api.bearer(api.token(user, organization=organization))


def flagged_scans(
    api: Api,
    points: list[tuple[float, float]],
    state: NigerianState,
    *,
    product: Product | None = None,
    verdict: Verdict = Verdict.SUSPICIOUS,
    llm_used: bool = False,
) -> list[UUID]:
    """Completed scans written straight to the database at the given (coarse) locations."""
    ids = []
    with api.sync_sessions() as session, session.begin():
        for latitude, longitude in points:
            scan_id = uuid7()
            session.add(
                Scan(
                    id=scan_id,
                    created_at=uuid7_time(scan_id),
                    status=ScanStatus.COMPLETED,
                    verdict=verdict,
                    risk_score=0.7,
                    confidence=0.8,
                    state=state,
                    location_coarse=f"SRID=4326;POINT({longitude} {latitude})",
                    matched_product_id=product.id if product else None,
                    llm_used=llm_used,
                    scorer_version="weighted-rules-v2",
                )
            )
            ids.append(scan_id)
    return ids


# ------------------------------------------------------------------------- verdict override
async def test_a_lab_result_can_override_the_verdict_and_is_audited(api: Api) -> None:
    admin, organization = await regulator(api, ["NG-EB"])
    scan_id = await unregistered_scan(api, "NG-EB")
    case = open_case(api, scan_id)
    assert case is not None
    headers = auth(api, admin, organization)
    url = f"/api/v1/cases/{case.id}"

    not_allowed = await api.client.post(
        f"{url}/lab-result",
        json={"outcome": "counterfeit", "override_verdict": "LIKELY_GENUINE"},
        headers=headers,
    )
    decided = await api.client.post(
        f"{url}/lab-result",
        json={"outcome": "counterfeit", "override_verdict": "SUSPICIOUS"},
        headers=headers,
    )
    scan = (await api.client.get(f"/api/v1/scans/{scan_id}", headers=headers)).json()

    assert not_allowed.status_code == 422
    assert decided.status_code == 200, decided.text
    assert decided.json()["scan"]["verdict"] == "UNREGISTERED"
    assert decided.json()["scan"]["verdict_override"] == "SUSPICIOUS"
    assert scan["verdict"] == "SUSPICIOUS"
    assert scan["verdict_overridden"] is True
    assert scan["analysis"]["pipeline_verdict"] == "UNREGISTERED"
    assert scan["analysis"]["override_case_id"] == str(case.id)
    with api.sync_sessions() as session:
        entry = session.scalar(
            select(AuditLog).where(
                AuditLog.action == "scan.verdict_overridden", AuditLog.resource_id == str(scan_id)
            )
        )
    assert entry is not None
    assert entry.metadata_["from"] == "UNREGISTERED"
    assert entry.metadata_["to"] == "SUSPICIOUS"


async def test_inconclusive_results_cannot_override(api: Api) -> None:
    admin, organization = await regulator(api, ["NG-EB"])
    case = open_case(api, await unregistered_scan(api, "NG-EB"))
    assert case is not None

    response = await api.client.post(
        f"/api/v1/cases/{case.id}/lab-result",
        json={"outcome": "inconclusive", "override_verdict": "SUSPICIOUS"},
        headers=auth(api, admin, organization),
    )

    assert response.status_code == 422


async def test_reviewers_download_the_photo_and_the_download_is_audited(api: Api) -> None:
    admin, organization = await regulator(api, ["NG-EK"])
    outsider, other = await regulator(api, ["NG-KO"])
    scan_id = await unregistered_scan(api, "NG-EK")
    case = open_case(api, scan_id)
    assert case is not None

    image = await api.client.get(
        f"/api/v1/cases/{case.id}/image", headers=auth(api, admin, organization)
    )
    hidden = await api.client.get(
        f"/api/v1/cases/{case.id}/image", headers=auth(api, outsider, other)
    )
    detail = await api.client.get(
        f"/api/v1/cases/{case.id}", headers=auth(api, admin, organization)
    )
    with api.sync_sessions() as session, session.begin():
        stored = session.scalar(select(Scan).where(Scan.id == scan_id))
        assert stored is not None
        stored.image_s3_key = None  # past the retention period
    gone = await api.client.get(
        f"/api/v1/cases/{case.id}/image", headers=auth(api, admin, organization)
    )

    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert image.headers["cache-control"] == "no-store"
    assert image.content[:2] == b"\xff\xd8"
    assert hidden.status_code == 404
    assert detail.json()["scan"]["has_image"] is True
    assert detail.json()["scan"]["signals"] is not None
    assert gone.status_code == 404
    assert "case.image_viewed" in await api.audit_actions(admin.id)


# ------------------------------------------------------------------------- hotspots
async def test_hotspots_cluster_flagged_scans_and_are_cached(api: Api) -> None:
    product = await api.create_product()
    flagged_scans(api, MAIDUGURI, NigerianState.BORNO, product=product)
    flagged_scans(api, YOLA, NigerianState.ADAMAWA, verdict=Verdict.UNREGISTERED)
    flagged_scans(api, [MUBI], NigerianState.ADAMAWA)
    flagged_scans(api, MAIDUGURI[:3], NigerianState.BORNO, verdict=Verdict.LIKELY_GENUINE)
    admin = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)
    headers = auth(api, admin)

    borno = await api.client.get("/api/v1/hotspots?state=NG-BO&days=1", headers=headers)
    adamawa = await api.client.get("/api/v1/hotspots?state=NG-AD&days=1", headers=headers)

    assert borno.status_code == 200, borno.text
    [cluster] = borno.json()["clusters"]
    assert cluster["scan_count"] == 5  # LIKELY_GENUINE scans are not counted
    assert cluster["severity"] == "medium"
    assert cluster["states"] == ["NG-BO"]
    assert abs(cluster["centroid"]["latitude"] - 11.834) < 0.01
    assert 500 < cluster["radius_m"] < 2500
    assert cluster["top_products"][0]["product_id"] == str(product.id)
    assert cluster["top_products"][0]["scan_count"] == 5
    [yola] = adamawa.json()["clusters"]  # Mubi is noise
    assert yola["scan_count"] == 3
    assert yola["top_products"] == []

    flagged_scans(api, MAIDUGURI[:2], NigerianState.BORNO)
    again = await api.client.get("/api/v1/hotspots?state=NG-BO&days=1", headers=headers)
    other = await api.client.get(
        "/api/v1/hotspots?state=NG-BO&days=1&eps_m=500&min_points=4", headers=headers
    )
    assert again.json() == borno.json()  # cached for 5 minutes
    assert other.json()["generated_at"] != borno.json()["generated_at"]


async def test_regulators_only_see_hotspots_in_their_jurisdiction(api: Api) -> None:
    flagged_scans(api, MAIDUGURI, NigerianState.BORNO)
    user, organization = await regulator(api, ["NG-AD"])
    consumer = await api.create_user()

    mine = await api.client.get("/api/v1/hotspots?days=1", headers=auth(api, user, organization))
    asked = await api.client.get(
        "/api/v1/hotspots?days=1&state=NG-BO", headers=auth(api, user, organization)
    )
    refused = await api.client.get("/api/v1/hotspots", headers=auth(api, consumer))

    assert all("NG-BO" not in cluster["states"] for cluster in mine.json()["clusters"])
    assert asked.json()["clusters"] == []
    assert refused.status_code == 403


async def test_the_scheduled_recompute_keeps_snapshots(api: Api) -> None:
    flagged_scans(api, MAIDUGURI, NigerianState.BORNO)
    now = datetime.now(UTC)

    with api.sync_sessions() as session, session.begin():
        count = record_snapshots(session, now=now)
        stored = session.scalars(
            select(HotspotSnapshot).where(HotspotSnapshot.window_end == window_end(now))
        ).all()
        session.execute(
            delete(HotspotSnapshot).where(HotspotSnapshot.id.in_([s.id for s in stored]))
        )

    assert count == len(stored) >= 1
    assert any("NG-BO" in snapshot.states for snapshot in stored)
    assert all(snapshot.algorithm_params["srid"] == 32632 for snapshot in stored)


# ------------------------------------------------------------------------- reports
async def test_the_summary_counts_verdicts_states_and_llm_use(api: Api) -> None:
    flagged_scans(api, YOLA, NigerianState.ADAMAWA, llm_used=True)
    admin = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)
    user, organization = await regulator(api, ["NG-BO"])

    everything = await api.client.get("/api/v1/stats/summary?days=1", headers=auth(api, admin))
    scoped = await api.client.get(
        "/api/v1/stats/summary?days=1", headers=auth(api, user, organization)
    )

    body = everything.json()
    assert everything.status_code == 200, everything.text
    assert body["total_scans"] == sum(body["by_verdict"].values())
    adamawa = next(row for row in body["by_state"] if row["state"] == "NG-AD")
    assert adamawa["flagged"] >= 3
    assert body["llm"]["scans_needing_llm"] >= 3
    assert 0 < body["llm"]["share_of_scans"] <= 1
    assert sum(day["total"] for day in body["by_day"]) == body["total_scans"]
    assert {row["state"] for row in scoped.json()["by_state"]} <= {"NG-BO"}


async def test_manufacturers_see_only_aggregates_of_their_own_products(api: Api) -> None:
    manufacturer = await api.create_org(OrganizationType.MANUFACTURER)
    staff = await api.create_user()
    await api.add_member(staff, manufacturer, RoleName.MANUFACTURER_STAFF)
    own = await api.create_product(manufacturer=manufacturer)
    competitor = await api.create_product(
        manufacturer=await api.create_org(OrganizationType.MANUFACTURER)
    )
    flagged_scans(api, MAIDUGURI + MAIDUGURI[:1], NigerianState.BORNO, product=own)  # 6
    flagged_scans(api, YOLA[:2], NigerianState.ADAMAWA, product=own)  # 2: suppressed
    flagged_scans(api, MAIDUGURI, NigerianState.BORNO, product=competitor)
    flagged_scans(api, MAIDUGURI, NigerianState.BORNO, product=own, verdict=Verdict.LIKELY_GENUINE)
    analyst, regulator_org = await regulator(api, ["NG-BO"])
    admin = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)

    report = await api.client.get(
        "/api/v1/manufacturer/scans", headers=auth(api, staff, manufacturer)
    )
    someone_else = await api.client.get(
        f"/api/v1/manufacturer/scans?organization_id={uuid4()}",
        headers=auth(api, staff, manufacturer),
    )
    as_regulator = await api.client.get(
        "/api/v1/manufacturer/scans", headers=auth(api, analyst, regulator_org)
    )
    admin_without_org = await api.client.get("/api/v1/manufacturer/scans", headers=auth(api, admin))
    admin_with_org = await api.client.get(
        f"/api/v1/manufacturer/scans?organization_id={manufacturer.id}", headers=auth(api, admin)
    )

    body = report.json()
    assert report.status_code == 200, report.text
    [cell] = body["cells"]
    assert cell["product_id"] == str(own.id)
    assert (cell["state"], cell["flagged_scans"]) == ("NG-BO", 6)
    assert body["suppressed_cells"] == 1
    assert "user_id" not in str(body)
    assert "geocell" not in str(body)
    assert someone_else.status_code == 403
    assert as_regulator.status_code == 403  # regulators hold no reports:read_own
    assert admin_without_org.status_code == 422
    assert admin_with_org.json()["cells"] == body["cells"]

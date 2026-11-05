"""Cases: opened for high-risk scans, routed by jurisdiction, reviewed by regulators."""

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from medlens.core.identifiers import NigerianState
from medlens.modules.cases.models import Case
from medlens.modules.cases.service import open_case_for_scan
from medlens.modules.hitl.models import Label
from medlens.modules.organizations.models import Organization, OrganizationType
from medlens.modules.rbac.models import RoleName
from medlens.modules.users.models import User
from tests.integration.api_support import Api, photo

pytestmark = pytest.mark.integration


async def unregistered_scan(api: Api, state: str | None) -> UUID:
    """A processed guest scan whose NAFDAC number is not registered (always opens a case)."""
    data = {"state": state} if state else {}
    response = await api.client.post(
        "/api/v1/scans",
        files={"image": ("pack.jpg", photo(), "image/jpeg")},
        data=data,
        headers={"X-Device-Id": f"device-{uuid4().hex}"},
    )
    scan_id = UUID(response.json()["id"])
    api.worker(text="NAFDAC REG NO: Z9-999999").process_scan(scan_id)
    return scan_id


def open_case(api: Api, scan_id: UUID) -> Case | None:
    with api.sync_sessions() as session, session.begin():
        case = open_case_for_scan(session, scan_id)
        if case is not None:
            session.expunge(case)
        return case


async def regulator(
    api: Api, states: list[str], role: RoleName = RoleName.REGULATOR_ADMIN
) -> tuple[User, Organization]:
    organization = await api.create_org(OrganizationType.REGULATOR, jurisdiction=states)
    user = await api.create_user()
    await api.add_member(user, organization, role)
    return user, organization


def auth(api: Api, user: User, organization: Organization) -> dict[str, str]:
    return api.bearer(api.token(user, organization=organization))


# ------------------------------------------------------------------------- opening and routing
async def test_a_high_risk_scan_opens_one_case_for_the_most_local_regulator(api: Api) -> None:
    # Zamfara is used by no other test: the one-state agency must win over the two-state one
    # and over national regulators.
    await regulator(api, ["NG-ZA", "NG-SO"])
    _, zamfara = await regulator(api, ["NG-ZA"])
    scan_id = await unregistered_scan(api, "NG-ZA")

    first = open_case(api, scan_id)
    second = open_case(api, scan_id)

    assert first is not None
    assert second is not None
    assert first.id == second.id  # idempotent
    assert first.priority.value == "high"
    assert first.state is NigerianState.ZAMFARA
    assert first.assigned_org_id == zamfara.id


async def test_scans_without_a_state_go_to_a_national_regulator(api: Api) -> None:
    await regulator(api, [state.value for state in NigerianState])
    scan_id = await unregistered_scan(api, None)

    case = open_case(api, scan_id)

    assert case is not None
    with api.sync_sessions() as session:
        organization = session.get(Organization, case.assigned_org_id)
    assert organization is not None
    assert len(organization.jurisdiction_states or []) == len(NigerianState)


async def test_low_risk_scans_open_no_case(api: Api) -> None:
    product = await api.create_product()
    response = await api.client.post(
        "/api/v1/scans",
        files={"image": ("pack.jpg", photo(), "image/jpeg")},
        headers={"X-Device-Id": f"device-{uuid4().hex}"},
    )
    scan_id = UUID(response.json()["id"])
    api.worker(text=f"{product.brand_name} NAFDAC NO: {product.nafdac_number}").process_scan(
        scan_id
    )

    assert open_case(api, scan_id) is None


# ------------------------------------------------------------------------- reviewing
async def test_cases_are_scoped_by_jurisdiction(api: Api) -> None:
    oyo_user, oyo = await regulator(api, ["NG-OY"])
    ekiti_user, ekiti = await regulator(api, ["NG-EK"])
    case = open_case(api, await unregistered_scan(api, "NG-OY"))
    assert case is not None

    oyo_view = await api.client.get(f"/api/v1/cases/{case.id}", headers=auth(api, oyo_user, oyo))
    ekiti_view = await api.client.get(
        f"/api/v1/cases/{case.id}", headers=auth(api, ekiti_user, ekiti)
    )
    ekiti_list = await api.client.get(
        "/api/v1/cases", params={"limit": 100}, headers=auth(api, ekiti_user, ekiti)
    )
    consumer = await api.client.get(
        "/api/v1/cases", headers=api.bearer(api.token(await api.create_user()))
    )

    assert oyo_view.status_code == 200
    assert oyo_view.json()["scan"]["final_verdict"] == "UNREGISTERED"
    assert ekiti_view.status_code == 404
    assert str(case.id) not in {item["id"] for item in ekiti_list.json()["items"]}
    assert consumer.status_code == 403


async def test_review_workflow_ends_with_a_training_label(api: Api) -> None:
    admin, organization = await regulator(api, ["NG-AB"])
    analyst = await api.create_user()
    await api.add_member(analyst, organization, RoleName.REGULATOR_ANALYST)
    outsider = await api.create_user()
    case = open_case(api, await unregistered_scan(api, "NG-AB"))
    assert case is not None
    url = f"/api/v1/cases/{case.id}"

    assigned = await api.client.post(
        f"{url}/assign", json={"user_id": str(analyst.id)}, headers=auth(api, admin, organization)
    )
    bad_assignee = await api.client.post(
        f"{url}/assign", json={"user_id": str(outsider.id)}, headers=auth(api, admin, organization)
    )
    by_analyst = await api.client.post(
        f"{url}/assign", json={"user_id": str(analyst.id)}, headers=auth(api, analyst, organization)
    )
    claimed = await api.client.post(f"{url}/claim", headers=auth(api, analyst, organization))
    noted = await api.client.patch(
        url,
        json={"notes": "<script>x</script>Seal broken"},
        headers=auth(api, analyst, organization),
    )
    result = await api.client.post(
        f"{url}/lab-result",
        json={"outcome": "counterfeit", "lab_reference": "LAB/AB/1"},
        headers=auth(api, analyst, organization),
    )
    after = await api.client.patch(
        url, json={"status": "closed"}, headers=auth(api, analyst, organization)
    )

    assert assigned.status_code == 200, assigned.text
    assert assigned.json()["assigned_to_user_id"] == str(analyst.id)
    assert bad_assignee.status_code == 422
    assert by_analyst.status_code == 403  # case:assign is for regulator admins
    assert claimed.json()["status"] == "in_review"
    assert noted.json()["notes"] == "Seal broken"
    assert result.json()["status"] == "confirmed_counterfeit"
    assert result.json()["lab_result"]["lab_reference"] == "LAB/AB/1"
    assert after.status_code == 409  # resolved cases are final
    with api.sync_sessions() as session:
        label = session.scalar(select(Label).where(Label.case_id == case.id))
    assert label is not None
    assert label.outcome.value == "counterfeit"
    assert label.labelled_by == analyst.id


async def test_closing_a_case_resolves_it(api: Api) -> None:
    admin, organization = await regulator(api, ["NG-BO"])
    case = open_case(api, await unregistered_scan(api, "NG-BO"))
    assert case is not None

    closed = await api.client.patch(
        f"/api/v1/cases/{case.id}",
        json={"status": "closed", "notes": "Duplicate"},
        headers=auth(api, admin, organization),
    )
    filtered = await api.client.get(
        "/api/v1/cases",
        params={"status": "closed", "limit": 100},
        headers=auth(api, admin, organization),
    )

    assert closed.json()["status"] == "closed"
    assert closed.json()["resolved_at"]
    assert str(case.id) in {item["id"] for item in filtered.json()["items"]}

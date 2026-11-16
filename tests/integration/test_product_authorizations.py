"""Phase 4 of the register check: batches and reference images stay tied to the registration
holder, or organisations it authorised (recorded by a platform admin); products without a holder
link keep working with a warning; inactive registrations are flagged, never purged."""

from datetime import date

import pytest
from sqlalchemy import select
from structlog.testing import capture_logs

from medlens.modules.audit.models import AuditLog
from medlens.modules.organizations.models import OrganizationType
from medlens.modules.products.models import Product, RegistrationStatus
from medlens.modules.rbac.models import RoleName
from tests.integration.api_support import Api
from tests.integration.test_products_api import auth, manufacturer_with_product, mfa

pytestmark = pytest.mark.integration
BATCH = {"expires_on": "2028-01-31", "manufactured_on": "2026-01-15"}


async def test_an_authorised_organisation_declares_batches_until_revoked(api: Api) -> None:
    _, holder_org, product = await manufacturer_with_product(api)
    contractor_admin, contractor = await api.user_with_role(RoleName.MANUFACTURER_ADMIN)
    assert contractor is not None
    admin = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)
    url = f"/api/v1/products/{product.nafdac_number}"
    as_contractor = auth(api, contractor_admin, contractor)

    before = await api.client.put(f"{url}/batches/CT-001", json=BATCH, headers=as_contractor)
    not_admin = await api.client.post(
        f"{url}/authorizations", json={"organization_id": str(contractor.id)}, headers=as_contractor
    )
    holder_itself = await api.client.post(
        f"{url}/authorizations",
        json={"organization_id": str(holder_org.id)},
        headers=mfa(api, admin),
    )
    granted = await api.client.post(
        f"{url}/authorizations",
        json={"organization_id": str(contractor.id), "note": "Holder letter 2026-11-01"},
        headers=mfa(api, admin),
    )
    twice = await api.client.post(
        f"{url}/authorizations",
        json={"organization_id": str(contractor.id)},
        headers=mfa(api, admin),
    )
    declared = await api.client.put(f"{url}/batches/CT-001", json=BATCH, headers=as_contractor)
    listed = await api.client.get(f"{url}/batches", headers=as_contractor)
    authorization_id = granted.json()["id"]
    revoked = await api.client.post(
        f"{url}/authorizations/{authorization_id}/revoke", headers=mfa(api, admin)
    )
    after = await api.client.put(f"{url}/batches/CT-002", json=BATCH, headers=as_contractor)
    history = await api.client.get(f"{url}/authorizations", headers=mfa(api, admin))
    with api.sync_sessions() as session:
        actions = set(
            session.scalars(select(AuditLog.action).where(AuditLog.resource_id == str(product.id)))
        )

    assert before.status_code == 403
    assert not_admin.status_code == 403
    assert holder_itself.status_code == 409
    assert granted.status_code == 201, granted.text
    assert granted.json()["active"] is True
    assert twice.status_code == 409
    assert declared.status_code == 201, declared.text
    assert [item["batch_number"] for item in listed.json()["items"]] == ["CT-001"]
    assert revoked.json()["active"] is False
    assert after.status_code == 403
    assert [item["active"] for item in history.json()] == [False]
    assert {"product.authorization_granted", "product.authorization_revoked"} <= actions


async def test_authorisations_need_a_linked_holder_and_a_verified_organisation(api: Api) -> None:
    unlinked = await api.create_product()
    _, _, linked = await manufacturer_with_product(api)
    pending = await api.create_org(OrganizationType.MANUFACTURER, status="pending_verification")  # type: ignore[arg-type]
    other = await api.create_org(OrganizationType.MANUFACTURER)
    admin = mfa(api, await api.create_user(global_role=RoleName.PLATFORM_ADMIN))

    no_holder = await api.client.post(
        f"/api/v1/products/{unlinked.nafdac_number}/authorizations",
        json={"organization_id": str(other.id)},
        headers=admin,
    )
    unverified = await api.client.post(
        f"/api/v1/products/{linked.nafdac_number}/authorizations",
        json={"organization_id": str(pending.id)},
        headers=admin,
    )

    assert no_holder.status_code == 409
    assert "approve the holder's product claim" in no_holder.json()["detail"]
    assert unverified.status_code == 404


async def test_writes_to_products_without_a_holder_link_are_logged(api: Api) -> None:
    product = await api.create_product()
    admin = await api.create_user(global_role=RoleName.PLATFORM_ADMIN)

    with capture_logs() as logs:
        response = await api.client.put(
            f"/api/v1/products/{product.nafdac_number}/batches/LEGACY-1",
            json=BATCH,
            headers=mfa(api, admin),
        )

    assert response.status_code == 201  # existing data keeps working...
    assert any(
        log["event"] == "products.no_holder_link" and log["action"] == "batch_write" for log in logs
    )  # ...with a warning


async def test_inactive_registrations_keep_their_batches_but_are_flagged(api: Api) -> None:
    admin_user, organization, product = await manufacturer_with_product(api)
    headers = auth(api, admin_user, organization)
    url = f"/api/v1/products/{product.nafdac_number}"
    await api.client.put(f"{url}/batches/SUSP-1", json=BATCH, headers=headers)
    with api.sync_sessions() as session, session.begin():
        stored = session.get(Product, product.id)
        assert stored is not None
        stored.registration_status = RegistrationStatus.SUSPENDED

    batches = await api.client.get(f"{url}/batches", headers=headers)
    looked_up = await api.client.get(url)

    assert [item["registration_inactive"] for item in batches.json()["items"]] == [True]
    assert looked_up.json()["registration_inactive"] is True
    assert looked_up.json()["registration_status"] == "suspended"


async def test_the_product_lookup_shows_the_register_entry(api: Api) -> None:
    product = await api.create_product()
    with api.sync_sessions() as session, session.begin():
        stored = session.get(Product, product.id)
        assert stored is not None
        stored.active_ingredients = [
            {"name": "artemether", "strength_value": 20.0, "strength_unit": "mg"}
        ]
        stored.source_url = "https://greenbook.nafdac.gov.ng/"
        stored.retrieved_on = date(2026, 11, 10)

    body = (await api.client.get(f"/api/v1/products/{product.nafdac_number}")).json()

    assert body["active_ingredients"] == [
        {"name": "artemether", "strength_value": 20.0, "strength_unit": "mg"}
    ]
    assert body["source_url"] == "https://greenbook.nafdac.gov.ng/"
    assert body["retrieved_on"] == "2026-11-10"
    assert body["registration_inactive"] is False

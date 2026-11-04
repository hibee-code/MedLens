"""Products (public), batches (manufacturer only), claims and reference-image review."""

from typing import Any

import pytest

from medlens.modules.organizations.models import (
    Organization,
    OrganizationStatus,
    OrganizationType,
)
from medlens.modules.products.models import Product, ReferenceImage
from medlens.modules.rbac.models import RoleName
from medlens.modules.users.models import User
from tests.integration.api_support import Api, photo

pytestmark = pytest.mark.integration


async def manufacturer_with_product(api: Api) -> tuple[User, Organization, Product]:
    admin, organization = await api.user_with_role(RoleName.MANUFACTURER_ADMIN)
    assert organization is not None
    product = await api.create_product(manufacturer=organization)
    return admin, organization, product


def auth(api: Api, user: User, organization: Organization | None = None) -> dict[str, str]:
    return api.bearer(api.token(user, organization=organization))


def mfa(api: Api, user: User, organization: Organization | None = None) -> dict[str, str]:
    return api.bearer(api.token(user, organization=organization, auth_methods=("pwd", "mfa")))


# ------------------------------------------------------------------------- public lookup
async def test_products_are_public_and_found_by_number_or_name(api: Api) -> None:
    product = await api.create_product(brand="Zentramol Forte")

    by_number = await api.client.get(f"/api/v1/products/{product.nafdac_number.lower()}")
    exact = await api.client.get("/api/v1/products", params={"query": product.nafdac_number})
    fuzzy = await api.client.get("/api/v1/products", params={"query": "zentramol forte"})
    missing = await api.client.get("/api/v1/products/Z9-000000")

    assert by_number.status_code == 200
    assert by_number.json()["manufacturer_onboarded"] is False
    assert exact.json()["items"][0]["match_score"] == 1.0
    assert exact.json()["has_more"] is False
    assert product.nafdac_number in {item["nafdac_number"] for item in fuzzy.json()["items"]}
    assert missing.status_code == 404


async def test_listing_without_a_query_pages(api: Api) -> None:
    await api.create_product()
    await api.create_product()

    first = await api.client.get("/api/v1/products", params={"limit": 1})

    assert first.json()["has_more"] is True
    assert first.json()["next_cursor"]


# ------------------------------------------------------------------------- batches
BATCH = {
    "expires_on": "2027-12-31",
    "manufactured_on": "2025-01-15",
    "distribution_states": ["Lagos", "NG-OG"],
}


async def test_manufacturers_declare_batches_others_cannot(api: Api) -> None:
    admin, organization, product = await manufacturer_with_product(api)
    rival, rival_org = await api.user_with_role(RoleName.MANUFACTURER_ADMIN)
    url = f"/api/v1/products/{product.nafdac_number}/batches"

    created = await api.client.put(
        f"{url}/b2309-14", json=BATCH, headers=auth(api, admin, organization)
    )
    updated = await api.client.put(
        f"{url}/B2309-14", json=BATCH | {"quantity": 5}, headers=auth(api, admin, organization)
    )
    hijack = await api.client.put(f"{url}/X1", json=BATCH, headers=auth(api, rival, rival_org))

    assert created.status_code == 201, created.text
    assert created.json()["batch_number"] == "B2309-14"
    assert created.json()["distribution_states"] == ["NG-LA", "NG-OG"]
    assert updated.status_code == 200
    assert updated.json()["quantity"] == 5
    assert hijack.status_code == 403


async def test_batches_are_hidden_from_the_public(api: Api) -> None:
    admin, organization, product = await manufacturer_with_product(api)
    url = f"/api/v1/products/{product.nafdac_number}/batches"
    await api.client.put(f"{url}/B1", json=BATCH, headers=auth(api, admin, organization))
    consumer = await api.create_user()
    regulator, regulator_org = await api.user_with_role(RoleName.REGULATOR_ANALYST)

    as_consumer = await api.client.get(url, headers=auth(api, consumer))
    as_regulator = await api.client.get(url, headers=auth(api, regulator, regulator_org))
    as_owner = await api.client.get(url, headers=auth(api, admin, organization))

    assert as_consumer.status_code == 404
    assert [item["batch_number"] for item in as_regulator.json()["items"]] == ["B1"]
    assert as_owner.status_code == 200


async def test_csv_import_is_all_or_nothing_with_a_row_report(api: Api) -> None:
    admin, organization, product = await manufacturer_with_product(api)
    url = f"/api/v1/products/{product.nafdac_number}/batches"
    headers = auth(api, admin, organization)
    bad = (
        "batch_number,expires_on,manufactured_on,distribution_states\n"
        "A1,2027-01-31,2025-01-01,NG-LA\n"
        "A2,2024-01-01,2025-01-01,NG-LA\n"  # expires before manufacture
        "A1,2027-01-31,,\n"  # duplicate
        "A3,soon,,Atlantis\n"
    )
    good = (
        "batch_number,expires_on,quantity,distribution_states\n"
        "A1,2027-01-31,10,Lagos;Ogun\n"
        "A2,2027-02-28,,\n"
    )

    rejected = await api.client.post(
        f"{url}/import", files={"file": ("batches.csv", bad, "text/csv")}, headers=headers
    )
    listed_after_rejection = await api.client.get(url, headers=headers)
    accepted = await api.client.post(
        f"{url}/import", files={"file": ("batches.csv", good, "text/csv")}, headers=headers
    )

    errors = rejected.json()["errors"]
    assert rejected.status_code == 422
    assert [tuple(error["loc"][2:]) for error in errors] == [
        (3, "row"),
        (4, "batch_number"),
        (5, "expires_on"),
        (5, "distribution_states"),
    ]
    assert errors[0]["msg"] == "expires_on must be after manufactured_on"
    assert listed_after_rejection.json()["items"] == []
    assert accepted.status_code == 200, accepted.text
    assert accepted.json() == {"rows": 2, "created": 2, "updated": 0}


async def test_csv_headers_are_checked(api: Api) -> None:
    admin, organization, product = await manufacturer_with_product(api)

    response = await api.client.post(
        f"/api/v1/products/{product.nafdac_number}/batches/import",
        files={"file": ("b.csv", "lot,expiry\nA,2027-01-01\n", "text/csv")},
        headers=auth(api, admin, organization),
    )

    assert response.status_code == 422
    assert {error["msg"] for error in response.json()["errors"]} == {
        "missing column",
        "unknown column",
    }


# ------------------------------------------------------------------------- claims
async def test_a_claim_links_the_product_after_review(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.MANUFACTURER_ADMIN)
    product = await api.create_product()
    reviewer, reviewer_org = await api.user_with_role(RoleName.REGULATOR_ADMIN)
    analyst, analyst_org = await api.user_with_role(RoleName.REGULATOR_ANALYST)
    body = {"nafdac_number": product.nafdac_number, "note": "<b>We</b> hold it"}

    claim = await api.client.post(
        "/api/v1/product-claims", json=body, headers=auth(api, admin, organization)
    )
    duplicate = await api.client.post(
        "/api/v1/product-claims", json=body, headers=auth(api, admin, organization)
    )
    claim_id = claim.json()["id"]
    by_analyst = await api.client.post(
        f"/api/v1/product-claims/{claim_id}/review",
        json={"decision": "approve"},
        headers=mfa(api, analyst, analyst_org),
    )
    approved = await api.client.post(
        f"/api/v1/product-claims/{claim_id}/review",
        json={"decision": "approve", "note": "Certificate checked."},
        headers=mfa(api, reviewer, reviewer_org),
    )
    twice = await api.client.post(
        f"/api/v1/product-claims/{claim_id}/review",
        json={"decision": "reject"},
        headers=mfa(api, reviewer, reviewer_org),
    )
    batch = await api.client.put(
        f"/api/v1/products/{product.nafdac_number}/batches/C1",
        json=BATCH,
        headers=auth(api, admin, organization),
    )

    assert claim.status_code == 201, claim.text
    assert claim.json()["note"] == "We hold it"  # HTML stripped
    assert duplicate.status_code == 409
    assert by_analyst.status_code == 403
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert twice.status_code == 409
    assert batch.status_code == 201
    assert (await api.reload(Product, product.id)).manufacturer_org_id == organization.id


async def test_only_verified_manufacturers_can_claim(api: Api) -> None:
    product = await api.create_product()
    pending = await api.create_org(
        OrganizationType.MANUFACTURER, status=OrganizationStatus.PENDING_VERIFICATION
    )
    admin = await api.create_user()
    await api.add_member(admin, pending, RoleName.MANUFACTURER_ADMIN)
    pharmacist, pharmacy = await api.user_with_role(RoleName.PHARMACY_ADMIN)

    as_pending = await api.client.post(
        "/api/v1/product-claims",
        json={"nafdac_number": product.nafdac_number},
        headers=auth(api, admin, pending),
    )
    as_pharmacy = await api.client.post(
        "/api/v1/product-claims",
        json={"nafdac_number": product.nafdac_number},
        headers=auth(api, pharmacist, pharmacy),
    )

    assert as_pending.status_code == 403  # an unverified organization grants no permissions
    assert as_pharmacy.status_code == 403


async def test_claims_are_listed_per_audience(api: Api) -> None:
    admin, organization = await api.user_with_role(RoleName.MANUFACTURER_ADMIN)
    product = await api.create_product()
    created = await api.client.post(
        "/api/v1/product-claims",
        json={"nafdac_number": product.nafdac_number},
        headers=auth(api, admin, organization),
    )
    other, other_org = await api.user_with_role(RoleName.MANUFACTURER_ADMIN)
    reviewer, reviewer_org = await api.user_with_role(RoleName.REGULATOR_ADMIN)
    consumer = await api.create_user()

    async def ids(headers: dict[str, str]) -> Any:
        response = await api.client.get(
            "/api/v1/product-claims", params={"status": "pending", "limit": 100}, headers=headers
        )
        return response.status_code, {item["id"] for item in response.json().get("items", [])}

    claim_id = created.json()["id"]
    assert claim_id in (await ids(auth(api, admin, organization)))[1]
    assert claim_id not in (await ids(auth(api, other, other_org)))[1]
    assert claim_id in (await ids(auth(api, reviewer, reviewer_org)))[1]
    assert (await ids(auth(api, consumer)))[0] == 403


# ------------------------------------------------------------------------- reference images
async def test_reference_images_wait_for_review(api: Api) -> None:
    admin, organization, product = await manufacturer_with_product(api)
    reviewer, reviewer_org = await api.user_with_role(RoleName.REGULATOR_ADMIN)
    url = f"/api/v1/products/{product.nafdac_number}/reference-images"

    uploaded = await api.client.post(
        url,
        files={"image": ("front.jpg", photo(), "image/jpeg")},
        data={"side": "front"},
        headers=auth(api, admin, organization),
    )
    image_id = uploaded.json()["id"]
    listed = await api.client.get(url, headers=auth(api, reviewer, reviewer_org))
    reviewed = await api.client.post(
        f"/api/v1/reference-images/{image_id}/review",
        json={"decision": "approve"},
        headers=mfa(api, reviewer, reviewer_org),
    )
    consumer_view = await api.client.get(url, headers=auth(api, await api.create_user()))

    stored = await api.reload(ReferenceImage, image_id)
    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["status"] == "pending"
    assert stored.s3_key in api.storage.objects
    assert [item["id"] for item in listed.json()["items"]] == [image_id]
    assert reviewed.json()["status"] == "approved"
    assert stored.status.value == "approved"
    assert consumer_view.status_code == 404


async def test_only_the_manufacturer_uploads_references(api: Api) -> None:
    _, _, product = await manufacturer_with_product(api)
    rival, rival_org = await api.user_with_role(RoleName.MANUFACTURER_ADMIN)

    response = await api.client.post(
        f"/api/v1/products/{product.nafdac_number}/reference-images",
        files={"image": ("front.jpg", photo(), "image/jpeg")},
        data={"side": "front"},
        headers=auth(api, rival, rival_org),
    )

    assert response.status_code == 403
    assert api.storage.objects == {}

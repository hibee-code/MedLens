"""Row-level policies, evaluated on in-memory actors and compiled SQL."""

from uuid import UUID, uuid4

import pytest
from sqlalchemy.dialects import postgresql

from medlens.core.identifiers import NigerianState
from medlens.modules.auth.principal import Principal
from medlens.modules.organizations.models import OrganizationType
from medlens.modules.products.models import Product
from medlens.modules.rbac import policies
from medlens.modules.rbac.access import Actor, OrganizationContext
from medlens.modules.rbac.models import RoleName
from medlens.modules.rbac.permissions import ROLE_PERMISSIONS
from medlens.modules.scans.models import Scan

ALL_STATES = frozenset(state.value for state in NigerianState)


def actor(
    role: RoleName,
    *,
    org_type: OrganizationType | None = None,
    states: frozenset[str] | None = None,
    org_id: UUID | None = None,
) -> Actor:
    organization = (
        OrganizationContext(
            id=org_id or uuid4(), type=org_type, role=role.value, jurisdiction_states=states
        )
        if org_type
        else None
    )
    return Actor(
        principal=Principal(
            kind="user", user_id=uuid4(), organization_id=organization.id if organization else None
        ),
        permissions=ROLE_PERMISSIONS[role],
        global_role=None,
        organization=organization,
        perm_ver=1,
    )


def sql(clause: object) -> str:
    compiled = clause.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})  # type: ignore[attr-defined]
    return str(compiled)


def lagos_regulator(role: RoleName = RoleName.REGULATOR_ANALYST) -> Actor:
    return actor(role, org_type=OrganizationType.REGULATOR, states=frozenset({"NG-LA", "NG-OG"}))


def test_scan_visibility_per_role() -> None:
    assert sql(policies.scan_visibility(actor(RoleName.PLATFORM_ADMIN))) == "true"
    national = actor(
        RoleName.REGULATOR_ANALYST, org_type=OrganizationType.REGULATOR, states=ALL_STATES
    )
    assert sql(policies.scan_visibility(national)) == "true"
    assert "scans.state IN ('NG-LA', 'NG-OG')" in sql(policies.scan_visibility(lagos_regulator()))
    pharmacy = actor(RoleName.PHARMACY_STAFF, org_type=OrganizationType.PHARMACY)
    clause = sql(policies.scan_visibility(pharmacy))
    assert "scans.organization_id =" in clause
    assert "scans.user_id =" in clause
    consumer = sql(policies.scan_visibility(actor(RoleName.CONSUMER)))
    assert consumer.startswith("scans.user_id =")


def test_object_level_scan_checks() -> None:
    lagos = lagos_regulator()
    consumer = actor(RoleName.CONSUMER)
    pharmacy = actor(RoleName.PHARMACY_STAFF, org_type=OrganizationType.PHARMACY)

    assert policies.can_view_scan(lagos, Scan(state=NigerianState.LAGOS))
    assert not policies.can_view_scan(lagos, Scan(state=NigerianState.KANO))
    assert policies.can_view_scan(consumer, Scan(user_id=consumer.user_id))
    assert not policies.can_view_scan(consumer, Scan(user_id=uuid4()))
    assert pharmacy.organization is not None
    assert policies.can_view_scan(pharmacy, Scan(organization_id=pharmacy.organization.id))
    assert not policies.can_view_scan(pharmacy, Scan(organization_id=uuid4()))


@pytest.mark.parametrize("role", list(RoleName))
def test_only_regulators_and_platform_admins_see_signals(role: RoleName) -> None:
    expected = role in {
        RoleName.REGULATOR_ANALYST,
        RoleName.REGULATOR_ADMIN,
        RoleName.PLATFORM_ADMIN,
    }

    assert policies.can_view_scan_signals(actor(role)) is expected


def test_jurisdiction_and_hotspots() -> None:
    lagos = lagos_regulator(RoleName.REGULATOR_ADMIN)

    assert policies.jurisdiction(lagos) == {"NG-LA", "NG-OG"}
    assert policies.jurisdiction(actor(RoleName.PLATFORM_ADMIN)) is None
    assert (
        policies.jurisdiction(actor(RoleName.PHARMACY_ADMIN, org_type=OrganizationType.PHARMACY))
        == frozenset()
    )
    assert policies.hotspot_states(lagos) == {"NG-LA", "NG-OG"}
    assert policies.hotspot_states(actor(RoleName.CONSUMER)) == frozenset()


def test_cases_are_visible_to_the_assigned_regulator() -> None:
    regulator = lagos_regulator()

    assert "cases.assigned_org_id =" in sql(policies.case_visibility(regulator))
    assert "cases.state IN" in sql(policies.case_visibility(regulator))
    national = actor(
        RoleName.REGULATOR_ANALYST, org_type=OrganizationType.REGULATOR, states=ALL_STATES
    )
    assert sql(policies.case_visibility(national)) == "true"
    assert sql(policies.case_visibility(actor(RoleName.PLATFORM_ADMIN))) == "true"
    assert sql(policies.case_visibility(actor(RoleName.CONSUMER))) == "false"


def test_manufacturers_write_only_their_own_products() -> None:
    own_org = uuid4()
    manufacturer = actor(
        RoleName.MANUFACTURER_STAFF, org_type=OrganizationType.MANUFACTURER, org_id=own_org
    )

    assert policies.can_write_product(manufacturer, Product(manufacturer_org_id=own_org))
    assert not policies.can_write_product(manufacturer, Product(manufacturer_org_id=uuid4()))
    assert not policies.can_write_product(manufacturer, Product(manufacturer_org_id=None))
    assert policies.can_write_product(
        actor(RoleName.PLATFORM_ADMIN), Product(manufacturer_org_id=uuid4())
    )


def test_organization_scoping() -> None:
    own_org = uuid4()
    admin = actor(RoleName.PHARMACY_ADMIN, org_type=OrganizationType.PHARMACY, org_id=own_org)
    staff = actor(RoleName.PHARMACY_STAFF, org_type=OrganizationType.PHARMACY, org_id=own_org)

    assert policies.can_manage_members(admin, own_org)
    assert not policies.can_manage_members(admin, uuid4())
    assert not policies.can_manage_members(staff, own_org)
    assert policies.can_read_organization(staff, own_org)
    assert not policies.can_read_organization(staff, uuid4())
    assert policies.can_manage_members(actor(RoleName.PLATFORM_ADMIN), uuid4())


def test_actor_staleness_and_tenant() -> None:
    def user_actor(token_perm_ver: int, current_perm_ver: int) -> Actor:
        principal = Principal(
            kind="user", user_id=uuid4(), organization_id=None, perm_ver=token_perm_ver
        )
        return Actor(principal, frozenset(), "consumer", None, perm_ver=current_perm_ver)

    assert user_actor(1, 2).stale  # roles changed after the token was issued
    assert not user_actor(2, 2).stale
    pharmacy = actor(RoleName.PHARMACY_ADMIN, org_type=OrganizationType.PHARMACY)
    assert pharmacy.organization is not None
    assert pharmacy.tenant.organization_id == pharmacy.organization.id
    assert not pharmacy.tenant.unrestricted
    assert actor(RoleName.PLATFORM_ADMIN).tenant.unrestricted

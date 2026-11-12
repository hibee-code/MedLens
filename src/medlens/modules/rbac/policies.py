"""Resource-level authorization rules.

Called from services, never from routers, so the API and the workers apply the same rules.
Query helpers return SQL conditions to add to a repository query; ``can_*`` helpers decide on a
loaded object. Code that hides a row the caller may not see should answer 404, not 403.
"""

from uuid import UUID

from sqlalchemy import ColumnElement, false, or_, true

from medlens.core.identifiers import NigerianState
from medlens.modules.cases.models import Case
from medlens.modules.organizations.models import OrganizationType
from medlens.modules.products.models import Product
from medlens.modules.rbac.access import Actor
from medlens.modules.rbac.permissions import Permission
from medlens.modules.scans.models import Scan

_ALL_STATES = frozenset(state.value for state in NigerianState)


def jurisdiction(actor: Actor) -> frozenset[str] | None:
    """States a regulator may see: None means everywhere (platform admins, national regulators),
    an empty set means nowhere (not a regulator)."""
    if actor.is_platform_admin:
        return None
    organization = actor.organization
    if organization is None or organization.type is not OrganizationType.REGULATOR:
        return frozenset()
    states = organization.jurisdiction_states or frozenset()
    return None if states >= _ALL_STATES else states


# ------------------------------------------------------------------ scans
def scan_visibility(actor: Actor) -> ColumnElement[bool]:
    if actor.is_platform_admin:
        return true()
    if actor.can(Permission.SCAN_READ_ALL):
        states = jurisdiction(actor)
        if states is None:
            return true()
        if states:
            return Scan.state.in_(sorted(states))
    conditions: list[ColumnElement[bool]] = []
    if actor.can(Permission.SCAN_READ_ORG) and actor.organization is not None:
        conditions.append(Scan.organization_id == actor.organization.id)
    if actor.can(Permission.SCAN_READ_OWN) and actor.user_id is not None:
        conditions.append(Scan.user_id == actor.user_id)
    return or_(*conditions) if conditions else false()


def can_view_scan(actor: Actor, scan: Scan) -> bool:
    if actor.is_platform_admin:
        return True
    if actor.can(Permission.SCAN_READ_ALL):
        states = jurisdiction(actor)
        if states is None or (scan.state is not None and scan.state.value in states):
            return True
    if (
        actor.can(Permission.SCAN_READ_ORG)
        and actor.organization is not None
        and scan.organization_id == actor.organization.id
    ):
        return True
    return (
        actor.can(Permission.SCAN_READ_OWN)
        and actor.user_id is not None
        and scan.user_id == actor.user_id
    )


def can_view_scan_signals(actor: Actor) -> bool:
    """Detailed signals stay with regulators and platform admins, so counterfeiters cannot learn
    which checks their packs fail; everyone else gets reason codes."""
    return actor.can(Permission.SCAN_READ_SIGNALS)


# ------------------------------------------------------------------ cases and hotspots
def case_visibility(actor: Actor) -> ColumnElement[bool]:
    """Cases in the regulator's jurisdiction (national regulators: all), plus any case routed
    to its organization."""
    if actor.is_platform_admin:
        return true()
    if not actor.can(Permission.CASE_READ) or actor.organization is None:
        return false()
    states = jurisdiction(actor)
    if states is None:
        return true()
    assigned = Case.assigned_org_id == actor.organization.id
    return or_(Case.state.in_(sorted(states)), assigned) if states else assigned


def hotspot_states(actor: Actor) -> frozenset[str] | None:
    """States whose hotspots the actor may see (None: all)."""
    if not actor.can(Permission.HOTSPOT_READ):
        return frozenset()
    return jurisdiction(actor)


# ------------------------------------------------------------------ products
def can_write_product(actor: Actor, product: Product) -> bool:
    """Manufacturers change only products linked to their own organization."""
    if actor.is_platform_admin:
        return True
    organization = actor.organization
    return (
        actor.can(Permission.PRODUCT_WRITE)
        and organization is not None
        and organization.type is OrganizationType.MANUFACTURER
        and product.manufacturer_org_id == organization.id
    )


def owns_product(actor: Actor, product: Product) -> bool:
    organization = actor.organization
    return (
        organization is not None
        and organization.type is OrganizationType.MANUFACTURER
        and product.manufacturer_org_id == organization.id
    )


def can_read_batches(actor: Actor, product: Product) -> bool:
    """Batch numbers would help counterfeiters label fakes, so they are not public: the
    product's manufacturer, regulators and platform admins only."""
    if actor.is_platform_admin or owns_product(actor, product):
        return True
    organization = actor.organization
    return organization is not None and organization.type is OrganizationType.REGULATOR


def can_write_batches(actor: Actor, product: Product) -> bool:
    if actor.is_platform_admin:
        return True
    return actor.can(Permission.BATCH_WRITE) and owns_product(actor, product)


def can_upload_reference_images(actor: Actor, product: Product) -> bool:
    if actor.is_platform_admin:
        return True
    return actor.can(Permission.REFERENCE_IMAGE_WRITE) and owns_product(actor, product)


def can_review_products(actor: Actor) -> bool:
    """Product claims and reference images are approved by people (CLAUDE.md §5)."""
    return actor.is_user and actor.can(Permission.PRODUCT_REVIEW)


def can_read_reference_images(actor: Actor, product: Product) -> bool:
    return can_review_products(actor) or owns_product(actor, product)


# ------------------------------------------------------------------ organizations
def can_read_organization(actor: Actor, organization_id: UUID) -> bool:
    if actor.is_platform_admin:
        return True
    return (
        actor.can(Permission.ORG_READ)
        and actor.organization is not None
        and actor.organization.id == organization_id
    )


def can_invite_members(actor: Actor, organization_id: UUID) -> bool:
    if actor.is_platform_admin:
        return True
    return (
        actor.can(Permission.MEMBER_INVITE)
        and actor.organization is not None
        and actor.organization.id == organization_id
    )


def can_manage_organization(actor: Actor, organization_id: UUID) -> bool:
    """Organization details and branches."""
    if actor.is_platform_admin:
        return True
    return (
        actor.can(Permission.ORG_MANAGE)
        and actor.organization is not None
        and actor.organization.id == organization_id
    )


def can_manage_members(actor: Actor, organization_id: UUID) -> bool:
    if actor.is_platform_admin:
        return True
    return (
        actor.can(Permission.MEMBER_MANAGE)
        and actor.organization is not None
        and actor.organization.id == organization_id
    )

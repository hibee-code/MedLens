"""Development seed data: ``python -m medlens.seed`` (``make seed``).

Loads seed/products.csv (a 50-product sample of the real NAFDAC Greenbook, see
seed/PROVENANCE.md and ``medlens.seed.greenbook``), creates one organization per registration
holder, a national regulator, a test pharmacy, the synthetic batches in seed/batches.csv
(numbers start with SEED-; one valid and one expired batch per product), the sample knowledge
documents in seed/documents (marked as samples, not official notices; indexed without vectors,
which the hourly backfill adds) and one test user per role.

Idempotent: everything is looked up or upserted by its natural key, so running it again changes
nothing. Refuses to run in staging or production.
"""

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from medlens.core.config import KnowledgeSettings
from medlens.core.crypto import phone_lookup_hash
from medlens.core.identifiers import NigerianState
from medlens.core.security import hash_password
from medlens.modules.knowledge.ingestion import index_pages
from medlens.modules.knowledge.models import DocumentKind, KnowledgeDocument, Visibility
from medlens.modules.knowledge.text import extract_pages
from medlens.modules.organizations.models import (
    Membership,
    Organization,
    OrganizationStatus,
    OrganizationType,
)
from medlens.modules.products.models import (
    Batch,
    ImageSide,
    Product,
    ReferenceImage,
    ReviewStatus,
)
from medlens.modules.rbac.models import RoleName
from medlens.modules.users.models import User
from medlens.ports.storage import SyncStoragePort

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "seed"
REGULATOR_NAME = "National Agency for Food and Drug Administration and Control (NAFDAC)"
PHARMACY_NAME = "MedLens Test Pharmacy (seed)"
MANUFACTURER_FOR_TEST_USERS = "Emzor Pharmaceutical Industries Limited"


@dataclass(frozen=True, slots=True)
class TestUser:
    role: RoleName
    email: str | None = None
    phone: str | None = None
    organization: str | None = None  # "regulator", "pharmacy" or "manufacturer"


TEST_USERS = (
    TestUser(RoleName.CONSUMER, phone="+2348030000001"),
    TestUser(RoleName.HEALTH_WORKER, phone="+2348030000002"),
    TestUser(RoleName.PHARMACY_STAFF, email="pharmacy.staff@medlens.test", organization="pharmacy"),
    TestUser(RoleName.PHARMACY_ADMIN, email="pharmacy.admin@medlens.test", organization="pharmacy"),
    TestUser(
        RoleName.MANUFACTURER_STAFF,
        email="manufacturer.staff@medlens.test",
        organization="manufacturer",
    ),
    TestUser(
        RoleName.MANUFACTURER_ADMIN,
        email="manufacturer.admin@medlens.test",
        organization="manufacturer",
    ),
    TestUser(
        RoleName.REGULATOR_ANALYST, email="regulator.analyst@medlens.test", organization="regulator"
    ),
    TestUser(
        RoleName.REGULATOR_ADMIN, email="regulator.admin@medlens.test", organization="regulator"
    ),
    TestUser(RoleName.PLATFORM_ADMIN, email="platform.admin@medlens.test"),
)


@dataclass(frozen=True, slots=True)
class SeedSummary:
    products: int
    organizations: int
    batches: int
    users: int
    memberships: int
    documents: int = 0


def read_products(path: Path) -> list[dict[str, object]]:
    """Product rows from a Greenbook sample or snapshot CSV."""
    # Imported here so `python -m medlens.seed.greenbook` does not load that module twice.
    from medlens.seed.greenbook import product_values

    with path.open(newline="", encoding="utf-8") as handle:
        return [product_values(record) for record in csv.DictReader(handle)]


def run_seed(
    session: Session,
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    staff_password: str | None = None,
    reset_staff_logins: bool = False,
    storage: SyncStoragePort | None = None,
) -> SeedSummary:
    """Seed inside the caller's transaction. ``staff_password`` becomes the password of every
    e-mail test user that has none yet (phone users sign in with SMS codes); with
    ``reset_staff_logins`` it replaces their passwords and switches their MFA off (demos on a
    database seeded earlier). With ``storage``, the synthetic reference images in
    seed/reference_images are stored and registered as approved (the embedding backfill embeds
    them)."""
    products = read_products(data_dir / "products.csv")
    manufacturers = {
        name: _organization(session, str(name), OrganizationType.MANUFACTURER)
        for name in sorted(
            {str(p["manufacturer_name"]) for p in products if p["manufacturer_name"]}
        )
    }
    regulator = _organization(
        session,
        REGULATOR_NAME,
        OrganizationType.REGULATOR,
        jurisdiction_states=[state.value for state in NigerianState],
    )
    pharmacy = _organization(
        session, PHARMACY_NAME, OrganizationType.PHARMACY, state=NigerianState.LAGOS
    )

    for product in products:
        organization = manufacturers.get(str(product["manufacturer_name"]))
        product["manufacturer_org_id"] = organization.id if organization else None
    from medlens.seed.greenbook import upsert_products

    upsert_products(session, products)

    _seed_batches(session, data_dir / "batches.csv")
    _seed_documents(session, data_dir / "documents")

    memberships = {
        "regulator": regulator,
        "pharmacy": pharmacy,
        "manufacturer": manufacturers.get(MANUFACTURER_FOR_TEST_USERS)
        or next(iter(manufacturers.values())),
    }
    password_hash = hash_password(staff_password) if staff_password else None
    for test_user in TEST_USERS:
        user = _user(session, test_user)
        if password_hash and test_user.email and (reset_staff_logins or user.password_hash is None):
            user.password_hash = password_hash
            if reset_staff_logins:
                user.mfa_enabled = False
                user.mfa_secret = None
        if test_user.organization:
            session.execute(
                insert(Membership)
                .values(
                    user_id=user.id,
                    organization_id=memberships[test_user.organization].id,
                    role=test_user.role.value,
                )
                .on_conflict_do_nothing(index_elements=["user_id", "organization_id"])
            )
    session.flush()
    if storage is not None:
        admin = session.scalar(select(User).where(User.email == "platform.admin@medlens.test"))
        _seed_reference_images(
            session, data_dir / "reference_images", storage, admin.id if admin else None
        )
    return SeedSummary(
        products=_count(session, Product),
        organizations=_count(session, Organization),
        batches=_count(session, Batch),
        users=_count(session, User),
        memberships=_count(session, Membership),
        documents=_count(session, KnowledgeDocument),
    )


def _organization(
    session: Session,
    name: str,
    kind: OrganizationType,
    *,
    state: NigerianState | None = None,
    jurisdiction_states: list[str] | None = None,
) -> Organization:
    existing = session.scalars(
        select(Organization).where(Organization.type == kind, Organization.name == name)
    ).first()
    if existing is not None:
        return existing
    organization = Organization(
        name=name,
        type=kind,
        state=state,
        jurisdiction_states=jurisdiction_states,
        status=OrganizationStatus.ACTIVE,
        verified_at=datetime.now(UTC),
    )
    session.add(organization)
    session.flush()
    return organization


def _seed_batches(session: Session, path: Path) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        declared = list(csv.DictReader(handle))
    products = {
        product.nafdac_number: product
        for product in session.scalars(
            select(Product).where(
                Product.nafdac_number.in_({row["nafdac_number"] for row in declared})
            )
        )
    }
    rows = [
        {
            "product_id": product.id,
            "batch_number": row["batch_number"],
            "manufactured_on": date.fromisoformat(row["manufactured_on"]),
            "expires_on": date.fromisoformat(row["expires_on"]),
            "distribution_states": row["distribution_states"].split(";"),
            "declared_by_org_id": product.manufacturer_org_id,
        }
        for row in declared
        if (product := products.get(row["nafdac_number"])) is not None
    ]
    if rows:
        session.execute(
            insert(Batch).on_conflict_do_nothing(index_elements=["product_id", "batch_number"]),
            rows,
        )


def _seed_documents(session: Session, directory: Path) -> None:
    index = directory / "index.csv"
    if not index.exists():
        return
    with index.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        exists = session.scalar(
            select(KnowledgeDocument.id).where(
                KnowledgeDocument.title == row["title"], KnowledgeDocument.source == row["source"]
            )
        )
        if exists is not None:
            continue
        data = (directory / row["file"]).read_bytes()
        document = KnowledgeDocument(
            title=row["title"],
            kind=DocumentKind(row["kind"]),
            visibility=Visibility(row["visibility"]),
            source=row["source"],
            published_on=date.fromisoformat(row["published_on"]) if row["published_on"] else None,
            official=row.get("official") == "true",
            active=row.get("active", "true") == "true",
            media_type="text/plain",
            size_bytes=len(data),
        )
        session.add(document)
        session.flush()
        index_pages(session, document, extract_pages(data, "text/plain"), None, KnowledgeSettings())


def _seed_reference_images(
    session: Session, directory: Path, storage: SyncStoragePort, reviewer: UUID | None
) -> None:
    index = directory / "index.csv"
    if not index.exists():
        return
    with index.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        product = session.scalar(
            select(Product).where(Product.nafdac_number == row["nafdac_number"])
        )
        key = f"reference-images/seed/{row['file']}"
        if product is None or session.scalar(
            select(ReferenceImage.id).where(ReferenceImage.s3_key == key)
        ):
            continue
        storage.put(key, (directory / row["file"]).read_bytes(), content_type="image/jpeg")
        session.add(
            ReferenceImage(
                product_id=product.id,
                side=ImageSide(row["side"]),
                s3_key=key,
                status=ReviewStatus.APPROVED,  # synthetic seed data, for local demos only
                reviewed_at=datetime.now(UTC),
                verified_by=reviewer,
            )
        )
    session.flush()


def _user(session: Session, test_user: TestUser) -> User:
    if test_user.phone:
        lookup = User.phone_hash == phone_lookup_hash(test_user.phone)
    else:
        lookup = User.email == test_user.email
    existing = session.scalars(select(User).where(lookup)).first()
    if existing is not None:
        return existing
    global_role = (
        test_user.role
        if test_user.organization is None
        else RoleName.CONSUMER  # organization roles are held through the membership
    )
    user = User(
        email=test_user.email,
        full_name=f"Seed {test_user.role.value.replace('_', ' ')}",
        global_role=global_role.value,
    )
    user.set_phone(test_user.phone)
    session.add(user)
    session.flush()
    return user


def _count(session: Session, model: type[object]) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0

"""Development seed data: ``python -m medlens.seed`` (``make seed``).

Loads seed/products.csv (a 50-product sample of the real NAFDAC Greenbook, see
seed/PROVENANCE.md and ``medlens.seed.greenbook``), creates one organization per registration
holder, a national regulator, a test pharmacy, the synthetic batches in seed/batches.csv
(numbers start with SEED-; one valid and one expired batch per product), the sample knowledge
documents in seed/documents (marked as samples, not official notices; indexed without vectors,
which the hourly backfill adds). It creates no users: the first platform admin is invited with
`make bootstrap-admin`, and staff join the seeded organizations by invitation (scripts/demo.sh
does both).

Idempotent: everything is looked up or upserted by its natural key, so running it again changes
nothing. Refuses to run in staging or production.
"""

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from medlens.core.config import KnowledgeSettings
from medlens.core.identifiers import NigerianState
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
    Product,
    ReferenceImage,
    RegisterSource,
)
from medlens.modules.users.models import User

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "seed"
REGULATOR_NAME = "National Agency for Food and Drug Administration and Control (NAFDAC)"
PHARMACY_NAME = "MedLens Test Pharmacy (seed)"


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


def run_seed(session: Session, data_dir: Path = DEFAULT_DATA_DIR) -> SeedSummary:
    """Seed inside the caller's transaction. The seed adds no users and no reference images:
    synthetic images are for tests only, and those an earlier seed approved are withdrawn."""
    products = read_products(data_dir / "products.csv")
    manufacturers = {
        name: _organization(session, str(name), OrganizationType.MANUFACTURER)
        for name in sorted(
            {str(p["manufacturer_name"]) for p in products if p["manufacturer_name"]}
        )
    }
    _organization(
        session,
        REGULATOR_NAME,
        OrganizationType.REGULATOR,
        jurisdiction_states=[state.value for state in NigerianState],
    )
    _organization(session, PHARMACY_NAME, OrganizationType.PHARMACY, state=NigerianState.LAGOS)

    for product in products:
        organization = manufacturers.get(str(product["manufacturer_name"]))
        product["manufacturer_org_id"] = organization.id if organization else None
    from medlens.seed.greenbook import upsert_products

    # The committed sample comes from a snapshot: hand-typed rows are kept as they are.
    upsert_products(session, products, RegisterSource.SNAPSHOT)

    _seed_batches(session, data_dir / "batches.csv")
    _seed_documents(session, data_dir / "documents")

    session.flush()
    _withdraw_synthetic_references(session)
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


SYNTHETIC_REFERENCE_PREFIX = "reference-images/seed/"


def _withdraw_synthetic_references(session: Session) -> None:
    """Seeds of 2026-11-04 stored drawn reference packs as approved; synthetic images must never
    count as genuine references (CLAUDE.md decision log), so they are removed."""
    session.execute(
        delete(ReferenceImage).where(ReferenceImage.s3_key.startswith(SYNTHETIC_REFERENCE_PREFIX))
    )


def _count(session: Session, model: type[object]) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0

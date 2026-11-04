"""The pipeline's Catalog on Postgres, for Celery workers (sync session; CLAUDE.md §7)."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import ColumnElement, distinct, func, select
from sqlalchemy.orm import Session

from medlens.modules.knowledge.models import PublishedAlert
from medlens.modules.products.models import Batch, Product, ReferenceImage, ReviewStatus
from medlens.modules.scans.models import Scan, ScanStatus
from medlens.modules.verification.catalog import AlertFacts, BatchFacts, BatchSpread, ProductFacts

CANDIDATES = 5
# pg_trgm thresholds for the two fuzzy lookups.
NUMBER_SIMILARITY = 0.4
NAME_WORD_SIMILARITY = 0.6
MIN_BRAND_LENGTH = 4  # shorter names would match words in almost any text
NEAR_DUPLICATE_BITS = 6  # perceptual hashes this close are the same photo


def _facts(product: Product) -> ProductFacts:
    return ProductFacts(
        id=product.id,
        nafdac_number=product.nafdac_number,
        brand_name=product.brand_name,
        generic_name=product.generic_name,
        registration_status=product.registration_status,
    )


@dataclass(frozen=True, slots=True)
class EarlierScan:
    scan: Scan
    distance: int


class SqlCatalog:
    def __init__(self, session: Session) -> None:
        self._session = session

    def product_by_nafdac(self, nafdac_number: str) -> ProductFacts | None:
        product = self._session.scalar(
            select(Product).where(Product.nafdac_number == nafdac_number)
        )
        return _facts(product) if product else None

    def similar_nafdac(self, nafdac_number: str) -> list[tuple[ProductFacts, float]]:
        score = func.similarity(Product.nafdac_number, nafdac_number)
        statement = (
            select(Product, score)
            .where(score >= NUMBER_SIMILARITY)
            .order_by(score.desc())
            .limit(CANDIDATES)
        )
        return [
            (_facts(product), float(value)) for product, value in self._session.execute(statement)
        ]

    def products_named_in(self, text: str) -> list[tuple[ProductFacts, float]]:
        """word_similarity: how well the brand name matches some stretch of the text."""
        text = text[:4000]
        score = func.word_similarity(Product.brand_name, text)
        statement = (
            select(Product, score)
            .where(
                func.length(Product.brand_name) >= MIN_BRAND_LENGTH, score >= NAME_WORD_SIMILARITY
            )
            .order_by(score.desc(), func.length(Product.brand_name).desc())
            .limit(CANDIDATES)
        )
        return [
            (_facts(product), float(value)) for product, value in self._session.execute(statement)
        ]

    def declared_batch_count(self, product_id: UUID) -> int:
        count = self._session.scalar(
            select(func.count()).select_from(Batch).where(Batch.product_id == product_id)
        )
        return int(count or 0)

    def batch(self, product_id: UUID, batch_number: str) -> BatchFacts | None:
        batch = self._session.scalar(
            select(Batch).where(Batch.product_id == product_id, Batch.batch_number == batch_number)
        )
        if batch is None:
            return None
        return BatchFacts(
            id=batch.id,
            batch_number=batch.batch_number,
            expires_on=batch.expires_on,
            distribution_states=frozenset(batch.distribution_states),
        )

    def batch_spread(self, batch_id: UUID, since: datetime, *, exclude_scan: UUID) -> BatchSpread:
        row = self._session.execute(
            select(func.count(distinct(Scan.geocell)), func.array_agg(distinct(Scan.state))).where(
                Scan.matched_batch_id == batch_id,
                Scan.created_at >= since,
                Scan.id != exclude_scan,
            )
        ).one()
        geocells, states = row
        return BatchSpread(
            geocells=int(geocells or 0),
            states=frozenset(str(state) for state in (states or []) if state is not None),
        )

    def published_alert(self, nafdac_number: str, batch_number: str | None) -> AlertFacts | None:
        batches = ["", batch_number] if batch_number else [""]
        alert = self._session.scalar(
            select(PublishedAlert)
            .where(
                PublishedAlert.nafdac_number == nafdac_number,
                PublishedAlert.batch_number.in_(batches),
            )
            .order_by(
                (PublishedAlert.batch_number == "").asc(),
                PublishedAlert.published_on.desc().nulls_last(),
            )
            .limit(1)
        )
        if alert is None:
            return None
        return AlertFacts(
            document_id=alert.document_id,
            kind=alert.kind.value,
            batch_specific=alert.batch_number != "",
            published_on=alert.published_on,
        )

    def reference_similarity(self, product_id: UUID, embedding: list[float]) -> float | None:
        distance = ReferenceImage.embedding.cosine_distance(embedding)
        best = self._session.scalar(
            select(func.min(distance)).where(
                ReferenceImage.product_id == product_id,
                ReferenceImage.status == ReviewStatus.APPROVED,
                ReferenceImage.deleted_at.is_(None),
                ReferenceImage.embedding.is_not(None),
            )
        )
        return None if best is None else 1.0 - float(best)

    def near_duplicate(
        self, phash: int, *, owner: ColumnElement[bool], since: datetime, exclude_scan: UUID
    ) -> EarlierScan | None:
        """The latest finished scan by the same person or device whose photo is perceptually the
        same: probing with slightly edited photos gets the first answer again."""
        candidates = self._session.scalars(
            select(Scan)
            .where(
                owner,
                Scan.created_at >= since,
                Scan.id != exclude_scan,
                Scan.image_phash.is_not(None),
                Scan.status.in_([ScanStatus.COMPLETED, ScanStatus.REJECTED]),
            )
            .order_by(Scan.created_at.desc())
            .limit(50)
        )
        for scan in candidates:
            distance = hamming(phash, scan.image_phash or 0)
            if distance <= NEAR_DUPLICATE_BITS:
                return EarlierScan(scan, distance)
        return None


def hamming(a: int, b: int) -> int:
    return ((a ^ b) & 0xFFFF_FFFF_FFFF_FFFF).bit_count()

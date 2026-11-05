"""What the pipeline needs to know about registered products, batches and earlier scans.

A Protocol so the pipeline stays pure: the worker implements it with SQL
(``verification/repository.py``), tests with dictionaries.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol
from uuid import UUID

from medlens.modules.products.models import RegistrationStatus


@dataclass(frozen=True, slots=True)
class ProductFacts:
    id: UUID
    nafdac_number: str
    brand_name: str
    generic_name: str
    registration_status: RegistrationStatus


@dataclass(frozen=True, slots=True)
class BatchFacts:
    id: UUID
    batch_number: str
    expires_on: date
    distribution_states: frozenset[str]


@dataclass(frozen=True, slots=True)
class AlertFacts:
    """A published alert or recall naming the product (and the batch, if batch_specific)."""

    document_id: UUID
    kind: str
    batch_specific: bool
    published_on: date | None


@dataclass(frozen=True, slots=True)
class BatchSpread:
    geocells: int
    states: frozenset[str]


class Catalog(Protocol):
    def product_by_nafdac(self, nafdac_number: str) -> ProductFacts | None: ...

    def similar_nafdac(self, nafdac_number: str) -> list[tuple[ProductFacts, float]]:
        """Products whose number is close to ``nafdac_number`` (OCR errors), best first."""
        ...

    def products_named_in(self, text: str) -> list[tuple[ProductFacts, float]]:
        """Products whose brand name appears in ``text``, best first, with a 0..1 score."""
        ...

    def declared_batch_count(self, product_id: UUID) -> int: ...

    def batch(self, product_id: UUID, batch_number: str) -> BatchFacts | None: ...

    def batch_spread(self, batch_id: UUID, since: datetime, *, exclude_scan: UUID) -> BatchSpread:
        """Distinct places where this batch was scanned since ``since``, other scans only."""
        ...

    def published_alerts(self, nafdac_number: str, batch_number: str | None) -> list[AlertFacts]:
        """Official, active alerts and recalls naming this product, for every batch or for
        ``batch_number``."""
        ...

    def reference_similarity(
        self, product_id: UUID, embedding: list[float], *, model_version: str
    ) -> float | None:
        """Best cosine similarity to the product's approved reference images embedded by the
        same model, or None when it has none (vectors of two models are never compared)."""
        ...

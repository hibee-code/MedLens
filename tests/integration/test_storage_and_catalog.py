"""Object storage adapters on MinIO, and the pipeline's SQL catalog on Postgres."""

from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from purgatory import AsyncCircuitBreakerFactory
from sqlalchemy.orm import Session

from medlens.adapters.s3 import create_sync_s3_client, open_s3_client
from medlens.adapters.storage import S3Storage, SyncS3Storage
from medlens.core.config import S3Settings, Settings
from medlens.core.ids import uuid7, uuid7_time
from medlens.db.session import create_sync_engine, create_sync_session_factory
from medlens.modules.products.models import (
    Batch,
    ImageSide,
    Product,
    ReferenceImage,
    RegistrationStatus,
    ReviewStatus,
)
from medlens.modules.scans.models import Scan, ScanStatus
from medlens.modules.verification.repository import SqlCatalog, hamming
from medlens.ports.storage import ObjectNotFoundError, StorageUnavailableError
from tests.integration.api_support import new_nafdac_number
from tests.integration.test_readiness import closed_port, ensure_bucket

pytestmark = pytest.mark.integration


# ------------------------------------------------------------------------- object storage
async def test_s3_storage_round_trip(live_groups: dict[str, dict[str, Any]]) -> None:
    await ensure_bucket(live_groups["s3"])
    settings = S3Settings(**live_groups["s3"])
    key = f"tests/{uuid4()}.jpg"

    async with open_s3_client(settings) as client:
        storage = S3Storage(client, settings.bucket)
        await storage.put(key, b"jpeg bytes", content_type="image/jpeg")
        fetched = await storage.get(key)
        sync = SyncS3Storage(create_sync_s3_client(settings), settings.bucket)
        assert sync.get(key) == b"jpeg bytes"
        sync.put(f"{key}.copy", b"copy", content_type="image/jpeg")
        await storage.delete(key)
        with pytest.raises(ObjectNotFoundError):
            await storage.get(key)
        with pytest.raises(ObjectNotFoundError):
            sync.get(key)

    assert fetched == b"jpeg bytes"


async def test_an_unreachable_store_opens_the_breaker(
    live_groups: dict[str, dict[str, Any]],
) -> None:
    settings = S3Settings(
        **live_groups["s3"]
        | {
            "endpoint_url": f"http://127.0.0.1:{closed_port()}",
            "max_attempts": 1,
            "connect_timeout_seconds": 0.2,
        }
    )
    breakers = AsyncCircuitBreakerFactory(default_threshold=2, default_ttl=30)

    async with open_s3_client(settings) as client:
        storage = S3Storage(client, settings.bucket, breakers)
        for _ in range(2):
            with pytest.raises(StorageUnavailableError):
                await storage.get("anything")
        with pytest.raises(StorageUnavailableError, match="circuit breaker"):
            await storage.get("anything")


# ------------------------------------------------------------------------- SQL catalog
@pytest.fixture
def session(migrated_settings: Settings) -> Any:
    engine = create_sync_engine(migrated_settings.database, application_name="medlens-tests")
    factory = create_sync_session_factory(engine)
    with factory() as session, session.begin():
        yield session
        session.rollback()  # leave the shared database as it was
    engine.dispose()


def product(session: Session, brand: str, number: str | None = None) -> Product:
    item = Product(
        nafdac_number=number or new_nafdac_number(),
        brand_name=brand,
        generic_name="Artemether, Lumefantrine",
        registration_status=RegistrationStatus.ACTIVE,
    )
    session.add(item)
    session.flush()
    return item


def test_numbers_and_names_are_matched_fuzzily(session: Session) -> None:
    brand = f"Qwertasol {uuid4().hex[:6]}"
    target = product(session, brand, "Q7-482915")
    catalog = SqlCatalog(session)

    exact = catalog.product_by_nafdac("Q7-482915")
    close = catalog.similar_nafdac("Q7-482975")  # one OCR error
    named = catalog.products_named_in(f"PACK TEXT {brand.upper()} 20/120 TABLETS EXP 12/27")

    assert exact is not None
    assert exact.id == target.id
    assert close[0][0].id == target.id
    assert close[0][1] >= 0.4
    assert named[0][0].id == target.id
    assert named[0][1] == pytest.approx(1.0)
    assert catalog.product_by_nafdac("Q7-000000") is None


def test_batches_and_their_spread(session: Session) -> None:
    item = product(session, f"Spreadol {uuid4().hex[:6]}")
    batch = Batch(
        product_id=item.id,
        batch_number="SP-1",
        expires_on=date(2027, 1, 31),
        distribution_states=["NG-LA"],
    )
    session.add(batch)
    session.flush()
    now = datetime.now(UTC)
    for state, cell in (("NG-LA", "s0aaaa"), ("NG-KN", "s0bbbb"), ("NG-KN", "s0bbbb")):
        scan_id = uuid7()
        session.add(
            Scan(
                id=scan_id,
                created_at=uuid7_time(scan_id),
                matched_batch_id=batch.id,
                state=state,
                geocell=cell,
                status=ScanStatus.COMPLETED,
            )
        )
    session.flush()
    catalog = SqlCatalog(session)

    facts = catalog.batch(item.id, "SP-1")
    spread = catalog.batch_spread(batch.id, now - timedelta(days=1), exclude_scan=uuid4())

    assert catalog.declared_batch_count(item.id) == 1
    assert facts is not None
    assert facts.distribution_states == frozenset({"NG-LA"})
    assert catalog.batch(item.id, "NOPE") is None
    assert spread.geocells == 2
    assert spread.states == frozenset({"NG-LA", "NG-KN"})


def test_only_approved_reference_images_count(
    session: Session, migrated_settings: Settings
) -> None:
    item = product(session, f"Visiol {uuid4().hex[:6]}")
    dimension = migrated_settings.embeddings.image_dimension
    genuine = [1.0] + [0.0] * (dimension - 1)
    unrelated = [0.0, 1.0] + [0.0] * (dimension - 2)
    for status, vector in ((ReviewStatus.APPROVED, genuine), (ReviewStatus.PENDING, unrelated)):
        session.add(
            ReferenceImage(
                product_id=item.id,
                side=ImageSide.FRONT,
                s3_key=f"reference-images/{uuid4()}.jpg",
                embedding=vector,
                status=status,
            )
        )
    session.flush()
    catalog = SqlCatalog(session)

    assert catalog.reference_similarity(item.id, genuine) == pytest.approx(1.0)
    assert catalog.reference_similarity(item.id, unrelated) == pytest.approx(0.0)
    assert catalog.reference_similarity(uuid4(), genuine) is None


def test_hamming_distance_handles_signed_hashes() -> None:
    assert hamming(-1, 0) == 64
    assert hamming(0b1011, 0b0001) == 2

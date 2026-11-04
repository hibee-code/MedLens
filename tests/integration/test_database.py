"""Database features: trigram search, HNSW search, field encryption, append-only audit log,
partition routing and the composite foreign key from cases to scans."""

import math
import random
from datetime import UTC, datetime

import pytest
from geoalchemy2 import WKTElement
from sqlalchemy import select, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from medlens.core.config import Settings
from medlens.core.crypto import DecryptionError, field_cipher, phone_lookup_hash
from medlens.core.ids import uuid7
from medlens.modules.audit.models import AuditLog
from medlens.modules.cases.models import Case
from medlens.modules.products.models import ImageSide, Product, ReferenceImage, RegistrationStatus
from medlens.modules.products.repository import ProductRepository
from medlens.modules.scans.models import Scan, Verdict
from medlens.modules.users.models import User

pytestmark = pytest.mark.integration


def product(nafdac_number: str, brand_name: str, generic_name: str) -> Product:
    return Product(
        nafdac_number=nafdac_number,
        brand_name=brand_name,
        generic_name=generic_name,
        registration_status=RegistrationStatus.ACTIVE,
    )


async def uses_index(
    session: AsyncSession, sql: str, params: dict[str, object], index: str
) -> bool:
    await session.execute(text("SET LOCAL enable_seqscan = off"))
    plan = "\n".join((await session.execute(text(f"EXPLAIN {sql}"), params)).scalars())
    return index in plan


# ------------------------------------------------------------------------- trigram search
async def test_trigram_search_absorbs_ocr_errors(db_session: AsyncSession) -> None:
    db_session.add_all(
        [
            product("04-0005", "Panadol Extra Caplet", "Paracetamol; Caffeine"),
            product("04-0411", "Emzor Paracetamol Tablet", "Paracetamol"),
            product("A4-1680", "Coartem 20/120 Dispersible Tablets", "Artemether; Lumefantrine"),
        ]
    )
    await db_session.flush()

    results = await ProductRepository(db_session).search_names("Panad0l Extr")

    assert results[0][0].brand_name == "Panadol Extra Caplet"
    assert results[0][1] > 0.3
    assert "Coartem 20/120 Dispersible Tablets" not in [p.brand_name for p, _ in results]
    assert await uses_index(
        db_session,
        "SELECT id FROM products WHERE brand_name % :q",
        {"q": "Panad0l Extr"},
        "ix_products_brand_name_trgm",
    )


async def test_exact_nafdac_lookup(db_session: AsyncSession) -> None:
    db_session.add(product("04-0005", "Panadol Extra Caplet", "Paracetamol; Caffeine"))
    await db_session.flush()

    repository = ProductRepository(db_session)

    assert (await repository.get_by_nafdac_number("04-0005")) is not None
    assert (await repository.get_by_nafdac_number("04-0006")) is None


# ------------------------------------------------------------------------- vector search
def unit_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


async def test_hnsw_nearest_neighbour(
    db_session: AsyncSession, migrated_settings: Settings
) -> None:
    dimension = migrated_settings.embeddings.image_dimension
    rng = random.Random(7)
    vectors = [unit_vector([rng.gauss(0, 1) for _ in range(dimension)]) for _ in range(40)]

    item = product("04-0411", "Emzor Paracetamol Tablet", "Paracetamol")
    db_session.add(item)
    await db_session.flush()
    images = [
        ReferenceImage(
            product_id=item.id, side=ImageSide.FRONT, s3_key=f"reference/{i}.jpg", embedding=vector
        )
        for i, vector in enumerate(vectors)
    ]
    db_session.add_all(images)
    await db_session.flush()

    query = [value + rng.gauss(0, 0.05) for value in vectors[17]]
    await db_session.execute(text("SET LOCAL hnsw.ef_search = 100"))
    nearest = (
        await db_session.scalars(
            select(ReferenceImage)
            .order_by(ReferenceImage.embedding.cosine_distance(query))
            .limit(3)
        )
    ).all()

    assert nearest[0].id == images[17].id
    assert await uses_index(
        db_session,
        "SELECT id FROM reference_images ORDER BY embedding <=> CAST(:q AS vector) LIMIT 3",
        {"q": str(query)},
        "ix_reference_images_embedding",
    )


async def test_embedding_dimension_is_enforced(
    db_session: AsyncSession, migrated_settings: Settings
) -> None:
    dimension = migrated_settings.embeddings.image_dimension
    item = product("04-0411", "Emzor Paracetamol Tablet", "Paracetamol")
    db_session.add(item)
    await db_session.flush()

    db_session.add(
        ReferenceImage(product_id=item.id, side=ImageSide.BACK, s3_key="x.jpg", embedding=[0.1] * 3)
    )
    with pytest.raises(DBAPIError, match=f"expected {dimension} dimensions"):
        await db_session.flush()


# ------------------------------------------------------------------------- field encryption
async def test_phone_is_encrypted_at_rest_and_found_by_hash(db_session: AsyncSession) -> None:
    user = User()
    user.set_phone("+2348031234567")
    db_session.add(user)
    await db_session.flush()

    stored = (
        await db_session.execute(
            text("SELECT phone_e164, phone_hash FROM users WHERE id = :id"), {"id": user.id}
        )
    ).one()
    assert stored.phone_e164.startswith(f"{field_cipher().active_key_id}:")
    assert "8031234567" not in stored.phone_e164
    assert stored.phone_hash == phone_lookup_hash("+2348031234567")

    db_session.expunge_all()
    found = (
        await db_session.scalars(
            select(User).where(User.phone_hash == phone_lookup_hash("+2348031234567"))
        )
    ).one()
    assert found.phone_e164 == "+2348031234567"


async def test_ciphertext_cannot_be_moved_to_another_column(db_session: AsyncSession) -> None:
    user = User(email="moved@medlens.test")
    user.set_phone("+2348031234567")
    db_session.add(user)
    await db_session.flush()

    # Copy the phone ciphertext into mfa_secret, as an attacker with SQL access might.
    await db_session.execute(
        text("UPDATE users SET mfa_secret = phone_e164 WHERE id = :id"), {"id": user.id}
    )
    db_session.expunge_all()

    with pytest.raises(DecryptionError, match=r"users\.mfa_secret"):
        await db_session.scalar(select(User.mfa_secret).where(User.id == user.id))


async def test_users_need_an_identity(db_session: AsyncSession) -> None:
    db_session.add(User())

    with pytest.raises(DBAPIError, match="ck_users_identity"):
        await db_session.flush()


# ------------------------------------------------------------------------- audit log
async def _audit_entry(session: AsyncSession) -> AuditLog:
    entry = AuditLog(action="auth.login_failed", metadata_={"reason": "bad_password"})
    session.add(entry)
    await session.flush()
    return entry


async def test_audit_log_accepts_inserts(db_session: AsyncSession) -> None:
    entry = await _audit_entry(db_session)

    assert entry.created_at is not None
    assert (await db_session.get(AuditLog, entry.id)) is not None


@pytest.mark.parametrize(
    "statement",
    ["UPDATE audit_logs SET action = 'tampered'", "DELETE FROM audit_logs", "TRUNCATE audit_logs"],
)
async def test_audit_log_rejects_changes_even_for_the_owner(
    db_session: AsyncSession, statement: str
) -> None:
    await _audit_entry(db_session)

    with pytest.raises(DBAPIError, match="audit_logs is append-only"):
        await db_session.execute(text(statement))


@pytest.mark.parametrize(
    "statement", ["UPDATE audit_logs SET action = 'x'", "DELETE FROM audit_logs"]
)
async def test_app_role_has_no_update_or_delete_on_audit_log(
    db_session: AsyncSession, statement: str
) -> None:
    entry = await _audit_entry(db_session)
    await db_session.execute(text("SET LOCAL ROLE medlens_app"))

    await db_session.execute(  # the app role may still insert and read
        text("INSERT INTO audit_logs (id, action) VALUES (:id, 'auth.logout')"), {"id": uuid7()}
    )
    assert (await db_session.get(AuditLog, entry.id)) is not None
    with pytest.raises(DBAPIError, match="permission denied for table audit_logs"):
        await db_session.execute(text(statement))


# ------------------------------------------------------------------------- scans and cases
async def test_scans_route_to_partitions_and_cases_reference_them(
    db_session: AsyncSession,
) -> None:
    now = datetime.now(UTC)
    current = Scan(
        verdict=Verdict.SUSPICIOUS,
        risk_score=0.72,
        location_coarse=WKTElement("POINT(3.379 6.524)", srid=4326),
        geocell="s14jgp",
        state="NG-LA",
    )
    far_future = Scan(created_at=datetime(2031, 1, 1, tzinfo=UTC))
    db_session.add_all([current, far_future])
    await db_session.flush()

    rows = await db_session.execute(
        text("SELECT id, tableoid::regclass::text FROM scans WHERE id IN (:a, :b)"),
        {"a": current.id, "b": far_future.id},
    )
    partitions = dict(rows.all())
    assert partitions[current.id] == f"scans_y{now.year}m{now.month:02d}"
    assert partitions[far_future.id] == "scans_default"

    case = Case(scan_id=current.id, scan_created_at=current.created_at)
    db_session.add(case)
    await db_session.flush()
    assert case.status == "open"

    with pytest.raises(DBAPIError, match="fk_cases_scan_id_scan_created_at_scans"):
        await db_session.execute(
            update(Scan).where(Scan.id == current.id).values(created_at=far_future.created_at)
        )


async def test_scan_scores_are_bounded(db_session: AsyncSession) -> None:
    db_session.add(Scan(risk_score=1.5))

    with pytest.raises(DBAPIError, match="ck_scans_risk_score"):
        await db_session.flush()

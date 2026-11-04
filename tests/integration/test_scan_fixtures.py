"""The M3 fixture packs end to end: real readers (OpenCV quality gate, zxing-cpp DataMatrix),
the vision LLM played by FakeLLM reading each pack's printed text, the real scorer and the
database. Images: tests/fixtures/packs (``scripts/make_test_packs.py``)."""

import csv
from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from medlens.adapters.fakes import FakeLLM
from medlens.adapters.vision import OpenCvImageAnalysis, ZxingBarcode
from medlens.db.session import UnitOfWork
from medlens.modules.products.models import Batch, Product, RegistrationStatus
from medlens.modules.rbac.models import RoleName
from tests.integration.api_support import Api

pytestmark = pytest.mark.integration

PACKS = Path(__file__).resolve().parents[1] / "fixtures" / "packs"
# The seed products and batches the fixtures print (seed/products.csv, seed/batches.csv).
PRODUCTS = {
    "04-0289": "Emzor Paracetamol Syrup",
    "04-8655": "Augmentin 1 G",
    "A4-2374": "ADALAT LA 30MG TABLETS",
}
BATCHES = {
    "SEED-04-0289-A": ("04-0289", date(2028, 1, 14)),
    "SEED-04-0289-B": ("04-0289", date(2025, 5, 31)),
    "SEED-04-8655-A": ("04-8655", date(2028, 1, 14)),
    "SEED-A4-2374-A": ("A4-2374", date(2028, 1, 14)),
}


def fixtures() -> list[dict[str, str]]:
    with (PACKS / "packs.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reading(pack: dict[str, str]) -> str:
    """What a vision model reads off the pack: its printed fields."""
    return FakeLLM.reading(
        confidence=0.95,
        nafdac_number=pack["nafdac_number"],
        product_name=pack["brand"],
        batch_number=pack["batch_number"],
        expiry_date=pack["expiry"],
    )


async def ensure_seed_products(api: Api) -> None:
    """Get-or-create: other tests may have run the seed on this shared database."""
    async with UnitOfWork(api.resources.session_factory) as uow:
        session = uow.session
        await session.execute(
            insert(Product)
            .values(
                [
                    {
                        "nafdac_number": number,
                        "brand_name": brand,
                        "generic_name": "Seeded for fixtures",
                        "registration_status": RegistrationStatus.ACTIVE,
                    }
                    for number, brand in PRODUCTS.items()
                ]
            )
            .on_conflict_do_nothing(index_elements=["nafdac_number"])
        )
        ids = dict(
            (
                await session.execute(
                    select(Product.nafdac_number, Product.id).where(
                        Product.nafdac_number.in_(PRODUCTS)
                    )
                )
            ).all()
        )
        await session.execute(
            insert(Batch)
            .values(
                [
                    {
                        "product_id": ids[number],
                        "batch_number": batch,
                        "expires_on": expires_on,
                        "distribution_states": ["NG-LA"],
                    }
                    for batch, (number, expires_on) in BATCHES.items()
                ]
            )
            .on_conflict_do_nothing(index_elements=["product_id", "batch_number"])
        )
        await uow.commit()


async def test_each_fixture_pack_gets_its_expected_verdict(api: Api) -> None:
    await ensure_seed_products(api)
    packs = fixtures()
    images = {pack["file"]: (PACKS / pack["file"]).read_bytes() for pack in packs}
    llm = FakeLLM()  # the API re-encodes uploads, so readings are scripted per scan
    worker = api.worker(readers=(OpenCvImageAnalysis(), ZxingBarcode()), llm=llm)
    regulator, organization = await api.user_with_role(RoleName.REGULATOR_ANALYST)
    staff = api.bearer(api.token(regulator, organization=organization))

    results = {}
    for pack in packs:
        headers = {"X-Device-Id": f"device-{uuid4().hex}"}
        submitted = await api.client.post(
            "/api/v1/scans",
            files={"image": (pack["file"], images[pack["file"]], "image/jpeg")},
            data={"state": "NG-LA"},
            headers=headers,
        )
        scan_id = submitted.json()["id"]
        llm.default = reading(pack)
        worker.process_scan(UUID(scan_id))
        results[pack["file"]] = (
            await api.client.get(f"/api/v1/scans/{scan_id}", headers=staff)
        ).json()

    verdicts = {name: result["verdict"] for name, result in results.items()}
    assert verdicts == {pack["file"]: pack["expected_verdict"] for pack in packs}
    for pack in packs:
        for reason in filter(None, pack["expected_reasons"].split(";")):
            assert reason in results[pack["file"]]["reason_codes"], pack["file"]

    genuine = results["genuine.jpg"]["analysis"]
    assert genuine["extracted"]["batch_number"]["source"] == "barcode"  # read by zxing
    assert genuine["llm_used"] is True  # the NAFDAC number is printed, not encoded
    no_barcode = results["no_barcode.jpg"]["analysis"]
    assert no_barcode["extracted"]["batch_number"]["source"] == "llm"
    blurry = results["blurry.jpg"]["analysis"]
    assert blurry["llm_used"] is False  # a bad photo never reaches the model
    assert results["blurry.jpg"]["advice_codes"][0] == "RETAKE_PHOTO"
    assert all(result["disclaimer_code"] for result in results.values())

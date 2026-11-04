"""Small Phase 5 building blocks: locations, ids, idempotency keys, device ids, fakes."""

import struct
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import Depends, FastAPI
from geoalchemy2 import WKBElement, WKTElement

from medlens.adapters.fakes import FakeLLM, InMemoryStorage, SyncView
from medlens.core.bootstrap import configure_process
from medlens.core.config import Settings
from medlens.core.devices import device_hash
from medlens.core.errors import ValidationFailed, register_exception_handlers
from medlens.core.geo import CELL_DEGREES, coarsen, geohash, point_coordinates
from medlens.core.idempotency import request_fingerprint, validate_key
from medlens.core.ids import uuid7, uuid7_time
from medlens.ports.llm import LLMUnavailableError
from medlens.ports.storage import ObjectNotFoundError


# ------------------------------------------------------------------------- locations
def test_points_snap_to_cell_centres() -> None:
    a = coarsen(6.524379, 3.379206)
    b = coarsen(6.5249, 3.3751)  # the same ~1 km cell

    assert (a.latitude, a.longitude) == (6.525, 3.375)
    assert a == b
    assert abs(a.latitude - 6.524379) <= CELL_DEGREES
    assert a.geocell == geohash(6.525, 3.375)
    assert len(a.geocell) == 6
    assert a.wkt == "SRID=4326;POINT(3.375 6.525)"


@pytest.mark.parametrize(
    ("latitude", "longitude", "expected"),
    [(57.64911, 10.40744, "u4pruy"), (-25.382708, -49.265506, "6gkzwg"), (0.0, 0.0, "s00000")],
)
def test_geohash_matches_the_reference_algorithm(
    latitude: float, longitude: float, expected: str
) -> None:
    assert geohash(latitude, longitude) == expected


def test_out_of_range_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError, match="latitude"):
        coarsen(91.0, 3.0)


def test_point_coordinates_read_ewkb_and_wkt() -> None:
    ewkb = struct.pack("<BIIdd", 1, 1 | 0x20000000, 4326, 3.3515, 6.6018)
    big_endian_wkb = struct.pack(">BIdd", 0, 1, 3.3515, 6.6018)

    assert point_coordinates(WKBElement(ewkb, srid=4326, extended=True)) == (6.6018, 3.3515)
    assert point_coordinates(big_endian_wkb) == (6.6018, 3.3515)
    assert point_coordinates(ewkb.hex()) == (6.6018, 3.3515)
    assert point_coordinates(WKTElement("POINT(3.3515 6.6018)", srid=4326)) == (6.6018, 3.3515)
    assert point_coordinates(None) is None
    with pytest.raises(ValueError, match="not a point"):
        point_coordinates(struct.pack("<BI", 1, 3) + b"\0" * 16)
    with pytest.raises(TypeError):
        point_coordinates(42)


# ------------------------------------------------------------------------- ids
def test_uuid7_time_is_the_creation_time() -> None:
    before = datetime.now(UTC).replace(microsecond=0)

    made = uuid7_time(uuid7())

    assert before <= made <= datetime.now(UTC)
    with pytest.raises(ValueError, match="UUIDv7"):
        uuid7_time(uuid4())


# ------------------------------------------------------------------------- idempotency
@pytest.mark.parametrize("key", ["short", "x" * 129, "has space here", "semi;colon-key"])
def test_bad_idempotency_keys_are_rejected(key: str) -> None:
    with pytest.raises(ValidationFailed):
        validate_key(key)


def test_fingerprints_separate_fields() -> None:
    assert request_fingerprint("ab", "c") != request_fingerprint("a", "bc")
    assert request_fingerprint("a", None) == request_fingerprint("a", b"")
    assert validate_key("retry-0190a8d4:1") == "retry-0190a8d4:1"


# ------------------------------------------------------------------------- device ids
async def test_device_ids_are_hashed_and_validated(settings: Settings) -> None:
    configure_process(settings)
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/")
    def echo(device: Annotated[str | None, Depends(device_hash)]) -> dict[str, str | None]:
        return {"device": device}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        good = await client.get(
            "/", headers={"X-Device-Id": "3f1c2b5e-9a7d-4e21-8c3b-1d2e3f4a5b6c"}
        )
        absent = await client.get("/")
        bad = await client.get("/", headers={"X-Device-Id": "short"})

    assert len(good.json()["device"]) == 64
    assert "3f1c2b5e" not in good.json()["device"]
    assert absent.json() == {"device": None}
    assert bad.status_code == 422


# ------------------------------------------------------------------------- fakes
async def test_in_memory_storage_serves_both_ports() -> None:
    storage = InMemoryStorage()
    sync = SyncView(storage)

    await storage.put("a", b"1", content_type="image/jpeg")
    sync.put("b", b"2", content_type="image/jpeg")

    assert await storage.get("b") == b"2"
    assert sync.get("a") == b"1"
    await storage.delete("a")
    with pytest.raises(ObjectNotFoundError):
        sync.get("a")


def test_an_unscripted_llm_is_unavailable() -> None:
    llm = FakeLLM()

    with pytest.raises(LLMUnavailableError):
        llm.extract_pack_fields(b"x", fields=["nafdac_number"])
    assert llm.calls == [("nafdac_number",)]


def test_scan_ids_round_trip_through_their_time() -> None:
    scan_id = uuid7()

    assert UUID(str(scan_id)) == scan_id
    assert uuid7_time(scan_id).tzinfo is UTC

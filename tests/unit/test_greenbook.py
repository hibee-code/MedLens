"""Greenbook snapshots: cleaning register rows, checksummed files, the committed sample."""

import csv
import gzip
import hashlib
import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from medlens.core.identifiers import normalize_nafdac_number
from medlens.modules.products.models import RegistrationStatus
from medlens.seed import greenbook
from medlens.seed.greenbook import (
    COLUMNS,
    Manifest,
    SnapshotError,
    clean,
    decode_snapshot,
    deduplicate,
    encode_snapshot,
    export,
    fetch_register,
    fetch_source,
    normalize_row,
    product_values,
    sample,
)

ROOT = Path(__file__).resolve().parents[2]
FETCHED = date(2026, 11, 3)


def raw(**overrides: object) -> dict[str, Any]:
    """A register row as the Greenbook endpoint returns it."""
    row: dict[str, Any] = {
        "NAFDAC": "04-0005",
        "product_name": "Panadol Extra Caplet",
        "ingredient_name": "Paracetamol, Caffeine",
        "strength": "500 mg/65 mg",
        "form_name": "Tablet",
        "pack_size": "1 x 10",
        "applicant_name": "GlaxoSmithKline Consumer Nigeria PLC",
        "category_name": "Drugs",
        "status": "Active",
        "expiry_date": "2029-01-31",
        "product_id": 17,
    }
    return row | overrides


def record(**overrides: object) -> dict[str, str]:
    result = normalize_row(raw(**overrides), FETCHED)
    assert result is not None
    return result


def manifest_for(data: bytes, rows: int) -> Manifest:
    return Manifest(
        file="snapshot.csv.gz",
        sha256=hashlib.sha256(data).hexdigest(),
        rows=rows,
        fetched_on=FETCHED.isoformat(),
    )


# ------------------------------------------------------------------------- cleaning
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ADALAT LA 30MG TABLETS $", "ADALAT LA 30MG TABLETS"),
        ("BW OMEPRAZOLE $2", "BW OMEPRAZOLE"),
        ("Vamoxil Suspension_", "Vamoxil Suspension"),
        ("KRISPINE CAPLET HAEMOGLOBIN_(updates needed)", "KRISPINE CAPLET HAEMOGLOBIN"),
        ("Amoxil* 500 # (check dosage form)", "Amoxil 500"),
        ("Fe&amp;Folic  \n Tablets", "Fe&Folic Tablets"),
        ("Co-trimoxazole_480", "Co-trimoxazole_480"),
        (None, ""),
    ],
)
def test_clean_removes_register_markers(value: object, expected: str) -> None:
    assert clean(value) == expected


def test_rows_become_snapshot_records() -> None:
    result = record(NAFDAC=" a4-2374 ")

    assert result["nafdac_number"] == "A4-2374"
    assert result["registration_status"] == "active"
    assert result["category"] == "Drugs"
    assert result["fetched_on"] == "2026-11-03"
    assert list(result) == list(COLUMNS)


@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        ({"expiry_date": "2026-11-02"}, "expired"),
        ({"status": "Inactive"}, "suspended"),
        ({"status": "Inactive", "expiry_date": "2020-01-01"}, "expired"),
        ({"expiry_date": ""}, "active"),
    ],
)
def test_status_is_derived_as_of_the_fetch_date(overrides: dict[str, object], status: str) -> None:
    assert record(**overrides)["registration_status"] == status


@pytest.mark.parametrize(
    "overrides",
    [{"NAFDAC": "not a number"}, {"NAFDAC": ""}, {"product_name": " * "}, {"expiry_date": "soon"}],
)
def test_unusable_rows_are_dropped(overrides: dict[str, object]) -> None:
    assert normalize_row(raw(**overrides), FETCHED) is None


def test_a_missing_generic_name_falls_back_to_the_brand() -> None:
    assert record(ingredient_name="")["generic_name"] == "Panadol Extra Caplet"


def test_duplicates_keep_the_active_entry_with_the_latest_expiry() -> None:
    expired = record(expiry_date="2025-01-01", product_id=1)
    active_soon = record(expiry_date="2027-01-01", product_id=2)
    active_late = record(expiry_date="2028-01-01", product_id=3)
    other = record(NAFDAC="A4-1000", product_id=4)

    kept = deduplicate([expired, active_soon, other, active_late])

    assert [r["greenbook_product_id"] for r in kept] == ["3", "4"]  # sorted by NAFDAC number


# ------------------------------------------------------------------------- snapshot files
def test_snapshots_are_deterministic_and_round_trip() -> None:
    records = [record(), record(NAFDAC="A4-1000")]

    data = encode_snapshot(records)

    assert data == encode_snapshot(records)  # no gzip timestamp
    assert decode_snapshot(data, manifest_for(data, 2)) == records


def test_a_tampered_snapshot_is_refused() -> None:
    data = encode_snapshot([record()])
    manifest = manifest_for(data, 1)
    tampered = encode_snapshot([record(product_name="Fake Panadol")])

    with pytest.raises(SnapshotError, match="checksum mismatch"):
        decode_snapshot(tampered, manifest)


def test_a_row_count_mismatch_is_refused() -> None:
    data = encode_snapshot([record()])

    with pytest.raises(SnapshotError, match="has 1 rows, expected 2"):
        decode_snapshot(data, manifest_for(data, 2))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"plain text", "not a gzipped UTF-8 CSV"),
        (gzip.compress(b"a,b\n1,2\n", mtime=0), "snapshot columns"),
    ],
)
def test_malformed_snapshots_are_refused(data: bytes, message: str) -> None:
    with pytest.raises(SnapshotError, match=message):
        decode_snapshot(data, manifest_for(data, 1))


def test_manifests_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = Manifest(file="f.csv.gz", sha256="0" * 64, rows=3, fetched_on="2026-11-03")

    manifest.write(path)

    assert Manifest.read(path) == manifest
    with pytest.raises(SnapshotError, match="cannot read"):
        Manifest.read(tmp_path / "missing.json")


# ------------------------------------------------------------------------- sources
def test_sources_can_be_local_paths(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.csv.gz"
    path.write_bytes(b"bytes")

    assert fetch_source(str(path)) == b"bytes"
    assert fetch_source(path.as_uri()) == b"bytes"
    with pytest.raises(SnapshotError, match="cannot read"):
        fetch_source(str(tmp_path / "missing.csv.gz"))


def test_sources_can_be_https(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def get(url: str, params: object = None) -> httpx.Response:
        seen.append(url)
        return httpx.Response(200, content=b"remote")

    monkeypatch.setattr(greenbook, "_http_get", get)

    assert fetch_source("https://files.example/snapshot.csv.gz") == b"remote"
    assert seen == ["https://files.example/snapshot.csv.gz"]


class FakeS3:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803 - boto3 API
        self.calls.append((Bucket, Key))
        return {"Body": io.BytesIO(b"stored")}


def test_sources_can_be_object_storage() -> None:
    s3 = FakeS3()

    assert fetch_source("s3://reference-data/greenbook/2026.csv.gz", s3_client=s3) == b"stored"
    assert s3.calls == [("reference-data", "greenbook/2026.csv.gz")]
    with pytest.raises(SnapshotError, match="needs object storage"):
        fetch_source("s3://reference-data/x")


def test_unsupported_sources_are_refused() -> None:
    with pytest.raises(SnapshotError, match="unsupported"):
        fetch_source("ftp://example.org/snapshot.csv.gz")


def test_http_get_retries_transport_errors_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def flaky(url: str, **kwargs: object) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("reset")
        return httpx.Response(200, request=httpx.Request("GET", url), content=b"ok")

    monkeypatch.setattr(httpx, "get", flaky)
    monkeypatch.setattr(greenbook._http_get.retry, "sleep", lambda seconds: None)  # type: ignore[attr-defined]

    assert greenbook._http_get("https://example.org").content == b"ok"
    assert calls == 2


# ------------------------------------------------------------------------- export
def test_the_register_is_fetched_page_by_page(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = {0: [raw(), raw()], 2: [raw()]}
    starts: list[int] = []

    def get(url: str, params: dict[str, int]) -> httpx.Response:
        starts.append(params["start"])
        body = {"recordsTotal": 3, "data": pages.get(params["start"], [])}
        return httpx.Response(200, json=body)

    monkeypatch.setattr(greenbook, "_http_get", get)
    monkeypatch.setattr(greenbook.time, "sleep", lambda seconds: None)

    assert len(list(fetch_register(page_size=2))) == 3
    assert starts == [0, 2]


def test_an_empty_page_ends_the_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        greenbook, "_http_get", lambda url, params: httpx.Response(200, json={"recordsTotal": 9})
    )

    assert list(fetch_register()) == []


def test_export_writes_a_snapshot_and_its_manifest(tmp_path: Path) -> None:
    rows = [raw(), raw(product_id=2), raw(NAFDAC="bad"), raw(NAFDAC="A4-1000")]

    result = export(iter(rows), tmp_path, FETCHED)

    data = result.path.read_bytes()
    assert result.path.name == "greenbook-2026-11-03.csv.gz"
    assert (result.fetched, result.skipped, result.duplicates) == (4, ["bad"], 1)
    assert result.manifest.rows == 2
    assert result.manifest.sha256 == hashlib.sha256(data).hexdigest()
    assert len(decode_snapshot(data, result.manifest)) == 2


# ------------------------------------------------------------------------- sample
def test_the_sample_takes_the_closest_registered_drug_per_brand() -> None:
    records = [
        record(product_name="Panadol Extra Soluble", NAFDAC="A4-0001"),
        record(product_name="Panadol Extra", NAFDAC="A4-0002"),
        record(product_name="Panadol Extra", NAFDAC="A4-0003", category_name="Veterinary"),
        record(product_name="Coartem 80/480", NAFDAC="A4-0004"),
    ]

    picked = sample(records)

    assert [r["nafdac_number"] for r in picked] == ["A4-0004", "A4-0002"]
    assert len(sample(records, size=1)) == 1


def test_product_values_validate_the_status() -> None:
    values = product_values(record())

    assert values["registration_status"] is RegistrationStatus.ACTIVE
    assert values["registration_expires_on"] == date(2029, 1, 31)
    with pytest.raises(ValueError, match="not-a-status"):
        product_values(record() | {"registration_status": "not-a-status"})


# ------------------------------------------------------------------------- CLI
def test_cli_export_writes_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(greenbook, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(greenbook, "fetch_register", lambda: iter([raw()]))

    assert greenbook.main(["export", "--out", str(tmp_path / "out")]) == 0

    manifest = Manifest.read(tmp_path / "manifest.json")
    assert manifest.rows == 1
    assert (tmp_path / "out" / manifest.file).exists()
    assert manifest.sha256 in capsys.readouterr().out


def test_cli_sample_verifies_the_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = encode_snapshot([record(product_name="Coartem 20/120")])
    snapshot = tmp_path / "snapshot.csv.gz"
    snapshot.write_bytes(data)
    manifest_for(data, 1).write(tmp_path / "manifest.json")
    monkeypatch.setattr(greenbook, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(greenbook, "SAMPLE_PATH", tmp_path / "products.csv")

    assert greenbook.main(["sample", "--snapshot", str(snapshot)]) == 0
    snapshot.write_bytes(encode_snapshot([record(product_name="Tampered")]))
    assert greenbook.main(["sample", "--snapshot", str(snapshot)]) == 1

    with (tmp_path / "products.csv").open() as handle:
        assert [row["brand_name"] for row in csv.DictReader(handle)] == ["Coartem 20/120"]


def test_cli_import_needs_a_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    Manifest(file="f.csv.gz", sha256="0" * 64, rows=1, fetched_on="2026-11-03").write(
        tmp_path / "manifest.json"
    )
    monkeypatch.setattr(greenbook, "MANIFEST_PATH", tmp_path / "manifest.json")

    assert greenbook.main(["import"]) == 1
    assert "no location yet" in capsys.readouterr().err


# ------------------------------------------------------------------------- committed files
def test_the_committed_manifest_is_valid() -> None:
    manifest = Manifest.read()

    assert len(manifest.sha256) == 64
    assert manifest.file == f"greenbook-{manifest.fetched_on}.csv.gz"
    assert manifest.location is None or manifest.location.startswith(("s3://", "https://"))


def test_the_committed_sample_is_clean_and_small() -> None:
    with greenbook.SAMPLE_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == COLUMNS
    assert len(rows) == greenbook.SAMPLE_SIZE
    for row in rows:
        assert row["nafdac_number"] == normalize_nafdac_number(row["nafdac_number"])
        assert row["category"] == "Drugs"
        assert clean(row["brand_name"]) == row["brand_name"]
        product_values(row)  # valid status and dates


def test_the_sample_matches_the_snapshot_when_it_is_available() -> None:
    manifest = Manifest.read()
    snapshot = greenbook.DEFAULT_EXPORT_DIR / manifest.file
    if not snapshot.exists():
        pytest.skip(f"{manifest.file} is not downloaded (kept outside Git)")
    records = decode_snapshot(snapshot.read_bytes(), manifest)
    with greenbook.SAMPLE_PATH.open(newline="", encoding="utf-8") as handle:
        committed = list(csv.DictReader(handle))

    assert committed == sample(records)


def test_the_provenance_file_states_source_date_and_terms() -> None:
    text = (ROOT / "seed" / "PROVENANCE.md").read_text()
    manifest = json.loads((ROOT / "seed" / "greenbook-snapshot.json").read_text())

    assert "https://greenbook.nafdac.gov.ng/" in text
    assert manifest["fetched_on"] in text
    assert "## Terms" in text
    assert f"{manifest['rows']:,}" in text

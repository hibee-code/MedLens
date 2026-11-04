"""NAFDAC Greenbook snapshots: export the public register, derive the committed sample, and import
a full snapshot into ``products``.

    uv run python -m medlens.seed.greenbook export [--out data/greenbook]  # network, about 1 min
    uv run python -m medlens.seed.greenbook sample --snapshot FILE          # seed/products.csv
    uv run python -m medlens.seed.greenbook import [--source PATH | s3://... | https://...]

Only a ~50-row sample (seed/products.csv) is committed. Full snapshots are kept outside Git, in
private object storage: seed/greenbook-snapshot.json names the current one and its SHA-256, and
the importer refuses any file that does not match it. Source, date and terms: seed/PROVENANCE.md.

The register is untrusted input: every NAFDAC number must pass our own normaliser, HTML entities
are decoded, text is whitespace-normalised, and the registry's display markers (*, #, $, $1, $2
and trailing underscores) and internal review notes such as "(check dosage form)" or
"(updates needed)" are removed. Registration status (the
Greenbook only says Active or Inactive) is derived as of the fetch date:

    expiry date before the fetch date        -> expired
    otherwise Active                         -> active
    otherwise (Inactive, not yet expired)    -> suspended
"""

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
import re
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from medlens.core.identifiers import normalize_nafdac_number
from medlens.modules.products.models import Product, RegistrationStatus

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "seed" / "greenbook-snapshot.json"
SAMPLE_PATH = ROOT / "seed" / "products.csv"
DEFAULT_EXPORT_DIR = ROOT / "data" / "greenbook"  # gitignored

GREENBOOK_URL = "https://greenbook.nafdac.gov.ng/"
USER_AGENT = "MedLens-greenbook-snapshot/1.0 (registered-products reference data)"
PAGE_SIZE = 250
REQUEST_INTERVAL_SECONDS = 1.0  # be polite to a public service
UPSERT_CHUNK = 1000

COLUMNS = (
    "nafdac_number",
    "brand_name",
    "generic_name",
    "strength",
    "dosage_form",
    "pack_size",
    "manufacturer",
    "category",
    "registration_status",
    "registration_expires_on",
    "greenbook_product_id",
    "fetched_on",
)

# Widely sold brands and generics, ending with medicines often found counterfeit in Nigeria;
# the sample takes the best registered drug for each.
SAMPLE_SIZE = 50
SAMPLE_BRANDS = (
    "Panadol Extra", "Panadol Advance", "Emzor Paracetamol", "Coartem", "Lonart", "Amatem",
    "Lumartem", "P-Alaxin", "Camosunate", "Artequin", "Augmentin", "Amoxil", "Ampiclox",
    "Ciprotab", "Ciprinol", "Flagyl", "Septrin", "Zinnat", "Rocephin", "Zithromax", "Postinor",
    "Ventolin", "Buscopan", "Feldene", "Voltaren", "Cataflam", "Brufen", "Loxagyl",
    "Glucophage", "Daonil", "Norvasc", "Adalat", "Aldomet", "Moduretic", "Zestril", "Lasix",
    "Procold", "Coldcap", "Tutolin", "Benylin", "Andrews Liver Salt", "Alabukun", "Vermox",
    "Ketoconazole", "Fluconazole", "Ibuprofen", "Metformin", "Amlodipine", "Lisinopril",
    "Vitamin C", "Folic Acid", "Ferrous Sulphate", "Omeprazole", "Gestid", "Flagentyl",
    "Diclofenac", "Chloroquine", "Quinine", "Mebendazole", "Albendazole", "Artesunate",
    "Sulfadoxine", "Oxytocin", "Ceftriaxone", "Amoxicillin", "Ciprofloxacin", "Metronidazole",
    "Oral Rehydration",
)  # fmt: skip

Record = dict[str, str]


class SnapshotError(Exception):
    """A snapshot is missing, does not match its manifest, or cannot be parsed."""


@dataclass(frozen=True, slots=True)
class Manifest:
    """seed/greenbook-snapshot.json: which full snapshot is current."""

    file: str
    sha256: str
    rows: int
    fetched_on: str
    source: str = GREENBOOK_URL
    # Where the file is stored (s3://bucket/key or an https URL); set after uploading it.
    location: str | None = None

    @classmethod
    def read(cls, path: Path | None = None) -> "Manifest":
        path = path or MANIFEST_PATH
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            raise SnapshotError(f"cannot read {path.name}: {exc}") from None

    def write(self, path: Path | None = None) -> None:
        (path or MANIFEST_PATH).write_text(json.dumps(asdict(self), indent=2) + "\n", "utf-8")


# --------------------------------------------------------------------------- normalising
_REVIEW_NOTE = re.compile(r"\s*\((?:check|updates? needed)[^)]*\)?", re.IGNORECASE)
_MARKERS = re.compile(r"[*#]|\$\d*|_+(?=\s|$)")


def clean(value: object) -> str:
    text = _REVIEW_NOTE.sub("", html.unescape(str(value or "")))
    return re.sub(r"\s+", " ", _MARKERS.sub("", text)).strip()


def registration_status(row: dict[str, Any], fetched_on: date) -> str:
    expiry = clean(row.get("expiry_date"))[:10]
    if expiry and date.fromisoformat(expiry) < fetched_on:
        return "expired"
    return "active" if clean(row.get("status")).lower() == "active" else "suspended"


def normalize_row(row: dict[str, Any], fetched_on: date) -> Record | None:
    """One register row as a snapshot record, or None when it has no valid NAFDAC number."""
    try:
        nafdac = normalize_nafdac_number(clean(row.get("NAFDAC")))
        status = registration_status(row, fetched_on)
    except ValueError:
        return None
    brand = clean(row.get("product_name"))
    if not brand:
        return None
    return {
        "nafdac_number": nafdac,
        "brand_name": brand,
        "generic_name": clean(row.get("ingredient_name")) or brand,
        "strength": clean(row.get("strength")),
        "dosage_form": clean(row.get("form_name")),
        "pack_size": clean(row.get("pack_size")),
        "manufacturer": clean(row.get("applicant_name")),
        "category": clean(row.get("category_name")),
        "registration_status": status,
        "registration_expires_on": clean(row.get("expiry_date"))[:10],
        "greenbook_product_id": clean(row.get("product_id")),
        "fetched_on": fetched_on.isoformat(),
    }


_STATUS_RANK = {"active": 0, "suspended": 1, "expired": 2}


def deduplicate(records: list[Record]) -> list[Record]:
    """One record per NAFDAC number (the products table's key), sorted by it. When the register
    lists a number twice, the active entry with the latest expiry wins."""
    best: dict[str, Record] = {}
    for record in records:
        key = record["nafdac_number"]
        current = best.get(key)
        if current is None or _preference(record) < _preference(current):
            best[key] = record
    return [best[key] for key in sorted(best)]


def _preference(record: Record) -> tuple[int, int, int]:
    """Smaller is better: active first, then the latest expiry, then the oldest entry."""
    expires = record["registration_expires_on"]
    expiry_day = date.fromisoformat(expires).toordinal() if expires else 0
    return (
        _STATUS_RANK.get(record["registration_status"], len(_STATUS_RANK)),
        -expiry_day,
        int(record["greenbook_product_id"] or 0),
    )


# --------------------------------------------------------------------------- snapshot files
def encode_snapshot(records: Sequence[Record]) -> bytes:
    """Gzipped CSV. Deterministic (no gzip timestamp), so equal data gives an equal checksum."""
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(text.getvalue().encode("utf-8"))
    return buffer.getvalue()


def decode_snapshot(data: bytes, manifest: Manifest) -> list[Record]:
    """Verify ``data`` against the manifest (checksum, then row count) and parse it."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest.sha256:
        raise SnapshotError(
            f"checksum mismatch for {manifest.file}: expected {manifest.sha256}, got {digest}"
        )
    try:
        text = gzip.decompress(data).decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SnapshotError(f"{manifest.file} is not a gzipped UTF-8 CSV: {exc}") from None
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise SnapshotError(f"{manifest.file} does not have the snapshot columns")
    records = list(reader)
    if len(records) != manifest.rows:
        raise SnapshotError(f"{manifest.file} has {len(records)} rows, expected {manifest.rows}")
    return records


def fetch_source(location: str, *, s3_client: Any = None) -> bytes:
    """Bytes of a snapshot from a local path, ``s3://bucket/key`` or an https URL."""
    parsed = urlparse(location)
    if parsed.scheme == "s3":
        if s3_client is None:
            raise SnapshotError("an s3:// source needs object storage settings")
        response = s3_client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        return bytes(response["Body"].read())
    if parsed.scheme == "https":
        response = _http_get(location)
        return response.content
    if parsed.scheme in {"", "file"}:
        path = Path(parsed.path if parsed.scheme == "file" else location)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise SnapshotError(f"cannot read {path}: {exc.strerror}") from None
    raise SnapshotError(f"unsupported snapshot source {parsed.scheme}://")


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    reraise=True,
)
def _http_get(url: str, params: dict[str, str | int] | None = None) -> httpx.Response:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json, */*"}
    if params is not None:
        headers["X-Requested-With"] = "XMLHttpRequest"  # the register's JSON endpoint
    response = httpx.get(url, params=params, headers=headers, timeout=30.0)
    response.raise_for_status()
    return response


# --------------------------------------------------------------------------- export
def fetch_register(*, page_size: int = PAGE_SIZE) -> Iterator[dict[str, Any]]:
    """Every row of the public register, page by page."""
    start, total = 0, None
    while total is None or start < total:
        params: dict[str, str | int] = {
            "draw": 1,
            "start": start,
            "length": page_size,
            "search[value]": "",
            "columns[0][data]": "product_name",
            "columns[0][name]": "product_name",
            "columns[0][searchable]": "true",
        }
        body = _http_get(GREENBOOK_URL, params).json()
        total = int(body["recordsTotal"])
        rows: list[dict[str, Any]] = body.get("data") or []
        if not rows:
            break
        yield from rows
        start += len(rows)
        time.sleep(REQUEST_INTERVAL_SECONDS)


@dataclass(frozen=True, slots=True)
class ExportResult:
    path: Path
    manifest: Manifest
    fetched: int
    skipped: list[str]  # NAFDAC values of rows dropped as invalid
    duplicates: int


def export(rows: Iterator[dict[str, Any]], out_dir: Path, fetched_on: date) -> ExportResult:
    fetched = 0
    skipped: list[str] = []
    records: list[Record] = []
    for row in rows:
        fetched += 1
        record = normalize_row(row, fetched_on)
        if record is None:
            skipped.append(clean(row.get("NAFDAC")))
        else:
            records.append(record)
    unique = deduplicate(records)
    data = encode_snapshot(unique)
    name = f"greenbook-{fetched_on.isoformat()}.csv.gz"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_bytes(data)
    manifest = Manifest(
        file=name,
        sha256=hashlib.sha256(data).hexdigest(),
        rows=len(unique),
        fetched_on=fetched_on.isoformat(),
    )
    return ExportResult(path, manifest, fetched, skipped, len(records) - len(unique))


# --------------------------------------------------------------------------- sample
def sample(records: Sequence[Record], size: int = SAMPLE_SIZE) -> list[Record]:
    """The best registered drug for each of SAMPLE_BRANDS, up to ``size``, sorted by brand."""
    drugs = [record for record in records if record["category"] == "Drugs"]
    picked: dict[str, Record] = {}
    for brand in SAMPLE_BRANDS:
        if len(picked) == size:
            break
        needle = brand.lower()
        matches = [
            record
            for record in drugs
            if needle in record["brand_name"].lower() and record["nafdac_number"] not in picked
        ]
        if matches:
            # The closest name wins ("Panadol Extra" over "Panadol Extra Soluble").
            best = min(matches, key=lambda r: (len(r["brand_name"]), _preference(r)))
            picked[best["nafdac_number"]] = best
    return sorted(picked.values(), key=lambda record: record["brand_name"].lower())


def write_sample(records: Sequence[Record], path: Path | None = None) -> None:
    with (path or SAMPLE_PATH).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


# --------------------------------------------------------------------------- import
def product_values(record: Record) -> dict[str, object]:
    """Columns of ``products`` from a snapshot or sample record (ValueError when invalid)."""
    expires = record["registration_expires_on"]
    return {
        "nafdac_number": normalize_nafdac_number(record["nafdac_number"]),
        "brand_name": record["brand_name"],
        "generic_name": record["generic_name"],
        "strength": record["strength"] or None,
        "dosage_form": record["dosage_form"] or None,
        "pack_size": record["pack_size"] or None,
        "manufacturer_name": record["manufacturer"] or None,
        "registration_status": RegistrationStatus(record["registration_status"]),
        "registration_expires_on": date.fromisoformat(expires) if expires else None,
    }


def upsert_products(session: Session, products: Sequence[dict[str, object]]) -> int:
    """Insert or update products by NAFDAC number in the caller's transaction; returns how many
    were new. Only the columns present in the rows change, so a register import never touches
    ``manufacturer_org_id`` (set by approved manufacturer claims)."""
    if not products:
        return 0
    before = session.scalar(select(func.count()).select_from(Product)) or 0
    statement = insert(Product)
    statement = statement.on_conflict_do_update(
        index_elements=[Product.nafdac_number],
        set_={
            column: statement.excluded[column]
            for column in products[0]
            if column != "nafdac_number"
        }
        | {"updated_at": func.now()},
    )
    for start in range(0, len(products), UPSERT_CHUNK):
        session.execute(statement, list(products[start : start + UPSERT_CHUNK]))
    after = session.scalar(select(func.count()).select_from(Product)) or 0
    return after - before


def import_snapshot(session: Session, data: bytes, manifest: Manifest) -> tuple[int, int]:
    """Verify and load a full snapshot; returns (new, updated)."""
    products = [product_values(record) for record in decode_snapshot(data, manifest)]
    new = upsert_products(session, products)
    return new, len(products) - new


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NAFDAC Greenbook snapshots.")
    commands = parser.add_subparsers(dest="command", required=True)
    export_cmd = commands.add_parser("export", help="fetch the full register (network)")
    export_cmd.add_argument("--out", type=Path, help=f"default: {DEFAULT_EXPORT_DIR}")
    sample_cmd = commands.add_parser("sample", help="rewrite seed/products.csv from a snapshot")
    sample_cmd.add_argument("--snapshot", required=True, help="path, s3:// or https:// source")
    import_cmd = commands.add_parser("import", help="load the current snapshot into products")
    import_cmd.add_argument("--source", help="path, s3:// or https:// (default: the manifest)")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            return _export(args.out or DEFAULT_EXPORT_DIR)
        if args.command == "sample":
            return _sample(args.snapshot)
        return _import(args.source)
    except SnapshotError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1


def _export(out_dir: Path) -> int:
    fetched_on = date.today()  # noqa: DTZ011 - a calendar date for the snapshot
    result = export(fetch_register(), out_dir, fetched_on)
    result.manifest.write()
    sys.stdout.write(
        f"wrote {result.path} ({result.manifest.rows} products; {len(result.skipped)} invalid "
        f"rows and {result.duplicates} duplicates dropped of {result.fetched})\n"
        f"invalid NAFDAC numbers: {', '.join(repr(value) for value in result.skipped)}\n"
        f"sha256 {result.manifest.sha256}\n"
        f"updated {MANIFEST_PATH.name}. Next: upload the file to private object "
        'storage, set "location" in the manifest, and commit the manifest.\n'
    )
    return 0


def _sample(source: str) -> int:
    records = decode_snapshot(_read_source(source), Manifest.read())
    picked = sample(records)
    write_sample(picked)
    sys.stdout.write(f"wrote {len(picked)} products to {SAMPLE_PATH.name}\n")
    return 0


def _import(source: str | None) -> int:
    from medlens.core.bootstrap import configure_process
    from medlens.core.config import get_settings
    from medlens.db.session import (
        create_sync_engine,
        create_sync_session_factory,
        sync_session_scope,
    )

    manifest = Manifest.read()
    location = source or manifest.location
    if not location:
        raise SnapshotError(
            f"{MANIFEST_PATH.name} has no location yet; pass --source PATH|s3://...|https://..."
        )
    settings = get_settings()
    configure_process(settings)
    data = _read_source(location)
    engine = create_sync_engine(settings.database, application_name="medlens-greenbook-import")
    try:
        with sync_session_scope(create_sync_session_factory(engine)) as session:
            new, updated = import_snapshot(session, data, manifest)
    finally:
        engine.dispose()
    sys.stdout.write(
        f"imported {manifest.file} (fetched {manifest.fetched_on}): {new} new products, "
        f"{updated} updated\n"
    )
    return 0


def _read_source(location: str) -> bytes:
    if urlparse(location).scheme != "s3":
        return fetch_source(location)
    from medlens.adapters.s3 import create_sync_s3_client
    from medlens.core.config import get_settings

    return fetch_source(location, s3_client=create_sync_s3_client(get_settings().s3))


if __name__ == "__main__":
    sys.exit(main())

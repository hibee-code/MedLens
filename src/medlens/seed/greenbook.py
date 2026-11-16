"""NAFDAC Greenbook snapshots: export the public register, derive the committed sample, and import
a full snapshot into ``products``.

    uv run python -m medlens.seed.greenbook export [--out data/greenbook]  # network, about 1 min
    uv run python -m medlens.seed.greenbook sample --snapshot FILE          # seed/products.csv
    uv run python -m medlens.seed.greenbook import [--source PATH | s3://... | https://...]
                                       # make greenbook-import-snapshot; hand-typed rows kept

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
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from medlens.core.identifiers import normalize_nafdac_number
from medlens.modules.products.models import Product, RegisterSource, RegistrationStatus
from medlens.modules.products.register import parse_ingredients

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
    """Columns of ``products`` from a snapshot, sample or manual register record (ValueError
    when the NAFDAC number, status or a date is invalid). Active ingredients are parsed from
    generic_name and strength; a pair that does not parse leaves the list empty here (a full
    snapshot must not fail on one odd row), while the manual import refuses it (register_import).
    Rows without a source_url come from the Greenbook itself; retrieved_on falls back to the
    snapshot's fetched_on. GTINs are not product columns (products.ProductGtin): only the
    hand-typed import sets them."""
    expires = record["registration_expires_on"]
    registered = record.get("registration_date") or ""
    retrieved = record.get("retrieved_on") or record.get("fetched_on") or ""
    try:
        ingredients = [
            item.as_json() for item in parse_ingredients(record["generic_name"], record["strength"])
        ]
    except ValueError:
        ingredients = []
    values: dict[str, object] = {
        "nafdac_number": normalize_nafdac_number(record["nafdac_number"]),
        "brand_name": record["brand_name"],
        "generic_name": record["generic_name"],
        "strength": record["strength"] or None,
        "dosage_form": record["dosage_form"] or None,
        "pack_size": record["pack_size"] or None,
        "manufacturer_name": record["manufacturer"] or None,
        "registration_status": RegistrationStatus(record["registration_status"]),
        "registration_expires_on": date.fromisoformat(expires) if expires else None,
        "active_ingredients": ingredients,
        "registration_date": date.fromisoformat(registered) if registered else None,
        "source_url": record.get("source_url") or GREENBOOK_URL,
        "retrieved_on": date.fromisoformat(retrieved) if retrieved else None,
    }
    return values


# Register facts compared when a snapshot row and a hand-typed row describe the same product.
FACT_FIELDS = (
    "brand_name",
    "generic_name",
    "strength",
    "active_ingredients",
    "dosage_form",
    "pack_size",
    "manufacturer_name",
    "registration_status",
    "registration_date",
    "registration_expires_on",
    "source_url",
)


def incoming_wins(
    incoming: RegisterSource,
    incoming_retrieved: date | None,
    stored: RegisterSource,
    stored_retrieved: date | None,
) -> bool:
    """Whether an incoming register row replaces the stored one (owner's rule): from the same
    source, always (the newer import refreshes); between a snapshot and a hand-typed row, the
    one retrieved most recently wins, and on the same day the snapshot wins. A row without a
    retrieval date loses to one with a date."""
    if incoming is stored or stored_retrieved is None:
        return True
    if incoming_retrieved is None:
        return False
    if incoming_retrieved != stored_retrieved:
        return incoming_retrieved > stored_retrieved
    return incoming is RegisterSource.SNAPSHOT


@dataclass(frozen=True, slots=True)
class Conflict:
    """A product a snapshot and the hand-typed file both describe, differently."""

    nafdac_number: str
    fields: dict[str, tuple[object, object]]  # name -> (stored, incoming)
    stored_source: RegisterSource
    stored_retrieved_on: date | None
    incoming_retrieved_on: date | None
    incoming_wins: bool

    def as_json(self) -> dict[str, Any]:
        def show(value: object) -> object:
            return value.isoformat() if isinstance(value, date) else value

        return {
            "nafdac_number": self.nafdac_number,
            "stored_source": self.stored_source.value,
            "stored_retrieved_on": show(self.stored_retrieved_on),
            "incoming_retrieved_on": show(self.incoming_retrieved_on),
            "winner": "incoming" if self.incoming_wins else "stored",
            "fields": {name: [show(a), show(b)] for name, (a, b) in self.fields.items()},
        }


@dataclass
class Plan:
    """What an upsert of register rows from ``source`` would do."""

    source: RegisterSource
    new: list[dict[str, object]] = field(default_factory=list)
    updated: list[dict[str, object]] = field(default_factory=list)
    unchanged: int = 0
    kept: list[str] = field(default_factory=list)  # the other source's newer rows
    conflicts: list[Conflict] = field(default_factory=list)


def plan_upsert(
    session: Session, products: Sequence[dict[str, object]], source: RegisterSource
) -> Plan:
    """Compare incoming rows with the stored ones: new, updated, unchanged, kept (the stored row
    of the other source is more recent, incoming_wins) and every conflicting field between a
    snapshot row and a hand-typed row, whoever wins."""
    plan = Plan(source)
    numbers = [str(product["nafdac_number"]) for product in products]
    stored: dict[str, Product] = {}
    for start in range(0, len(numbers), UPSERT_CHUNK):
        chunk = numbers[start : start + UPSERT_CHUNK]
        for product in session.scalars(select(Product).where(Product.nafdac_number.in_(chunk))):
            stored[product.nafdac_number] = product
    for row in products:
        current = stored.get(str(row["nafdac_number"]))
        if current is None:
            plan.new.append(row)
            continue
        retrieved = row.get("retrieved_on")
        incoming_retrieved = retrieved if isinstance(retrieved, date) else None
        differs = {
            name: (getattr(current, name), row[name])
            for name in FACT_FIELDS
            if name in row and getattr(current, name) != row[name]
        }
        wins = incoming_wins(
            source, incoming_retrieved, current.register_source, current.retrieved_on
        )
        if current.register_source is not source and differs:
            plan.conflicts.append(
                Conflict(
                    current.nafdac_number,
                    differs,
                    current.register_source,
                    current.retrieved_on,
                    incoming_retrieved,
                    wins,
                )
            )
        if not wins:
            plan.kept.append(current.nafdac_number)
        elif (
            differs
            or current.register_source is not source
            or current.retrieved_on != incoming_retrieved
        ):
            plan.updated.append(row)
        else:
            plan.unchanged += 1
    return plan


@dataclass(frozen=True, slots=True)
class Upserted:
    new: int
    updated: int
    kept: int  # rows of the other source that were retrieved more recently


def upsert_products(
    session: Session,
    products: Sequence[dict[str, object]],
    source: RegisterSource,
    plan: Plan | None = None,
) -> Upserted:
    """Insert or update products by NAFDAC number in the caller's transaction, by the rule of
    ``incoming_wins`` (a newer row of the other source is kept), recording ``source`` on every
    row written. Only the columns present in the rows change, so a register import never
    touches ``manufacturer_org_id`` (set by approved manufacturer claims). Nothing is ever
    deleted: a hand-typed row missing from a snapshot stays."""
    plan = plan or plan_upsert(session, products, source)
    rows = [{**product, "register_source": source} for product in plan.new + plan.updated]
    if rows:
        statement = insert(Product)
        stored, excluded = Product.__table__.c, statement.excluded
        newer = (
            excluded.retrieved_on >= stored.retrieved_on
            if source is RegisterSource.SNAPSHOT
            else excluded.retrieved_on > stored.retrieved_on
        )
        statement = statement.on_conflict_do_update(
            index_elements=[Product.nafdac_number],
            set_={column: excluded[column] for column in rows[0] if column != "nafdac_number"}
            | {"updated_at": func.now()},
            # The same rule in SQL, should the table change between the plan and the write.
            where=or_(
                stored.register_source == source.value,
                stored.retrieved_on.is_(None),
                and_(excluded.retrieved_on.is_not(None), newer),
            ),
        )
        for start in range(0, len(rows), UPSERT_CHUNK):
            session.execute(statement, rows[start : start + UPSERT_CHUNK])
    return Upserted(len(plan.new), len(plan.updated), len(plan.kept))


def import_snapshot(
    session: Session, data: bytes, manifest: Manifest, *, apply: bool = True
) -> tuple[Plan, Upserted | None]:
    """Verify a full snapshot and plan its import; apply it when ``apply``. A hand-typed row
    retrieved more recently than the snapshot is kept; every conflicting field is in the plan."""
    products = [product_values(record) for record in decode_snapshot(data, manifest)]
    plan = plan_upsert(session, products, RegisterSource.SNAPSHOT)
    if not apply:
        return plan, None
    return plan, upsert_products(session, products, RegisterSource.SNAPSHOT, plan)


def render_conflicts(conflicts: Sequence[Conflict], incoming: str) -> list[str]:
    """Report lines: every conflicting field, both values, both dates and the winner."""
    lines: list[str] = []
    if conflicts:
        lines.append(
            f"  conflicts with stored rows of the other source ({len(conflicts)}; the most "
            "recently retrieved wins, on the same day the snapshot):"
        )
    for conflict in conflicts:
        stored = conflict.stored_source.value
        winner = incoming if conflict.incoming_wins else f"stored {stored} row"
        lines.append(
            f"  <> {conflict.nafdac_number}: stored {stored} retrieved "
            f"{_day(conflict.stored_retrieved_on)}, {incoming} retrieved "
            f"{_day(conflict.incoming_retrieved_on)} -> {winner} wins"
        )
        lines += [
            f"      {name}: {_shown(a)!r} (stored) vs {_shown(b)!r} ({incoming})"
            for name, (a, b) in conflict.fields.items()
        ]
    return lines


def _day(value: date | None) -> str:
    return value.isoformat() if value else "(no date)"


def _shown(value: object) -> object:
    if isinstance(value, date):
        return value.isoformat()
    return value.value if isinstance(value, RegistrationStatus) else value


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NAFDAC Greenbook snapshots.")
    commands = parser.add_subparsers(dest="command", required=True)
    export_cmd = commands.add_parser("export", help="fetch the full register (network)")
    export_cmd.add_argument("--out", type=Path, help=f"default: {DEFAULT_EXPORT_DIR}")
    sample_cmd = commands.add_parser("sample", help="rewrite seed/products.csv from a snapshot")
    sample_cmd.add_argument("--snapshot", required=True, help="path, s3:// or https:// source")
    import_cmd = commands.add_parser(
        "import", help="report on the current snapshot; load it into products with --confirm"
    )
    import_cmd.add_argument("--source", help="path, s3:// or https:// (default: the manifest)")
    import_cmd.add_argument("--confirm", action="store_true", help="apply (else report only)")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            return _export(args.out or DEFAULT_EXPORT_DIR)
        if args.command == "sample":
            return _sample(args.snapshot)
        return _import(args.source, confirm=args.confirm)
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


def _import(source: str | None, *, confirm: bool = False) -> int:
    from medlens.core.bootstrap import configure_process
    from medlens.core.config import get_settings
    from medlens.db.session import create_sync_engine, create_sync_session_factory

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
        with create_sync_session_factory(engine)() as session:
            plan, result = import_snapshot(session, data, manifest, apply=confirm)
            sys.stdout.write(
                "\n".join(
                    [
                        f"{manifest.file} (fetched {manifest.fetched_on}): {len(plan.new)} new, "
                        f"{len(plan.updated)} updated, {plan.unchanged} unchanged, "
                        f"{len(plan.kept)} hand-typed rows retrieved more recently kept",
                        *render_conflicts(plan.conflicts, "snapshot"),
                    ]
                )
                + "\n"
            )
            if result is None:
                sys.stderr.write("Report only. Run again with CONFIRM=1 to apply.\n")
                return 0
            session.commit()
    finally:
        engine.dispose()
    clear_cached_products(settings)
    sys.stderr.write(f"Applied: {result.new} new, {result.updated} updated.\n")
    return 0


def clear_cached_products(settings: Any) -> None:
    """After product facts changed: scans must not keep using the old ones until they expire."""
    from medlens.cache.redis import create_sync_redis_client
    from medlens.modules.products.cache import clear_product_facts

    redis = create_sync_redis_client(settings.redis, db=settings.redis.cache_db)
    try:
        clear_product_facts(redis)
    finally:
        redis.close()


def _read_source(location: str) -> bytes:
    if urlparse(location).scheme != "s3":
        return fetch_source(location)
    from medlens.adapters.s3 import create_sync_s3_client
    from medlens.core.config import get_settings

    return fetch_source(location, s3_client=create_sync_s3_client(get_settings().s3))


if __name__ == "__main__":
    sys.exit(main())

"""Import register (Greenbook) entries typed in by hand: validate every row, report the
difference with the products table, and apply only when asked.

    make greenbook-add-rows FILE=seed/greenbook_manual.csv            # report only
    make greenbook-add-rows FILE=seed/greenbook_manual.csv CONFIRM=1  # report, then apply

The file has the columns of the Greenbook snapshot (medlens.seed.greenbook.COLUMNS) plus
registration_date, source_url, retrieved_on (fetched_on is accepted for retrieved_on) and the
optional columns gtin (the product's barcodes, "; "-separated: a GTIN belongs to one product;
with the column, an empty cell clears the product's GTINs, shown in the report) and gtin_source
(where they come from: register or manufacturer, required with any GTIN). One row
per registration, as the Greenbook shows it: several active ingredients and their strengths as
parallel lists separated by "; " ("Artemether; Lumefantrine" / "20 mg; 120 mg"); dates as
YYYY-MM-DD; status active, expired, suspended or revoked; source_url the Greenbook page the row
was read from. Nothing is fetched from the Greenbook (CLAUDE.md decision log).

Rows: added (new NAFDAC number), changed (with each field's old and new value), unchanged,
invalid (with reasons). Products in the register but not in the file are listed and kept: the
import never deletes a registration (set its status to suspended or revoked instead) and never
touches the link to the holder's organisation (set by approved product claims). The import is all
or nothing: with any invalid row nothing is applied, even with CONFIRM=1. An applied import is
audited (register.imported) and clears the scan pipeline's cached product facts.

Rows written by this import are manual (products.register_source). Where the file and a stored
snapshot row describe the same product, the row retrieved most recently wins (retrieved_on), and
on the same day the snapshot (greenbook.incoming_wins); the report lists every conflicting field
with both values, both dates and the winner before anything is applied. A file row older than
the stored snapshot row is not applied ("superseded"); its GTINs still are (snapshots carry
none). A snapshot import (make greenbook-import-snapshot) follows the same rule and never
removes a hand-typed row.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from medlens.core.identifiers import normalize_gtin, normalize_nafdac_number
from medlens.modules.audit.models import AuditLog
from medlens.modules.products.models import (
    GtinSource,
    Product,
    ProductGtin,
    RegisterSource,
    RegistrationStatus,
)
from medlens.modules.products.register import dosage_form_family, parse_ingredients
from medlens.seed.greenbook import (
    FACT_FIELDS,
    Conflict,
    clean,
    incoming_wins,
    product_values,
    render_conflicts,
    upsert_products,
)

MANUAL_COLUMNS = (
    "nafdac_number",
    "brand_name",
    "generic_name",
    "strength",
    "dosage_form",
    "pack_size",
    "manufacturer",
    "category",
    "registration_status",
    "registration_date",
    "registration_expires_on",
    "greenbook_product_id",
    "source_url",
    "retrieved_on",
    "gtin",  # optional: the product's barcodes, "; "-separated
    "gtin_source",  # with any GTIN: register or manufacturer (where the mapping comes from)
)
REQUIRED = (
    "nafdac_number",
    "brand_name",
    "generic_name",
    "dosage_form",
    "manufacturer",
    "registration_status",
    "registration_expires_on",
    "source_url",
    "retrieved_on",
)
GREENBOOK_HOST = "greenbook.nafdac.gov.ng"
# Fields compared with the products table (column names), in report order.
COMPARED = (
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
    "retrieved_on",
)


@dataclass
class Row:
    line: int
    nafdac_number: str
    values: dict[str, object] | None = None
    # GTIN -> source; None when the file has no gtin column (stored GTINs stay as they are).
    gtins: dict[str, GtinSource] | None = None
    # False when a stored snapshot row was retrieved more recently: its facts win.
    facts_apply: bool = True
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Report:
    added: list[Row] = field(default_factory=list)
    changed: list[tuple[Row, dict[str, tuple[object, object]]]] = field(default_factory=list)
    unchanged: list[Row] = field(default_factory=list)
    invalid: list[Row] = field(default_factory=list)
    not_in_file: list[str] = field(default_factory=list)
    warnings: list[Row] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    superseded: list[Row] = field(default_factory=list)  # a newer snapshot row wins

    @property
    def applicable(self) -> bool:
        return not self.invalid

    def as_json(self) -> dict[str, Any]:
        return {
            "added": [row.nafdac_number for row in self.added],
            "changed": {
                row.nafdac_number: {name: [_show(a), _show(b)] for name, (a, b) in fields.items()}
                for row, fields in self.changed
            },
            "unchanged": [row.nafdac_number for row in self.unchanged],
            "invalid": [
                {"line": row.line, "nafdac_number": row.nafdac_number, "reasons": row.problems}
                for row in self.invalid
            ],
            "not_in_file": self.not_in_file,
            "conflicts": [conflict.as_json() for conflict in self.conflicts],
            "superseded": [row.nafdac_number for row in self.superseded],
            "warnings": [
                {"line": row.line, "nafdac_number": row.nafdac_number, "warnings": row.warnings}
                for row in self.warnings
            ],
        }


def _show(value: object) -> object:
    if isinstance(value, date):
        return value.isoformat()
    return value.value if isinstance(value, RegisterSource) else value


# --------------------------------------------------------------------------- validation
def _date(value: str, name: str, problems: list[str]) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        problems.append(f"{name} {value!r} is not a date (YYYY-MM-DD)")
        return None


def validate_row(line: int, raw: dict[str, str], today: date) -> Row:
    record = {name: clean(raw.get(name, "")) for name in MANUAL_COLUMNS}
    if not record["retrieved_on"]:
        record["retrieved_on"] = clean(raw.get("fetched_on", ""))
    problems: list[str] = []
    warnings: list[str] = []
    try:
        number = normalize_nafdac_number(record["nafdac_number"])
    except ValueError:
        number = record["nafdac_number"] or "(none)"
        if record["nafdac_number"]:
            problems.append(f"nafdac_number {record['nafdac_number']!r} is not a NAFDAC number")
    problems += [f"{name} is required" for name in REQUIRED if not record[name]]
    status = record["registration_status"].lower()
    if status and status not in {item.value for item in RegistrationStatus}:
        problems.append(
            f"registration_status {status!r} is not active, expired, suspended or revoked"
        )
    registered = _date(record["registration_date"], "registration_date", problems)
    expires = _date(record["registration_expires_on"], "registration_expires_on", problems)
    retrieved = _date(record["retrieved_on"], "retrieved_on", problems)
    if registered and expires and registered > expires:
        problems.append("registration_date is after registration_expires_on")
    if retrieved and retrieved > today:
        problems.append("retrieved_on is in the future")
    if registered and retrieved and registered > retrieved:
        problems.append("registration_date is after retrieved_on")
    if expires and retrieved and status == "active" and expires < retrieved:
        problems.append("status active, but the registration expired before it was retrieved")
    if expires and retrieved and status == "expired" and expires >= retrieved:
        problems.append("status expired, but the expiry date had not passed when retrieved")
    url = urlparse(record["source_url"])
    if record["source_url"] and (url.scheme != "https" or url.hostname != GREENBOOK_HOST):
        problems.append(f"source_url must be an https://{GREENBOOK_HOST}/ page")
    if record["generic_name"]:
        try:
            parse_ingredients(record["generic_name"], record["strength"])
        except ValueError as exc:
            problems.append(f"active ingredients: {exc}")
    gtins = _read_gtins(record, problems) if "gtin" in raw else None
    if record["dosage_form"] and dosage_form_family(record["dosage_form"]) is None:
        warnings.append(f"dosage_form {record['dosage_form']!r} is not a known form (kept as is)")
    row = Row(line=line, nafdac_number=number, problems=problems, warnings=warnings)
    if not problems:
        record["nafdac_number"] = number
        record["registration_status"] = status
        row.values = product_values(record)
        row.gtins = gtins
    return row


def _read_gtins(record: dict[str, str], problems: list[str]) -> dict[str, GtinSource]:
    found: list[str] = []
    for part in record["gtin"].split(";"):
        if part.strip():
            try:
                found.append(normalize_gtin(part))
            except ValueError:
                problems.append(f"gtin {part.strip()!r} is not a GTIN with a valid check digit")
    source = record["gtin_source"].lower()
    if found and source not in {item.value for item in GtinSource}:
        problems.append("gtin_source must be register or manufacturer when a gtin is given")
        return {}
    return dict.fromkeys(found, GtinSource(source)) if found else {}


def read_rows(path: Path, today: date) -> tuple[list[Row], list[str]]:
    """(rows, file-level problems such as missing columns)."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        if "retrieved_on" not in header and "fetched_on" in header:
            header.add("retrieved_on")
        missing = [name for name in REQUIRED if name not in header]
        if missing:
            return [], [f"missing columns: {', '.join(missing)}"]
        rows = [validate_row(line, raw, today) for line, raw in enumerate(reader, start=2)]
    lines_by_number: dict[str, list[int]] = {}
    for row in rows:
        lines_by_number.setdefault(row.nafdac_number, []).append(row.line)
    for row in rows:
        lines = lines_by_number[row.nafdac_number]
        if len(lines) > 1:  # which of them is right is for a person to decide
            row.problems.append(
                "NAFDAC number on more than one line: " + ", ".join(map(str, lines))
            )
            row.values = None
    owners: dict[str, set[str]] = {}
    for row in rows:
        for gtin in _gtins(row):
            owners.setdefault(gtin, set()).add(row.nafdac_number)
    for row in rows:
        shared = [g for g in _gtins(row) if len(owners[g]) > 1]
        if shared:
            row.problems.append(f"GTIN {', '.join(shared)} is on more than one product in the file")
            row.values = None
    if not rows:
        return [], ["no rows"]
    return rows, []


def _gtins(row: Row) -> list[str]:
    return list(row.gtins or {}) if row.values is not None else []


def _shown_gtins(gtins: dict[str, GtinSource]) -> list[str]:
    return [f"{gtin} ({source.value})" for gtin, source in sorted(gtins.items())]


# --------------------------------------------------------------------------- diff
def _current(product: Product) -> dict[str, object]:
    return {name: getattr(product, name) for name in COMPARED}


def diff(session: Session, rows: Sequence[Row]) -> Report:
    report = Report()
    numbers = [row.nafdac_number for row in rows]
    existing = {
        product.nafdac_number: product
        for product in session.scalars(select(Product).where(Product.nafdac_number.in_(numbers)))
    }
    wanted = sorted({gtin for row in rows for gtin in _gtins(row)})
    registered: dict[str, str] = {}
    if wanted:
        statement = (
            select(ProductGtin.gtin, Product.nafdac_number)
            .join(Product, Product.id == ProductGtin.product_id)
            .where(ProductGtin.gtin.in_(wanted))
        )
        registered = dict(session.execute(statement).tuples().all())
    stored: dict[str, dict[str, GtinSource]] = {}
    for gtin, number, source in session.execute(
        select(ProductGtin.gtin, Product.nafdac_number, ProductGtin.source)
        .join(Product, Product.id == ProductGtin.product_id)
        .where(Product.nafdac_number.in_(numbers))
    ):
        stored.setdefault(number, {})[gtin] = source
    for row in rows:
        taken = [
            f"GTIN {gtin} is registered to {registered[gtin]}"
            for gtin in _gtins(row)
            if registered.get(gtin, row.nafdac_number) != row.nafdac_number
        ]
        if taken:
            row.problems += taken
            row.values = None
    for row in rows:
        if row.warnings:
            report.warnings.append(row)
        if row.values is None:
            report.invalid.append(row)
            continue
        current = existing.get(row.nafdac_number)
        if current is None:
            report.added.append(row)
            continue
        before = _current(current)
        changes: dict[str, tuple[object, object]] = {
            name: (before[name], row.values[name])
            for name in COMPARED
            if name in row.values and before[name] != row.values[name]
        }
        retrieved = row.values.get("retrieved_on")
        file_retrieved = retrieved if isinstance(retrieved, date) else None
        row.facts_apply = incoming_wins(
            RegisterSource.MANUAL, file_retrieved, current.register_source, current.retrieved_on
        )
        if current.register_source is RegisterSource.SNAPSHOT:
            differs = {
                name: (getattr(current, name), row.values[name])
                for name in FACT_FIELDS
                if name in row.values and getattr(current, name) != row.values[name]
            }
            if differs:
                report.conflicts.append(
                    Conflict(
                        row.nafdac_number,
                        differs,
                        current.register_source,
                        current.retrieved_on,
                        file_retrieved,
                        row.facts_apply,
                    )
                )
        if not row.facts_apply:
            if changes:  # the stored snapshot row is newer: its facts stay
                report.superseded.append(row)
            changes = {}
        old_gtins = stored.get(row.nafdac_number, {})
        if row.gtins is not None and row.gtins != old_gtins:
            changes["gtins"] = (_shown_gtins(old_gtins), _shown_gtins(row.gtins))
        if changes:
            report.changed.append((row, changes))
        elif row not in report.superseded:
            report.unchanged.append(row)
    in_file = set(numbers)
    report.not_in_file = sorted(
        number for number in session.scalars(select(Product.nafdac_number)) if number not in in_file
    )
    return report


def apply(session: Session, report: Report, *, file_name: str, sha256: str) -> None:
    """Upsert the added and changed rows in the caller's transaction, and audit it."""
    rows = list(report.added) + [row for row, _ in report.changed]
    upsert_products(
        session,
        [row.values for row in rows if row.values is not None and row.facts_apply],
        RegisterSource.MANUAL,
    )
    with_gtins = {row.nafdac_number: row.gtins for row in rows if row.gtins is not None}
    if with_gtins:
        ids = dict(
            session.execute(
                select(Product.nafdac_number, Product.id).where(
                    Product.nafdac_number.in_(list(with_gtins))
                )
            )
            .tuples()
            .all()
        )
        session.execute(delete(ProductGtin).where(ProductGtin.product_id.in_(ids.values())))
        session.flush()
        session.add_all(
            ProductGtin(gtin=gtin, product_id=ids[number], source=source)
            for number, gtins in with_gtins.items()
            for gtin, source in gtins.items()
        )
    session.add(
        AuditLog(
            action="register.imported",
            resource_type="register_file",
            resource_id=sha256[:64],
            metadata_={
                "file": file_name,
                "sha256": sha256,
                "added": [row.nafdac_number for row in report.added],
                "changed": [row.nafdac_number for row, _ in report.changed],
                "superseded": [row.nafdac_number for row in report.superseded],
                "unchanged": len(report.unchanged),
            },
        )
    )
    session.flush()


# --------------------------------------------------------------------------- report
def render(report: Report, problems: Sequence[str]) -> str:
    lines: list[str] = []
    if problems:
        return "\n".join(["The file cannot be read:", *(f"  {p}" for p in problems)])
    lines.append(
        f"added {len(report.added)}, changed {len(report.changed)}, unchanged "
        f"{len(report.unchanged)}, invalid {len(report.invalid)}, superseded by newer snapshot "
        f"rows {len(report.superseded)}; {len(report.not_in_file)} registered product(s) not in "
        "this file (kept)"
    )
    lines += render_conflicts(report.conflicts, "file")
    for row in report.added:
        values = row.values or {}
        lines.append(f"  + {row.nafdac_number}  {values.get('brand_name')}")
    for row, changes in report.changed:
        lines.append(f"  ~ {row.nafdac_number}")
        lines += [f"      {name}: {_show(a)!r} -> {_show(b)!r}" for name, (a, b) in changes.items()]
    for row in report.invalid:
        lines.append(f"  ! line {row.line} {row.nafdac_number}: " + "; ".join(row.problems))
    for row in report.warnings:
        lines.append(f"  ? line {row.line} {row.nafdac_number}: " + "; ".join(row.warnings))
    if report.not_in_file:
        shown = ", ".join(report.not_in_file[:15])
        more = f" and {len(report.not_in_file) - 15} more" if len(report.not_in_file) > 15 else ""
        lines.append(f"  = not in this file, kept: {shown}{more}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("--confirm", action="store_true", help="apply when every row is valid")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    from medlens.core.bootstrap import configure_process
    from medlens.core.config import get_settings
    from medlens.db.session import create_sync_engine, create_sync_session_factory
    from medlens.seed.greenbook import clear_cached_products

    data = args.file.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    rows, problems = read_rows(args.file, datetime.now(UTC).date())
    settings = get_settings()
    configure_process(settings)
    engine = create_sync_engine(settings.database, application_name="medlens-register-import")
    try:
        with create_sync_session_factory(engine)() as session:
            report = diff(session, rows) if rows else Report()
            output = (
                json.dumps(report.as_json(), indent=2) if args.json else render(report, problems)
            )
            sys.stdout.write(output + "\n")
            if problems or not report.applicable:
                if args.confirm:
                    sys.stderr.write("Nothing applied: fix the invalid rows first.\n")
                return 2 if args.confirm else 0
            if not args.confirm:
                sys.stderr.write("Report only. Run again with CONFIRM=1 to apply.\n")
                return 0
            if not report.added and not report.changed:
                sys.stderr.write("Nothing to apply.\n")
                return 0
            apply(session, report, file_name=args.file.name, sha256=sha256)
            session.commit()
    finally:
        engine.dispose()
    clear_cached_products(settings)
    sys.stderr.write(
        f"Applied: {len(report.added)} added, {len(report.changed)} changed (audited).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

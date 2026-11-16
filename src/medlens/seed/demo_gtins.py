"""Load the demo's synthetic barcodes (seed/test_packs/gtins.csv) into a local database, so the
demo can show the cloned-barcode and same-holder checks. Local only, and only rows marked
synthetic: these GTINs are drawn for the test packs and are not real products' barcodes, so they
never go through the register import or into another environment.

    uv run python -m medlens.seed.demo_gtins seed/test_packs/gtins.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from medlens.core.config import Environment
from medlens.core.identifiers import normalize_gtin, normalize_nafdac_number
from medlens.modules.products.models import GtinSource, Product, ProductGtin


def load(session: Session, path: Path) -> tuple[int, list[str]]:
    """Upsert the file's synthetic GTINs; returns (loaded, problems). A GTIN moves to the
    product the file names; rows not marked synthetic or naming no product are refused."""
    loaded, problems = 0, []
    with path.open(newline="", encoding="utf-8") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            if row.get("synthetic") != "yes":
                problems.append(f"line {line}: not marked synthetic (refused)")
                continue
            number = normalize_nafdac_number(row["nafdac_number"])
            product_id = session.scalar(select(Product.id).where(Product.nafdac_number == number))
            if product_id is None:
                problems.append(f"line {line}: {number} is not a registered product")
                continue
            statement = insert(ProductGtin).values(
                gtin=normalize_gtin(row["gtin"]),
                product_id=product_id,
                source=GtinSource(row["source"]),
            )
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=[ProductGtin.gtin],
                    set_={
                        "product_id": statement.excluded.product_id,
                        "source": statement.excluded.source,
                        "updated_at": func.now(),
                    },
                )
            )
            loaded += 1
    return loaded, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file", type=Path)
    args = parser.parse_args(argv)

    from medlens.core.config import get_settings
    from medlens.db.session import create_sync_engine, create_sync_session_factory

    settings = get_settings()
    if settings.app.environment is not Environment.LOCAL:
        sys.stderr.write("refusing: synthetic demo barcodes are for local databases only\n")
        return 1
    engine = create_sync_engine(settings.database, application_name="medlens-demo-gtins")
    try:
        with create_sync_session_factory(engine)() as session:
            loaded, problems = load(session, args.file)
            if problems:
                sys.stderr.write("\n".join(problems) + "\nNothing loaded.\n")
                return 2
            session.commit()
    finally:
        engine.dispose()
    sys.stdout.write(f"loaded {loaded} synthetic demo barcodes\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

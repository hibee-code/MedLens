"""Building blocks for hand-written migrations.

Old revisions import these, so only ever change them in backwards-compatible ways. Constraint
and index names come from the naming convention on the target metadata (db/base.py).
"""

from typing import Any

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geography
from sqlalchemy.dialects.postgresql import JSONB

# ISO 3166-2:NG codes of the 36 states and the FCT.
STATE_CODES = (
    "NG-AB", "NG-AD", "NG-AK", "NG-AN", "NG-BA", "NG-BY", "NG-BE", "NG-BO", "NG-CR", "NG-DE",
    "NG-EB", "NG-ED", "NG-EK", "NG-EN", "NG-FC", "NG-GO", "NG-IM", "NG-JI", "NG-KD", "NG-KN",
    "NG-KT", "NG-KE", "NG-KO", "NG-KW", "NG-LA", "NG-NA", "NG-NI", "NG-OG", "NG-ON", "NG-OS",
    "NG-OY", "NG-PL", "NG-RI", "NG-SO", "NG-TA", "NG-YO", "NG-ZA",
)  # fmt: skip
STATE_CODES_ARRAY = "ARRAY[" + ", ".join(f"'{code}'" for code in STATE_CODES) + "]::varchar[]"


def enum(name: str, *values: str) -> sa.Enum:
    """VARCHAR + CHECK constraint ck_<table>_<name>, matching db/types.py:str_enum."""
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=True)


def state_enum() -> sa.Enum:
    return enum("state", *STATE_CODES)


def states_check(column: str, name: str) -> sa.CheckConstraint:
    return sa.CheckConstraint(f"{column} <@ {STATE_CODES_ARRAY}", name=name)


def id_column() -> sa.Column[Any]:
    return sa.Column("id", sa.Uuid(), primary_key=True)


def created_at() -> sa.Column[Any]:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )


def updated_at() -> sa.Column[Any]:
    return sa.Column(
        "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )


def deleted_at() -> sa.Column[Any]:
    return sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True)


def point(name: str, *, nullable: bool = True) -> sa.Column[Any]:
    """geography(Point, 4326). Spatial indexes are created explicitly (GiST, named)."""
    return sa.Column(
        name, Geography(geometry_type="POINT", srid=4326, spatial_index=False), nullable=nullable
    )


def foreign_key(
    column: str, target: str, *, nullable: bool, ondelete: str | None = None
) -> sa.Column[Any]:
    return sa.Column(column, sa.Uuid(), sa.ForeignKey(target, ondelete=ondelete), nullable=nullable)


def empty_array(name: str, item_type: sa.types.TypeEngine[Any]) -> sa.Column[Any]:
    return sa.Column(name, sa.ARRAY(item_type), server_default=sa.text("'{}'"), nullable=False)


def empty_json(name: str) -> sa.Column[Any]:
    return sa.Column(name, JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False)


def resize_vectors(
    table: str, column: str, dimension: int, *, index: str, model_column: str
) -> None:
    """Make ``table.column`` a vector of ``dimension`` if it is not one already. Vectors of
    another size cannot be compared, so they are cleared (with their model version, so the
    embedding jobs recompute them) and the HNSW index is rebuilt. Idempotent."""
    current = op.get_bind().scalar(
        sa.text(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = CAST(:table AS regclass) AND attname = :column"
        ).bindparams(table=table, column=column)
    )
    if current == dimension:
        return
    op.drop_index(index, table_name=table)
    op.execute(
        f"ALTER TABLE {table} ALTER COLUMN {column} TYPE vector({int(dimension)}) USING NULL"
    )
    op.execute(f"UPDATE {table} SET {model_column} = NULL")
    op.create_index(
        index,
        table,
        [column],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={column: "vector_cosine_ops"},
    )

"""Enable PostgreSQL extensions.

postgis     geography columns and clustering (hotspots, branches, scan locations)
vector      pgvector: pack-image embeddings with HNSW indexes
pg_trgm     trigram similarity for fuzzy product-name and NAFDAC-number lookup
citext      case-insensitive e-mail addresses
btree_gist  exclusion and GiST indexes that mix scalar and spatial columns

Revision ID: 0001
Revises:
Create Date: 2026-11-03
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EXTENSIONS = ("postgis", "vector", "pg_trgm", "citext", "btree_gist")


def upgrade() -> None:
    for extension in EXTENSIONS:
        op.execute(f'CREATE EXTENSION IF NOT EXISTS "{extension}"')


def downgrade() -> None:
    for extension in reversed(EXTENSIONS):
        op.execute(f'DROP EXTENSION IF EXISTS "{extension}"')

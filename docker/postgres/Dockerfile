# PostgreSQL 17 with PostGIS and pgvector. pg_trgm, citext and btree_gist ship with the base
# image's contrib package. Extensions are created by the first Alembic migration (Phase 2).
FROM postgres:17.11-trixie

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      "postgresql-${PG_MAJOR}-postgis-3" \
      "postgresql-${PG_MAJOR}-postgis-3-scripts" \
      "postgresql-${PG_MAJOR}-pgvector" \
 && rm -rf /var/lib/apt/lists/*

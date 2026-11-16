# PostgreSQL 17 with PostGIS and pgvector. pg_trgm, citext and btree_gist ship with the base
# image's contrib package. Extensions are created by the first Alembic migration (Phase 2).
FROM postgres:17.11-trixie

# PostGIS and pgvector are pinned to builds for this server version. On 2026-11-12 the
# PostgreSQL apt repository's current builds (pgvector 0.8.7-1.pgdg13+2, PostGIS
# 3.7.0+dfsg-1.pgdg13+2) were compiled against postgresql-17 17.12, which neither the repository
# nor the official image had yet: pgvector would not install, PostGIS would not load ("undefined
# symbol: heap_datum_check"). The pinned builds come from the repository's long-term archive.
# Remove the pins once the base image is 17.12 or newer.
ARG PGDG_POOL=https://apt-archive.postgresql.org/pub/repos/apt/pool/main/p
ARG PGVECTOR_VERSION=0.8.7-1.pgdg13+1
ARG POSTGIS_VERSION=3.7.0+dfsg-1.pgdg13+1

RUN set -eux \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && arch="$(dpkg --print-architecture)" \
 && cd /tmp \
 && curl -fsSLO "${PGDG_POOL}/pgvector/postgresql-${PG_MAJOR}-pgvector_${PGVECTOR_VERSION}_${arch}.deb" \
 && curl -fsSLO "${PGDG_POOL}/postgis/postgresql-${PG_MAJOR}-postgis-3_${POSTGIS_VERSION}_${arch}.deb" \
 && curl -fsSLO "${PGDG_POOL}/postgis/postgresql-${PG_MAJOR}-postgis-3-scripts_${POSTGIS_VERSION}_all.deb" \
 && apt-get install -y --no-install-recommends /tmp/*.deb \
 && apt-get purge -y --auto-remove curl \
 && rm -rf /var/lib/apt/lists/* /tmp/*.deb

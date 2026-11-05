# syntax=docker/dockerfile:1.7
#
# Multi-stage build. uv installs the locked environment into /opt/venv in a builder stage; runtime
# stages copy only that virtualenv and run as an unprivileged user. No dev dependencies ship.
#
#   docker build --target api    -t medlens-api .      # FastAPI, one Uvicorn process
#   docker build --target worker -t medlens-worker .   # Celery worker: vision, ONNX models

ARG PYTHON_IMAGE=python:3.12.16-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.6

FROM ${UV_IMAGE} AS uv

# ------------------------------------------------------------------------------ builders
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src

FROM builder AS builder-api
# Dependencies first (cached until uv.lock changes), then the project itself. No dependency
# groups: dev and vision are default groups for development only.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --no-install-project
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable

# The models, exported to ONNX and checked against torch (scripts/export_onnx.py). torch only
# lives in this stage: the worker runs ONNX Runtime. Cached until the lockfile, the export
# script or the model names change.
FROM builder AS models
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --group ml --group ml-export --no-install-project
ARG IMAGE_MODEL=clip-ViT-B-32
ARG TEXT_MODEL=intfloat/multilingual-e5-small
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENV HF_HOME=/tmp/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ORT_DISABLE_TELEMETRY=1
COPY scripts/export_onnx.py /export/scripts/export_onnx.py
COPY src/medlens/adapters/onnx_models.py /export/src/medlens/adapters/onnx_models.py
RUN /opt/venv/bin/python /export/scripts/export_onnx.py --out /opt/models/onnx \
      --image-model "$IMAGE_MODEL" --text-model "$TEXT_MODEL" --reranker "$RERANKER_MODEL"

FROM builder AS builder-worker
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --group vision --group ml --no-install-project
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group vision --group ml --no-editable

# ------------------------------------------------------------------------------ runtime
FROM ${PYTHON_IMAGE} AS runtime
# libmagic: python-magic sniffs the real type of uploaded files.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libmagic1t64 \
 && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 10001 medlens \
 && useradd --system --uid 10001 --gid medlens --home-dir /app --no-create-home \
      --shell /usr/sbin/nologin medlens
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

FROM runtime AS api
COPY --from=builder-api /opt/venv /opt/venv
# Migrations ship with the API image so a release job can run `alembic upgrade head`.
COPY alembic.ini ./
COPY migrations ./migrations
USER medlens
EXPOSE 8000
# --proxy-headers trusts X-Forwarded-For/-Proto only from FORWARDED_ALLOW_IPS. Set it to the load
# balancer's address in each deployment; the default trusts nobody but localhost.
ENV FORWARDED_ALLOW_IPS=127.0.0.1
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]
# One Uvicorn process per container (no Gunicorn, no --workers): scale by running more containers.
CMD ["uvicorn", "medlens.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

FROM runtime AS worker
# OpenMP runtime for LightGBM and ONNX Runtime; Tesseract (English) for OCR.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder-worker /opt/venv /opt/venv
COPY --from=models /opt/models/onnx /opt/models/onnx
# ORT_DISABLE_TELEMETRY: ONNX Runtime (1.31+) otherwise reports usage events.
ENV MEDLENS_EMBEDDINGS__BACKEND=onnx \
    MEDLENS_EMBEDDINGS__ONNX_DIR=/opt/models/onnx \
    ORT_DISABLE_TELEMETRY=1 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus-multiproc
USER medlens
# Prometheus metrics of all pool processes (MEDLENS_OBSERVABILITY__WORKER_METRICS_PORT).
EXPOSE 9100
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import socket, sys; from medlens.queue.celery_app import celery_app; sys.exit(0 if celery_app.control.ping(destination=[f'celery@{socket.gethostname()}'], timeout=5) else 1)"]
# The metrics directory is emptied before Celery imports anything: prometheus_client creates
# its sample files at import, and samples from a previous run would be summed into this one.
CMD ["sh", "-c", "rm -rf \"$PROMETHEUS_MULTIPROC_DIR\" && mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\" && exec celery -A medlens.queue.celery_app:celery_app worker --loglevel=INFO --queues=scans.pipeline,ml.ocr,ml.embed,llm,events,maintenance"]

# MedLens developer commands. Run `make` for the list.
.DEFAULT_GOAL := help
UV ?= uv
COMPOSE ?= docker compose
# e.g. `make up PROFILES=observability`
PROFILES ?=
SEED_ARGS ?=
# e.g. `make greenbook-import GREENBOOK_SOURCE=s3://bucket/greenbook-2026-11-03.csv.gz`
GREENBOOK_SOURCE ?=
INFRA := postgres pgbouncer redis minio

.PHONY: help dev-secrets env install install-ml dev test test-unit test-integration lint format typecheck check \
	openapi migrate seed embeddings greenbook-import up infra down logs eval demo test-packs

help: ## Show this help
	@grep -hE '^[a-zA-Z_.-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2}'

dev-secrets: ## Create or update the local .env: new variables, random secrets; keeps your values
	$(UV) run python scripts/sync_env.py

env: dev-secrets ## Same as dev-secrets

# Targets that need configuration depend on .env, which is re-synced whenever the template changes.
.env: .env.example
	$(UV) run python scripts/sync_env.py

install: ## Install main + dev dependencies into .venv
	$(UV) sync

install-ml: ## Also install the ml group (embedding models; large, worker image only)
	$(UV) sync --group ml

dev: .env ## Run the API with auto-reload on :8000 (needs `make infra`)
	$(UV) run uvicorn medlens.main:create_app --factory --reload --host 127.0.0.1 --port 8000

test: ## All tests with coverage (integration tests need Docker)
	$(UV) run pytest --cov --cov-report=term-missing

test-unit: ## Unit tests only (no Docker)
	$(UV) run pytest tests/unit

test-integration: ## Integration tests only (testcontainers)
	$(UV) run pytest tests/integration

lint: ## Ruff lint and format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Apply Ruff fixes and formatting
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## mypy --strict on src
	$(UV) run mypy src

check: lint typecheck test ## Everything the Definition of Done requires

openapi: ## Regenerate the committed openapi.json after API changes
	$(UV) run python scripts/export_openapi.py

migrate: .env ## Apply database migrations (alembic upgrade head)
	$(UV) run alembic upgrade head

seed: .env ## Load development seed data (idempotent). Staff logins: SEED_ARGS="--staff-password ..."
	$(UV) run python -m medlens.seed $(SEED_ARGS)

embeddings: .env ## Queue embeddings for reference images and document chunks that have none (workers must run)
	$(UV) run celery -A medlens.queue.celery_app:celery_app call medlens.reference_images.backfill
	$(UV) run celery -A medlens.queue.celery_app:celery_app call medlens.knowledge.backfill_embeddings

greenbook-import: .env ## Load the full Greenbook snapshot (seed/PROVENANCE.md); checksum-verified
	$(UV) run python -m medlens.seed.greenbook import $(if $(GREENBOOK_SOURCE),--source $(GREENBOOK_SOURCE))

eval: .env ## Score the pipeline on seed/test_packs and the assistant on seed/eval_questions.csv
	$(UV) run python scripts/eval.py

demo: .env ## Start the stack, seed it, sign in as each role and run the demo requests
	bash scripts/demo.sh

test-packs: ## Redraw the synthetic pack photos (tests/fixtures/packs, seed/test_packs)
	$(UV) run python scripts/make_test_packs.py

up: .env ## Build and start the full stack and wait until healthy
	$(COMPOSE) $(if $(PROFILES),--profile $(PROFILES)) up -d --build --wait

infra: .env ## Start only Postgres, PgBouncer, Redis and MinIO (for `make dev`)
	$(COMPOSE) up -d --wait $(INFRA)

down: ## Stop the stack (data volumes are kept)
	$(COMPOSE) --profile observability down

logs: ## Follow logs of all services
	$(COMPOSE) logs -f --tail=100

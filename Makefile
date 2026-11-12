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
	openapi migrate seed embeddings greenbook-import up infra down logs eval demo test-packs \
	audit ci dashboard onnx test-models retrieval-eval ocr-benchmark load-test

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

install-ml: ## Also install the model groups: ONNX Runtime (ml) and torch for exports (ml-export)
	$(UV) sync --group ml --group ml-export

onnx: ## Export the models to ONNX into models/onnx (worker image: done while building)
	$(UV) run --group ml --group ml-export python scripts/export_onnx.py --out models/onnx

test-models: ## The ONNX exports against the original models, cosine > 0.99 (needs `make onnx`)
	MEDLENS_MODEL_TESTS=1 $(UV) run --group ml --group ml-export pytest -m models tests/models

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

audit: ## Known vulnerabilities in the locked dependencies (pip-audit) and security lint (bandit)
	@requirements=$$(mktemp) && \
	$(UV) export --format requirements-txt --no-hashes --all-groups --no-emit-project --quiet \
		| sed -E 's/\+cpu//' > $$requirements && \
	$(UV) run pip-audit -r $$requirements --disable-pip --no-deps --progress-spinner off; \
	status=$$?; rm -f $$requirements; exit $$status
	$(UV) run bandit -c pyproject.toml -r src --severity-level medium --quiet

ci: check audit ## What CI runs (plus the eval and image builds, see .github/workflows/ci.yml)
	$(UV) run python scripts/eval.py --skip-assistant --min-accuracy 0.9

dashboard: ## Regenerate the Grafana dashboard from scripts/make_dashboard.py
	$(UV) run python scripts/make_dashboard.py

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

QUESTIONS ?= seed/eval_questions.csv
retrieval-eval: .env ## Recall/MRR per reranker and the evidence bar's calibration (QUESTIONS=file.csv)
	$(UV) run --group ml --group ml-export python scripts/retrieval_eval.py --questions $(QUESTIONS)

SCENARIO ?= mixed
RATE ?= 50
DURATION ?= 60
load-test: ## Open-loop load test of a running stack (SCENARIO=reads|scans|mixed RATE=req/s DURATION=s)
	$(UV) run python scripts/load_test.py --scenario $(SCENARIO) --rate $(RATE) --duration $(DURATION)

PHOTOS ?= seed/test_packs
ENGINES ?= tesseract
ocr-benchmark: .env ## OCR engines on labelled photos, in the worker image (PHOTOS=dir with labels.csv)
	docker compose run --rm --no-deps -T -v "$(CURDIR)/scripts:/app/scripts:ro" \
	  -v "$(abspath $(PHOTOS)):/photos:ro" worker sh -c 'mkdir -p "$$PROMETHEUS_MULTIPROC_DIR" && \
	  python scripts/ocr_benchmark.py --photos /photos \
	  --labels /photos/$(if $(filter seed/test_packs,$(PHOTOS)),expected.csv,labels.csv) \
	  --engines $(ENGINES)'

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

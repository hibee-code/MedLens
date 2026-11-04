CLAUDE.md — MedLens AI Backend

Project memory for Claude Code. Read this file fully before every task. It defines what we are building, the stack, the architecture rules and the definition of done. If a request conflicts with this file, stop and ask.

1. What we are building

MedLens is the backend service for a counterfeit-medicine verification product in Nigeria.

A buyer holding a drug pack has no fast, reliable way to tell whether it is genuine before using it. Fake drugs cause treatment failure, drug resistance, deaths and financial loss. NAFDAC scratch codes do not cover every product and can be copied.

The service takes a photo of a drug pack and answers one question:

"Is this pack likely genuine, suspicious or unregistered, and how confident are we?"

As a side effect, it records where suspicious scans happen so regulators (NAFDAC and state regulators) can see hotspots.

Core capabilities
Extract: read NAFDAC number, product name, batch number and expiry from a photo (barcode/DataMatrix first, then OCR, then a vision LLM for hard cases).
Verify: look the extracted details up in a registered-products database that stands in for the NAFDAC Greenbook (exact match first, then fuzzy match to absorb OCR errors).
Score: combine signals (record found or not, expired, batch seen in too many places, visual mismatch with genuine reference packs, image quality) into a risk level with a confidence value.
Report: log every scan with a privacy-safe location and expose hotspot endpoints that cluster suspicious scans by place and time.
Assist (AI layer for professional users only): regulators, pharmacists and manufacturers get an evidence-grounded assistant and investigation agents. It answers from a knowledge base of regulatory documents (NAFDAC public alerts, recall notices, Greenbook exports, product leaflets, SOPs) with citations, drafts case summaries and hotspot briefings, and is exposed through API endpoints and a Model Context Protocol server. Every consequential action goes through human approval.
Why RAG and document ingestion are in scope

The verdict on a pack is not produced by RAG or an agent; it comes from the deterministic pipeline and the RiskScorer. RAG exists because regulators and pharmacists need answers grounded in documents that change weekly and are mostly PDFs and web notices ("Has NAFDAC issued an alert for this brand?", "What does the recall notice for batch X say?"). Ingested alerts also feed a scan signal: a match against a published counterfeit alert raises risk.

Out of scope (do NOT build)
Any frontend, mobile app, web UI or admin dashboard.
WhatsApp, USSD, SMS conversations, voice, push notifications or any other user channel.
Multi-language user-facing text (store reason codes; translation is a client concern).
Chat UIs. The assistant exists only as API endpoints (with streaming) and an MCP server.
Any AI feature for anonymous consumers beyond the scan verdict. Consumers never talk to an LLM directly.

The only outbound SMS allowed is the OTP for login, behind a provider interface, with a console provider in development.

The API (OpenAPI docs at /docs) and the test suite are the product surface for now.

2. Tech stack (Python only)

Use these tools. Do not add a dependency outside this list without asking first and explaining why.

Layer	Tool	Notes
Runtime / packaging	Python 3.12+, uv	uv for venv, lockfile (uv.lock) and scripts
Web framework	FastAPI + Uvicorn, one Uvicorn process per container (no Gunicorn)	Async everywhere on the request path; scale by adding containers
Validation	Pydantic v2	Request/response schemas, JSON Schema for LLM structured output
Config	pydantic-settings	Missing or invalid env var must fail startup
Database	PostgreSQL 17 + pgvector + PostGIS + pg_trgm	Vectors for pack embeddings, PostGIS for hotspots, trigram for fuzzy names
ORM / migrations	SQLAlchemy 2.0 (async) + asyncpg + Alembic + GeoAlchemy2 + pgvector-python	Real Vector and Geography column types
Worker DB driver	psycopg 3 (sync engine)	Celery tasks use a sync session, not asyncpg
Pooling	PgBouncer (transaction mode)	asyncpg must disable prepared statement caches (see §7)
Cache / counters	Redis via redis.asyncio	Cache, rate limits, batch-velocity counters, token revocation
Jobs	Celery + Redis broker + Celery Beat + Flower	Retries, routing, priorities, schedules
Object storage	S3 / Cloudflare R2, MinIO locally, via aioboto3 (API) and boto3 (workers)	Lifecycle rules for retention
Barcodes / DataMatrix	zxing-cpp (fallback pyzbar)	Deterministic and cheap: always try before OCR
Image quality	OpenCV (Laplacian variance blur, brightness), Pillow	Quality gate runs in the Celery worker, never in the API; bad photos get a retake reason
OCR	Tesseract via pytesseract (primary) behind the OcrEngine interface	PaddleOCR is a blocked candidate, decided in the OCR phase by a benchmark on 100+ real labelled photos against a vision-LLM fallback
Vision LLM	anthropic Python SDK behind our own LLMGateway; LiteLLM optional adapter	Structured JSON output validated by Pydantic
Image embeddings	DINOv2-small (384-d) or OpenCLIP ViT-B/32 (512-d) on ONNX Runtime	Dimension is config, not hard-coded
Duplicate detection	SHA-256 in the API (exact duplicates); imagehash perceptual hash in the worker	Same photo resubmitted, replay attempts
Text embeddings	multilingual-e5-small (384-d) via sentence-transformers, exported to ONNX Runtime for serving	Local, free, CPU-friendly; model name and dimension in the model registry
Reranking	bge-reranker-base cross-encoder (sentence-transformers / ONNX)	Re-orders top-k retrieval results
Vector store	pgvector in the same Postgres (HNSW, halfvec for large tables) + Postgres full-text (tsvector) for hybrid search	One database to secure, back up and scope by tenant; Qdrant only if pgvector stops scaling
Document ingestion	pypdf + pdfplumber (tables), PaddleOCR for scanned PDFs, trafilatura (HTML notices), python-docx, openpyxl	Custom structure-aware chunker
PII detection	presidio-analyzer + presidio-anonymizer	Scrub PII before embedding, logging or sending to an LLM
LLM providers	anthropic SDK (primary) behind LLMGateway; LiteLLM adapter optional	Model IDs live only in the model registry/config, never in code
Agents	Own agent loop on the Anthropic tool-use API with Pydantic tool schemas (optional: Pydantic AI)	Small, auditable, easy to pause for human approval
Model Context Protocol	mcp Python SDK (FastMCP), streamable HTTP transport, mounted on its own ASGI app	Read-heavy tools for regulators, OAuth/JWT-protected
Evaluation	pytest golden sets + custom metrics; ragas optional for RAG metrics	Runs in CI and nightly
Token counting	Anthropic count_tokens API, tiktoken only as a rough local estimate	Budget checks before calls
Risk scoring	Weighted rules first, then scikit-learn / LightGBM with calibration	Behind a RiskScorer interface; version every model
Hotspots	PostGIS ST_ClusterDBSCAN (primary), scikit-learn HDBSCAN (offline analysis)	
Auth	PyJWT (EdDSA/Ed25519 via cryptography), pwdlib[argon2], pyotp (TOTP MFA)	Phone OTP for consumers via Termii / Africa's Talking provider interface
Rate limiting	fastapi-limiter or a custom Redis sliding-window (Lua)	Distributed limits
Sanitising	nh3	Strip HTML from any free-text field
Upload safety	FastAPI UploadFile + python-multipart (>= 0.0.18) + python-magic + Pillow re-encode	Check real type by content, reject polyglots, re-encode, strip EXIF; only re-encoded bytes are stored
Resilience	purgatory (async circuit breaker) + tenacity (retries)	Around LLM, OCR service, S3, SMS
Logging	structlog (JSON)	request_id, trace_id, user_id, org_id on every line
Metrics	prometheus-fastapi-instrumentator + prometheus_client	Custom business metrics
Tracing	OpenTelemetry (FastAPI, SQLAlchemy, Redis, Celery, httpx instrumentations)	OTLP export
Errors	Sentry SDK	FastAPI + Celery integrations
Request context	contextvars	No globals for per-request state
Quality	Ruff (lint + format), mypy (strict), pytest, pytest-asyncio, pytest-cov, httpx, testcontainers, pre-commit, pip-audit, bandit	Branch coverage, fail_under = 85
Packaging	Docker (multi-stage, non-root), docker compose	
3. Architecture

Modular monolith with one codebase and two process types: the API (FastAPI) and workers (Celery). Modules are organised by domain, and each module is layered.

Router (HTTP only) → Service (business logic, transactions) → Repository (SQL only) → Models
                         ↘ Ports (interfaces) → Adapters (OCR, LLM, S3, SMS, embeddings)

Rules:

Routers parse input, call one service method, return a schema. No SQL, no business rules.
Services own business logic and transaction boundaries. They depend on repositories and ports (Protocols), never on concrete adapters.
Repositories are the only place that writes SQL. Every query that touches tenant data takes a TenantContext and filters by it.
Adapters implement ports and are swapped through dependency injection. Every adapter has a fake for tests.
Schemas (Pydantic) and models (SQLAlchemy) are separate. Never return an ORM object from a route.
Modules talk to each other through service interfaces or domain events, never by importing another module's repository.
Side effects that must survive a crash (enqueue job, emit event) go through the transactional outbox.
All LLM calls go through LLMGateway (the model control plane). No module imports the anthropic SDK directly. The gateway handles model routing, budgets, caching, retries, circuit breaking, guardrails, logging and cost metering.
Naming: "MCP" is ambiguous here. The Model Control Plane lives in modules/control_plane/; the Model Context Protocol server lives in mcp_server/. Never call either just "mcp" in code.
AI layer
Client / MCP host
   → API or MCP server (auth, RBAC, rate limit)
      → Agent runtime (plan → tool calls → guardrails → HITL approvals)
         → Tools (typed service calls, scoped by TenantContext)
         → RAG (hybrid retrieve → rerank → grounded generate → cite)
         → Memory (session, case, user/org long-term)
      → LLMGateway / Model Control Plane (registry, router, budget, cache, breaker, logging)
         → Providers (Anthropic, local ONNX models)
Folder structure
medlens/
├── pyproject.toml
├── uv.lock
├── alembic.ini
├── Dockerfile
├── docker-compose.yml
├── docker/
│   ├── postgres/Dockerfile          # postgres:17 + postgis + pgvector
│   ├── pgbouncer/
│   └── observability/               # otel-collector, prometheus, grafana, jaeger configs
├── migrations/                      # Alembic
├── seed/                            # products.csv, reference images, fake test cases
├── src/medlens/
│   ├── main.py                      # app factory create_app()
│   ├── core/
│   │   ├── config.py                # pydantic-settings
│   │   ├── logging.py               # structlog setup
│   │   ├── context.py               # contextvars: request_id, user, tenant
│   │   ├── errors.py                # domain exceptions + RFC 9457 handlers
│   │   ├── security.py              # hashing, JWT, key rotation
│   │   ├── middleware.py            # request id, security headers, body limit, timing
│   │   ├── telemetry.py             # OTel + Sentry + Prometheus
│   │   └── pagination.py            # cursor pagination
│   ├── db/
│   │   ├── base.py                  # DeclarativeBase, naming conventions, mixins
│   │   ├── session.py               # async engine/session (API), sync engine (workers)
│   │   └── types.py                 # custom column types
│   ├── cache/                       # redis client, cache-aside helpers, locks
│   ├── ratelimit/
│   ├── queue/
│   │   ├── celery_app.py
│   │   ├── outbox.py
│   │   └── beat_schedule.py
│   ├── ports/                       # Protocols: OcrEngine, LLMPort, EmbeddingPort, StoragePort, SMSPort, BarcodePort
│   ├── adapters/                    # real + fake implementations of each port
│   ├── ai/
│   │   ├── gateway/                 # LLMGateway: routing, budgets, cache, breaker, logging
│   │   ├── guardrails/              # input/output/tool guards, injection & PII filters
│   │   ├── prompts/                 # versioned prompt templates (files) + registry loader
│   │   └── evals/                   # golden sets, metrics, eval runner
│   ├── mcp_server/                  # Model Context Protocol server (FastMCP)
│   ├── modules/
│   │   ├── control_plane/           # model registry, routes, flags, budgets, kill switches
│   │   ├── embeddings/              # embedding models, versions, re-embedding jobs
│   │   ├── knowledge/               # sources, documents, ingestion pipeline, chunks
│   │   ├── rag/                     # retrieval, rerank, grounded answers, citations
│   │   ├── agents/                  # agent definitions, runtime, tools, runs, steps
│   │   ├── hitl/                    # approval requests, review tasks, feedback, labels
│   │   ├── memory/                  # conversations, case memory, long-term memories
│   │   ├── auth/                    # router, service, schemas, repository, models
│   │   ├── users/
│   │   ├── organizations/           # orgs, memberships, branches
│   │   ├── rbac/                    # roles, permissions, policies
│   │   ├── products/                # registered products, batches, reference images
│   │   ├── scans/                   # upload, status, results
│   │   ├── verification/            # pipeline stages + risk scoring
│   │   ├── cases/                   # human review queue for high-risk scans
│   │   ├── hotspots/
│   │   └── audit/
│   └── workers/                     # Celery task modules
└── tests/
    ├── unit/
    ├── integration/                 # testcontainers Postgres + Redis
    ├── e2e/
    └── fixtures/                    # sample pack images (genuine, fake, blurry)
4. Domain model (summary)

Entities

User: consumer, health worker, org staff, regulator, platform admin. Phone is the consumer identity; email+password for staff.
Organization with a type: pharmacy, ppmv (patent medicine vendor), clinic, manufacturer, distributor, regulator. Regulators have a jurisdiction (national or a list of states).
Membership: user ↔ organization with a role. A user can belong to several organizations.
Branch / Location: physical premises of an org (PostGIS point).
Product: registered product (NAFDAC number unique, brand name, generic name, strength, dosage form, pack size, manufacturer org, registration status, expiry of registration).
Batch: manufacturer-declared batch (batch number, manufacture date, expiry date, quantity, distribution states).
ReferenceImage: verified genuine pack image of a product (S3 key, embedding vector, perceptual hash, side: front/back/blister).
Scan: one verification request (image key, phash, quality metrics, extracted fields, matched product/batch, signals, risk level, risk score, confidence, scorer version, privacy-safe location, status).
Case: review item opened for high-risk scans, assigned to a regulator or pharmacist reviewer, with lab-result outcome (feeds future model training labels).
HotspotSnapshot: periodic cluster result (centroid, radius, scan count, products involved, time window, severity).
AuditLog: append-only record of security-relevant actions.
OutboxEvent, IdempotencyKey, RefreshToken, ApiKey.

AI entities

EmbeddingModel: registered embedding model (name, modality image/text, dimension, version, status active/shadow/retired).
KnowledgeSource (e.g. NAFDAC alerts page, Greenbook export, manufacturer leaflets, internal SOPs) with visibility (public, org, regulator-only) and trust level.
Document (source, title, uri, content hash, published_at, version, status) and Chunk (text, section path, page, token count, embedding, tsvector, metadata such as NAFDAC numbers, batch numbers and brands mentioned).
ModelRoute / PromptTemplate / FeatureFlag / Budget: control-plane configuration.
LLMCall: one metered call (route, model, prompt version, tokens in/out, cache hits, cost, latency, outcome, guardrail results).
AgentRun / AgentStep: an agent execution and each tool call or model turn in it.
ApprovalRequest: an agent or system action waiting for a human decision.
Feedback / Label: human judgements on answers, verdicts and cases (training and eval data).
Conversation / Message / Memory: short-term and long-term memory records.

Verdicts (never say "safe" or "100% genuine"):

LIKELY_GENUINE
SUSPICIOUS
UNREGISTERED
INCONCLUSIVE (unreadable or poor-quality photo; tell the client to retake)
5. AI system principles (non-negotiable)
The verdict on a pack is deterministic and explainable. LLMs may extract fields from a hard image (validated by schema) but never decide the verdict. Agents and RAG never change a scan's verdict; only a human case review can.
Grounded or silent. RAG answers cite chunk ids; if retrieval is weak, the answer is "I don't have enough evidence", not a guess.
Least privilege for agents. Tools are typed service calls scoped by the caller's TenantContext and permissions. Agents cannot do anything the human using them cannot do.
Humans approve consequential actions: publishing an alert, notifying a manufacturer, closing a case, changing a verdict, approving reference images, adding knowledge from an untrusted source.
Untrusted text is data, not instructions. Pack text, OCR output, documents, web pages and memories are wrapped and labelled as untrusted in prompts and never allowed to change tools, roles or policy.
Every AI output is traceable: model, model version, prompt version, retrieved chunk ids, tools called, cost and the human who approved it.
Budgets are enforced before the call, not discovered on the invoice.
Every AI feature has a fallback and a kill switch and degrades to a safe, non-AI behaviour.
No PII in prompts, embeddings or memories unless the feature requires it and the data is minimised first.
6. Coding conventions
Type hints everywhere; mypy --strict passes.
Ruff for lint and format; no unused imports, no bare except.
async def on the request path; never call blocking I/O or CPU-heavy work inside the event loop. OCR, ONNX inference, the OpenCV quality gate and perceptual hashing run in Celery workers; the ml dependency group is installed only in the worker image. The API's only image work is Pillow validation and re-encoding plus a SHA-256 hash, inside run_in_threadpool.
Client guidance (not built here): the mobile app should run an on-device blur check before uploading to save a round trip. The worker's quality gate stays authoritative.
Uploads: only the Pillow re-encoded JPEG from validate_image_upload (core/uploads.py) is ever stored or passed on, never the original upload. JPEGs may carry trailing data (phones append depth maps and motion-photo video); the polyglot checks stay as extra defence. The request body limit (BodySizeLimitMiddleware) runs before multipart parsing. python-multipart must stay >= 0.0.18.
Domain errors are exceptions in core/errors.py mapped to RFC 9457 application/problem+json responses. Never leak stack traces, SQL or internal IDs of other tenants.
Problem types are urn:medlens:problem:<slug>. The prefix exists only as PROBLEM_TYPE_PREFIX in core/errors.py.
Deployment: one Uvicorn process per container, no Gunicorn: uvicorn medlens.main:create_app --factory --host 0.0.0.0 --port 8000 --proxy-headers. Scale by running more containers; set FORWARDED_ALLOW_IPS to the load balancer's address.
Testing: pytest-cov in the dev group, branch coverage, fail_under = 85 (pyproject.toml). CI runs pytest --cov.
IDs are UUIDv7 (time-ordered). Timestamps are timestamptz in UTC.
All list endpoints use cursor pagination.
All public routes live under /api/v1.
No secrets in code, tests or logs. Use SecretStr in settings.
Every external call (LLM, SMS, S3, OCR service) has a timeout, retry policy and circuit breaker.
Log events, not prose: log.info("scan.scored", scan_id=..., verdict=..., score=...).
Phone numbers are normalised to E.164 (+234...).
7. Known gotchas (follow these)
PgBouncer transaction mode + asyncpg: set statement_cache_size=0 in connect_args and prepared_statement_cache_size=0 in the SQLAlchemy URL/engine options, or queries will fail randomly under load.
Celery is not async: workers use the sync SQLAlchemy engine (psycopg 3). Do not call asyncio.run per task.
Load ONNX / OCR models once per worker process (worker_process_init signal), never per task.
pgvector: use an HNSW index with vector_cosine_ops; set hnsw.ef_search per query when needed.
PostGIS clustering: transform points to a metric CRS (EPSG:32632, UTM 32N covers most of Nigeria) before ST_ClusterDBSCAN so eps is in metres.
Partitioned scans table: the primary key must include the partition key (id, created_at).
Embedding dimensions differ per model. Store each embedding model's vectors in a table (or column) for that model version; never mix vectors from two models in one index. Re-embedding is a background migration, not an in-place update.
Anthropic prompt caching only pays off when the stable prefix (system prompt, tool definitions, static context) comes first and is byte-identical across calls. Put variable content last.
Model IDs change. Keep them in the model registry with aliases (fast, balanced, deep); check docs.claude.com for current IDs.
MCP streamable HTTP needs its own ASGI mount and lifespan; share auth with the API, not sessions.
Alembic autogenerate does not understand extensions, partitions, HNSW indexes or trigram indexes well: write those parts by hand with op.execute.
8. Definition of done (every task)
Code follows the architecture rules above.
uv run ruff check . && uv run ruff format --check . passes.
uv run mypy src passes.
uv run pytest --cov passes, with new tests for new behaviour (unit + integration where DB/Redis is involved). Coverage of changed code is 85% or higher.
AI features have an eval set, a fallback, a kill switch, cost metering and guardrail tests (prompt injection at minimum).
New endpoints have summaries, descriptions, examples and documented error responses in OpenAPI.
Migrations upgrade and downgrade cleanly.
You finish with a short summary: what changed, how to run it, what is left, and any decision I need to make.

Work one phase at a time. Do not start the next phase until I say so.

MVP scope (M0-M6, 2026-11-03)

The MVP is built on this codebase, not as a separate project: everything above still applies, and where the MVP prompts (MVP_PROMPTS.md, CLAUDE_MVP.md) differ from the existing design, the existing design stays and the difference is logged below. The MVP adds:
M3: the LLMGateway (control plane: tiers, versioned prompt files, budgets, kill switch, metering in llm_calls, circuit breaker, structured output with one repair turn), real readers in the worker (OpenCV quality gate, zxing-cpp GS1 DataMatrix, Tesseract when installed), vision-LLM extraction of missing fields, CLIP image embeddings, a fixed disclaimer on scan results, synthetic test packs (tests/fixtures/packs, seed/test_packs).
M4: verdict overrides through case lab results (audited), the case evidence photo, hotspots (PostGIS DBSCAN, cached), the stats summary, the manufacturer report.
M5: the knowledge base (document upload, worker ingestion, chunks with vectors and full-text search, published alerts as a scan signal), hybrid search with Reciprocal Rank Fusion, the grounded assistant.
M6: distributed rate limits, security headers, HTTP and scan metrics (API and workers), the AI usage report, scripts/eval.py, scripts/demo.sh, README.md.
Out of scope for the MVP, as in §1: agents, the MCP server, memory, any UI or messaging channel.

9. Decision log

Newest last. Each entry overrides the phase prompts and anything above that it contradicts. Every hard-to-reverse choice gets an entry with: date and title, Decision, Why, Alternatives rejected, Revisit when, Decided by ("owner", or "agent-proposed, owner-approved"; anything still awaiting review says so).

2026-11-02 Tesseract-first OCR
  Decision: Tesseract (pytesseract) is the primary OCR engine behind the OcrEngine port. PaddleOCR is a blocked candidate: it needs the paddlepaddle runtime, which is not installed.
  Why: avoid committing to a heavy runtime before measuring accuracy on real Nigerian packs.
  Alternatives rejected: PaddleOCR now (heavy runtime, unmeasured gain); a vision LLM as the primary reader (cost and latency on every scan; LLMs are for hard cases only).
  Revisit when: the OCR phase benchmark on 100+ real labelled pack photos (Tesseract vs PaddleOCR vs the vision-LLM fallback) is in.
  Decided by: owner.

2026-11-02 One OpenCV build
  Decision: only opencv-python-headless (<5) is installed; opencv-contrib-python, pulled in by paddleocr/paddlex, is blocked with a uv override in pyproject.toml.
  Why: both builds ship the same cv2 module and whichever installs last silently wins; paddlex is tested against OpenCV 4.
  Alternatives rejected: the contrib build everywhere (larger, unneeded modules); letting both install (non-deterministic).
  Revisit when: a contrib-only module is needed, or PaddleOCR is adopted and requires contrib.
  Decided by: agent-proposed, owner-approved.

2026-11-02 Worker-side quality gate
  Decision: the API does Pillow validation and re-encoding plus SHA-256 only. The quality gate (blur, brightness, size) and the perceptual hash run in the worker; the ml group stays worker-only. Supersedes Phase 5 steps 2-3, which put them in the API.
  Why: keeps the API image slim and its event loop free of CPU-heavy work.
  Alternatives rejected: the gate in the API via run_in_threadpool (OpenCV in the API image, CPU contention with requests); client-side checks only (not authoritative).
  Revisit when: retake feedback through the asynchronous path proves too slow for users.
  Decided by: owner.

2026-11-02 URN problem types
  Decision: RFC 9457 problem types are urn:medlens:problem:<slug>; the prefix exists only as PROBLEM_TYPE_PREFIX in core/errors.py.
  Why: stable identifiers clients can branch on, without depending on a documentation domain.
  Alternatives rejected: https URLs on a docs domain (needs a permanent domain and hosted pages; breaks if either changes); about:blank with a custom code field (no machine-readable type).
  Revisit when: public developer docs get a permanent domain (problem pages can be linked from the docs; the URNs stay).
  Decided by: agent-proposed, owner-approved.

2026-11-02 Coverage tooling
  Decision: pytest-cov in the dev group; branch coverage; fail_under = 85; CI runs pytest --cov.
  Why: the Definition of Done requires measured coverage.
  Alternatives rejected: line coverage only (misses untested branches); no threshold.
  Revisit when: the suite is split across CI jobs (coverage must then be combined).
  Decided by: agent-proposed, owner-approved.

2026-11-03 No Gunicorn
  Decision: one Uvicorn process per container: uvicorn medlens.main:create_app --factory --host 0.0.0.0 --port 8000 --proxy-headers; uvicorn-worker removed; FORWARDED_ALLOW_IPS is the load balancer's address.
  Why: Uvicorn's Gunicorn worker is deprecated, and the orchestrator already restarts and scales containers, so a process manager inside adds nothing.
  Alternatives rejected: Gunicorn with uvicorn-worker (deprecated); uvicorn --workers (a second process manager; per-process metrics need multiprocess mode).
  Revisit when: deploying somewhere without an orchestrator, where one host must run several processes.
  Decided by: owner.

2026-11-03 Trailing JPEG data and upload policy
  Decision: JPEG trailing data is allowed; the polyglot checks stay as extra defence; only the Pillow re-encoded bytes are ever stored or passed on; the body-size limit runs before multipart parsing; python-multipart >= 0.0.18.
  Why: Android phones append depth maps and motion-photo video after the JPEG end marker; re-encoding discards everything that is not pixels; 0.0.18 fixes CVE-2024-53981 (multipart DoS).
  Alternatives rejected: rejecting trailing data (fails many genuine phone photos); storing originals (keeps attacker-controlled bytes and EXIF location).
  Revisit when: a decoder vulnerability shows re-encoding does not neutralise a payload, or investigations need original files (that would need a separate quarantined store).
  Decided by: agent-proposed, owner-approved.

2026-11-03 Phase 0-1 defaults
  Decision: health probes live at /health/* outside /api/v1. Module routers are built with core.routing.api_v1_router() and carry their full path. Local MinIO is Chainguard's image pinned by digest. Nigerian states are ISO 3166-2:NG codes (NG-LA); names are accepted on input.
  Why: probes must not change with the API version; FastAPI no longer flattens included routers, so a prefix added at include time is missing from route templates in logs and metrics; upstream MinIO stopped publishing images in October 2025; ISO codes are unambiguous.
  Alternatives rejected: probes under /api/v1; prefixes at include time; unpinned or unmaintained MinIO images; free-text state names.
  Revisit when: FastAPI flattens routers again, or Chainguard's free MinIO image goes away.
  Decided by: agent-proposed, owner-approved.

2026-11-03 Environment files (amended by "Secrets have no defaults" below)
  Decision: .env.example documents every variable (application and docker compose) with its default and Required. and Secret. markers, and holds no real values. Local values live only in the gitignored .env, created or updated by `make dev-secrets` (scripts/sync_env.py; `make env` is an alias): it keeps existing values, adds new variables, generates random local secrets, backs up the old file, writes mode 600, and refuses a .env for staging or production. docker compose takes credentials only from .env. Tests fail if the template drifts from the code or compose.
  Why: no secrets or environment-specific values in the repository, and one documented contract for every environment.
  Alternatives rejected: a committed .env with local values; compose fallback credentials.
  Revisit when: deployments move to a secret manager that should also feed local development.
  Decided by: owner (request), agent-proposed design, owner-approved.

2026-11-03 Secrets have no defaults
  Decision: every secret is a required setting with no default in code: database password, S3 secret key, JWT signing keys, field-encryption keys, HMAC pepper. .env.example shows the placeholder CHANGE_ME for them, which every environment rejects. Staging and production also reject well-known values, and secrets we choose ourselves shorter than 32 characters (provider-issued API keys and DSNs are exempt from the length rule). Tests generate random secrets per run. Non-secret settings keep their defaults.
  Why: a default secret in code is a secret in Git, and a deployment that forgets one should fail at startup, not run on a known key.
  Alternatives rejected: development defaults rejected only in staging and production (the previous approach: secrets still in code, local and test running on public keys); an empty template (no hint of the expected shape).
  Revisit when: a secret manager integration replaces environment variables.
  Decided by: owner.

2026-11-03 Token revocation while Redis is down
  Decision: session revocation lives in Redis. If Redis cannot answer, consumer actions fail open (a revoked consumer token works until it expires, at most 15 minutes). Sessions holding any permission beyond scan:create, scan:read_own and product:read, and routes that change credentials or mint tokens (password change, account deletion, MFA enrolment, switch organization), fall back to Postgres: access tokens issued before users.tokens_valid_after are rejected. That timestamp is stamped whenever all of a user's sessions end (log out everywhere, password change, erasure). Every skipped check increments medlens_auth_revocation_checks_skipped_total; fallbacks are counted by result; alerts MedLensRedisDown (critical) and MedLensRevocationChecksSkipped (warning) are in docker/observability/alerts.yml.
  Why: a Redis outage must not take scanning down; the consumer exposure is bounded by the token lifetime, while privileged actions are where a revoked token does damage.
  Alternatives rejected: fail closed (a Redis outage becomes an API outage); fail open for everyone (revoked staff tokens keep privileged access); always check Postgres (database load on the hottest path).
  Consequences, agent-proposed and awaiting owner review: readiness reports a Redis outage as "degraded" (HTTP 200) so instances stay in the load balancer; the API probes its dependencies every 15 s for the medlens_dependency_up gauge; /metrics is served on the API port and must not be routed publicly; switch-organization always requires a live session in Postgres; single-session logouts are visible in Redis only.
  Revisit when: the access-token lifetime grows past 15 minutes; Redis becomes highly available enough to fail closed; or revocations published during an outage need replaying (Phase 6 outbox).
  Decided by: owner (the policy, metric and alert); consequences as noted.

2026-11-03 Manufacturers cannot see raw scans
  Decision: manufacturer roles hold no scan:read_org, scan:read_all or scan:read_signals, not even for their own products. Roadmap: reports:read_own, an aggregated per-product report of flagged-scan counts by state and ISO week, with cells under 5 suppressed.
  Why: raw scans carry locations and reveal who checks which medicines; manufacturers need trends, not individual scans; signals would also teach counterfeiters what the checks look for.
  Alternatives rejected: raw scans of own products (re-identification risk); no manufacturer insight at all (they are key partners in recalls).
  Revisit when: the reports endpoint is built, or a manufacturer needs scan-level evidence (that goes through a regulator's case).
  Decided by: owner.

2026-11-03 Audit log access and immutability
  Decision: reading the audit log (audit:read) is platform-admin only for the MVP. audit_logs is append-only: a trigger blocks UPDATE, DELETE and TRUNCATE for every role (migration 0005), and the application role has none of those privileges (0006). Roadmap: audit:read_regulatory for regulators, limited to recall, approval and product-status entries in their jurisdiction.
  Why: audit entries hold security events of every tenant; immutability must not depend on application code.
  Alternatives rejected: regulators reading the whole log (other tenants' security events); append-only enforced in application code only (bypassable).
  Revisit when: regulators need recall and approval trails (ship audit:read_regulatory), or a retention law requires purging (needs a separate maintenance role, never the application role).
  Decided by: owner.

2026-11-03 No distributor role
  Decision: distributors use the pharmacy roles (pharmacy_staff and pharmacy_admin may be held in distributor organizations).
  Why: until bulk stock verification exists their needs match a pharmacy's; a separate role would duplicate the matrix.
  Alternatives rejected: a distributor role now (no distinct permissions yet); no access for distributors.
  Revisit when: bulk stock verification is built.
  Decided by: owner.

2026-11-03 Greenbook data handling
  Decision: Git holds only a 50-product sample (seed/products.csv) and its provenance (seed/PROVENANCE.md: source, date, terms). Full snapshots live outside Git; seed/greenbook-snapshot.json names the current one with its SHA-256 and row count, and `make greenbook-import` (python -m medlens.seed.greenbook import) refuses any file that does not match. The import upserts product facts only: it never creates organizations or removes a manufacturer link.
  Why: the register publishes no terms ("(c) NAFDAC, all rights reserved"); the checksum makes every import reproducible and tamper-evident; the repository stays small.
  Alternatives rejected: the full snapshot in Git or Git LFS (redistribution, repository bloat); fetching the live site during imports (not reproducible, depends on a third-party site's uptime).
  Revisit when: NAFDAC publishes terms or an API, or a data-sharing agreement is signed.
  Decided by: owner. Storage location agent-proposed, awaiting owner choice: private object storage rather than a public release file, until the terms allow redistribution.

2026-11-03 Scan submission contract (Phase 5)
  Decision: POST /api/v1/scans answers 202 with a status URL; the verdict comes from the worker and clients poll. Guests must send X-Device-Id (an app-generated id, stored only as a keyed hash): it lets them read their own scans and scopes duplicates and idempotency keys. Idempotency-Key replays for 24 hours; the same key with a different request is a 409 (urn:medlens:problem:idempotency-key-reused). The same photo (SHA-256) from the same person or device within 24 hours returns the earlier scan with 200; a near-identical photo (perceptual hash within 6 bits, checked in the worker) gets the earlier answer. Coordinates are snapped to a 0.01-degree cell plus geohash-6 before storage. The state comes from the client form field; coordinates are not mapped to a state until boundary data is loaded.
  Why: no consumer talks to an LLM or waits on OCR in the request path; guests need a stable handle without accounts; probing with edited photos must teach nothing.
  Alternatives rejected: synchronous verification in the request (ties API latency to OCR; the Phase 6 fast path covers short queues); guest result tokens per scan (a device id also serves dedupe and idempotency); nearest-state-capital lookup (wrong near borders, and wrong data is worse than none).
  Revisit when: state boundary data is loaded (derive the state from coordinates), or clients cannot send a device id.
  Decided by: agent-proposed, awaiting owner review.

2026-11-03 Manufacturer submissions need human review (Phase 5)
  Decision: a manufacturer is linked to a product only through a product claim approved by a person holding product:review (regulator_admin, platform_admin); reference images are used for visual matching only after the same approval. product:review cannot be delegated to API keys. Batch numbers are not public: only the product's manufacturer, regulators and platform admins see them. Batch CSV imports are all or nothing with a per-row report. Regulator organizations are created by platform admins only; other organizations start pending_verification and grant nothing until a platform admin verifies them.
  Why: claims and reference images change what the verdict trusts (CLAUDE.md §5: humans approve consequential actions); batch numbers would help counterfeiters label fakes.
  Alternatives rejected: self-service product links; partial CSV imports (a half-applied file is hard to reason about and to retry).
  Revisit when: manufacturers onboard at volume and review becomes the bottleneck (consider evidence-based auto-approval with sampling).
  Decided by: agent-proposed, awaiting owner review.

2026-11-03 Invitations without messaging (Phase 5)
  Decision: invitations go to a phone number or e-mail address, return their token once to the inviter, and are accepted by a signed-in account with that phone or e-mail. MedLens sends nothing (CLAUDE.md §1 allows only the login OTP). Tokens expire after 7 days; a new invitation replaces an open one for the same person.
  Alternatives rejected: sending invitation SMS or e-mail (out of scope); accepting by token alone (a leaked token would let anyone join).
  Revisit when: an e-mail channel is approved.
  Decided by: agent-proposed, awaiting owner review.

2026-11-03 Scoring v1 and case routing (Phase 5)
  Decision: WeightedRuleScorer v1 as documented in docs/scoring.md. Hard rules: no registered record with a clearly read number is UNREGISTERED; registration revoked and a NAFDAC number belonging to another product are SUSPICIOUS with high priority; an expired pack is SUSPICIOUS. A lapsed registration is a weighted signal, not a hard rule. Thresholds 0.3 and 0.6 are settings. SUSPICIOUS high-priority and UNREGISTERED scans open one case, routed to the most local active regulator covering the scan's state (national for none). Regulators see cases in their jurisdiction plus those routed to them. Lab results become labels and never change the stored verdict.
  Why: about a third of Greenbook registrations are past their expiry date while products stay on sale during renewal, so a hard rule would flag a third of the market; local agencies act faster.
  Alternatives rejected: expired registration as a hard rule (see Why); cases visible to the assigned regulator only (state agencies could not see cases routed nationally).
  Revisit when: enough lab-confirmed labels exist to fit and calibrate a model scorer.
  Decided by: agent-proposed, awaiting owner review.

2026-11-03 Queue architecture (Phase 6)
  Decision: six queues (scans.pipeline, ml.ocr, ml.embed, llm, events, maintenance) on Redis with JSON messages, late acknowledgement, re-delivery when a worker dies, prefetch 1, soft and hard time limits, a visibility timeout longer than the longest task, and ignored results (business results live in Postgres). The whole verification pipeline runs in one scans.pipeline task on ML-capable workers; ml.ocr and ml.embed take standalone heavy jobs (reference-image embeddings, benchmarks, re-embedding). Side effects go through the transactional outbox; Beat runs the relay every second (SELECT ... FOR UPDATE SKIP LOCKED, exponential backoff, dead-lettering after 10 attempts). Each outbox event stores the request context, so tasks log under the original request id. Tasks retry only transient errors (storage, database, Redis, timeouts) with exponential backoff and jitter; final failures land in failed_tasks, listed and requeued by platform admins. Backpressure: above 200 queued scans the estimated wait grows, above 2000 POST /scans answers 503 with Retry-After. The sync fast path (wait for the verdict on Redis pub/sub) exists but is off by default (MEDLENS_QUEUE__SYNC_WAIT_SECONDS=0). Revocations are also written to the outbox, so revocations made while Redis is down reach it later.
  Why: one task per scan fetches the photo once and avoids hand-offs between queues, at the cost of scaling OCR and orchestration together; queue separation keeps a backlog of one kind of work from starving the others.
  Alternatives rejected: a chain of tasks per pipeline stage on the ml queues (image shuttling and latency for little gain at MVP volume); storing results in the Celery backend (two sources of truth); enabling the fast path by default (ties API workers to the queue; measure first).
  Revisit when: OCR dominates worker CPU and needs its own scaling (split the OCR stage onto ml.ocr), or measured p50 scan latency makes the fast path worth enabling.
  Decided by: agent-proposed (the phase asked for this trade-off), awaiting owner review.

2026-11-03 MVP on the current codebase
  Decision: implement MVP phases M3-M6 on the existing modules; keep the existing design where the MVP prompts differ (logged in the entries below); add an MVP scope section to this file.
  Why: M0-M2 and most of M3-M4 already existed in a stricter form; a rewrite would lose tested security work.
  Alternatives rejected: a separate MVP codebase; replacing CLAUDE.md with CLAUDE_MVP.md.
  Revisit when: never for the MVP; later phases follow the main plan.
  Decided by: owner.

2026-11-03 MVP model and reader stack
  Decision: image embeddings are clip-ViT-B-32 (512-d) and text embeddings multilingual-e5-small (384-d), both through sentence-transformers on CPU torch (pytorch-cpu index) in the worker-only ml group; OCR is Tesseract (pytesseract, apt tesseract-ocr in the worker image) and is optional at run time; paddleocr moved to its own ocr-benchmark group, out of every image. Dimensions are settings read by migrations; migration 0012 resizes reference_images.embedding to the configured dimension and clears old vectors.
  Why: sentence-transformers loads both models with one dependency and no export step; the CPU wheel keeps the worker image far smaller than CUDA torch.
  Alternatives rejected: ONNX Runtime exports now (an export and parity check per model before any measurement); DINOv2 (no text tower if CLIP text search is wanted later); PaddleOCR in the worker (blocked by the OCR decision above).
  Revisit when: worker CPU or image size matters (export both models to ONNX Runtime), or the OCR benchmark is in.
  Decided by: agent-proposed, awaiting owner review.

2026-11-03 LLM gateway behaviour
  Decision: the Anthropic SDK's own retries (MEDLENS_LLM__MAX_RETRIES, transient errors only, honouring Retry-After) inside a request timeout, plus a purgatory circuit breaker, instead of tenacity; structured output through output_config.format with the Pydantic model's JSON schema and one repair turn; tiers fast (claude-haiku-5-5) and balanced (claude-sonnet-5-5) in settings with per-model prices; the system prompt is the cached prefix and untrusted content is wrapped in <untrusted_*> tags after it; server-side refusal fallbacks are enabled on the balanced tier (MEDLENS_LLM__REFUSAL_FALLBACK). AI is off by default (MEDLENS_LLM__ENABLED=false).
  Why: SDK retries already classify retryable errors correctly, and stacking tenacity on them multiplies attempts; schema-constrained output removes most repair turns.
  Alternatives rejected: tenacity around the SDK (double retries); prefilled JSON (not supported with structured output); LiteLLM (one provider today).
  Revisit when: a second provider is added (LiteLLM adapter), or refusal fallbacks change cost or behaviour noticeably.
  Decided by: agent-proposed, awaiting owner review.

2026-11-03 Sync fast path on (supersedes part of "Queue architecture")
  Decision: MEDLENS_QUEUE__SYNC_WAIT_SECONDS defaults to 8: POST /scans waits up to 8 s for the worker's result when the queue is short, then falls back to 202 and polling.
  Why: most scans finish in seconds; one round trip is a better client experience.
  Alternatives rejected: polling only (the previous default).
  Revisit when: API containers spend too long waiting under load (lower it or turn it off).
  Decided by: owner.

2026-11-03 Cases open through the outbox
  Decision: high-risk scans open their case in a follow-up task triggered by the scan.completed outbox event (same commit as the verdict), not in the scan's own transaction as M4 asks.
  Why: the event is committed with the verdict, so the case cannot be lost; opening it separately keeps routing and its failures out of the scoring transaction.
  Alternatives rejected: case creation in the scoring transaction (a routing error would roll back the verdict).
  Revisit when: regulators need the case visible at the same instant as the verdict.
  Decided by: agent-proposed, awaiting owner review.

2026-11-03 Verdict overrides (amends "Scoring v1 and case routing")
  Decision: a lab result recorded on a case may set override_verdict (counterfeit: SUSPICIOUS or UNREGISTERED; genuine: LIKELY_GENUINE; inconclusive: none). The pipeline's verdict stays stored; scans.verdict_override is what clients see, with verdict_overridden; regulators also see the pipeline's verdict and the case. Every override is audited (scan.verdict_overridden). Hotspots, statistics and reports count the verdict shown.
  Why: M4 requires that only a case decision overrides a verdict; keeping both values keeps the scorer's output auditable and re-scorable.
  Alternatives rejected: overwriting scans.verdict (loses the scorer's output); overrides without a lab result (no evidence behind them).
  Revisit when: overrides need a second reviewer.
  Decided by: agent-proposed, awaiting owner review.

2026-11-03 Scorer v2: published alerts
  Decision: weighted-rules-v2 adds one hard rule: a published alert or recall naming the product (all batches) or the pack's batch makes it SUSPICIOUS with high priority (reason PUBLISHED_ALERT). Alerts come from documents of kind alert or recall: labelled NAFDAC numbers paired with the labelled batch numbers (containing a digit) in the same chunk.
  Why: M5 requires it; high priority routes the scan to a regulator.
  Alternatives rejected: a weighted signal only (an alert is direct evidence).
  Revisit when: alerts produce false matches (then require a reviewer to confirm extracted alerts).
  Decided by: agent-proposed, awaiting owner review.

2026-11-03 Knowledge ingestion and search
  Decision: uploads are stored and ingested by a Celery task (outbox event knowledge.document_uploaded) on the ml.embed queue, not a FastAPI BackgroundTask; the API gets question embeddings from a worker (task medlens.knowledge.embed_query, result backend, 5 s timeout) and searches on keywords and NAFDAC numbers alone when that fails; chunks without vectors (no model, seeded documents) are embedded by an hourly backfill. Documents are professional or regulator-only. PDFs with scripts, attachments or forms are refused; scanned PDFs without a text layer fail as unreadable (no OCR for documents yet).
  Why: the embedding model is worker-only (CLAUDE.md §6), and a BackgroundTask dies with its API container.
  Alternatives rejected: embedding models in the API image; BackgroundTasks; failing searches when no worker answers.
  Revisit when: question latency matters (a small ONNX query encoder in the API), or scanned notices arrive (document OCR).
  Decided by: agent-proposed, awaiting owner review.

2026-11-03 Rate limiting
  Decision: a custom Redis sliding window (one Lua script, Redis clock) instead of slowapi: guest scans 10/min per IP, signed-in scans 30/min per user or API key, sign-in 5/min per IP (codes, passwords, MFA), assistant 10/min per user; 429 problem+json with Retry-After. Checks fail open when Redis is down, counted in medlens_ratelimit_checks_skipped_total. Client IPs come from Uvicorn, which trusts X-Forwarded-For only from FORWARDED_ALLOW_IPS; IPs are hashed in Redis keys. Rate limits are off in test settings except in their own tests.
  Why: slowapi's Redis storage is synchronous (it would block the event loop) and keys only by IP or user, not both per route.
  Alternatives rejected: slowapi; fastapi-limiter (fixed windows, extra dependency); failing closed (a Redis outage would stop scanning).
  Revisit when: abuse needs per-device or per-organization limits.
  Decided by: agent-proposed, awaiting owner review.

2026-11-03 MVP naming kept from the existing design
  Decision: reason codes keep their existing names (PACK_EXPIRED, BATCH_NOT_DECLARED, BATCH_SEEN_WIDELY, BLURRY/TOO_DARK/TOO_BRIGHT/TOO_SMALL) rather than the MVP examples (EXPIRED, BATCH_UNKNOWN, BATCH_SEEN_IN_MANY_STATES, PHOTO_UNCLEAR); the disclaimer is a stable code (AUTOMATED_ESTIMATE_NOT_A_LAB_TEST) plus English text; object storage stays S3/MinIO, not local disk; the manufacturer report is product x state x ISO week with cells under 5 suppressed.
  Why: clients translate stable codes (§1); the codes were already documented and tested.
  Alternatives rejected: renaming codes for the MVP.
  Revisit when: a client integration needs different codes (add aliases, never rename).
  Decided by: agent-proposed, awaiting owner review.

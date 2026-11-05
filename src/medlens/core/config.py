"""Application settings.

Settings come from environment variables (or a local ``.env`` file) named
``MEDLENS_<GROUP>__<FIELD>``, for example ``MEDLENS_DATABASE__HOST``. ``.env.example`` documents
every variable.

Invalid or missing settings stop the process at startup with a readable message (``SettingsError``)
that names the offending variables and never echoes their values.

Secrets have no defaults in code: every environment must provide them (locally, ``make dev-secrets``
generates them into ``.env``). ``.env.example`` shows ``SECRET_PLACEHOLDER`` in their place, and
that value is rejected everywhere.
"""

import base64
import binascii
import re
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Any, Literal, Self
from urllib.parse import quote

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "MEDLENS_"
ENV_NESTED_DELIMITER = "__"

# .env.example shows this value in place of every required secret. It is rejected everywhere.
SECRET_PLACEHOLDER = "CHANGE_ME"  # noqa: S105  # pragma: allowlist secret
# Staging and production reject shorter secrets we choose ourselves (see SelfChosenSecret).
MIN_SECRET_LENGTH = 32
# Rejected as secrets in staging and production.
_WEAK_SECRETS = frozenset(
    {
        "password",
        "changeme",
        "change-me",
        "change_me",
        "secret",
        "admin",
        "postgres",
        "minioadmin",
        "test",
    }
)
_PLACEHOLDER_MESSAGE = (
    "is still the .env.example placeholder: run `make dev-secrets` locally, or set a real secret"
)


def env_var(*path: str) -> str:
    """Environment variable for a settings path: ("app", "debug") -> MEDLENS_APP__DEBUG."""
    return ENV_PREFIX + ENV_NESTED_DELIMITER.join(part.upper() for part in path)


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class _Group(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _SelfChosen:
    """Marks a secret we choose ourselves rather than one a provider issues (API keys, DSNs).
    Staging and production require at least MIN_SECRET_LENGTH characters."""


def _reject_placeholder(value: SecretStr) -> SecretStr:
    if value.get_secret_value() == SECRET_PLACEHOLDER:
        raise ValueError(_PLACEHOLDER_MESSAGE)
    return value


_SELF_CHOSEN = _SelfChosen()
# A required secret with no default: .env.example shows the placeholder instead.
SelfChosenSecret = Annotated[SecretStr, AfterValidator(_reject_placeholder), _SELF_CHOSEN]

_KEY_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_KEY_BYTES = 32  # AES-256 keys and Ed25519 seeds


def _validate_key_map(value: dict[str, SecretStr], kind: str) -> dict[str, SecretStr]:
    """key id -> base64 of 32 bytes, at least one entry."""
    if not value:
        raise ValueError(f"at least one {kind} is required")
    for key_id, key in value.items():
        if not _KEY_ID.fullmatch(key_id):
            raise ValueError(f"key id {key_id!r} must be 1-32 letters, digits, '-' or '_'")
        if key.get_secret_value() == SECRET_PLACEHOLDER:
            raise ValueError(f"key {key_id!r} {_PLACEHOLDER_MESSAGE}")
        try:
            decoded = base64.b64decode(key.get_secret_value(), validate=True)
        except binascii.Error:
            raise ValueError(f"key {key_id!r} is not valid base64") from None
        if len(decoded) != _KEY_BYTES:
            raise ValueError(f"key {key_id!r} must decode to {_KEY_BYTES} bytes")
    return value


class AppSettings(_Group):
    environment: Environment
    debug: bool = False
    # None = automatic: docs are served everywhere except production.
    docs_enabled: bool | None = None
    cors_origins: list[str] = Field(default_factory=list)
    max_request_body_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    # Image uploads: file size, and decoded pixels (decompression-bomb guard).
    max_upload_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=50_000_000, gt=0)
    readiness_timeout_seconds: float = Field(default=2.0, gt=0, le=30)

    @model_validator(mode="after")
    def _upload_fits_in_body(self) -> Self:
        if self.max_upload_bytes >= self.max_request_body_bytes:
            raise ValueError(
                "max_upload_bytes must be smaller than max_request_body_bytes "
                "(multipart encoding adds overhead)"
            )
        return self

    @property
    def show_docs(self) -> bool:
        if self.docs_enabled is not None:
            return self.docs_enabled
        return self.environment is not Environment.PRODUCTION


class DatabaseSettings(_Group):
    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    name: str = "medlens"
    user: str = "medlens"
    password: SelfChosenSecret
    # True when connecting through PgBouncer in transaction mode (disables prepared statements).
    pgbouncer: bool = False
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=5, ge=0)
    pool_timeout_seconds: float = Field(default=10.0, gt=0)
    pool_recycle_seconds: int = Field(default=1800, gt=0)
    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    command_timeout_seconds: float = Field(default=30.0, gt=0)


class RedisSettings(_Group):
    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    password: Annotated[SecretStr | None, _SELF_CHOSEN] = None
    tls: bool = False
    # Separate logical databases so cache flushes never touch queued tasks.
    cache_db: int = Field(default=0, ge=0, le=15)
    broker_db: int = Field(default=1, ge=0, le=15)
    result_db: int = Field(default=2, ge=0, le=15)
    # Redis answers in milliseconds: short timeouts and one quick retry keep requests fast
    # when it is down (callers fail open), instead of the client's default ten retries.
    socket_timeout_seconds: float = Field(default=1.0, gt=0)
    socket_connect_timeout_seconds: float = Field(default=0.5, gt=0)
    retries: int = Field(default=1, ge=0, le=5)
    max_connections: int = Field(default=50, ge=1)

    @model_validator(mode="after")
    def _distinct_databases(self) -> Self:
        if len({self.cache_db, self.broker_db, self.result_db}) != 3:
            raise ValueError("cache_db, broker_db and result_db must all be different")
        return self

    def url(self, db: int) -> str:
        """Connection URL for one logical database. Contains the password: never log it."""
        scheme = "rediss" if self.tls else "redis"
        auth = ""
        if self.password is not None:
            auth = f":{quote(self.password.get_secret_value(), safe='')}@"
        return f"{scheme}://{auth}{self.host}:{self.port}/{db}"


class S3Settings(_Group):
    # None means AWS S3. Set it for MinIO (local) or Cloudflare R2.
    endpoint_url: str | None = None
    region: str = "us-east-1"
    access_key_id: str = "medlens-local"
    secret_access_key: SelfChosenSecret
    bucket: str = "medlens-local"
    addressing_style: Literal["auto", "path", "virtual"] = "path"
    connect_timeout_seconds: float = Field(default=2.0, gt=0)
    read_timeout_seconds: float = Field(default=10.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)


class AuthSettings(_Group):
    jwt_issuer: str = "medlens"
    jwt_audience: str = "medlens-api"
    # Ed25519 signing keys: key id -> base64 of a 32-byte seed (`python -m medlens.core.crypto`
    # prints one). Tokens are signed with jwt_active_key_id; every listed key still verifies,
    # so a retired key stays until the tokens it signed have expired.
    jwt_keys: dict[str, SecretStr]
    jwt_active_key_id: str = "local-dev"
    access_token_ttl_seconds: int = Field(default=900, ge=60, le=3600)
    refresh_token_ttl_days: int = Field(default=30, ge=1, le=90)
    mfa_token_ttl_seconds: int = Field(default=300, ge=60, le=900)
    mfa_issuer: str = "MedLens"
    otp_ttl_seconds: int = Field(default=300, ge=60, le=900)
    otp_max_attempts: int = Field(default=5, ge=1, le=10)
    # Progressive lockout: after login_max_failures failed logins within the window (per
    # account and per IP), lock for login_lockout_base_seconds, doubling each time up to the max.
    login_max_failures: int = Field(default=5, ge=1)
    login_failure_window_seconds: int = Field(default=900, ge=60)
    login_lockout_base_seconds: int = Field(default=60, ge=1)
    login_lockout_max_seconds: int = Field(default=3600, ge=1)

    @field_validator("jwt_keys")
    @classmethod
    def _valid_jwt_keys(cls, value: dict[str, SecretStr]) -> dict[str, SecretStr]:
        return _validate_key_map(value, "JWT signing key")

    @model_validator(mode="after")
    def _active_jwt_key_exists(self) -> Self:
        if self.jwt_active_key_id not in self.jwt_keys:
            raise ValueError(f"jwt_active_key_id {self.jwt_active_key_id!r} is not in jwt_keys")
        return self


class SmsSettings(_Group):
    # console logs messages instead of sending them (local and test only).
    provider: Literal["console", "termii", "africastalking"] = "console"
    sender_id: str = "MedLens"
    timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    termii_api_key: SecretStr | None = None
    termii_base_url: str = "https://api.ng.termii.com"
    africastalking_username: str | None = None
    africastalking_api_key: SecretStr | None = None
    africastalking_base_url: str = "https://api.africastalking.com"

    @model_validator(mode="after")
    def _provider_credentials(self) -> Self:
        if self.provider == "termii" and self.termii_api_key is None:
            raise ValueError("termii_api_key is required when provider is termii")
        if self.provider == "africastalking" and not (
            self.africastalking_username and self.africastalking_api_key
        ):
            raise ValueError(
                "africastalking_username and africastalking_api_key are required when provider "
                "is africastalking"
            )
        return self


class ModelPrice(_Group):
    """US dollars per million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_write: float = Field(ge=0)
    cache_read: float = Field(ge=0)


def _default_prices() -> dict[str, ModelPrice]:
    return {
        "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50, cache_write=0.125, cache_read=0.01),
        "claude-sonnet-5-5": ModelPrice(input=2.0, output=10.0, cache_write=2.5, cache_read=0.20),
    }


class LLMSettings(_Group):
    # Global AI kill switch. When off, every AI feature falls back to its non-AI behaviour.
    enabled: bool = False
    anthropic_api_key: SecretStr | None = None
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    # Model tiers: routes name a tier, never a model id (ai/gateway/registry.py).
    model_fast: str = Field(default="claude-haiku-5-5", min_length=1)
    model_balanced: str = Field(default="claude-sonnet-5-5", min_length=1)
    # Calls stop for the rest of the UTC day once the day's metered cost reaches this.
    daily_budget_usd: float = Field(default=5.0, ge=0)
    prices: dict[str, ModelPrice] = Field(default_factory=_default_prices)
    # Server-side refusal fallback on the balanced tier (Claude API only).
    refusal_fallback: bool = True
    # Validated answers of cacheable routes (the assistant) are reused for identical requests
    # (same model, prompt version, documents and question) for this long; 0 turns it off.
    answer_cache_seconds: int = Field(default=3600, ge=0)

    @model_validator(mode="after")
    def _priced(self) -> Self:
        missing = [m for m in (self.model_fast, self.model_balanced) if m not in self.prices]
        if missing:
            raise ValueError(f"prices has no entry for {', '.join(missing)}")
        return self


class ObservabilitySettings(_Group):
    service_name: str = "medlens"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # auto = JSON in staging/production, human-readable console output elsewhere.
    log_format: Literal["auto", "json", "console"] = "auto"
    otlp_endpoint: str | None = None
    traces_sample_rate: float = Field(default=0.1, ge=0, le=1)
    sentry_dsn: SecretStr | None = None
    # How often the API probes its dependencies for the medlens_dependency_up metric; 0 = never.
    dependency_probe_interval_seconds: float = Field(default=15.0, ge=0)
    # Celery workers serve their metrics on this port (0: off); see workers/metrics.py.
    worker_metrics_port: int = Field(default=9100, ge=0, le=65535)


class RateLimitSettings(_Group):
    enabled: bool = True
    # Requests per minute (sliding window). Client IPs come from Uvicorn, which trusts
    # X-Forwarded-For only from FORWARDED_ALLOW_IPS (the load balancer).
    guest_scans_per_minute: int = Field(default=10, ge=1)  # per client IP
    user_scans_per_minute: int = Field(default=30, ge=1)  # per user or API key
    logins_per_minute: int = Field(default=5, ge=1)  # per client IP
    assistant_per_minute: int = Field(default=10, ge=1)  # per user


class CryptoSettings(_Group):
    # Field-level encryption keys for PII columns: key id -> base64 of 32 random bytes
    # (`python -m medlens.core.crypto` prints a new one). New values are encrypted with
    # active_key_id; every listed key can still decrypt, which is how keys are rotated.
    field_keys: dict[str, SecretStr]
    active_key_id: str = "local-dev"
    # HMAC pepper for blind indexes (phone lookup) and IP hashes. Changing it orphans every
    # stored hash, so it is not rotated like the field keys.
    hmac_pepper: SelfChosenSecret

    @field_validator("field_keys")
    @classmethod
    def _valid_keys(cls, value: dict[str, SecretStr]) -> dict[str, SecretStr]:
        return _validate_key_map(value, "field encryption key")

    @field_validator("hmac_pepper")
    @classmethod
    def _long_pepper(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 16:
            raise ValueError("the HMAC pepper must be at least 16 characters")
        return value

    @model_validator(mode="after")
    def _active_key_exists(self) -> Self:
        if self.active_key_id not in self.field_keys:
            raise ValueError(f"active_key_id {self.active_key_id!r} is not in field_keys")
        return self


class EmbeddingSettings(_Group):
    # Models loaded once per worker process (the `ml` group). Without them, visual matching,
    # semantic search and reranking are unavailable and degrade gracefully.
    enabled: bool = True
    # onnx: the exports in onnx_dir (scripts/export_onnx.py, baked into the worker image);
    # torch: sentence-transformers. onnx falls back to torch when the exports are missing.
    backend: Literal["onnx", "torch"] = "onnx"
    onnx_dir: str = "/opt/models/onnx"
    onnx_threads: int = Field(default=2, ge=1, le=32)  # per model, per worker process
    # ONNX Runtime's CPU memory arena keeps peak allocations for reuse. Measured on the worker
    # image (three models, one process, after a large rerank batch): on, about 1.9 GB resident
    # and fastest; off, about 0.9 GB and 50-60% slower on large batches. Turn it off on hosts
    # short of memory.
    onnx_memory_arena: bool = True
    image_model: str = "clip-ViT-B-32"
    # Migrations read the dimensions when creating vector columns; changing a model's
    # dimension needs a new migration. pgvector HNSW indexes take at most 2000 dimensions.
    image_dimension: int = Field(default=512, ge=1, le=2000)
    text_model: str = "intfloat/multilingual-e5-small"
    text_dimension: int = Field(default=384, ge=1, le=2000)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class KnowledgeSettings(_Group):
    """Document ingestion and retrieval for the assistant (modules/knowledge)."""

    # Keep below MEDLENS_APP__MAX_REQUEST_BODY_BYTES, which applies first (413).
    max_document_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    # The assistant's own kill switch (MEDLENS_LLM__ENABLED switches off every AI feature).
    assistant_enabled: bool = True
    chunk_tokens: int = Field(default=400, ge=50, le=2000)
    chunk_overlap_tokens: int = Field(default=60, ge=0)
    # Candidates per retriever, results kept after fusion, and the evidence bar.
    candidates: int = Field(default=20, ge=1, le=200)
    top_k: int = Field(default=6, ge=1, le=50)
    min_similarity: float = Field(default=0.80, ge=0, le=1)
    rrf_k: int = Field(default=60, ge=1)
    # The API embeds questions through the worker; slower than this, search runs without it.
    query_embedding_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    # Re-order the best fused candidates with the cross-encoder (in a worker); slower than the
    # timeout, the fused order stands.
    rerank: bool = True
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    rerank_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    # The assistant's evidence bar on reranked results: passages the cross-encoder scores
    # below it are not shown to the model, and with none left it answers "not enough
    # evidence" without a model call. Passages naming the asked NAFDAC number always count.
    # Measured on seed/eval_questions.csv: answerable questions' best passage 0.96 or more,
    # unanswerable ones 0.02 or less. Cosine similarity cannot separate them (0.80-0.92).
    min_rerank_score: float = Field(default=0.1, ge=0, le=1)
    # Question embeddings are cached in Redis (per model) for this long; 0 turns it off.
    query_cache_seconds: int = Field(default=86400, ge=0)

    @model_validator(mode="after")
    def _overlap(self) -> Self:
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError("chunk_overlap_tokens must be below chunk_tokens")
        if self.top_k > self.candidates:
            raise ValueError("top_k must not exceed candidates")
        return self


class VerificationSettings(_Group):
    """Thresholds of the verification pipeline and WeightedRuleScorer (docs/scoring.md)."""

    # Risk score at or above which a pack is SUSPICIOUS, and at or above which it is also high
    # priority (opens a case).
    suspicious_threshold: float = Field(default=0.3, gt=0, lt=1)
    high_priority_threshold: float = Field(default=0.6, gt=0, le=1)
    # Fields read with less confidence are re-read by the next stage (OCR, then the LLM).
    min_field_confidence: float = Field(default=0.6, ge=0, le=1)
    # Quality gate (worker): shorter side in pixels, Laplacian variance, mean grey level 0-255.
    min_side_pixels: int = Field(default=480, ge=1)
    blur_threshold: float = Field(default=60.0, ge=0)
    dark_threshold: float = Field(default=40.0, ge=0, le=255)
    bright_threshold: float = Field(default=235.0, ge=0, le=255)
    # A batch scanned in more than this many states within the window is suspicious.
    spread_window_days: int = Field(default=30, ge=1, le=365)
    spread_state_limit: int = Field(default=3, ge=1, le=37)
    # The same photo (or a near-identical one) from the same person or device within this
    # window returns the earlier result.
    duplicate_window_hours: int = Field(default=24, ge=0, le=720)
    # Stages still pending after this long are skipped (degrading confidence, not failing).
    pipeline_timeout_seconds: float = Field(default=20.0, gt=0, le=300)
    # Registered products looked up by NAFDAC number are cached in Redis for this long. Their
    # facts change only through the Greenbook import, which waits this long to show. 0: off.
    product_cache_seconds: int = Field(default=60, ge=0, le=3600)
    # real: OpenCV quality gate, zxing barcodes, Tesseract when installed (the `vision` group);
    # fake: no readers (tests).
    readers: Literal["real", "fake"] = "real"

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.suspicious_threshold >= self.high_priority_threshold:
            raise ValueError("suspicious_threshold must be below high_priority_threshold")
        if self.dark_threshold >= self.bright_threshold:
            raise ValueError("dark_threshold must be below bright_threshold")
        return self


class QueueSettings(_Group):
    """Celery, the outbox relay, backpressure and retention (docs: queue/celery_app.py)."""

    # Outbox relay: run by Beat every interval; events failing this many times are dead-lettered.
    relay_interval_seconds: float = Field(default=1.0, gt=0, le=60)
    relay_batch_size: int = Field(default=100, ge=1, le=1000)
    relay_max_attempts: int = Field(default=10, ge=1, le=50)
    # Task limits. The broker re-delivers unacknowledged messages after the visibility
    # timeout, which must exceed the longest task.
    # Pool processes per worker container. Each loads the embedding models (about 1 GB), so
    # scale with containers rather than processes.
    worker_concurrency: int = Field(default=2, ge=1, le=32)
    task_soft_time_limit_seconds: int = Field(default=120, ge=1)
    task_time_limit_seconds: int = Field(default=180, ge=1)
    visibility_timeout_seconds: int = Field(default=3600, ge=60)
    # Backpressure on POST /scans: above the soft limit the estimated wait grows by this much
    # per queued scan; above the hard limit new scans get 503 with Retry-After.
    scans_soft_limit: int = Field(default=200, ge=0)
    scans_hard_limit: int = Field(default=2000, ge=1)
    seconds_per_queued_scan: float = Field(default=0.5, ge=0)
    # Sync fast path: when the queue is short, POST /scans waits up to this long for the
    # result before answering 202. 0 turns it off.
    sync_wait_seconds: float = Field(default=8.0, ge=0, le=20)
    # Scan photos older than this are no longer kept (the bucket's lifecycle rule deletes
    # them; the purge job clears their keys).
    scan_image_retention_days: int = Field(default=180, ge=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.task_time_limit_seconds <= self.task_soft_time_limit_seconds:
            raise ValueError("task_time_limit_seconds must exceed task_soft_time_limit_seconds")
        if self.visibility_timeout_seconds <= self.task_time_limit_seconds:
            raise ValueError("visibility_timeout_seconds must exceed task_time_limit_seconds")
        if self.scans_hard_limit <= self.scans_soft_limit:
            raise ValueError("scans_hard_limit must exceed scans_soft_limit")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        env_file=".env",
        env_file_encoding="utf-8",
        # .env is shared with docker compose: read only MEDLENS_* keys from it. Unknown MEDLENS_*
        # keys (typos) are still rejected by extra="forbid".
        dotenv_filtering="match_prefix",
        env_ignore_empty=True,
        extra="forbid",
        frozen=True,
    )

    app: AppSettings
    # Groups holding a required secret are required themselves.
    database: DatabaseSettings
    redis: RedisSettings = Field(default_factory=RedisSettings)
    s3: S3Settings
    auth: AuthSettings
    sms: SmsSettings = Field(default_factory=SmsSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    ratelimit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    crypto: CryptoSettings
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)

    @model_validator(mode="after")
    def _deployment_safety(self) -> Self:
        environment = self.app.environment
        problems: list[str] = []
        if environment is Environment.PRODUCTION:
            if self.app.debug:
                problems.append(f"{env_var('app', 'debug')} must be false in production")
            if "*" in self.app.cors_origins:
                problems.append(
                    f"{env_var('app', 'cors_origins')} must list explicit origins, not '*'"
                )
            if self.sms.provider == "console":
                problems.append(
                    f"{env_var('sms', 'provider')} must be a real SMS provider: the console "
                    "provider writes one-time codes to the logs"
                )
        if environment in {Environment.STAGING, Environment.PRODUCTION}:
            problems.extend(self._weak_secrets())
        if problems:
            listed = "\n".join(f"    - {problem}" for problem in problems)
            raise ValueError(f"unsafe settings for environment '{environment}':\n{listed}")
        return self

    def _weak_secrets(self) -> list[str]:
        found: list[str] = []
        for name, secret, self_chosen in self._secrets():
            value = secret.get_secret_value()
            if value.lower() in _WEAK_SECRETS:
                found.append(f"{name} uses a well-known value; set a real secret")
            elif self_chosen and len(value) < MIN_SECRET_LENGTH:
                found.append(f"{name} must be at least {MIN_SECRET_LENGTH} characters")
        return found

    def _secrets(self) -> list[tuple[str, SecretStr, bool]]:
        """(variable name, value, self-chosen) for every secret that is set."""
        found: list[tuple[str, SecretStr, bool]] = []
        for group_name in type(self).model_fields:
            group = getattr(self, group_name)
            for field_name, field in type(group).model_fields.items():
                value = getattr(group, field_name)
                name = env_var(group_name, field_name)
                self_chosen = any(isinstance(item, _SelfChosen) for item in field.metadata)
                if isinstance(value, SecretStr):
                    found.append((name, value, self_chosen))
                elif isinstance(value, dict):
                    # Key maps are validated as 32-byte keys on their own.
                    found.extend(
                        (f"{name}[{key}]", item, False)
                        for key, item in value.items()
                        if isinstance(item, SecretStr)
                    )
        return found


class SettingsError(SystemExit):
    """Invalid configuration. Subclasses SystemExit so any entrypoint exits with the message."""


def load_settings(**overrides: Any) -> Settings:
    """Build settings from the environment (plus overrides), failing fast with a clear message."""
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        # `from None`: the original error repeats input values, which may be secrets.
        raise SettingsError(_describe(exc)) from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def _describe(exc: ValidationError) -> str:
    lines = ["MedLens configuration is invalid:"]
    for error in exc.errors(include_url=False, include_input=False, include_context=False):
        message = error["msg"].removeprefix("Value error, ")
        locations = _missing_group_fields(error["loc"]) if error["type"] == "missing" else []
        for location in locations or [_location(error["loc"])]:
            lines.append(f"  - {location}: {message}" if location else f"  - {message}")
    lines.append("Set these variables in the environment or in .env (see .env.example).")
    return "\n".join(lines)


def _missing_group_fields(loc: tuple[int | str, ...]) -> list[str]:
    """A missing group (e.g. ``app``) is reported as its required variables."""
    field = Settings.model_fields.get(str(loc[0])) if len(loc) == 1 else None
    group = field.annotation if field is not None else None
    if not (isinstance(group, type) and issubclass(group, BaseModel)):
        return []
    return [env_var(str(loc[0]), name) for name, f in group.model_fields.items() if f.is_required()]


def _location(loc: tuple[int | str, ...]) -> str:
    names = [part for part in loc if isinstance(part, str)]
    indexes = "".join(f"[{part}]" for part in loc if isinstance(part, int))
    return f"{env_var(*names)}{indexes}" if names else ""

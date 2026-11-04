"""Auth metrics. Alert rules: docker/observability/alerts.yml."""

from prometheus_client import Counter

REVOCATION_CHECKS_SKIPPED = Counter(
    "medlens_auth_revocation_checks_skipped",
    "Access-token revocation checks that could not run because Redis was unavailable. "
    "Consumer actions continue (fail open); privileged ones fall back to Postgres.",
)
REVOCATION_FALLBACKS = Counter(
    "medlens_auth_revocation_fallbacks",
    "Revocation checks answered from Postgres (users.tokens_valid_after) instead of Redis.",
    ["result"],  # active | revoked
)

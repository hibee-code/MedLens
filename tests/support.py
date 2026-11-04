"""Settings for tests. Secrets are random, generated once per test run: none are committed."""

import secrets
from functools import cache
from typing import Any

from medlens.core.config import Settings
from medlens.core.crypto import generate_key


@cache
def secret_groups() -> dict[str, dict[str, Any]]:
    """A value for every required secret, as settings groups."""
    return {
        "database": {"password": secrets.token_urlsafe(32)},
        "s3": {"secret_access_key": secrets.token_urlsafe(32)},
        "auth": {"jwt_keys": {"test": generate_key()}, "jwt_active_key_id": "test"},
        "crypto": {
            "field_keys": {"test": generate_key()},
            "active_key_id": "test",
            "hmac_pepper": secrets.token_urlsafe(32),
        },
        # Not secrets: test defaults that keep requests fast and offline.
        "queue": {"sync_wait_seconds": 0.0},
        "verification": {"readers": "fake"},
        "embeddings": {"enabled": False},
        # Tests sign in and scan many times from one address; the limit has its own tests.
        "ratelimit": {"enabled": False},
    }


def settings_groups(**groups: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``groups`` merged field by field over the generated secrets, in the test environment
    unless ``app`` says otherwise."""
    merged = {name: {**values, **groups.pop(name, {})} for name, values in secret_groups().items()}
    merged["app"] = {"environment": "test", **groups.pop("app", {})}
    return {**merged, **groups}


def build_settings(**groups: dict[str, Any]) -> Settings:
    """Settings from explicit groups only (no .env file); see settings_groups."""
    return Settings(_env_file=None, **settings_groups(**groups))  # type: ignore[call-arg]

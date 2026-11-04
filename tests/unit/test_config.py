import secrets

import pytest
from pydantic import ValidationError

from medlens.core.config import (
    MIN_SECRET_LENGTH,
    SECRET_PLACEHOLDER,
    Environment,
    SettingsError,
    load_settings,
)
from medlens.core.crypto import generate_key
from tests.support import secret_groups, settings_groups

PRODUCTION_FIELD_KEY = generate_key()

# Production-grade values for every required secret, plus a real SMS provider.
PRODUCTION_SECRETS = {
    "database": {"password": secrets.token_urlsafe(32)},
    "s3": {"secret_access_key": secrets.token_urlsafe(32)},
    "crypto": {
        "field_keys": {"prod-2026-11": PRODUCTION_FIELD_KEY},
        "active_key_id": "prod-2026-11",
        "hmac_pepper": secrets.token_urlsafe(32),
    },
    "auth": {"jwt_keys": {"prod-jwt-1": generate_key()}, "jwt_active_key_id": "prod-jwt-1"},
    "sms": {"provider": "termii", "termii_api_key": secrets.token_hex(8)},  # provider-issued
}


def production(**app: object) -> dict[str, object]:
    return {"app": {"environment": "production", **app}, **PRODUCTION_SECRETS}


def test_local_defaults_load(settings_factory) -> None:
    settings = settings_factory(app={"environment": "local"})

    assert settings.app.environment is Environment.LOCAL
    assert settings.app.show_docs is True
    assert settings.database.pgbouncer is False


def test_environment_is_required() -> None:
    with pytest.raises(SettingsError) as raised:
        load_settings(_env_file=None, **secret_groups())

    assert "MEDLENS_APP__ENVIRONMENT: Field required" in str(raised.value)


def test_secrets_have_no_defaults() -> None:
    with pytest.raises(SettingsError) as raised:
        load_settings(_env_file=None, app={"environment": "local"})

    message = str(raised.value)
    for name in (
        "MEDLENS_DATABASE__PASSWORD",
        "MEDLENS_S3__SECRET_ACCESS_KEY",
        "MEDLENS_AUTH__JWT_KEYS",
        "MEDLENS_CRYPTO__FIELD_KEYS",
        "MEDLENS_CRYPTO__HMAC_PEPPER",
    ):
        assert f"{name}: Field required" in message
    assert "MEDLENS_DATABASE__HOST" not in message  # non-secret settings keep their defaults


def test_a_partly_configured_group_names_only_the_missing_secret() -> None:
    groups = settings_groups(app={"environment": "local"})
    groups["database"] = {"port": 5433}

    with pytest.raises(SettingsError) as raised:
        load_settings(_env_file=None, **groups)

    assert "MEDLENS_DATABASE__PASSWORD: Field required" in str(raised.value)


@pytest.mark.parametrize(
    ("groups", "variable"),
    [
        ({"database": {"password": SECRET_PLACEHOLDER}}, "MEDLENS_DATABASE__PASSWORD"),
        ({"crypto": {"hmac_pepper": SECRET_PLACEHOLDER}}, "MEDLENS_CRYPTO__HMAC_PEPPER"),
        ({"auth": {"jwt_keys": {"local": SECRET_PLACEHOLDER}}}, "MEDLENS_AUTH__JWT_KEYS"),
    ],
)
def test_placeholders_are_rejected_in_every_environment(
    groups: dict[str, dict[str, object]], variable: str
) -> None:
    with pytest.raises(SettingsError) as raised:
        load_settings(_env_file=None, **settings_groups(app={"environment": "local"}, **groups))

    assert variable in str(raised.value)
    assert "make dev-secrets" in str(raised.value)


def test_nested_values_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDLENS_APP__ENVIRONMENT", "local")
    monkeypatch.setenv("MEDLENS_DATABASE__PORT", "6432")
    monkeypatch.setenv("MEDLENS_APP__CORS_ORIGINS", '["https://medlens.example"]')

    settings = load_settings(_env_file=None, **secret_groups())

    assert settings.database.port == 6432
    assert settings.app.cors_origins == ["https://medlens.example"]


def test_invalid_value_names_the_variable_without_echoing_it() -> None:
    with pytest.raises(SettingsError) as raised:
        load_settings(_env_file=None, app={"environment": "local"}, database={"port": "not-a-port"})

    message = str(raised.value)
    assert "MEDLENS_DATABASE__PORT" in message
    assert "not-a-port" not in message


def test_production_with_real_secrets_is_accepted(settings_factory) -> None:
    settings = settings_factory(**production())

    assert settings.app.show_docs is False


def test_production_rejects_debug(settings_factory) -> None:
    with pytest.raises(ValidationError, match="MEDLENS_APP__DEBUG must be false in production"):
        settings_factory(**production(debug=True))


def test_production_rejects_debug_with_a_readable_startup_message() -> None:
    with pytest.raises(SettingsError) as raised:
        load_settings(_env_file=None, **production(debug=True))

    assert "unsafe settings for environment 'production'" in str(raised.value)
    assert "MEDLENS_APP__DEBUG must be false in production" in str(raised.value)


def test_production_rejects_wildcard_cors(settings_factory) -> None:
    with pytest.raises(ValidationError, match="MEDLENS_APP__CORS_ORIGINS"):
        settings_factory(**production(cors_origins=["https://medlens.example", "*"]))


SHORT_SECRET = "s" * (MIN_SECRET_LENGTH - 1)


def test_production_rejects_short_self_chosen_secrets_without_revealing_them() -> None:
    groups = production()
    groups["database"] = {"password": SHORT_SECRET}
    groups["crypto"] = {**PRODUCTION_SECRETS["crypto"], "hmac_pepper": SHORT_SECRET}
    groups["redis"] = {"password": SHORT_SECRET}

    with pytest.raises(SettingsError) as raised:
        load_settings(_env_file=None, **groups)

    message = str(raised.value)
    for name in (
        "MEDLENS_DATABASE__PASSWORD",
        "MEDLENS_CRYPTO__HMAC_PEPPER",
        "MEDLENS_REDIS__PASSWORD",
    ):
        assert f"{name} must be at least {MIN_SECRET_LENGTH} characters" in message
    assert SHORT_SECRET not in message


def test_local_accepts_short_self_chosen_secrets(settings_factory) -> None:
    settings = settings_factory(app={"environment": "local"}, database={"password": "short"})

    assert settings.database.password.get_secret_value() == "short"


def test_provider_issued_keys_have_no_length_rule(settings_factory) -> None:
    groups = production()
    groups["llm"] = {"anthropic_api_key": "provider-key"}  # pragma: allowlist secret

    assert settings_factory(**groups).llm.anthropic_api_key is not None


@pytest.mark.parametrize(
    ("crypto", "message"),
    [
        ({"field_keys": {}}, "at least one field encryption key"),
        ({"field_keys": {"k": "not base64!"}, "active_key_id": "k"}, "not valid base64"),
        ({"field_keys": {"k": "c2hvcnQ="}, "active_key_id": "k"}, "must decode to 32 bytes"),
        ({"field_keys": {"bad id!": PRODUCTION_FIELD_KEY}, "active_key_id": "bad id!"}, "key id"),
        ({"active_key_id": "missing"}, "is not in field_keys"),
        ({"hmac_pepper": "short"}, "at least 16 characters"),
    ],
)
def test_crypto_settings_are_validated(settings_factory, crypto, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        settings_factory(crypto=crypto)


@pytest.mark.parametrize("weak", ["password", "minioadmin", "ChangeMe", "change_me"])
def test_staging_rejects_well_known_secrets(settings_factory, weak: str) -> None:
    with pytest.raises(ValidationError, match="MEDLENS_REDIS__PASSWORD"):
        settings_factory(
            app={"environment": "staging"}, redis={"password": weak}, **PRODUCTION_SECRETS
        )


def test_staging_allows_debug(settings_factory) -> None:
    settings = settings_factory(app={"environment": "staging", "debug": True}, **PRODUCTION_SECRETS)

    assert settings.app.debug is True


def test_production_docs_can_be_enabled_explicitly(settings_factory) -> None:
    settings = settings_factory(**production(docs_enabled=True))

    assert settings.app.show_docs is True


def test_redis_databases_must_differ(settings_factory) -> None:
    with pytest.raises(ValidationError, match="must all be different"):
        settings_factory(redis={"cache_db": 1, "broker_db": 1})


def test_redis_url_escapes_password(settings_factory) -> None:
    settings = settings_factory(redis={"password": "p@ss/word", "tls": True})

    assert settings.redis.url(1) == "rediss://:p%40ss%2Fword@localhost:6379/1"


def test_chunk_overlap_must_be_below_the_chunk_size(settings_factory) -> None:
    assert settings_factory(knowledge={"chunk_tokens": 200, "chunk_overlap_tokens": 50})
    with pytest.raises(ValidationError, match="chunk_overlap_tokens"):
        settings_factory(knowledge={"chunk_tokens": 100, "chunk_overlap_tokens": 100})


def test_env_file_typos_are_rejected_but_compose_variables_ignored(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MEDLENS_APP__ENVIRONMENT=local\nAPI_HOST_PORT=8000\nMEDLENS_DATABSE__HOST=db\n"
    )

    with pytest.raises(SettingsError) as raised:
        load_settings(_env_file=env_file)

    assert "DATABSE" in str(raised.value)
    assert "API_HOST_PORT" not in str(raised.value)


@pytest.mark.parametrize(
    ("verification", "message"),
    [
        ({"suspicious_threshold": 0.6, "high_priority_threshold": 0.6}, "below high_priority"),
        ({"dark_threshold": 240, "bright_threshold": 230}, "dark_threshold must be below"),
    ],
)
def test_verification_thresholds_are_ordered(settings_factory, verification, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        settings_factory(verification=verification)

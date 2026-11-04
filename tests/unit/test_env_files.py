"""The configuration contract: .env.example documents everything and holds no real values (only
the placeholder for required secrets), its documented defaults match the code and docker compose,
and `make dev-secrets` produces a working .env."""

import json
import re
import stat
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from scripts.sync_env import (
    GENERATED_SECRETS,
    LOCAL_PROFILE,
    RefusedError,
    render,
    sync,
    template_keys,
)

from medlens.core.config import SECRET_PLACEHOLDER, Settings, SettingsError, env_var, load_settings

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = ROOT / ".env.example"
TEMPLATE = TEMPLATE_PATH.read_text()
COMPOSE = (ROOT / "docker-compose.yml").read_text()
# Last comment line above each variable: "[Secret. ]Required." or "[Secret. ]Default: <value>".
_CONTRACT = re.compile(
    r"^# (?P<secret>Secret\. )?(?:(?P<required>Required\.)|Default: (?P<default>.+))$"
)
_COMPOSE_VARIABLE = re.compile(
    r"(?<!\$)\$\{(?P<name>[A-Z][A-Z0-9_]*)(?::(?P<op>[-?])(?P<arg>[^}]*))?\}"
)


def contracts() -> dict[str, re.Match[str]]:
    lines = TEMPLATE.splitlines()
    found: dict[str, re.Match[str]] = {}
    for index, line in enumerate(lines):
        if line and not line.startswith("#"):
            key = line.split("=", 1)[0]
            contract = _CONTRACT.match(lines[index - 1])
            assert contract, f"{key}: the line above must state 'Required.' or 'Default: ...'"
            found[key] = contract
    return found


def app_fields() -> dict[str, FieldInfo]:
    fields: dict[str, FieldInfo] = {}
    for group_name, group_field in Settings.model_fields.items():
        group = group_field.annotation
        assert isinstance(group, type)
        assert issubclass(group, BaseModel)
        for name, field in group.model_fields.items():
            fields[env_var(group_name, name)] = field
    return fields


def as_env(value: Any) -> str:
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, list | dict):
        return json.dumps(value, default=lambda item: item.model_dump())
    return str(value)


def compose_variables() -> dict[str, tuple[str | None, str | None]]:
    return {m["name"]: (m["op"], m["arg"]) for m in _COMPOSE_VARIABLE.finditer(COMPOSE)}


def is_secret(field: FieldInfo) -> bool:
    return "SecretStr" in str(field.annotation)


def template_values() -> dict[str, str]:
    return {key: value or "" for key, value in dotenv_values(TEMPLATE_PATH).items()}


# ------------------------------------------------------------------------- the template
def test_template_holds_only_placeholders_for_required_secrets() -> None:
    documented = contracts()
    for key, value in template_values().items():
        required_secret = bool(documented[key]["secret"] and documented[key]["required"])
        if required_secret:
            assert SECRET_PLACEHOLDER in value, f"{key}: show the placeholder"
        else:
            assert value == "", f"{key}: the template holds no values"


def test_template_placeholders_have_the_settings_shape() -> None:
    """Key maps stay valid JSON, so a copied template fails on the placeholder, not on parsing."""
    fields = app_fields()
    for key, value in template_values().items():
        if key in fields and value.startswith("{"):
            assert set(json.loads(value).values()) == {SECRET_PLACEHOLDER}, key


def test_the_copied_template_is_rejected_by_name() -> None:
    with pytest.raises(SettingsError) as raised:
        load_settings(_env_file=TEMPLATE_PATH, app={"environment": "local"})

    message = str(raised.value)
    for key in ("DATABASE__PASSWORD", "AUTH__JWT_KEYS", "CRYPTO__HMAC_PEPPER"):
        assert f"MEDLENS_{key}" in message
    assert message.count("make dev-secrets") >= 5


def test_no_secret_has_a_default() -> None:
    """Secrets are required (no value in code) or optional and unset by default."""
    for key, field in app_fields().items():
        if is_secret(field) and not field.is_required():
            assert field.get_default(call_default_factory=True, validated_data={}) is None, key


def test_template_lists_each_variable_once() -> None:
    keys = template_keys(TEMPLATE)

    assert len(keys) == len(set(keys))


def test_template_documents_every_setting_and_nothing_else() -> None:
    documented = {key for key in template_keys(TEMPLATE) if key.startswith("MEDLENS_")}

    assert documented == set(app_fields())


def test_documented_defaults_match_the_code() -> None:
    documented = contracts()
    for key, field in app_fields().items():
        contract = documented[key]
        assert bool(contract["secret"]) == is_secret(field), f"{key}: 'Secret.' marker"
        if field.is_required():
            assert contract["required"], f"{key} is required"
            continue
        default = field.get_default(call_default_factory=True, validated_data={})
        assert contract["default"] == as_env(default), f"{key}: documented default"


def test_compose_reads_only_documented_variables_with_matching_defaults() -> None:
    documented = contracts()
    for name, (op, arg) in compose_variables().items():
        assert name in documented, f"compose reads {name}, which .env.example does not document"
        if name.startswith("MEDLENS_"):
            continue  # the application default is what the template states
        if op == "?":
            assert documented[name]["required"], f"{name} is required by compose"
        else:
            assert documented[name]["default"] == arg, f"{name}: compose default"


def test_compose_has_no_fallback_credentials() -> None:
    secret_names = {name for name, contract in contracts().items() if contract["secret"]}
    for name, (op, _) in compose_variables().items():
        if name in secret_names:
            assert op == "?", f"compose must require {name}, not fall back to a value"


# ------------------------------------------------------------------------- make dev-secrets
def test_local_profile_matches_compose_host_ports() -> None:
    defaults = {name: arg for name, (op, arg) in compose_variables().items() if op == "-"}

    assert LOCAL_PROFILE["MEDLENS_DATABASE__PORT"] == defaults["PGBOUNCER_HOST_PORT"]
    assert LOCAL_PROFILE["MEDLENS_REDIS__PORT"] == defaults["REDIS_HOST_PORT"]
    assert LOCAL_PROFILE["MEDLENS_S3__ENDPOINT_URL"].endswith(f":{defaults['MINIO_HOST_PORT']}")


def test_every_required_secret_is_generated() -> None:
    documented = contracts()
    required_secrets = {
        key for key, contract in documented.items() if contract["secret"] and contract["required"]
    }

    assert set(GENERATED_SECRETS) | set(LOCAL_PROFILE) <= set(documented)
    assert set(GENERATED_SECRETS) == required_secrets


def test_fresh_env_is_a_working_local_configuration(tmp_path: Path) -> None:
    template = tmp_path / ".env.example"
    template.write_text(TEMPLATE)
    first, second = tmp_path / "first.env", tmp_path / "second.env"

    assert sync(template, first) == "created first.env: 7 secrets generated"
    sync(template, second)

    settings = load_settings(_env_file=first)
    assert settings.app.environment == "local"
    assert settings.database.pgbouncer is True
    assert settings.crypto.active_key_id in settings.crypto.field_keys
    assert settings.auth.jwt_active_key_id in settings.auth.jwt_keys
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    first_secrets = _values(first)
    second_secrets = _values(second)
    for key in GENERATED_SECRETS:
        assert first_secrets[key], key
        assert first_secrets[key] != second_secrets[key], key


SPACED_VALUE = "a value # with spaces"  # needs quoting in .env


def test_resync_keeps_values_and_unknown_variables(tmp_path: Path) -> None:
    template = tmp_path / ".env.example"
    template.write_text(TEMPLATE)
    env = tmp_path / ".env"
    env.write_text(
        "MEDLENS_APP__ENVIRONMENT=local\n"
        f"MEDLENS_DATABASE__PASSWORD='{SPACED_VALUE}'\n"
        "MEDLENS_LLM__ANTHROPIC_API_KEY=\n"
        "MY_TOOL_TOKEN=keep-me\n"
    )

    message = sync(template, env)

    values = _values(env)
    assert values["MEDLENS_DATABASE__PASSWORD"] == SPACED_VALUE
    assert values["MEDLENS_LLM__ANTHROPIC_API_KEY"] == ""  # an explicit empty value is kept
    assert values["MY_TOOL_TOKEN"] == "keep-me"
    assert values["MEDLENS_CRYPTO__HMAC_PEPPER"]  # missing secret generated
    assert "existing values kept" in message
    assert "secrets generated" in message
    assert (tmp_path / ".env.bak").read_text().startswith("MEDLENS_APP__ENVIRONMENT=local")
    assert stat.S_IMODE((tmp_path / ".env.bak").stat().st_mode) == 0o600
    assert sync(template, env) == ".env is up to date"


def test_empty_and_placeholder_secrets_are_generated_and_others_kept(tmp_path: Path) -> None:
    template = tmp_path / ".env.example"
    template.write_text(TEMPLATE)
    env = tmp_path / ".env"
    env.write_text(TEMPLATE)  # someone copied the template
    keep = "a-password-chosen-by-hand"  # pragma: allowlist secret
    env.write_text(
        env.read_text()
        .replace("MEDLENS_APP__ENVIRONMENT=", "MEDLENS_APP__ENVIRONMENT=local")
        .replace("MEDLENS_DATABASE__PASSWORD=CHANGE_ME", f"MEDLENS_DATABASE__PASSWORD={keep}")
        .replace("GRAFANA_ADMIN_PASSWORD=CHANGE_ME", "GRAFANA_ADMIN_PASSWORD=")
    )

    message = sync(template, env)

    values = _values(env)
    assert values["MEDLENS_DATABASE__PASSWORD"] == keep
    assert values["GRAFANA_ADMIN_PASSWORD"]
    assert not any(SECRET_PLACEHOLDER in value for value in values.values())
    assert f"{len(GENERATED_SECRETS) - 1} secrets generated" in message
    assert load_settings(_env_file=env).crypto.field_keys


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_deployed_environments_are_refused(tmp_path: Path, environment: str) -> None:
    env = tmp_path / ".env"
    original = f"MEDLENS_APP__ENVIRONMENT={environment}\n"
    env.write_text(original)

    with pytest.raises(RefusedError, match=environment):
        sync(TEMPLATE_PATH, env)
    assert env.read_text() == original


def test_render_preserves_the_template_layout() -> None:
    content, added, _ = render(TEMPLATE, {"MEDLENS_APP__ENVIRONMENT": "test"})

    assert "MEDLENS_APP__ENVIRONMENT=test" in content
    assert "# Default: 10485760\nMEDLENS_APP__MAX_REQUEST_BODY_BYTES=\n" in content
    assert content.startswith("# =====")
    assert "never commit it" in content
    assert "MEDLENS_APP__ENVIRONMENT" not in added


def _values(path: Path) -> dict[str, str]:
    return {key: value or "" for key, value in dotenv_values(path, interpolate=False).items()}


@pytest.mark.parametrize("value", ['{"k":"v"}', "with space", "it's", "plain-token_123"])
def test_values_round_trip_through_quoting(tmp_path: Path, value: str) -> None:
    path = tmp_path / ".env"
    content, _, _ = render("MEDLENS_X=\n", {"MEDLENS_X": value})
    path.write_text(content)

    assert _values(path)["MEDLENS_X"] == value

"""Create or update the local .env from the .env.example template (``make dev-secrets``).

    uv run python scripts/sync_env.py [--template .env.example] [--env-file .env]

* .env.example holds no real values (required secrets show the CHANGE_ME placeholder); .env
  holds this machine's values and is never committed.
* .env gets the template's layout: every variable, documentation and order. Values come from:
    1. the existing .env, so re-running never changes a value you set (an empty value stays
       empty), except that a secret listed in GENERATED_SECRETS that is empty or still holds the
       placeholder is generated;
    2. otherwise LOCAL_PROFILE, which points the app at the `make infra` services;
    3. otherwise a freshly generated random secret (GENERATED_SECRETS);
    4. otherwise empty, meaning the application default documented in the template.
* Variables in .env that the template does not know are kept at the end, never dropped.
* The previous .env is copied to .env.bak before it is changed; the new file is chmod 600.
* A .env for staging or production is refused: those secrets belong in a secret manager.
* Values are never printed.

Generated secrets only take effect on fresh volumes: PostgreSQL and MinIO keep the credentials
they were initialised with (`docker compose down -v` resets them, deleting local data).
"""

import argparse
import json
import re
import secrets
import shutil
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple

from dotenv import dotenv_values

from medlens.core.config import SECRET_PLACEHOLDER, Environment
from medlens.core.crypto import generate_key

ROOT = Path(__file__).resolve().parents[1]
_ASSIGNMENT = re.compile(r"^(?P<key>[A-Z][A-Z0-9_]*)=(?P<value>.*)$")
_NEEDS_QUOTES = re.compile(r"[\s#'\"$\\]")

# Non-secret values for local development against `make infra` (ports match the compose
# defaults). Everything not listed falls back to the application default.
LOCAL_PROFILE: dict[str, str] = {
    "MEDLENS_APP__ENVIRONMENT": "local",
    "MEDLENS_DATABASE__HOST": "localhost",
    "MEDLENS_DATABASE__PORT": "6432",  # PgBouncer, as in production
    "MEDLENS_DATABASE__PGBOUNCER": "true",
    "MEDLENS_DATABASE__NAME": "medlens",
    "MEDLENS_DATABASE__USER": "medlens",
    "MEDLENS_REDIS__HOST": "localhost",
    "MEDLENS_REDIS__PORT": "6380",
    "MEDLENS_S3__ENDPOINT_URL": "http://localhost:9000",
    "MEDLENS_S3__ACCESS_KEY_ID": "medlens-local",
    "MEDLENS_S3__BUCKET": "medlens-local",
}

# Key maps use the default active key id ("local-dev"), so no *_ACTIVE_KEY_ID is needed.
GENERATED_SECRETS: dict[str, Callable[[], str]] = {
    "MEDLENS_DATABASE__PASSWORD": lambda: secrets.token_urlsafe(32),
    "MEDLENS_S3__SECRET_ACCESS_KEY": lambda: secrets.token_urlsafe(32),
    "MEDLENS_CRYPTO__FIELD_KEYS": lambda: json.dumps({"local-dev": generate_key()}),
    "MEDLENS_AUTH__JWT_KEYS": lambda: json.dumps({"local-dev": generate_key()}),
    "MEDLENS_CRYPTO__HMAC_PEPPER": lambda: secrets.token_urlsafe(32),
    "FLOWER_BASIC_AUTH": lambda: f"medlens:{secrets.token_urlsafe(18)}",
    "GRAFANA_ADMIN_PASSWORD": lambda: secrets.token_urlsafe(18),
}

ENV_HEADER = """\
# =============================================================================
#  MedLens local environment (.env)
# =============================================================================
#  This machine's configuration. Gitignored: never commit it or copy it to a
#  server. Generated and kept in sync with .env.example by `make dev-secrets`,
#  which adds new variables, fills secrets that are empty or still CHANGE_ME,
#  and never changes other values already set here.
#
#  Every variable is documented below. An empty value means the application
#  default stated above it.
# ============================================================================="""

EXTRA_HEADER = """\
# -----------------------------------------------------------------------------
# Not in .env.example (kept by `make dev-secrets`; remove when no longer needed)
# -----------------------------------------------------------------------------"""


def template_keys(template: str) -> list[str]:
    return [m["key"] for line in template.splitlines() if (m := _ASSIGNMENT.match(line))]


class Rendered(NamedTuple):
    content: str
    added: list[str]  # template variables that were not in the existing file
    generated: list[str]  # secrets generated, for new variables or empty / placeholder values


def render(template: str, existing: Mapping[str, str | None]) -> Rendered:
    """The new .env text for ``template`` filled from ``existing`` (see the module docstring)."""
    lines = template.splitlines()
    body_start = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    output = [ENV_HEADER]
    added: list[str] = []
    generated: list[str] = []
    for line in lines[body_start:]:
        match = _ASSIGNMENT.match(line)
        if match is None:
            output.append(line)
            continue
        key = match["key"]
        current = existing.get(key)
        generator = GENERATED_SECRETS.get(key)
        if key not in existing:
            added.append(key)
        if generator is not None and (not current or SECRET_PLACEHOLDER in current):
            generated.append(key)
            value = generator()
        elif key in existing:
            value = current or ""
        else:
            value = LOCAL_PROFILE.get(key, "")
        output.append(f"{key}={_quote(value)}")
    extra = [key for key in existing if key not in set(template_keys(template))]
    if extra:
        output += ["", EXTRA_HEADER, *(f"{key}={_quote(existing[key] or '')}" for key in extra)]
    return Rendered("\n".join(output).rstrip() + "\n", added, generated)


def _quote(value: str) -> str:
    if not value or not _NEEDS_QUOTES.search(value):
        return value
    if "'" not in value:
        return f"'{value}'"  # single quotes: no escapes, no ${} interpolation in compose
    return json.dumps(value)


class RefusedError(Exception):
    """The .env file belongs to a deployed environment."""


def sync(template_path: Path, env_path: Path) -> str:
    template = template_path.read_text(encoding="utf-8")
    existed = env_path.exists()
    existing = dotenv_values(env_path, interpolate=False) if existed else {}
    environment = existing.get("MEDLENS_APP__ENVIRONMENT")
    if environment in {Environment.STAGING, Environment.PRODUCTION}:
        raise RefusedError(
            f"{env_path.name} is for {environment}: refusing to write local secrets into it. "
            "Deployed environments take their secrets from a secret manager."
        )
    content, added, generated = render(template, existing)
    if existed and env_path.read_text(encoding="utf-8") == content:
        return f"{env_path.name} is up to date"
    if existed:
        backup = env_path.with_name(env_path.name + ".bak")
        shutil.copy2(env_path, backup)
        backup.chmod(0o600)  # it holds the same secrets
    env_path.write_text(content, encoding="utf-8")
    env_path.chmod(0o600)
    if not existed:
        return f"created {env_path.name}: {len(generated)} secrets generated"
    return (
        f"updated {env_path.name}: {len(added)} variables added, {len(generated)} secrets "
        f"generated; existing values kept; previous file saved as {env_path.name}.bak"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or update .env from .env.example.")
    parser.add_argument("--template", type=Path, default=ROOT / ".env.example")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    try:
        sys.stdout.write(sync(args.template, args.env_file) + "\n")
    except RefusedError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

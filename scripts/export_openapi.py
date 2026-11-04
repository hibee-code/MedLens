"""Write the OpenAPI document to openapi.json, or check that the committed copy is current.

    uv run python scripts/export_openapi.py          # regenerate (make openapi)
    uv run python scripts/export_openapi.py --check  # exit 1 if openapi.json is out of date

openapi.json is committed so every API change shows up in review as a diff of the contract.
"""

import argparse
import secrets
import sys
from pathlib import Path
from typing import Any

from medlens.core.config import Settings
from medlens.core.crypto import generate_key
from medlens.core.openapi import render_openapi
from medlens.main import create_app

OPENAPI_PATH = Path(__file__).resolve().parents[1] / "openapi.json"


def current_document() -> str:
    # Fixed settings: the document must not depend on the caller's .env. The app is never
    # started, so throwaway secrets suffice; none of them appears in the document.
    groups: dict[str, Any] = {
        "app": {"environment": "test"},
        "database": {"password": secrets.token_urlsafe(32)},
        "s3": {"secret_access_key": secrets.token_urlsafe(32)},
        "auth": {"jwt_keys": {"openapi": generate_key()}, "jwt_active_key_id": "openapi"},
        "crypto": {
            "field_keys": {"openapi": generate_key()},
            "active_key_id": "openapi",
            "hmac_pepper": secrets.token_urlsafe(32),
        },
    }
    settings = Settings(_env_file=None, **groups)
    return render_openapi(create_app(settings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if openapi.json is stale")
    args = parser.parse_args()

    document = current_document()
    if args.check:
        if not OPENAPI_PATH.exists() or OPENAPI_PATH.read_text() != document:
            sys.stderr.write("openapi.json is out of date; run `make openapi` and commit it.\n")
            return 1
        return 0
    OPENAPI_PATH.write_text(document)
    sys.stdout.write(f"wrote {OPENAPI_PATH.name}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

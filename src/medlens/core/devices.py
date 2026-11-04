"""The X-Device-Id header: an opaque id the app generates once per installation. It lets guests
read their own scans and scopes duplicates and idempotency keys. Only a keyed hash is stored."""

import re
from typing import Annotated

from fastapi import Header

from medlens.core.crypto import keyed_digest
from medlens.core.errors import ValidationFailed

DEVICE_HEADER = "X-Device-Id"
_DEVICE_ID = re.compile(r"^[A-Za-z0-9_\-]{16,128}$")


def device_hash(
    x_device_id: Annotated[
        str | None,
        Header(
            alias=DEVICE_HEADER,
            description=(
                "Opaque id generated once per app installation (16-128 letters, digits, - or _), "
                "e.g. a random UUID. Required for guests."
            ),
        ),
    ] = None,
) -> str | None:
    """FastAPI dependency: the keyed hash of the device id, or None when absent."""
    if x_device_id is None:
        return None
    if not _DEVICE_ID.fullmatch(x_device_id):
        raise ValidationFailed(
            f"{DEVICE_HEADER} must be 16-128 letters, digits, '-' or '_'.",
            errors=[{"loc": ["header", DEVICE_HEADER], "msg": "invalid", "type": "value"}],
        )
    return keyed_digest(x_device_id, purpose="device")

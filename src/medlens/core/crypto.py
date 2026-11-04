"""Field-level encryption for PII columns, and blind indexes for looking them up.

Encryption: AES-256-GCM with a random 96-bit nonce. A stored value looks like

    <key id>:<base64url(nonce || ciphertext || tag)>

The key id says which key decrypts it, so keys can be rotated: add a new key, make it active,
re-encrypt rows in the background, then remove the old key. The column name is bound as
associated data, so a ciphertext copied into another column (say phone -> mfa_secret) fails to
decrypt instead of silently revealing the wrong field.

Encrypted columns cannot be searched. Lookups use a blind index instead: an HMAC-SHA256 of the
normalised value with a secret pepper (``phone_lookup_hash``).

Run ``python -m medlens.core.crypto`` to print a new random key.
"""

import base64
import binascii
import hashlib
import hmac
import os
import sys
from collections.abc import Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from medlens.core.config import CryptoSettings

NONCE_BYTES = 12
KEY_BYTES = 32


class DecryptionError(Exception):
    """The value is corrupt, was encrypted for another column, or its key is unknown."""


class FieldCipher:
    def __init__(self, keys: Mapping[str, bytes], active_key_id: str) -> None:
        if active_key_id not in keys:
            raise ValueError(f"active key {active_key_id!r} is not configured")
        self._ciphers = {key_id: AESGCM(key) for key_id, key in keys.items()}
        self.active_key_id = active_key_id

    @classmethod
    def from_settings(cls, settings: CryptoSettings) -> "FieldCipher":
        keys = {
            key_id: base64.b64decode(key.get_secret_value())
            for key_id, key in settings.field_keys.items()
        }
        return cls(keys, settings.active_key_id)

    def encrypt(self, plaintext: str, *, context: str) -> str:
        nonce = os.urandom(NONCE_BYTES)
        sealed = self._ciphers[self.active_key_id].encrypt(
            nonce, plaintext.encode(), context.encode()
        )
        return f"{self.active_key_id}:{base64.urlsafe_b64encode(nonce + sealed).decode()}"

    def decrypt(self, token: str, *, context: str) -> str:
        key_id, separator, payload = token.partition(":")
        cipher = self._ciphers.get(key_id)
        if not separator or cipher is None:
            raise DecryptionError(f"unknown key id {key_id!r}")
        try:
            raw = base64.urlsafe_b64decode(payload.encode())
            plaintext = cipher.decrypt(raw[:NONCE_BYTES], raw[NONCE_BYTES:], context.encode())
        except (binascii.Error, InvalidTag, ValueError):
            raise DecryptionError(f"cannot decrypt value for {context}") from None
        return plaintext.decode()

    def needs_rotation(self, token: str) -> bool:
        """True when the value was encrypted with a key other than the active one."""
        return token.partition(":")[0] != self.active_key_id


def blind_index(value: str, *, pepper: bytes, purpose: str) -> str:
    """Deterministic, non-reversible lookup key. ``purpose`` separates domains (phone vs ip)."""
    message = purpose.encode() + b"\x00" + value.encode()
    return hmac.new(pepper, message, hashlib.sha256).hexdigest()


# Process-wide configuration, set once at startup (core/bootstrap.py). The SQLAlchemy column type
# needs it outside any request, so it cannot come from dependency injection.
_cipher: FieldCipher | None = None
_pepper: bytes | None = None


def configure_crypto(settings: CryptoSettings) -> None:
    global _cipher, _pepper  # noqa: PLW0603 - deliberate process-wide configuration
    _cipher = FieldCipher.from_settings(settings)
    _pepper = settings.hmac_pepper.get_secret_value().encode()


def field_cipher() -> FieldCipher:
    if _cipher is None:
        raise RuntimeError("field encryption is not configured; call configure_process() first")
    return _cipher


def _configured_pepper() -> bytes:
    if _pepper is None:
        raise RuntimeError("the HMAC pepper is not configured; call configure_process() first")
    return _pepper


def keyed_digest(value: str, *, purpose: str) -> str:
    """HMAC-SHA256 of ``value`` with the configured pepper: a lookup key that reveals nothing."""
    return blind_index(value, pepper=_configured_pepper(), purpose=purpose)


def phone_lookup_hash(phone_e164: str) -> str:
    return keyed_digest(phone_e164, purpose="phone")


def ip_hash(ip_address: str) -> str:
    return keyed_digest(ip_address, purpose="ip")


def generate_key() -> str:
    return base64.b64encode(os.urandom(KEY_BYTES)).decode()


if __name__ == "__main__":
    sys.stdout.write(generate_key() + "\n")

import base64

import pytest

from medlens.core import crypto
from medlens.core.crypto import DecryptionError, FieldCipher, blind_index, generate_key

KEY_A = b"a" * 32
KEY_B = b"b" * 32


def test_round_trip_with_key_id_prefix() -> None:
    cipher = FieldCipher({"2026-11": KEY_A}, "2026-11")

    token = cipher.encrypt("+2348031234567", context="users.phone_e164")

    assert token.startswith("2026-11:")
    assert "8031234567" not in token
    assert cipher.decrypt(token, context="users.phone_e164") == "+2348031234567"


def test_each_encryption_uses_a_fresh_nonce() -> None:
    cipher = FieldCipher({"k": KEY_A}, "k")

    first = cipher.encrypt("same", context="c")
    second = cipher.encrypt("same", context="c")

    assert first != second


def test_ciphertext_is_bound_to_its_column() -> None:
    cipher = FieldCipher({"k": KEY_A}, "k")
    token = cipher.encrypt("+2348031234567", context="users.phone_e164")

    with pytest.raises(DecryptionError):
        cipher.decrypt(token, context="users.mfa_secret")


def test_tampering_is_detected() -> None:
    cipher = FieldCipher({"k": KEY_A}, "k")
    key_id, payload = cipher.encrypt("secret", context="c").split(":")
    raw = bytearray(base64.urlsafe_b64decode(payload))
    raw[-1] ^= 0x01

    with pytest.raises(DecryptionError):
        cipher.decrypt(f"{key_id}:{base64.urlsafe_b64encode(bytes(raw)).decode()}", context="c")


@pytest.mark.parametrize("token", ["unknown:AAAA", "no-separator", "k:not base64!!", "k:"])
def test_malformed_or_unknown_tokens_are_rejected(token: str) -> None:
    with pytest.raises(DecryptionError):
        FieldCipher({"k": KEY_A}, "k").decrypt(token, context="c")


def test_rotation_keeps_old_values_readable() -> None:
    old = FieldCipher({"2026-01": KEY_A}, "2026-01")
    token = old.encrypt("+2348031234567", context="c")
    rotated = FieldCipher({"2026-01": KEY_A, "2026-11": KEY_B}, "2026-11")

    assert rotated.decrypt(token, context="c") == "+2348031234567"
    assert rotated.needs_rotation(token)
    assert not rotated.needs_rotation(rotated.encrypt("x", context="c"))


def test_active_key_must_exist() -> None:
    with pytest.raises(ValueError, match="not configured"):
        FieldCipher({"k": KEY_A}, "other")


def test_blind_index_is_deterministic_and_purpose_separated() -> None:
    pepper = b"pepper-for-tests-only"

    phone = blind_index("+2348031234567", pepper=pepper, purpose="phone")

    assert phone == blind_index("+2348031234567", pepper=pepper, purpose="phone")
    assert phone != blind_index("+2348031234567", pepper=pepper, purpose="ip")
    assert phone != blind_index("+2348031234567", pepper=b"another-pepper-1234", purpose="phone")
    assert len(phone) == 64


def test_unconfigured_process_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crypto, "_cipher", None)
    monkeypatch.setattr(crypto, "_pepper", None)

    with pytest.raises(RuntimeError, match="not configured"):
        crypto.field_cipher()
    with pytest.raises(RuntimeError, match="not configured"):
        crypto.phone_lookup_hash("+2348031234567")


def test_configure_from_settings(settings_factory) -> None:
    key = generate_key()
    crypto.configure_crypto(
        settings_factory(crypto={"field_keys": {"prod-1": key}, "active_key_id": "prod-1"}).crypto
    )
    try:
        token = crypto.field_cipher().encrypt("x", context="c")
        assert token.startswith("prod-1:")
        assert len(base64.b64decode(key)) == 32
        assert crypto.ip_hash("10.0.0.1") != crypto.phone_lookup_hash("10.0.0.1")
    finally:
        crypto.configure_crypto(settings_factory().crypto)  # back to the test defaults

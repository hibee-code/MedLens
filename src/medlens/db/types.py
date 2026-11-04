"""Custom column types."""

from enum import StrEnum
from typing import Any

from sqlalchemy import CheckConstraint, Enum, Text
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

from medlens.core.crypto import field_cipher
from medlens.core.identifiers import NigerianState


class EncryptedText(TypeDecorator[str]):
    """Text encrypted at rest (AES-256-GCM, see core/crypto.py), transparent to the model.

    ``context`` (conventionally "table.column") is bound into every ciphertext, so values cannot
    be moved between columns. Never filter on an encrypted column: each encryption of the same
    value differs. Search through a blind-index column instead (e.g. users.phone_hash).
    """

    impl = Text
    cache_ok = True

    def __init__(self, context: str) -> None:
        super().__init__()
        self.context = context

    def process_bind_param(self, value: str | None, dialect: Dialect) -> str | None:  # noqa: ARG002
        return None if value is None else field_cipher().encrypt(value, context=self.context)

    def process_result_value(self, value: Any, dialect: Dialect) -> str | None:  # noqa: ARG002
        return None if value is None else field_cipher().decrypt(value, context=self.context)


def str_enum(enum_class: type[StrEnum], name: str) -> Enum:
    """VARCHAR + CHECK constraint (named ``ck_<table>_<name>``) instead of a native PG enum.

    Adding a value is then an ordinary migration that replaces the CHECK constraint, rather
    than an ALTER TYPE with its transaction restrictions.
    """
    return Enum(
        enum_class,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        length=max(len(member.value) for member in enum_class),
        values_callable=lambda members: [member.value for member in members],
    )


STATE_CODES_SQL = "ARRAY[" + ", ".join(f"'{state.value}'" for state in NigerianState) + "]"


def states_array_check(column: str, name: str) -> CheckConstraint:
    """Every element of a text[] column is a valid NigerianState code."""
    return CheckConstraint(f"{column} <@ {STATE_CODES_SQL}::varchar[]", name=name)

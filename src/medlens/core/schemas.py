"""Shared building blocks for request and response schemas.

Naming convention, per entity:

    <Entity>Create   request body for POST           -> subclass InputModel
    <Entity>Update   request body for PATCH          -> subclass UpdateModel (every field optional)
    <Entity>Read     full representation             -> subclass ReadModel
    <Entity>Summary  compact item in list responses  -> subclass ReadModel

Inputs reject unknown fields (``extra="forbid"``) so typos fail loudly instead of being ignored.
Responses are always built from these schemas, never from ORM objects (``from_attributes`` lets
a service call ``ScanRead.model_validate(scan_row)``).

Reusable field types live at the bottom of this module. Their normalisation rules are in
``core/identifiers.py``.
"""

from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from medlens.core.identifiers import (
    NAFDAC_NUMBER_MAX_LENGTH,
    NigerianState,
    normalize_batch_number,
    normalize_nafdac_number,
    normalize_ng_phone,
    sanitize_text,
)

__all__ = [
    "APIModel",
    "BatchNumber",
    "EmailAddress",
    "InputModel",
    "Latitude",
    "Longitude",
    "NafdacNumber",
    "NigerianState",
    "NonEmptyStr",
    "PhoneNumberNG",
    "ReadModel",
    "SafeText",
    "UpdateModel",
]


class APIModel(BaseModel):
    """Base for every API schema."""

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        # Accept both field names and aliases (what `populate_by_name=True` did before 2.11).
        validate_by_name=True,
        validate_by_alias=True,
    )


class InputModel(APIModel):
    """Request bodies (<Entity>Create and other inputs). Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class UpdateModel(InputModel):
    """PATCH bodies (<Entity>Update): every field optional; only fields sent are applied."""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        required = [name for name, field in cls.model_fields.items() if field.is_required()]
        if required:
            raise TypeError(f"{cls.__name__}: PATCH fields must be optional: {required}")

    def changes(self) -> dict[str, Any]:
        """Fields the client actually sent (explicit nulls included)."""
        return self.model_dump(exclude_unset=True)


class ReadModel(APIModel):
    """Responses (<Entity>Read, <Entity>Summary)."""


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Deliberately loose (no deliverability check): sign-in and invitations compare it exactly.
EmailAddress = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, to_lower=True, max_length=254, pattern=r"^[^@\s]+@[^@\s]+$"
    ),
]

SafeText = Annotated[
    str,
    StringConstraints(max_length=10_000),
    AfterValidator(sanitize_text),
    Field(
        description="Free text. Any HTML is stripped; the result is plain text.",
        examples=["Seal was broken and the print looked faded."],
    ),
]

PhoneNumberNG = Annotated[
    str,
    StringConstraints(max_length=32),
    AfterValidator(normalize_ng_phone),
    Field(
        description=(
            "Nigerian mobile number in any common format (0803 123 4567, +234 803 123 4567). "
            "Stored and returned in E.164 form."
        ),
        examples=["+2348031234567"],
    ),
]

NafdacNumber = Annotated[
    str,
    StringConstraints(max_length=NAFDAC_NUMBER_MAX_LENGTH),
    AfterValidator(normalize_nafdac_number),
    Field(
        description=(
            "NAFDAC registration number, e.g. A4-1234 or 04-1234. Case, separators and common "
            "OCR confusions in the digits (O/0, I/1, S/5) are normalised."
        ),
        examples=["A4-1234", "04-1234"],
    ),
]

BatchNumber = Annotated[
    str,
    StringConstraints(max_length=64),
    AfterValidator(normalize_batch_number),
    Field(
        description="Manufacturer batch (lot) number. Upper-cased, inner spaces removed.",
        examples=["B2309-14"],
    ),
]

Latitude = Annotated[
    float,
    Field(ge=-90, le=90, allow_inf_nan=False, description="WGS 84 latitude in degrees."),
]

Longitude = Annotated[
    float,
    Field(ge=-180, le=180, allow_inf_nan=False, description="WGS 84 longitude in degrees."),
]

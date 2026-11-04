"""Annotated field types and schema base classes, exercised through Pydantic models."""

import pytest
from pydantic import ValidationError

from medlens.core.schemas import (
    BatchNumber,
    InputModel,
    Latitude,
    Longitude,
    NafdacNumber,
    NigerianState,
    NonEmptyStr,
    PhoneNumberNG,
    ReadModel,
    SafeText,
    UpdateModel,
)


class Fields(InputModel):
    nafdac: NafdacNumber | None = None
    batch: BatchNumber | None = None
    phone: PhoneNumberNG | None = None
    state: NigerianState | None = None
    lat: Latitude | None = None
    lng: Longitude | None = None
    name: NonEmptyStr | None = None
    notes: SafeText | None = None


def valid(**fields: object) -> Fields:
    return Fields.model_validate(fields)


def invalid(**fields: object) -> None:
    with pytest.raises(ValidationError):
        Fields.model_validate(fields)


# ------------------------------------------------------------------------- NAFDAC numbers
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A4-1234", "A4-1234"),
        ("04-1234", "04-1234"),
        ("a4-1234", "A4-1234"),
        ("  A4 1234 ", "A4-1234"),
        ("A4_1234", "A4-1234"),
        ("A4\u20131234", "A4-1234"),  # en dash
        ("A4-100459", "A4-100459"),
        ("A11-0234", "A11-0234"),
        # OCR confusions in digit positions: O->0, I->1, S->5
        ("A4-I234", "A4-1234"),
        ("A4-12S4", "A4-1254"),
        ("AO-1234", "A0-1234"),
        ("O4-1234", "04-1234"),  # two-character prefix starting with O reads as digits
        ("I4-OOI5", "14-0015"),
        ("S4-1234", "54-1234"),
        ("A1O-1234", "A10-1234"),  # three characters: letter slot kept, digits mapped
        ("A7-4942L", "A7-4942L"),  # L suffix, as in the Greenbook
        ("a7 4942l", "A7-4942L"),
    ],
)
def test_nafdac_number_normalisation(raw: str, expected: str) -> None:
    assert valid(nafdac=raw).nafdac == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "A4",
        "A41234",  # no separator: prefix and serial cannot be told apart
        "A4-123",  # serial too short
        "A4-1234567",  # serial too long
        "AB-1234",  # second prefix character must be a digit
        "A4-12B4",  # B is not an OCR confusion we correct
        "A4-1234-5",
        "AA11-1234",
        "4-1234X",
        "A7-4942LL",
        "A4-123L",  # the suffix does not count towards the serial
        "04-5112ugo",
        "4/1/9086",  # legacy-looking register values are not mapped
    ],
)
def test_nafdac_number_rejects_invalid(raw: str) -> None:
    invalid(nafdac=raw)


# ------------------------------------------------------------------------- batch numbers
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("b2309-14", "B2309-14"), (" LOT 22 / 7 ", "LOT22/7"), ("0OI5", "0OI5"), ("A.1", "A.1")],
)
def test_batch_number_normalisation(raw: str, expected: str) -> None:
    assert valid(batch=raw).batch == expected


@pytest.mark.parametrize("raw", ["", "-123", "B#12", "x" * 41, "<b>1</b>"])
def test_batch_number_rejects_invalid(raw: str) -> None:
    invalid(batch=raw)


# ------------------------------------------------------------------------- phone numbers
@pytest.mark.parametrize(
    "raw",
    [
        "08031234567",
        "0803 123 4567",
        "0803-123-4567",
        "8031234567",
        "2348031234567",
        "+2348031234567",
        "+234 803 123 4567",
        "+234 (0) 803 123 4567",
        "002348031234567",
    ],
)
def test_phone_normalises_to_e164(raw: str) -> None:
    assert valid(phone=raw).phone == "+2348031234567"


@pytest.mark.parametrize("prefix", ["0701", "0802", "0813", "0903", "0915"])
def test_phone_accepts_mobile_ranges(prefix: str) -> None:
    assert valid(phone=f"{prefix}1234567").phone == f"+234{prefix[1:]}1234567"


@pytest.mark.parametrize(
    "raw",
    [
        "0803123456",  # too short
        "080312345678",  # too long
        "+447911123456",  # not Nigerian
        "01234567890",  # landline (Lagos)
        "06031234567",  # unallocated range
        "0803123456a",  # pragma: allowlist secret
        "",
    ],
)
def test_phone_rejects_invalid(raw: str) -> None:
    invalid(phone=raw)


# ------------------------------------------------------------------------- states
def test_there_are_36_states_and_the_fct() -> None:
    assert len(NigerianState) == 37
    assert NigerianState.FCT.display_name == "Federal Capital Territory"
    assert NigerianState.AKWA_IBOM.display_name == "Akwa Ibom"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NG-LA", NigerianState.LAGOS),
        ("lagos", NigerianState.LAGOS),
        ("LA", NigerianState.LAGOS),
        ("ng-la", NigerianState.LAGOS),
        ("Akwa Ibom", NigerianState.AKWA_IBOM),
        ("cross-river", NigerianState.CROSS_RIVER),
        ("FCT", NigerianState.FCT),
        ("Abuja", NigerianState.FCT),
        ("Nassarawa", NigerianState.NASARAWA),
    ],
)
def test_state_accepts_codes_names_and_aliases(raw: str, expected: NigerianState) -> None:
    assert valid(state=raw).state is expected


@pytest.mark.parametrize("raw", ["Atlantis", "NG-XX", "", "Lagos State of Mind"])
def test_state_rejects_unknown(raw: str) -> None:
    invalid(state=raw)


def test_state_serialises_as_code() -> None:
    assert valid(state="kano").model_dump(mode="json")["state"] == "NG-KN"


# ------------------------------------------------------------------------- coordinates
@pytest.mark.parametrize(("lat", "lng"), [(6.5244, 3.3792), (-90, -180), (90, 180), (0, 0)])
def test_coordinates_accept_valid_range(lat: float, lng: float) -> None:
    fields = valid(lat=lat, lng=lng)
    assert (fields.lat, fields.lng) == (lat, lng)


@pytest.mark.parametrize(
    "fields",
    [
        {"lat": 90.0001},
        {"lat": -91},
        {"lng": 180.5},
        {"lng": -181},
        {"lat": float("nan")},
        {"lng": float("inf")},
    ],
)
def test_coordinates_reject_out_of_range(fields: dict[str, float]) -> None:
    invalid(**fields)


# ------------------------------------------------------------------------- text
def test_non_empty_str_strips_and_requires_content() -> None:
    assert valid(name="  Emzor  ").name == "Emzor"
    invalid(name="   ")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Seal broken", "Seal broken"),
        ("<b>Seal</b> broken", "Seal broken"),
        ("<script>alert(1)</script>Faded print", "Faded print"),
        ("&lt;script&gt;alert(1)&lt;/script&gt;Faded", "Faded"),  # entity-encoded markup
        ("<img src=x onerror=alert(1)>Torn box", "Torn box"),
        ("5 < 6 & 7 > 3", "5 < 6 & 7 > 3"),  # plain text is not mangled or escaped
        ("  padded  ", "padded"),
    ],
)
def test_safe_text_strips_html(raw: str, expected: str) -> None:
    assert valid(notes=raw).notes == expected


def test_safe_text_is_length_limited() -> None:
    invalid(notes="x" * 10_001)


# ------------------------------------------------------------------------- base classes
def test_inputs_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Fields.model_validate({"nafdac": "A4-1234", "is_admin": True})


def test_update_models_only_report_fields_that_were_sent() -> None:
    class ProductUpdate(UpdateModel):
        brand_name: NonEmptyStr | None = None
        strength: NonEmptyStr | None = None

    update = ProductUpdate.model_validate({"brand_name": "Panadol", "strength": None})

    assert update.changes() == {"brand_name": "Panadol", "strength": None}
    assert ProductUpdate().changes() == {}


def test_update_models_must_not_have_required_fields() -> None:
    with pytest.raises(TypeError, match="PATCH fields must be optional"):

        class BrokenUpdate(UpdateModel):
            brand_name: str


def test_read_models_build_from_attributes() -> None:
    class Row:
        brand_name = "Panadol Extra"
        nafdac_number = "A4-1234"

    class ProductSummary(ReadModel):
        brand_name: str
        nafdac_number: str

    assert ProductSummary.model_validate(Row()).brand_name == "Panadol Extra"

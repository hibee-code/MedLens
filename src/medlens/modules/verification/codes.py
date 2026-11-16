"""Stable codes stored with every scan. Clients translate them (CLAUDE.md §1: no user-facing
text here); never rename a value, only add new ones."""

from enum import StrEnum


class ReasonCode(StrEnum):
    # The photo
    BLURRY = "BLURRY"
    TOO_DARK = "TOO_DARK"
    TOO_BRIGHT = "TOO_BRIGHT"
    TOO_SMALL = "TOO_SMALL"
    UNREADABLE = "UNREADABLE"  # no NAFDAC number or product name could be read
    # The registration record
    NOT_IN_REGISTRY = "NOT_IN_REGISTRY"
    NAFDAC_NUMBER_MISMATCH = "NAFDAC_NUMBER_MISMATCH"  # number and name point at different products
    # The number is in no registration, but the pack names a registered brand (strict match).
    # Regulators only: consumers see DETAILS_DO_NOT_MATCH_REGISTER.
    NUMBER_DOES_NOT_MATCH_BRAND = "NUMBER_DOES_NOT_MATCH_BRAND"
    REGISTRATION_REVOKED = "REGISTRATION_REVOKED"
    REGISTRATION_SUSPENDED = "REGISTRATION_SUSPENDED"
    REGISTRATION_EXPIRED = "REGISTRATION_EXPIRED"
    # No longer reported (2026-11-13): folded into DETAILS_DO_NOT_MATCH_REGISTER; older scans
    # may carry it and are shown with the new code (scans/router.py).
    NAME_MISMATCH = "NAME_MISMATCH"
    # The pack and its batch
    PACK_EXPIRED = "PACK_EXPIRED"
    BATCH_NOT_DECLARED = "BATCH_NOT_DECLARED"
    EXPIRY_MISMATCH = "EXPIRY_MISMATCH"
    BATCH_SEEN_WIDELY = "BATCH_SEEN_WIDELY"
    OUTSIDE_DISTRIBUTION = "OUTSIDE_DISTRIBUTION"
    VISUAL_MISMATCH = "VISUAL_MISMATCH"
    PUBLISHED_ALERT = "PUBLISHED_ALERT"  # an official alert names this product and batch
    ALERT_FOR_THIS_PRODUCT = "ALERT_FOR_THIS_PRODUCT"  # an official alert names the product only
    RECALLED_BATCH = "RECALLED_BATCH"  # an official recall covers this batch
    # The pack's name, ingredients, strength, dosage form, holder, manufacture date or barcode
    # disagree with its registration. Deliberately coarse: the field is visible to regulators
    # only.
    DETAILS_DO_NOT_MATCH_REGISTER = "DETAILS_DO_NOT_MATCH_REGISTER"


class AdviceCode(StrEnum):
    RETAKE_PHOTO = "RETAKE_PHOTO"
    DO_NOT_USE = "DO_NOT_USE"
    KEEP_THE_PACK = "KEEP_THE_PACK"  # evidence for a regulator
    REPORT_TO_NAFDAC = "REPORT_TO_NAFDAC"
    ASK_A_PHARMACIST = "ASK_A_PHARMACIST"
    CHECK_EXPIRY_DATE = "CHECK_EXPIRY_DATE"
    DO_NOT_USE_RETURN = "DO_NOT_USE_RETURN"  # recalled: stop using it, return it where bought


# Codes no longer reported, and what clients see instead for scans that still carry them.
_RETIRED = {ReasonCode.NAME_MISMATCH.value: ReasonCode.DETAILS_DO_NOT_MATCH_REGISTER.value}
# Codes for regulators only, and what everyone else sees instead.
_REGULATOR_ONLY = {
    ReasonCode.NUMBER_DOES_NOT_MATCH_BRAND.value: ReasonCode.DETAILS_DO_NOT_MATCH_REGISTER.value
}


def current_reason_codes(codes: list[str], *, regulator: bool = False) -> list[str]:
    """Stored reason codes as reported today, without duplicates: retired codes replaced, and
    regulator-only codes replaced unless ``regulator`` (whoever may see the scan's analysis)."""
    replaced = _RETIRED if regulator else _RETIRED | _REGULATOR_ONLY
    shown: list[str] = []
    for stored in codes:
        code = replaced.get(stored, stored)
        if code not in shown:
            shown.append(code)
    return shown

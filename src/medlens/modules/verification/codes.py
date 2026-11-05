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
    REGISTRATION_REVOKED = "REGISTRATION_REVOKED"
    REGISTRATION_SUSPENDED = "REGISTRATION_SUSPENDED"
    REGISTRATION_EXPIRED = "REGISTRATION_EXPIRED"
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


class AdviceCode(StrEnum):
    RETAKE_PHOTO = "RETAKE_PHOTO"
    DO_NOT_USE = "DO_NOT_USE"
    KEEP_THE_PACK = "KEEP_THE_PACK"  # evidence for a regulator
    REPORT_TO_NAFDAC = "REPORT_TO_NAFDAC"
    ASK_A_PHARMACIST = "ASK_A_PHARMACIST"
    CHECK_EXPIRY_DATE = "CHECK_EXPIRY_DATE"
    DO_NOT_USE_RETURN = "DO_NOT_USE_RETURN"  # recalled: stop using it, return it where bought

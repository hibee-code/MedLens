from typing import Protocol


class SmsUnavailableError(Exception):
    """The provider could not be reached (after retries) or its circuit breaker is open."""


class SmsDeliveryError(Exception):
    """The provider refused the message (bad number, account problem); retrying will not help."""


class SMSPort(Protocol):
    """Sends one SMS. The only outbound SMS MedLens sends is the login OTP (CLAUDE.md §1)."""

    async def send(self, to_e164: str, message: str) -> None: ...

    async def aclose(self) -> None: ...

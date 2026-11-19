from typing import Protocol


class EmailUnavailableError(Exception):
    """The mail server could not be reached (after retries) or its circuit breaker is open."""


class EmailDeliveryError(Exception):
    """The mail server refused the message (bad address, account problem); retrying will not
    help."""


class EmailPort(Protocol):
    """Sends one plain-text e-mail. MedLens sends only staff sign-up codes, password-reset codes
    and invitations (decision log, "E-mail channel for staff")."""

    async def send(self, to: str, subject: str, text: str) -> None: ...

    async def aclose(self) -> None: ...

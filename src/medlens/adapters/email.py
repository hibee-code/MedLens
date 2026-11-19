"""E-mail adapters: console (local/test) and SMTP (any provider's SMTP server).

The SMTP adapter follows the same policy as the SMS adapters (CLAUDE.md §6): a timeout, up to
three attempts with jittered exponential backoff on temporary failures only (network errors,
timeouts, 4xx SMTP replies), and a circuit breaker that opens after five consecutive failed
sends. A refused message (5xx reply for the recipient) is not retried and does not trip the
breaker. smtplib is blocking, so each send runs in a worker thread, off the event loop.
"""

import asyncio
import smtplib
import ssl
from collections import deque
from email.message import EmailMessage
from email.utils import formataddr, make_msgid

import structlog
from purgatory import AsyncCircuitBreakerFactory
from purgatory.domain.model import OpenedState
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from medlens.core.config import EmailSettings
from medlens.ports.email import EmailDeliveryError, EmailPort, EmailUnavailableError

log = structlog.stdlib.get_logger(__name__)

MAX_ATTEMPTS = 3
BREAKER_THRESHOLD = 5
BREAKER_RESET_SECONDS = 30
# SMTP replies meaning the server wants (other) credentials: 530, 534, 535, 538.
AUTH_REPLIES = frozenset({530, 534, 535, 538})
# Top-level domains reserved for testing and documentation (RFC 2606, RFC 6761): mail to them can
# never arrive, and sending it would only cost the sending domain's reputation (bounces). Demo
# and test accounts use them (@medlens.test).
RESERVED_TLDS = frozenset({"test", "example", "invalid", "localhost"})


def is_reserved_address(address: str) -> bool:
    domain = address.rpartition("@")[2].rstrip(".").lower()
    return domain.rpartition(".")[2] in RESERVED_TLDS


class _TransientError(Exception):
    pass


class ConsoleEmail:
    """Logs messages instead of sending them and keeps the latest in ``sent``.

    For local development and tests only: production settings refuse this provider, because
    the log line contains the one-time code. The address is not logged.
    """

    def __init__(self, history: int = 100) -> None:
        self.sent: deque[tuple[str, str, str]] = deque(maxlen=history)

    async def send(self, to: str, subject: str, text: str) -> None:
        self.sent.append((to, subject, text))
        log.info("email.console_message", subject=subject, text=text)

    async def aclose(self) -> None:
        return None


class SmtpEmail:
    def __init__(self, settings: EmailSettings, breakers: AsyncCircuitBreakerFactory) -> None:
        self._settings = settings
        self._breakers = breakers

    async def send(self, to: str, subject: str, text: str) -> None:
        if is_reserved_address(to):
            log.info("email.reserved_domain_skipped")
            raise EmailDeliveryError("smtp: reserved test domain, not sent")
        message = self._message(to, subject, text)
        try:
            async with await self._breakers.get_breaker("email:smtp"):
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(_TransientError),
                    stop=stop_after_attempt(MAX_ATTEMPTS),
                    wait=wait_exponential_jitter(initial=0.5, max=4, jitter=0.5),
                    reraise=True,
                ):
                    with attempt:
                        await asyncio.to_thread(self._deliver, message)
        except OpenedState:
            log.warning("email.circuit_open")
            raise EmailUnavailableError("e-mail circuit breaker is open") from None
        except _TransientError as exc:
            log.warning("email.unavailable", error=str(exc))
            raise EmailUnavailableError(str(exc)) from None

    def _message(self, to: str, subject: str, text: str) -> EmailMessage:
        settings = self._settings
        message = EmailMessage()
        message["From"] = formataddr((settings.from_name, settings.from_address))
        message["To"] = to
        message["Subject"] = subject
        message["Message-ID"] = make_msgid(domain=settings.from_address.partition("@")[2])
        message.set_content(text)
        return message

    def _deliver(self, message: EmailMessage) -> None:
        settings = self._settings
        host, port, timeout = settings.smtp_host or "", settings.smtp_port, settings.timeout_seconds
        try:
            if settings.smtp_security == "ssl":
                server: smtplib.SMTP = smtplib.SMTP_SSL(
                    host, port, timeout=timeout, context=ssl.create_default_context()
                )
            else:
                server = smtplib.SMTP(host, port, timeout=timeout)
            with server:
                if settings.smtp_security == "starttls":
                    server.starttls(context=ssl.create_default_context())
                if settings.smtp_username and settings.smtp_password is not None:
                    server.login(settings.smtp_username, settings.smtp_password.get_secret_value())
                server.send_message(message)
        except smtplib.SMTPRecipientsRefused:
            raise EmailDeliveryError("smtp: recipient refused") from None
        except smtplib.SMTPAuthenticationError:
            # A configuration problem, not a passing one: no retry, but the API answers 503.
            raise EmailUnavailableError("smtp: authentication failed") from None
        except smtplib.SMTPResponseException as exc:
            if exc.smtp_code in AUTH_REPLIES:
                # Authentication missing or refused (e.g. no API key set yet): a configuration
                # problem, so the API answers 503 instead of pretending the code was sent.
                raise EmailUnavailableError(f"smtp: authentication ({exc.smtp_code})") from None
            if 400 <= exc.smtp_code < 500:
                raise _TransientError(f"smtp: {exc.smtp_code}") from None
            raise EmailDeliveryError(f"smtp: {exc.smtp_code}") from None
        except (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError) as exc:
            raise _TransientError(f"smtp: {type(exc).__name__}") from None
        except OSError as exc:  # connection refused, DNS, timeouts (socket.timeout)
            raise _TransientError(f"smtp: {type(exc).__name__}") from None

    async def aclose(self) -> None:
        return None


def create_email_adapter(settings: EmailSettings) -> EmailPort:
    if settings.provider == "console":
        return ConsoleEmail()
    breakers = AsyncCircuitBreakerFactory(
        default_threshold=BREAKER_THRESHOLD,
        default_ttl=BREAKER_RESET_SECONDS,
        exclude=[EmailDeliveryError],
    )
    return SmtpEmail(settings, breakers)

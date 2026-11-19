"""E-mail adapters: console (local), SMTP with a fake server (no network), and its retry,
refusal and circuit-breaker policy."""

import smtplib
from email.message import EmailMessage
from typing import Any, ClassVar

import pytest
from pydantic import SecretStr

from medlens.adapters import email as email_adapter
from medlens.adapters.email import ConsoleEmail, SmtpEmail, create_email_adapter
from medlens.core.config import EmailSettings
from medlens.ports.email import EmailDeliveryError, EmailUnavailableError


class FakeServer:
    """Stands in for smtplib.SMTP / SMTP_SSL: records the conversation, fails as scripted."""

    instances: ClassVar[list["FakeServer"]] = []
    failures: ClassVar[list[BaseException]] = []

    def __init__(self, host: str, port: int, timeout: float, **kwargs: Any) -> None:
        self.host, self.port, self.timeout, self.kwargs = host, port, timeout, kwargs
        self.calls: list[str] = []
        self.messages: list[EmailMessage] = []
        FakeServer.instances.append(self)

    def __enter__(self) -> "FakeServer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.calls.append("quit")

    def starttls(self, context: Any = None) -> None:
        self.calls.append("starttls")

    def login(self, user: str, password: str) -> None:
        self.calls.append(f"login:{user}")

    def send_message(self, message: EmailMessage) -> None:
        if FakeServer.failures:
            raise FakeServer.failures.pop(0)
        self.messages.append(message)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> type[FakeServer]:
    FakeServer.instances, FakeServer.failures = [], []
    monkeypatch.setattr(email_adapter.smtplib, "SMTP", FakeServer)
    monkeypatch.setattr(email_adapter.smtplib, "SMTP_SSL", FakeServer)
    return FakeServer


def smtp(**overrides: Any) -> SmtpEmail:
    settings = EmailSettings(
        **{
            "provider": "smtp",
            "smtp_host": "smtp.example.com",
            "smtp_username": "mailer",
            "smtp_password": SecretStr("smtp-key"),  # pragma: allowlist secret
            "from_address": "no-reply@medlens.ng",
            **overrides,
        }
    )
    adapter = create_email_adapter(settings)
    assert isinstance(adapter, SmtpEmail)
    return adapter


async def test_console_keeps_messages_and_logs_no_address() -> None:
    console = create_email_adapter(EmailSettings())
    assert isinstance(console, ConsoleEmail)

    await console.send("someone@example.com", "Subject", "Your code is 123456.")

    assert console.sent[-1] == ("someone@example.com", "Subject", "Your code is 123456.")


async def test_smtp_sends_over_starttls_with_login(server: type[FakeServer]) -> None:
    await smtp().send("pharmacist@example.com", "Your MedLens code", "Code: 123456")

    [conversation] = server.instances
    assert (conversation.host, conversation.port) == ("smtp.example.com", 587)
    assert conversation.calls == ["starttls", "login:mailer", "quit"]
    [message] = conversation.messages
    assert message["To"] == "pharmacist@example.com"
    assert message["From"] == "MedLens <no-reply@medlens.ng>"
    assert message["Subject"] == "Your MedLens code"
    assert "Code: 123456" in message.get_content()


async def test_implicit_tls_on_port_465(server: type[FakeServer]) -> None:
    await smtp(smtp_security="ssl", smtp_port=465).send("a@example.com", "s", "t")

    [conversation] = server.instances
    assert conversation.port == 465
    assert "context" in conversation.kwargs  # SMTP_SSL with a verified TLS context
    assert "starttls" not in conversation.calls


async def test_a_temporary_failure_is_retried(server: type[FakeServer]) -> None:
    server.failures = [smtplib.SMTPServerDisconnected("dropped")]

    await smtp().send("a@example.com", "s", "t")

    assert len(server.instances) == 2  # reconnected once, then delivered


async def test_a_refused_recipient_is_not_retried(server: type[FakeServer]) -> None:
    server.failures = [smtplib.SMTPRecipientsRefused({"a@example.com": (550, b"no such user")})]

    with pytest.raises(EmailDeliveryError):
        await smtp().send("a@example.com", "s", "t")

    assert len(server.instances) == 1


async def test_repeated_temporary_failures_make_mail_unavailable(server: type[FakeServer]) -> None:
    server.failures = [smtplib.SMTPResponseException(451, b"try later")] * 3

    with pytest.raises(EmailUnavailableError):
        await smtp().send("a@example.com", "s", "t")

    assert len(server.instances) == 3  # MAX_ATTEMPTS


async def test_bad_smtp_credentials_answer_unavailable_without_retries(
    server: type[FakeServer],
) -> None:
    server.failures = [smtplib.SMTPAuthenticationError(535, b"bad credentials")]

    with pytest.raises(EmailUnavailableError, match="authentication failed"):
        await smtp().send("a@example.com", "s", "t")

    assert len(server.instances) == 1


async def test_the_circuit_opens_after_repeated_failures(server: type[FakeServer]) -> None:
    adapter = smtp()
    server.failures = [OSError("connection refused")] * 15
    for _ in range(5):
        with pytest.raises(EmailUnavailableError):
            await adapter.send("a@example.com", "s", "t")
    sent_before = len(server.instances)

    with pytest.raises(EmailUnavailableError, match="circuit breaker is open"):
        await adapter.send("a@example.com", "s", "t")

    assert len(server.instances) == sent_before  # no connection attempted


async def test_a_server_that_wants_credentials_answers_unavailable(
    server: type[FakeServer],
) -> None:
    """E.g. Resend before its API key is set: 530, not a refused recipient, so the API says
    503 instead of claiming the code was sent."""
    server.failures = [smtplib.SMTPSenderRefused(530, b"authentication required", "a@b.c")]

    with pytest.raises(EmailUnavailableError, match=r"authentication \(530\)"):
        await smtp(smtp_password=None).send("a@example.com", "s", "t")

    assert len(server.instances) == 1  # not retried


async def test_reserved_test_domains_are_never_sent(server: type[FakeServer]) -> None:
    for address in ("demo.admin@medlens.test", "x@foo.example", "y@host.invalid", "z@localhost"):
        with pytest.raises(EmailDeliveryError, match="reserved"):
            await smtp().send(address, "s", "t")

    assert server.instances == []  # no connection was made
    assert not email_adapter.is_reserved_address("pharmacist@medlens.ng")

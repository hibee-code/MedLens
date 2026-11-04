"""SMS adapters against a mocked HTTP transport: payloads, retries and the circuit breaker."""

import json

import httpx
import pytest

from medlens.adapters.sms import ConsoleSMS, create_sms_adapter
from medlens.core.config import SmsSettings
from medlens.ports.sms import SmsDeliveryError, SmsUnavailableError


class Provider:
    """Scripted responses; records every request."""

    def __init__(self, *responses: int | httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.responses.pop(0) if self.responses else 200
        if isinstance(reply, httpx.Response):
            return reply
        if reply == 0:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(reply, json={"message": "ok"})


def termii(provider: Provider) -> object:
    settings = SmsSettings(provider="termii", termii_api_key="test-termii-key")  # type: ignore[arg-type]
    return create_sms_adapter(settings, transport=httpx.MockTransport(provider))


async def test_console_sms_records_messages() -> None:
    sms = ConsoleSMS()

    await sms.send("+2348031234567", "Your code is 123456")

    assert list(sms.sent) == [("+2348031234567", "Your code is 123456")]
    await sms.aclose()


async def test_termii_request_format() -> None:
    provider = Provider(200)
    sms = termii(provider)

    await sms.send("+2348031234567", "Your code is 123456")  # type: ignore[attr-defined]

    [request] = provider.requests
    assert request.url == "https://api.ng.termii.com/api/sms/send"
    body = json.loads(request.content)
    assert body == {
        "to": "2348031234567",
        "from": "MedLens",
        "sms": "Your code is 123456",
        "type": "plain",
        "channel": "dnd",
        "api_key": "test-termii-key",  # pragma: allowlist secret
    }


async def test_africastalking_request_format_and_recipient_status() -> None:
    accepted = httpx.Response(
        201,
        json={
            "SMSMessageData": {"Recipients": [{"status": "Success", "number": "+2348031234567"}]}
        },
    )
    rejected = httpx.Response(
        201, json={"SMSMessageData": {"Recipients": [{"status": "InvalidPhoneNumber"}]}}
    )
    provider = Provider(accepted, rejected)
    settings = SmsSettings(
        provider="africastalking",
        africastalking_username="medlens",
        africastalking_api_key="test-at-key",  # type: ignore[arg-type]
    )
    sms = create_sms_adapter(settings, transport=httpx.MockTransport(provider))

    await sms.send("+2348031234567", "hello")
    with pytest.raises(SmsDeliveryError):
        await sms.send("+2348031234567", "hello")

    request = provider.requests[0]
    assert request.url == "https://api.africastalking.com/version1/messaging"
    assert request.headers["apiKey"] == "test-at-key"  # pragma: allowlist secret
    assert b"username=medlens" in request.content
    assert b"to=%2B2348031234567" in request.content


async def test_transient_failures_are_retried() -> None:
    provider = Provider(503, 0, 200)  # 5xx, connection refused, success

    await termii(provider).send("+2348031234567", "hi")  # type: ignore[attr-defined]

    assert len(provider.requests) == 3


async def test_refused_messages_are_not_retried() -> None:
    provider = Provider(400)

    with pytest.raises(SmsDeliveryError):
        await termii(provider).send("+2348031234567", "hi")  # type: ignore[attr-defined]

    assert len(provider.requests) == 1


async def test_breaker_opens_after_repeated_outages() -> None:
    provider = Provider(*([500] * 100))
    sms = termii(provider)

    for _ in range(5):
        with pytest.raises(SmsUnavailableError):
            await sms.send("+2348031234567", "hi")  # type: ignore[attr-defined]
    calls = len(provider.requests)
    with pytest.raises(SmsUnavailableError, match="circuit breaker is open"):
        await sms.send("+2348031234567", "hi")  # type: ignore[attr-defined]

    assert len(provider.requests) == calls  # no request while open
    assert calls == 15  # 5 sends x 3 attempts


def test_provider_credentials_are_required() -> None:
    with pytest.raises(ValueError, match="termii_api_key is required"):
        SmsSettings(provider="termii")
    with pytest.raises(ValueError, match="africastalking_username"):
        SmsSettings(provider="africastalking")

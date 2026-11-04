"""SMS adapters: console (local/test), Termii and Africa's Talking.

HTTP adapters share one policy (CLAUDE.md §6): a client timeout, up to three attempts with
jittered exponential backoff on network errors, 5xx and 429 only, and a circuit breaker per
provider that opens after five consecutive failed sends. A refused message (4xx) is not
retried and does not trip the breaker.
"""

from collections import deque
from typing import Any

import httpx
import structlog
from purgatory import AsyncCircuitBreakerFactory
from purgatory.domain.model import OpenedState
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from medlens.core.config import SmsSettings
from medlens.ports.sms import SmsDeliveryError, SMSPort, SmsUnavailableError

log = structlog.stdlib.get_logger(__name__)

MAX_ATTEMPTS = 3
BREAKER_THRESHOLD = 5
BREAKER_RESET_SECONDS = 30


class _TransientError(Exception):
    pass


class ConsoleSMS:
    """Logs messages instead of sending them and keeps the latest in ``sent``.

    For local development and tests only: production settings refuse this provider, because
    the log line contains the one-time code.
    """

    def __init__(self, history: int = 100) -> None:
        self.sent: deque[tuple[str, str]] = deque(maxlen=history)

    async def send(self, to_e164: str, message: str) -> None:
        self.sent.append((to_e164, message))
        log.info("sms.console_message", to=to_e164, message=message)

    async def aclose(self) -> None:
        return None


class _HttpSms:
    provider: str

    def __init__(self, client: httpx.AsyncClient, breakers: AsyncCircuitBreakerFactory) -> None:
        self._client = client
        self._breakers = breakers

    async def send(self, to_e164: str, message: str) -> None:
        try:
            async with await self._breakers.get_breaker(f"sms:{self.provider}"):
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(_TransientError),
                    stop=stop_after_attempt(MAX_ATTEMPTS),
                    wait=wait_exponential_jitter(multiplier=0.2, max=2),
                    reraise=True,
                ):
                    with attempt:
                        await self._deliver(to_e164, message)
        except OpenedState:
            log.warning("sms.circuit_open", provider=self.provider)
            raise SmsUnavailableError(f"{self.provider} circuit breaker is open") from None
        except _TransientError as exc:
            log.warning("sms.unavailable", provider=self.provider, error=str(exc))
            raise SmsUnavailableError(str(exc)) from None
        log.info("sms.sent", provider=self.provider)

    async def _deliver(self, to_e164: str, message: str) -> None:
        raise NotImplementedError

    async def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.post(path, **kwargs)
        except httpx.TransportError as exc:
            raise _TransientError(f"{self.provider}: {type(exc).__name__}") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise _TransientError(f"{self.provider}: HTTP {response.status_code}")
        if response.status_code >= 400:
            raise SmsDeliveryError(f"{self.provider}: HTTP {response.status_code}")
        return response

    async def aclose(self) -> None:
        await self._client.aclose()


class TermiiSMS(_HttpSms):
    provider = "termii"

    def __init__(
        self,
        client: httpx.AsyncClient,
        breakers: AsyncCircuitBreakerFactory,
        *,
        api_key: str,
        sender_id: str,
    ) -> None:
        super().__init__(client, breakers)
        self._api_key = api_key
        self._sender_id = sender_id

    async def _deliver(self, to_e164: str, message: str) -> None:
        await self._post(
            "/api/sms/send",
            json={
                "to": to_e164.removeprefix("+"),
                "from": self._sender_id,
                "sms": message,
                "type": "plain",
                "channel": "dnd",  # delivers to numbers on the Do-Not-Disturb register
                "api_key": self._api_key,
            },
        )


class AfricasTalkingSMS(_HttpSms):
    provider = "africastalking"

    def __init__(
        self,
        client: httpx.AsyncClient,
        breakers: AsyncCircuitBreakerFactory,
        *,
        username: str,
        api_key: str,
        sender_id: str,
    ) -> None:
        super().__init__(client, breakers)
        self._username = username
        self._api_key = api_key
        self._sender_id = sender_id

    async def _deliver(self, to_e164: str, message: str) -> None:
        response = await self._post(
            "/version1/messaging",
            data={
                "username": self._username,
                "to": to_e164,
                "message": message,
                "from": self._sender_id,
            },
            headers={"apiKey": self._api_key, "Accept": "application/json"},
        )
        recipients = response.json().get("SMSMessageData", {}).get("Recipients", [])
        if not recipients or recipients[0].get("status") != "Success":
            raise SmsDeliveryError("africastalking: message not accepted for the recipient")


def create_sms_adapter(
    settings: SmsSettings, *, transport: httpx.AsyncBaseTransport | None = None
) -> SMSPort:
    if settings.provider == "console":
        return ConsoleSMS()
    breakers = AsyncCircuitBreakerFactory(
        default_threshold=BREAKER_THRESHOLD,
        default_ttl=BREAKER_RESET_SECONDS,
        exclude=[SmsDeliveryError],
    )

    def client(base_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url, timeout=settings.timeout_seconds, transport=transport
        )

    # SmsSettings guarantees the credentials of the selected provider are present.
    if settings.provider == "termii" and settings.termii_api_key is not None:
        return TermiiSMS(
            client(settings.termii_base_url),
            breakers,
            api_key=settings.termii_api_key.get_secret_value(),
            sender_id=settings.sender_id,
        )
    if (
        settings.provider == "africastalking"
        and settings.africastalking_username
        and settings.africastalking_api_key is not None
    ):
        return AfricasTalkingSMS(
            client(settings.africastalking_base_url),
            breakers,
            username=settings.africastalking_username,
            api_key=settings.africastalking_api_key.get_secret_value(),
            sender_id=settings.sender_id,
        )
    raise ValueError(f"SMS provider {settings.provider!r} is missing its credentials")

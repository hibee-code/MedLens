"""Phone OTP sign-in (consumers and health workers)."""

import statistics
import time

import pytest

from tests.integration.api_support import Api, new_phone

pytestmark = pytest.mark.integration


async def test_full_otp_flow_creates_the_account_once(api: Api) -> None:
    phone = new_phone()

    requested = await api.client.post("/api/v1/auth/otp/request", json={"phone": phone})
    assert requested.status_code == 202
    assert requested.json()["expires_in"] == 300
    code = api.last_code(phone)
    verified = await api.client.post("/api/v1/auth/otp/verify", json={"phone": phone, "code": code})

    assert verified.status_code == 200, verified.text
    tokens = verified.json()
    assert tokens["status"] == "authenticated"
    assert tokens["token_type"] == "Bearer"
    me = (
        await api.client.get("/api/v1/users/me", headers=api.bearer(tokens["access_token"]))
    ).json()
    assert me["phone"] == phone
    assert me["global_role"] == "consumer"

    again = await api.otp_login(phone)
    me_again = (
        await api.client.get("/api/v1/users/me", headers=api.bearer(again["access_token"]))
    ).json()
    assert me_again["id"] == me["id"]  # same account, no duplicate
    actions = await api.audit_actions()
    assert "auth.otp_requested" in actions
    assert "auth.login_succeeded" in actions


async def test_phone_formats_are_normalised(api: Api) -> None:
    phone = new_phone()
    local_format = "0" + phone[4:7] + " " + phone[7:10] + " " + phone[10:]

    await api.client.post("/api/v1/auth/otp/request", json={"phone": local_format})

    assert api.last_code(phone)  # the SMS went to the E.164 form


async def test_five_wrong_codes_lock_the_code(api: Api) -> None:
    phone = new_phone()
    await api.client.post("/api/v1/auth/otp/request", json={"phone": phone})
    code = api.last_code(phone)
    wrong = "000000" if code != "000000" else "111111"

    for _ in range(5):
        response = await api.client.post(
            "/api/v1/auth/otp/verify", json={"phone": phone, "code": wrong}
        )
        assert response.status_code == 401
    locked = await api.client.post("/api/v1/auth/otp/verify", json={"phone": phone, "code": code})

    assert locked.status_code == 401  # even the right code: request a new one
    assert locked.json()["detail"] == "The code is invalid or has expired."
    await api.otp_login(phone)  # a new code works


async def test_verifying_without_a_code_fails(api: Api) -> None:
    response = await api.client.post(
        "/api/v1/auth/otp/verify", json={"phone": new_phone(), "code": "123456"}
    )

    assert response.status_code == 401


@pytest.mark.parametrize("phone", ["+447911123456", "0101234567", "not a phone"])
async def test_invalid_numbers_are_rejected(api: Api, phone: str) -> None:
    response = await api.client.post("/api/v1/auth/otp/request", json={"phone": phone})

    assert response.status_code == 422


async def test_otp_request_does_not_reveal_registered_numbers(api: Api) -> None:
    registered = new_phone()
    await api.create_user(phone=registered)

    async def timed(phone: str) -> tuple[float, int, dict[str, object]]:
        started = time.perf_counter()
        response = await api.client.post("/api/v1/auth/otp/request", json={"phone": phone})
        return time.perf_counter() - started, response.status_code, response.json()

    known = [await timed(registered) for _ in range(5)]
    unknown = [await timed(new_phone()) for _ in range(5)]

    assert {(status, str(body)) for _, status, body in known} == {
        (status, str(body)) for _, status, body in unknown
    }
    ratio = statistics.median(t for t, _, _ in known) / statistics.median(t for t, _, _ in unknown)
    assert 0.5 < ratio < 2.0

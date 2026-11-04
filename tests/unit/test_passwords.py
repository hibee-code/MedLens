import pytest

from medlens.modules.auth.passwords import MAX_LENGTH, password_problems


@pytest.mark.parametrize(
    ("password", "email", "problem"),
    [
        ("short", None, "at least 12 characters"),
        ("x" * (MAX_LENGTH + 1), None, f"at most {MAX_LENGTH} characters"),
        ("passwordpassword", None, "too common"),
        ("PasswordPassword", None, "too common"),  # case does not help
        ("chiamaka.obi.2026!", "chiamaka.obi@example.ng", "e-mail address"),
    ],
)
def test_weak_passwords_are_explained(password: str, email: str | None, problem: str) -> None:
    problems = password_problems(password, email=email)

    assert any(problem in message for message in problems), problems


@pytest.mark.parametrize(
    ("password", "email"),
    [
        ("Kola-nut tea at 7 o'clock", None),
        ("a-long-unusual-test-passphrase-42", "ab@example.ng"),  # short local parts are ignored
    ],
)
def test_strong_passwords_pass(password: str, email: str | None) -> None:
    assert password_problems(password, email=email) == []

"""The bootstrap-admin command's output: the token only where it may be shown."""

from typing import Any

import pytest

from medlens.cli import bootstrap_admin
from medlens.core.config import Settings
from medlens.core.errors import Conflict


def test_bootstrap_command(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from medlens.core import bootstrap as process_bootstrap
    from medlens.core import config

    monkeypatch.setattr(process_bootstrap, "configure_process", lambda settings: None)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    calls: list[str] = []

    async def fake(settings: Any, email: str) -> tuple[str, bool, str]:
        calls.append(email)
        return "tok-123", email.startswith("sent"), "se***@example.ng"

    async def refused(settings: Any, email: str) -> tuple[str, bool, str]:
        raise Conflict("A platform admin already exists.")

    monkeypatch.setattr(bootstrap_admin, "bootstrap", fake)
    assert bootstrap_admin.main(["--email", "not-an-address"]) == 2
    assert bootstrap_admin.main(["--email", "Sent@Example.NG"]) == 0
    assert calls == ["sent@example.ng"]
    out = capsys.readouterr().out
    assert "e-mailed to se***@example.ng" in out
    assert "tok-123" in out  # test environment: the token is printed

    monkeypatch.setattr(bootstrap_admin, "PRINTS_TOKEN", frozenset())
    assert bootstrap_admin.main(["--email", "sent@example.ng"]) == 0
    assert "tok-123" not in capsys.readouterr().out
    assert bootstrap_admin.main(["--email", "unsent@example.ng"]) == 1  # not sent, not shown
    assert "tok-123" not in capsys.readouterr().out

    monkeypatch.setattr(bootstrap_admin, "bootstrap", refused)
    assert bootstrap_admin.main(["--email", "sent@example.ng"]) == 1
    assert "already exists" in capsys.readouterr().err

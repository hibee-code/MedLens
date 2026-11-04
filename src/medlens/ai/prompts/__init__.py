"""Versioned prompt templates. A prompt is a file ``<name>.<version>.md`` next to this module;
its identity ``<name>@<version>`` is recorded with every model call. Never edit a released
version: add the next one and point the route at it."""

from functools import cache
from importlib import resources


@cache
def load_prompt(name: str, version: str) -> str:
    """The prompt text, byte-identical across calls (prompt caching depends on it)."""
    return (resources.files(__package__) / f"{name}.{version}.md").read_text(encoding="utf-8")

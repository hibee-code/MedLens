"""Redis key naming: ``medlens:v1:<domain>:<part>...``. Bumping the version orphans every key."""

KEY_NAMESPACE = "medlens:v1"


def redis_key(domain: str, *parts: object) -> str:
    return ":".join([KEY_NAMESPACE, domain, *(str(part) for part in parts)])

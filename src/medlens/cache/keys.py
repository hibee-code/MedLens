"""Redis key naming: ``medlens:v1:<domain>:<part>...``. Bumping the version orphans every key."""

KEY_NAMESPACE = "medlens:v1"


def redis_key(domain: str, *parts: object) -> str:
    return ":".join([KEY_NAMESPACE, domain, *(str(part) for part in parts)])


# The text model version the workers run, published by each worker at start: the API's
# question-embedding cache drops entries made by another version.
TEXT_MODEL_VERSION_KEY = redis_key("models", "text_version")

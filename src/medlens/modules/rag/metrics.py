from prometheus_client import Counter

ASSISTANT_REQUESTS = Counter(
    "medlens_assistant_requests",
    "Assistant questions by outcome (answered, insufficient_evidence, degraded, blocked).",
    ["outcome"],
)

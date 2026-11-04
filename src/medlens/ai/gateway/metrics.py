from prometheus_client import Counter, Histogram

LLM_CALLS = Counter(
    "medlens_llm_calls",
    "Model calls through the gateway, by route and outcome (blocked: no call was made).",
    ["route", "outcome"],
)
LLM_COST_USD = Counter("medlens_llm_cost_usd", "Metered model cost in US dollars.", ["route"])
LLM_LATENCY = Histogram(
    "medlens_llm_latency_seconds",
    "Model call latency, repair attempt included.",
    ["route"],
    buckets=(0.5, 1, 2, 4, 8, 16, 32, 64),
)

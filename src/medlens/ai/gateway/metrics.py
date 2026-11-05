from prometheus_client import Counter, Gauge, Histogram

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
# Today's spend (shared by every process: the max is the current value) and the budget.
LLM_SPENT_TODAY = Gauge(
    "medlens_llm_spent_today_usd",
    "Metered model cost since 00:00 UTC, as last read before a call.",
    multiprocess_mode="max",
)
LLM_DAILY_BUDGET = Gauge(
    "medlens_llm_daily_budget_usd",
    "MEDLENS_LLM__DAILY_BUDGET_USD: calls stop when the day's spend reaches it.",
    multiprocess_mode="max",
)
LLM_CACHE_HITS = Counter(
    "medlens_llm_cache_hits", "Answers served from the gateway's cache (no call).", ["route"]
)

"""Routes: what each kind of model call is allowed to use. Routes name a tier, never a model id;
the tier's model comes from settings (MEDLENS_LLM__MODEL_FAST / _BALANCED)."""

from dataclasses import dataclass
from enum import StrEnum

from medlens.core.config import LLMSettings


class Tier(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"


@dataclass(frozen=True, slots=True)
class Route:
    name: str
    tier: Tier
    prompt: str
    version: str
    max_tokens: int
    effort: str  # output_config.effort: low | medium | high

    @property
    def prompt_version(self) -> str:
        return f"{self.prompt}@{self.version}"


ROUTES: dict[str, Route] = {
    route.name: route
    for route in (
        # Reading fields off a pack photo: short structured output, cheapest tier.
        Route("pack_extraction", Tier.FAST, "pack_extraction", "v1", 2048, "low"),
        # Grounded answers with citations for professional users.
        Route("assistant_answer", Tier.BALANCED, "assistant_answer", "v1", 8000, "medium"),
    )
}


def model_for(tier: Tier, settings: LLMSettings) -> str:
    return settings.model_fast if tier is Tier.FAST else settings.model_balanced

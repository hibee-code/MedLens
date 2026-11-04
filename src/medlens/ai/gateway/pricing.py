from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from medlens.core.config import ModelPrice

_MILLION = Decimal(1_000_000)
_MICRO_DOLLAR = Decimal("0.000001")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input + other.input,
            self.output + other.output,
            self.cache_read + other.cache_read,
            self.cache_write + other.cache_write,
        )


def cost_usd(price: ModelPrice, usage: TokenUsage) -> Decimal:
    total = (
        Decimal(usage.input) * Decimal(str(price.input))
        + Decimal(usage.output) * Decimal(str(price.output))
        + Decimal(usage.cache_read) * Decimal(str(price.cache_read))
        + Decimal(usage.cache_write) * Decimal(str(price.cache_write))
    ) / _MILLION
    return total.quantize(_MICRO_DOLLAR, rounding=ROUND_HALF_UP)

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ImageAssessment:
    width: int
    height: int
    blur: float  # variance of the Laplacian: lower is blurrier
    brightness: float  # mean grey level, 0..255
    phash: int  # 64-bit perceptual hash, as a signed integer (Postgres BIGINT)

    def as_json(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "blur": round(self.blur, 2),
            "brightness": round(self.brightness, 2),
        }


class ImageAnalysisPort(Protocol):
    """Quality metrics and perceptual hash. Runs in the worker only (CLAUDE.md decision log)."""

    def assess(self, image: bytes) -> ImageAssessment: ...

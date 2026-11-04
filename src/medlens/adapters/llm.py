"""LLMPort on the LLM gateway: pack-field extraction with the vision model."""

from collections.abc import Sequence

from medlens.ai.gateway.gateway import ImagePart, LLMGateway, Subject, TextPart
from medlens.modules.verification.pipeline import PackExtraction


class GatewayPackReader:
    def __init__(self, gateway: LLMGateway) -> None:
        self._gateway = gateway

    def extract_pack_fields(
        self, image: bytes, *, fields: Sequence[str], subject_id: str | None = None
    ) -> str:
        extraction = self._gateway.run(
            "pack_extraction",
            [
                ImagePart(image),
                TextPart(
                    "Read the pack in this photo. The fields still missing are: "
                    + ", ".join(fields)
                    + ". Report all four fields."
                ),
            ],
            PackExtraction,
            subject=Subject("scan", subject_id) if subject_id else None,
        )
        return extraction.model_dump_json()

"""The worker's real readers and models, and how the runtime picks them. Embedding models are
replaced by stand-ins (no downloads); OpenCV and zxing-cpp run for real on the fixture packs."""

from pathlib import Path
from typing import Any

import pytest

from medlens.adapters import vision
from medlens.adapters.embeddings import (
    SentenceTransformerImageEmbedder,
    SentenceTransformerTextEmbedder,
)
from medlens.adapters.fakes import FakeBarcode, FakeImageAnalysis, FakeReranker
from medlens.modules.verification.extraction import from_gs1
from medlens.workers import runtime
from tests.support import build_settings

PACKS = Path(__file__).resolve().parents[1] / "fixtures" / "packs"


class StandIn:
    """A sentence-transformers model that returns fixed vectors and records its inputs."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.inputs: list[Any] = []

    def encode(self, inputs: Any, **kwargs: Any) -> list[list[float]]:
        self.inputs.extend(inputs)
        return [self.vector for _ in inputs]


# ------------------------------------------------------------------------- OpenCV and zxing
def test_the_quality_metrics_tell_sharp_from_blurred() -> None:
    analysis = vision.OpenCvImageAnalysis()

    sharp = analysis.assess((PACKS / "genuine.jpg").read_bytes())
    blurred = analysis.assess((PACKS / "blurry.jpg").read_bytes())

    assert (sharp.width, sharp.height) == (1200, 1600)
    assert sharp.blur > 300 > 60 > blurred.blur
    assert 100 < sharp.brightness < 235
    assert -(2**63) <= sharp.phash < 2**63  # fits a Postgres BIGINT
    assert sharp.phash != blurred.phash


def test_the_gs1_datamatrix_is_decoded_and_parsed() -> None:
    reads = vision.ZxingBarcode().decode((PACKS / "genuine.jpg").read_bytes())

    assert [read.format for read in reads] == ["DataMatrix"]
    fields = from_gs1(reads[0].text)
    assert fields.batch_number is not None
    assert fields.batch_number.value == "SEED-04-0289-A"
    assert fields.expiry_date is not None
    assert fields.expiry_date.value.isoformat() == "2028-01-14"
    assert vision.ZxingBarcode().decode((PACKS / "no_barcode.jpg").read_bytes()) == []


# ------------------------------------------------------------------------- Tesseract
def test_tesseract_output_keeps_printed_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "text": ["", "Emzor", "Syrup", "NAFDAC", "04-0289", "smudge"],
        "conf": ["-1", "96", "90", "88", "86", "-1"],
        "block_num": [0, 1, 1, 1, 1, 1],
        "par_num": [0, 1, 1, 1, 1, 1],
        "line_num": [0, 1, 1, 2, 2, 3],
    }
    calls: list[dict[str, Any]] = []

    def image_to_data(image: Any, **kwargs: Any) -> dict[str, list[Any]]:
        calls.append(kwargs)
        return data

    monkeypatch.setattr(vision.pytesseract, "image_to_data", image_to_data)

    result = vision.TesseractOcr(timeout_seconds=3).read((PACKS / "genuine.jpg").read_bytes())

    assert result.text == "Emzor Syrup\nNAFDAC 04-0289"
    assert result.confidence == 0.9
    assert calls[0]["timeout"] == 3
    assert calls[0]["config"] == "--psm 11"


def test_tesseract_with_no_words_has_no_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = {"text": [""], "conf": ["-1"], "block_num": [0], "par_num": [0], "line_num": [0]}
    monkeypatch.setattr(vision.pytesseract, "image_to_data", lambda image, **kwargs: empty)

    result = vision.TesseractOcr().read((PACKS / "genuine.jpg").read_bytes())

    assert (result.text, result.confidence) == ("", 0.0)


def test_tesseract_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> None:
        raise vision.pytesseract.TesseractNotFoundError

    monkeypatch.setattr(vision.pytesseract, "get_tesseract_version", lambda: "5.4.0")
    assert vision.tesseract_available()
    monkeypatch.setattr(vision.pytesseract, "get_tesseract_version", missing)
    assert not vision.tesseract_available()


# ------------------------------------------------------------------------- embeddings
def test_image_embeddings_check_their_dimension() -> None:
    model = StandIn([0.6, 0.8])
    embedder = SentenceTransformerImageEmbedder("clip-test", 2, load=lambda name: model)

    assert embedder.embed_image((PACKS / "genuine.jpg").read_bytes()) == [0.6, 0.8]
    assert embedder.model_version == "clip-test"
    wrong = SentenceTransformerImageEmbedder("clip-test", 3, load=lambda name: model)
    with pytest.raises(ValueError, match="2 dimensions"):
        wrong.embed_image((PACKS / "genuine.jpg").read_bytes())


def test_text_embeddings_use_the_e5_prefixes() -> None:
    model = StandIn([1.0, 0.0, 0.0])
    embedder = SentenceTransformerTextEmbedder("e5-test", 3, load=lambda name: model)

    assert embedder.embed_passages(["recall notice", "alert"]) == [[1.0, 0.0, 0.0]] * 2
    assert embedder.embed_query("is batch X recalled?") == [1.0, 0.0, 0.0]
    assert model.inputs == [
        "passage: recall notice",
        "passage: alert",
        "query: is batch X recalled?",
    ]
    with pytest.raises(ValueError, match="wrong dimension"):
        SentenceTransformerTextEmbedder("e5-test", 4, load=lambda name: model).embed_query("q")


# ------------------------------------------------------------------------- the runtime
def test_readers_follow_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = build_settings(verification={"readers": "fake"})
    real = build_settings(verification={"readers": "real"})
    monkeypatch.setattr(vision, "tesseract_available", lambda: False)

    analyzer, barcode, ocr = runtime.build_readers(fake)
    assert isinstance(analyzer, FakeImageAnalysis)
    assert isinstance(barcode, FakeBarcode)
    assert ocr is None
    analyzer, barcode, ocr = runtime.build_readers(real)
    assert isinstance(analyzer, vision.OpenCvImageAnalysis)
    assert isinstance(barcode, vision.ZxingBarcode)
    assert ocr is None  # no tesseract binary: barcodes and the vision LLM read the pack
    monkeypatch.setattr(vision, "tesseract_available", lambda: True)
    assert isinstance(runtime.build_readers(real)[2], vision.TesseractOcr)


def test_the_torch_models_are_an_optional_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """MEDLENS_EMBEDDINGS__BACKEND=torch (or missing ONNX exports): sentence-transformers."""
    disabled = runtime.build_models(build_settings(embeddings={"enabled": False}))
    assert disabled == runtime.Models()

    def unavailable(name: str) -> Any:
        raise OSError("no network to download the model")

    import medlens.adapters.embeddings as embeddings

    torch = {"enabled": True, "backend": "torch"}
    monkeypatch.setattr(
        embeddings.SentenceTransformerImageEmbedder.__init__, "__defaults__", (unavailable,)
    )
    assert runtime.build_models(build_settings(embeddings=torch)) == runtime.Models()

    settings = build_settings(embeddings=torch)
    image_model = StandIn([0.0] * settings.embeddings.image_dimension)
    text_model = StandIn([0.0] * settings.embeddings.text_dimension)
    monkeypatch.setattr(
        embeddings.SentenceTransformerImageEmbedder.__init__,
        "__defaults__",
        (lambda n: image_model,),
    )
    monkeypatch.setattr(
        embeddings.SentenceTransformerTextEmbedder.__init__, "__defaults__", (lambda n: text_model,)
    )
    monkeypatch.setattr(
        embeddings.SentenceTransformerReranker.__init__, "__defaults__", (lambda n: object(),)
    )
    models = runtime.build_models(settings)
    assert models.image is not None
    assert models.text is not None
    assert models.text.model_version == "intfloat/multilingual-e5-small"
    assert models.reranker is not None

    wrong = StandIn([0.0] * 3)  # a model whose vectors would not fit the columns
    monkeypatch.setattr(
        embeddings.SentenceTransformerImageEmbedder.__init__, "__defaults__", (lambda n: wrong,)
    )
    models = runtime.build_models(settings)
    assert models.image is None
    assert models.text is not None


def test_missing_onnx_exports_fall_back_to_torch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    fallback = runtime.Models(reranker=FakeReranker())
    monkeypatch.setattr(runtime, "_torch_models", lambda settings: fallback)

    settings = build_settings(embeddings={"enabled": True, "onnx_dir": str(tmp_path)})

    assert runtime.build_models(settings) == fallback


def test_the_runtime_wires_the_gateway_and_readers() -> None:
    built = runtime.build_runtime(build_settings(verification={"readers": "fake"}))
    try:
        assert built.gateway is not None
        assert built.embedder is None  # embeddings are off in test settings
        assert isinstance(built.analyzer, FakeImageAnalysis)
    finally:
        built.close()

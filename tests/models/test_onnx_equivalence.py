"""The ONNX exports the workers run against the original models (CLAUDE.md decision log: cosine
similarity above 0.99 on every sample).

Real models, so not part of the default run: needs the `ml` and `ml-export` groups, the exports
in models/onnx (`make onnx`) and MEDLENS_MODEL_TESTS=1. CI runs it in its own job, with the
model downloads cached. Samples are the seed's questions, document passages and pack photos.

    MEDLENS_MODEL_TESTS=1 uv run pytest -m models
"""

import csv
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        os.environ.get("MEDLENS_MODEL_TESTS") != "1", reason="real models: MEDLENS_MODEL_TESTS=1"
    ),
]

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = Path(os.environ.get("MEDLENS_EMBEDDINGS__ONNX_DIR", ROOT / "models" / "onnx"))
MIN_COSINE = 0.99
MAX_SCORE_DIFFERENCE = 0.01


def questions() -> list[str]:
    path = ROOT / "seed" / "eval_questions.csv"
    return [row["question"] for row in csv.DictReader(path.open(encoding="utf-8"))]


def passages() -> list[str]:
    """The seed documents cut by the production chunker: the passages search really embeds."""
    from medlens.modules.knowledge.text import chunk, extract_pages

    found = []
    for path in sorted((ROOT / "seed" / "documents").glob("*.md")):
        found += [piece.text for piece in chunk(extract_pages(path.read_bytes(), "text/plain"))]
    return found


def photos() -> list[bytes]:
    folders = (ROOT / "seed" / "test_packs", ROOT / "tests" / "fixtures" / "reference_images")
    return [path.read_bytes() for folder in folders for path in sorted(folder.glob("*.jpg"))]


def cosines(original: list[list[float]], exported: list[list[float]]) -> np.ndarray:
    a, b = np.asarray(original), np.asarray(exported)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def test_text_embeddings_match_the_original() -> None:
    from medlens.adapters.embeddings import SentenceTransformerTextEmbedder
    from medlens.adapters.onnx_models import OnnxTextEmbedder

    original = SentenceTransformerTextEmbedder("intfloat/multilingual-e5-small", 384)
    exported = OnnxTextEmbedder(EXPORTS / "text", 384)
    texts = passages()

    similarity = np.concatenate(
        [
            cosines(original.embed_passages(texts), exported.embed_passages(texts)),
            cosines(
                [original.embed_query(q) for q in questions()],
                [exported.embed_query(q) for q in questions()],
            ),
        ]
    )

    assert len(similarity) >= 15
    assert similarity.min() > MIN_COSINE, f"min cosine {similarity.min():.4f}"


def test_image_embeddings_match_the_original() -> None:
    from medlens.adapters.embeddings import SentenceTransformerImageEmbedder
    from medlens.adapters.onnx_models import OnnxImageEmbedder

    original = SentenceTransformerImageEmbedder("clip-ViT-B-32", 512)
    exported = OnnxImageEmbedder(EXPORTS / "image", 512)
    images = photos()

    similarity = cosines(
        [original.embed_image(image) for image in images],
        [exported.embed_image(image) for image in images],
    )

    assert len(similarity) >= 17
    assert similarity.min() > MIN_COSINE, f"min cosine {similarity.min():.4f}"


def test_reranker_scores_match_the_original() -> None:
    from medlens.adapters.embeddings import SentenceTransformerReranker
    from medlens.adapters.onnx_models import OnnxReranker

    original = SentenceTransformerReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    exported = OnnxReranker(EXPORTS / "reranker")
    texts = passages()
    reference, scores = [], []
    for question in questions():
        reference += original.rerank(question, texts)
        scores += exported.rerank(question, texts)

    similarity = float(cosines([reference], [scores])[0])
    difference = float(np.abs(np.asarray(reference) - np.asarray(scores)).max())

    assert similarity > MIN_COSINE, f"cosine {similarity:.4f}"
    assert difference <= MAX_SCORE_DIFFERENCE, f"max score difference {difference:.4f}"

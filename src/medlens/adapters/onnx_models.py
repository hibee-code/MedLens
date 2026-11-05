"""The worker's models on ONNX Runtime: text embeddings, the reranker and CLIP image embeddings.

Exported at image build time by scripts/export_onnx.py (one folder per model under
MEDLENS_EMBEDDINGS__ONNX_DIR: model.onnx, the tokenizer or image settings, meta.json), and
checked there against the torch originals, through these classes. At run time only ONNX
Runtime, the tokenizers library and Pillow are needed: no torch.

Exported models report their own model version (``<source>@onnx``, ``<source>@onnx-int8``):
preprocessing and quantization make their vectors differ slightly from the torch original's,
and vectors of two models are never mixed in one index (the backfill re-embeds).

ONNX Runtime's own usage telemetry is switched off (ORT_DISABLE_TELEMETRY, read when the
library loads; the worker image also sets it).
"""

import io
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

MAX_TOKENS = 512


def _session(path: Path, threads: int, arena: bool) -> ort.InferenceSession:
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = arena  # MEDLENS_EMBEDDINGS__ONNX_MEMORY_ARENA
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _meta(directory: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((directory / "meta.json").read_text())
    return meta


def model_version(meta: dict[str, Any]) -> str:
    quantized = meta.get("quantized")
    return f"{meta['source']}@onnx" + (f"-{quantized}" if quantized else "")


def available(directory: Path) -> bool:
    return (directory / "model.onnx").is_file() and (directory / "meta.json").is_file()


def _tokenizer(directory: Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.enable_truncation(MAX_TOKENS)
    tokenizer.enable_padding()
    return tokenizer


def _batch(encodings: Sequence[Any], names: Sequence[str]) -> dict[str, np.ndarray]:
    fields = {"input_ids": "ids", "attention_mask": "attention_mask", "token_type_ids": "type_ids"}
    return {
        name: np.asarray([getattr(item, fields[name]) for item in encodings], dtype=np.int64)
        for name in names
    }


class OnnxTextEmbedder:
    """TextEmbeddingPort: e5 embeddings, mean-pooled and normalised inside the graph."""

    def __init__(
        self, directory: Path, dimension: int, *, threads: int = 2, arena: bool = True
    ) -> None:
        meta = _meta(directory)
        self._session = _session(directory / "model.onnx", threads, arena)
        self._inputs = [item.name for item in self._session.get_inputs()]
        self._tokenizer = _tokenizer(directory)
        self._dimension = dimension
        self.model_version = model_version(meta)

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}"])[0]

    def probe_dimension(self) -> int:
        return len(self._encode(["query: probe"], check=False)[0])

    def _encode(self, texts: list[str], *, check: bool = True) -> list[list[float]]:
        encodings = self._tokenizer.encode_batch(texts)
        (vectors,) = self._session.run(None, _batch(encodings, self._inputs))
        rows = [[float(value) for value in row] for row in vectors]
        if check and any(len(row) != self._dimension for row in rows):
            raise ValueError(f"{self.model_version} returned the wrong dimension")
        return rows


class OnnxReranker:
    """RerankPort: a cross-encoder's relevance (sigmoid of its logit) for each passage."""

    def __init__(self, directory: Path, *, threads: int = 2, arena: bool = True) -> None:
        meta = _meta(directory)
        self._session = _session(directory / "model.onnx", threads, arena)
        self._inputs = [item.name for item in self._session.get_inputs()]
        self._tokenizer = _tokenizer(directory)
        self.model_version = model_version(meta)

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        encodings = self._tokenizer.encode_batch([(query, passage) for passage in passages])
        (logits,) = self._session.run(None, _batch(encodings, self._inputs))
        scores = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
        return [round(float(score), 6) for score in scores]


class OnnxImageEmbedder:
    """EmbeddingPort: CLIP's image tower, with CLIP's own preprocessing (shortest side to the
    model size with bicubic resampling, centre crop, scale to 0-1, normalise per channel)."""

    def __init__(
        self, directory: Path, dimension: int, *, threads: int = 2, arena: bool = True
    ) -> None:
        meta = _meta(directory)
        config = json.loads((directory / "preprocessor_config.json").read_text())
        self._session = _session(directory / "model.onnx", threads, arena)
        self._size = int(config["size"]["shortest_edge"])
        self._crop = (int(config["crop_size"]["width"]), int(config["crop_size"]["height"]))
        self._mean = np.asarray(config["image_mean"], dtype=np.float32)
        self._std = np.asarray(config["image_std"], dtype=np.float32)
        self._resample = int(config.get("resample", Image.Resampling.BICUBIC))
        self._dimension = dimension
        self.model_version = model_version(meta)

    def embed_image(self, image: bytes) -> list[float]:
        with Image.open(io.BytesIO(image)) as opened:
            vector = self._embed(opened.convert("RGB"))
        if len(vector) != self._dimension:
            raise ValueError(f"{self.model_version} returned {len(vector)} dimensions")
        return vector

    def probe_dimension(self) -> int:
        return len(self._embed(Image.new("RGB", (32, 32))))

    def pixels(self, picture: Image.Image) -> np.ndarray:
        width, height = picture.size
        scale = self._size / min(width, height)
        resized = picture.resize(
            (max(self._size, round(width * scale)), max(self._size, round(height * scale))),
            resample=Image.Resampling(self._resample),
        )
        crop_width, crop_height = self._crop
        left = (resized.width - crop_width) // 2
        top = (resized.height - crop_height) // 2
        cropped = resized.crop((left, top, left + crop_width, top + crop_height))
        array = np.asarray(cropped, dtype=np.float32) / 255.0
        array = (array - self._mean) / self._std
        return array.transpose(2, 0, 1)[np.newaxis]

    def _embed(self, picture: Image.Image) -> list[float]:
        (vectors,) = self._session.run(None, {"pixel_values": self.pixels(picture)})
        return [float(value) for value in vectors[0]]

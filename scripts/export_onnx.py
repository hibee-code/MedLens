"""Export the worker's models to ONNX Runtime (run while building the worker image).

    python scripts/export_onnx.py --out /opt/models/onnx

* text: the text embedding model (multilingual-e5-small), mean pooling and normalisation inside
  the graph, int8 weights;
* reranker: the cross-encoder (ms-marco-MiniLM-L-6-v2, 23M parameters), one relevance logit
  per pair, fp32 (small enough that int8 is not worth its drift);
* image: CLIP's image tower (clip-ViT-B-32) with its projection and normalisation, fp32 (image
  similarity is what the visual check compares against a threshold, so it stays exact).

Each model is checked against the torch original through the classes the worker uses at run
time (medlens.adapters.onnx_models: tokenizer, preprocessing and session together); a model
that drifts further than its tolerance fails the export, and so the image build. The output
directory has one folder per model: model.onnx, the tokenizer or image processor, meta.json.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image


def _runtime_adapters() -> Any:
    """medlens.adapters.onnx_models, also where medlens is not installed (the image's export
    stage copies just that file): it imports nothing from medlens."""
    try:
        from medlens.adapters import onnx_models
    except ImportError:
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "src/medlens/adapters/onnx_models.py"
        spec = importlib.util.spec_from_file_location("onnx_models", path)
        if spec is None or spec.loader is None:
            raise
        onnx_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(onnx_models)
    return onnx_models


_adapters = _runtime_adapters()
OnnxImageEmbedder = _adapters.OnnxImageEmbedder
OnnxReranker = _adapters.OnnxReranker
OnnxTextEmbedder = _adapters.OnnxTextEmbedder

OPSET = 17
TEXT_TOLERANCE = 0.98  # minimum cosine between int8 and torch embeddings
IMAGE_TOLERANCE = 0.99  # the runtime's numpy preprocessing differs slightly from transformers'
RERANK_TOLERANCE = 0.01  # maximum difference of sigmoid scores

SAMPLE_TEXTS = [
    "query: Has NAFDAC issued an alert for counterfeit Coartem?",
    "passage: NAFDAC alerts the public to counterfeit Augmentin 1 G tablets in Lagos.",
    "passage: Batch No: B2309-14. Expiry date: 12/2027. Store below 30 degrees.",
    "query: Are other batches of ADALAT LA affected by the recall?",
]
SAMPLE_PAIRS = [
    ("Has an alert been issued for Augmentin?", "NAFDAC alerts the public to fake Augmentin."),
    ("Has an alert been issued for Augmentin?", "Store the syrup below 30 degrees Celsius."),
    ("Which batch was recalled?", "Batch No: SEED-A4-2374-B is recalled for dissolution."),
]


def _export(module: torch.nn.Module, args: tuple[Any, ...], path: Path, **names: Any) -> None:
    with torch.no_grad():
        torch.onnx.export(module, args, str(path), opset_version=OPSET, dynamo=False, **names)


def _quantize(source: Path, target: Path) -> None:
    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)


def _write_meta(directory: Path, **meta: Any) -> None:
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


# --------------------------------------------------------------------------- text
class _MeanPooled(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)


def export_text(name: str, out: Path, scratch: Path) -> dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    wrapped = _MeanPooled(AutoModel.from_pretrained(name).eval()).eval()
    batch = tokenizer(SAMPLE_TEXTS, padding=True, truncation=True, return_tensors="pt")
    fp32 = scratch / "text.onnx"
    _export(
        wrapped,
        (batch["input_ids"], batch["attention_mask"]),
        fp32,
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "tokens"},
            "attention_mask": {0: "batch", 1: "tokens"},
            "embedding": {0: "batch"},
        },
    )
    out.mkdir(parents=True, exist_ok=True)
    _quantize(fp32, out / "model.onnx")
    with torch.no_grad():
        reference = wrapped(batch["input_ids"], batch["attention_mask"]).numpy()
    tokenizer.save_pretrained(out)
    dimension = int(reference.shape[1])
    meta: dict[str, Any] = {"source": name, "dimension": dimension, "quantized": "int8"}
    _write_meta(out, **meta)
    runtime = OnnxTextEmbedder(out, dimension)
    exported = np.asarray(runtime._encode(SAMPLE_TEXTS))
    cosine = float((reference * exported).sum(-1).min())
    if cosine < TEXT_TOLERANCE:
        raise SystemExit(f"{name}: int8 export drifts (min cosine {cosine:.4f})")
    meta["min_cosine"] = round(cosine, 5)
    _write_meta(out, **meta)
    return meta


# --------------------------------------------------------------------------- reranker
class _Logit(torch.nn.Module):
    """BERT-style cross-encoders also take segment ids (query = 0, passage = 1)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        ).logits[:, 0]


def export_reranker(name: str, out: Path, scratch: Path) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    del scratch  # fp32: written straight to the output
    tokenizer = AutoTokenizer.from_pretrained(name)
    wrapped = _Logit(AutoModelForSequenceClassification.from_pretrained(name).eval()).eval()
    queries, passages = zip(*SAMPLE_PAIRS, strict=True)
    batch = tokenizer(
        list(queries),
        list(passages),
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=True,
    )
    inputs = ("input_ids", "attention_mask", "token_type_ids")
    out.mkdir(parents=True, exist_ok=True)
    _export(
        wrapped,
        tuple(batch[name] for name in inputs),
        out / "model.onnx",
        input_names=list(inputs),
        output_names=["logit"],
        dynamic_axes={
            **{name: {0: "batch", 1: "tokens"} for name in inputs},
            "logit": {0: "batch"},
        },
    )
    with torch.no_grad():
        reference = torch.sigmoid(wrapped(*(batch[name] for name in inputs))).numpy()
    tokenizer.save_pretrained(out)
    meta: dict[str, Any] = {"source": name, "quantized": None}
    _write_meta(out, **meta)
    runtime = OnnxReranker(out)
    exported = np.asarray([runtime.rerank(query, [passage])[0] for query, passage in SAMPLE_PAIRS])
    drift = float(np.abs(reference - exported).max())
    if drift > RERANK_TOLERANCE:
        raise SystemExit(f"{name}: export drifts (max score difference {drift:.5f})")
    meta["max_score_drift"] = round(drift, 6)
    _write_meta(out, **meta)
    return meta


# --------------------------------------------------------------------------- image
class _ImageTower(torch.nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.model.get_image_features(pixel_values=pixel_values)
        if not isinstance(features, torch.Tensor):  # newer transformers return an output object
            features = features.pooler_output
        return torch.nn.functional.normalize(features, dim=-1)


def _sample_images() -> list[Image.Image]:
    rng = np.random.default_rng(7)
    noise = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    return [noise, Image.new("RGB", (640, 480), (200, 120, 40)), noise.rotate(90, expand=True)]


def export_image(name: str, out: Path, scratch: Path) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    del scratch  # fp32: written straight to the output

    model = SentenceTransformer(name, device="cpu")
    clip = model[0]
    processor = clip.processor
    images = _sample_images()
    pixels = processor(images=images, return_tensors="pt")["pixel_values"]
    tower = _ImageTower(clip.model.eval()).eval()
    out.mkdir(parents=True, exist_ok=True)
    _export(
        tower,
        (pixels,),
        out / "model.onnx",
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
    )
    reference = np.asarray(model.encode(images, normalize_embeddings=True))
    processor.image_processor.save_pretrained(out)
    meta: dict[str, Any] = {"source": name, "dimension": int(reference.shape[1]), "quantized": None}
    _write_meta(out, **meta)
    runtime = OnnxImageEmbedder(out, int(reference.shape[1]))
    exported = np.asarray([runtime._embed(image) for image in images])
    cosine = float((reference * exported).sum(-1).min())
    if cosine < IMAGE_TOLERANCE:
        raise SystemExit(f"{name}: export drifts (min cosine {cosine:.5f})")
    meta["min_cosine"] = round(cosine, 6)
    _write_meta(out, **meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-model", default="clip-ViT-B-32")
    parser.add_argument("--text-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    args = parser.parse_args(argv)
    torch.set_grad_enabled(False)
    with tempfile.TemporaryDirectory() as scratch:
        report = {
            "text": export_text(args.text_model, args.out / "text", Path(scratch)),
            "reranker": export_reranker(args.reranker, args.out / "reranker", Path(scratch)),
            "image": export_image(args.image_model, args.out / "image", Path(scratch)),
        }
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

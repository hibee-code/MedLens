"""Write the tiny ONNX models in tests/fixtures/onnx (``python scripts/make_onnx_fixtures.py``).

They have the inputs and outputs of the real exports (scripts/export_onnx.py) but a few
arithmetic nodes instead of a network, so the runtime adapters can be tested without
downloading models:

* text: every embedding is [0.5, 0.5, 0.5, 0.5] (4 dimensions);
* reranker: logit = (passage tokens) - 4, so longer passages score higher;
* image: the mean of each colour channel (3 dimensions, not normalised).

Needs the onnx and tokenizers packages (`make install-ml`).
"""

import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from tokenizers import Tokenizer, models, pre_tokenizers, processors

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "onnx"
OPSET = [helper.make_opsetid("", 17)]
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "query", "passage", ":", "alert", "batch", "recall"]


def _tokenizer() -> Tokenizer:
    vocab = {word: index for index, word in enumerate(VOCAB)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))  # noqa: S106 - a token
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return tokenizer


def _ints(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", "tokens"])


def _save_tokenizer(directory: Path) -> None:
    # A final newline, as the pre-commit end-of-file hook wants.
    (directory / "tokenizer.json").write_text(_tokenizer().to_str(pretty=True) + "\n")


def _save(graph: onnx.GraphProto, directory: Path, meta: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model = helper.make_model(graph, opset_imports=OPSET)
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    (directory / "meta.json").write_text(json.dumps(meta) + "\n")


def text() -> None:
    nodes = [
        helper.make_node("Cast", ["attention_mask"], ["mask"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["mask"], ["sum"], keepdims=1),  # (1, 1) over all
        helper.make_node("Div", ["mask", "sum"], ["share"]),
        helper.make_node("ReduceSum", ["share", "axis1"], ["row"], keepdims=1),  # (b, 1)
        helper.make_node("Mul", ["row", "zero"], ["zeros"]),
        helper.make_node("Add", ["zeros", "half"], ["embedding"]),  # (b, 4) of 0.5
    ]
    graph = helper.make_graph(
        nodes,
        "text",
        [_ints("input_ids"), _ints("attention_mask")],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 4])],
        initializer=[
            helper.make_tensor("axis1", TensorProto.INT64, [1], [1]),
            helper.make_tensor("zero", TensorProto.FLOAT, [1, 4], [0.0] * 4),
            helper.make_tensor("half", TensorProto.FLOAT, [1, 4], [0.5] * 4),
        ],
    )
    directory = OUT / "text"
    _save(graph, directory, {"source": "fixture/text", "dimension": 4, "quantized": "int8"})
    _save_tokenizer(directory)


def reranker() -> None:
    nodes = [
        helper.make_node("Cast", ["token_type_ids"], ["types"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["types", "axis1"], ["count"], keepdims=0),  # (b,)
        helper.make_node("Sub", ["count", "four"], ["logit"]),
    ]
    graph = helper.make_graph(
        nodes,
        "reranker",
        [_ints("input_ids"), _ints("attention_mask"), _ints("token_type_ids")],
        [helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["batch"])],
        initializer=[
            helper.make_tensor("axis1", TensorProto.INT64, [1], [1]),
            helper.make_tensor("four", TensorProto.FLOAT, [], [4.0]),
        ],
    )
    directory = OUT / "reranker"
    _save(graph, directory, {"source": "fixture/reranker", "quantized": None})
    _save_tokenizer(directory)


def image() -> None:
    nodes = [
        helper.make_node("ReduceMean", ["pixel_values"], ["embedding"], axes=[2, 3], keepdims=0)
    ]
    graph = helper.make_graph(
        nodes,
        "image",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["batch", 3, 224, 224])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 3])],
    )
    directory = OUT / "image"
    _save(graph, directory, {"source": "fixture/image", "dimension": 3, "quantized": None})
    (directory / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "size": {"shortest_edge": 224},
                "crop_size": {"height": 224, "width": 224},
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "resample": 3,
            }
        )
        + "\n"
    )


def main() -> int:
    text()
    reranker()
    image()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

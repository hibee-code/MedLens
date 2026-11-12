"""The evaluation scripts' own logic: OCR benchmark label checks and engine comparison, and the
retrieval metrics and evidence-bar calibration (scripts/ocr_benchmark.py,
scripts/retrieval_eval.py). The runs themselves need real photos, models and a database."""

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ocr = script("ocr_benchmark")
retrieval = script("retrieval_eval")


def labels(tmp_path: Path, text: str, photos: tuple[str, ...] = ("a.jpg", "b.jpg")) -> Path:
    for photo in photos:
        (tmp_path / photo).write_bytes(b"jpeg")
    path = tmp_path / "labels.csv"
    path.write_text(text)
    return path


def test_good_labels_pass(tmp_path: Path) -> None:
    path = labels(
        tmp_path,
        "file,nafdac_number,brand,batch_number,expiry\n"
        "a.jpg,A4-1234,Coartem,F2034,11/2027\n"
        "b.jpg,,Panadol,,\n",  # empty: not printed or not legible
    )

    assert ocr.label_problems(path, tmp_path) == []
    assert [label.nafdac_number for label in ocr.load_labels(path)] == ["A4-1234", ""]


def test_bad_labels_are_listed(tmp_path: Path) -> None:
    path = labels(
        tmp_path,
        "file,nafdac_number,brand,batch_number,expiry\n"
        "a.jpg,not-a-number,Coartem,F2034,2027-11-01\n"
        "a.jpg,A4-1234,Coartem,F2034,11/2027\n"
        "missing.jpg,A4-1234,Coartem,F2034,11/2027\n",
    )

    problems = ocr.label_problems(path, tmp_path)

    assert any("nafdac_number 'not-a-number'" in problem for problem in problems)
    assert any("expiry '2027-11-01'" in problem for problem in problems)
    assert any("a.jpg is labelled twice" in problem for problem in problems)
    assert any("no photo missing.jpg" in problem for problem in problems)
    no_columns = labels(tmp_path, "file,brand\na.jpg,X\n")
    assert ocr.label_problems(no_columns, tmp_path) == [
        "missing columns: nafdac_number, batch_number, expiry"
    ]


def test_the_best_engine_per_field_is_reported() -> None:
    engines = [
        {"engine": "tesseract", "nafdac_number": 0.9, "batch_number": 0.7, "expiry_month": 0.8,
         "brand_found": 0.1},
        {"engine": "paddle", "skipped": "not installed"},
        {"engine": "llm", "nafdac_number": 0.9, "batch_number": 0.8, "expiry_month": 0.9,
         "brand_found": 0.9},
    ]  # fmt: skip

    assert ocr.best_per_field(engines) == {
        "nafdac_number": "tesseract",  # a tie goes to the first (cheapest) engine
        "batch_number": "llm",
        "expiry_month": "llm",
        "brand_found": "llm",
    }
    assert ocr.best_per_field([])["nafdac_number"] is None


def test_retrieval_metrics() -> None:
    ranks = [1, 2, None, 4]  # first relevant passage per question; None: not retrieved

    assert retrieval.ranking_metrics(ranks, top_k=6) == {
        "recall@1": 25.0,
        "recall@3": 50.0,
        "recall@6": 75.0,
        "mrr": round(100 * (1 + 0.5 + 0.25) / 4, 1),
    }
    assert retrieval.first_relevant(["a", "b", "b"], "b") == 2
    assert retrieval.first_relevant(["a"], "b") is None


def test_the_evidence_bar_range_and_its_effect() -> None:
    separable = retrieval.bar_range([0.96, 0.99], [0.01, 0.02])
    overlapping = retrieval.bar_range([0.5, 0.9], [0.6])

    assert separable["separable"] is True
    assert separable["bar_between"] == [0.02, 0.96]
    assert overlapping["separable"] is False
    assert overlapping["bar_between"] is None
    assert retrieval.at_bar([0.96, 0.05], [0.02, 0.2], 0.1) == {
        "answerable_kept": "1/2",
        "unanswerable_dropped": "1/2",
    }


def test_questions_name_documents_by_file_or_title(tmp_path: Path) -> None:
    path = tmp_path / "questions.csv"
    path.write_text(
        "question,expected_document,answerable\n"
        "Is batch X recalled?,sample-recall-adalat.md,yes\n"
        "What does circular 12 say?,NAFDAC Circular 12/2026,yes\n"
        "What is the price?,,no\n"
    )

    questions = retrieval.load_questions(path)

    assert questions[0].expected_title is not None
    assert questions[0].expected_title.startswith("SAMPLE Recall Notice")
    assert questions[1].expected_title == "NAFDAC Circular 12/2026"
    assert (questions[2].expected_title, questions[2].answerable) == (None, False)


load = script("load_test")


def test_load_test_statistics_merge_across_client_processes() -> None:
    first, second = load.Stats(), load.Stats()
    for seconds in (0.010, 0.020, 0.030):
        first.record("GET /products/{nafdac}", "200", seconds)
    second.record("GET /products/{nafdac}", "503", 0.5)
    first.end_to_end, first.fast_path = [2.0, 4.0], 1
    second.queue_peak, second.dropped = 12.0, 3

    merged, elapsed = load.merge([(first, 10.0), (second, 12.0)])
    report = load.summary(merged, elapsed)

    reads = report["GET /products/{nafdac}"]
    assert reads["count"] == 4
    assert reads["statuses"] == {"200": 3, "503": 1}
    assert reads["success_rate"] == 0.75
    assert reads["latency_ms"]["max"] == 500
    assert report["scan_to_verdict_ms"]["fast_path_share"] == 0.5
    assert (report["scan_queue_peak"], report["dropped_by_client"]) == (12.0, 3)
    assert load.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 3.0


def test_load_test_photos_are_all_different() -> None:
    images = load.photos(5)

    assert len(set(images)) == 5  # no scan is answered as a duplicate

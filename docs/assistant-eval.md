# Assistant evaluation with real questions

The assistant answers only from passages its reranker scores at least
`MEDLENS_KNOWLEDGE__MIN_RERANK_SCORE` (0.1), and otherwise says it has not enough evidence
(CLAUDE.md decision log, "Reranker"). That bar was set on `seed/eval_questions.csv`: ten
synthetic questions over three sample documents, where every retriever already ranks the right
document first. It has to be checked on real questions over real documents before anyone relies
on the assistant's silence.

## Collecting questions

* Ask pharmacists, PPMV operators and regulator staff what they would ask: at least **50
  answerable** questions (the answer is in a document the knowledge base holds) and **20
  unanswerable** ones (dosing, prices, products no notice mentions: the assistant must stay
  silent). Keep their wording, typos and Pidgin included.
* Load the documents the answers come from (NAFDAC alerts, recall notices, SOPs) through
  `POST /api/v1/knowledge/documents`, so search can find them.
* No names, phone numbers, patients or premises in the questions. The assistant scrubs contact
  details, but the file is shared and kept.

## The file

`docs/templates/assistant-questions.csv`: `question`, `expected_document` (the title of the
document that answers it, or a file name from `seed/documents/index.csv`; empty when
unanswerable), `answerable` (`yes`/`no`). Keep it outside Git if the questions came from people
who did not agree to publication.

## Running

Retrieval and the evidence bar (needs the knowledge base and the models, `make install-ml`
and `make onnx`):

```bash
make retrieval-eval QUESTIONS=/path/to/questions.csv
```

Per reranker (and the fused order without one): recall@1, recall@3, recall@6 and MRR, and the
evidence bar's calibration: the lowest best score of an answerable question, the highest of an
unanswerable one, the range of bars that separates them, and what the current bar keeps and
drops. Then the answers themselves (with `ANTHROPIC_API_KEY`, the real model; it spends
money):

```bash
uv run python scripts/eval.py --skip-scans --questions /path/to/questions.csv
```

## Deciding

* The bar: pick a value inside the separating range, nearer the unanswerable side if missed
  answers cost less than wrong ones. If the ranges overlap, no bar is safe: record which
  questions overlap and why (missing documents, wording, the reranker's weakness on
  procedural questions noted in the decision log) before changing anything.
* The reranker: the decision log keeps ms-marco-MiniLM-L-6-v2 while its recall and MRR are
  within 3 points of bge-reranker-base on the eval set; re-run the comparison on these
  questions (`--rerankers cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base`).
* Record the numbers and the decision in CLAUDE.md §9.

# OCR benchmark

The decision log keeps Tesseract as the primary OCR engine and PaddleOCR blocked until a
benchmark on **100+ real, labelled photos of Nigerian packs** compares Tesseract, PaddleOCR and
the vision-LLM fallback. `scripts/ocr_benchmark.py` runs that benchmark; the photos have to come
from the field.

## Collecting photos

* At least 100 packs, photographed with ordinary phones in ordinary light, the way buyers will:
  some at an angle, some with glare, some blurred. Include blister strips, bottles, small cartons
  and packs printed light on dark.
* Cover the products people actually buy (antimalarials, antibiotics, analgesics), several
  manufacturers, and some known counterfeits if a regulator can share them.
* No people, faces or receipts in the frame. Strip location data (`exiftool -all= *.jpg`).
* Keep the photos and labels outside Git (they may show premises or stock); record where they
  are and who may use them next to the results.

## Labelling

One CSV row per photo, the fields as printed on the pack:

```csv
file,nafdac_number,brand,batch_number,expiry
IMG_0001.jpg,A4-1234,Coartem 20/120,F2034,11/2027
```

Leave a field empty when it is not on the pack or not legible to a person either.

## Running

Tesseract runs in the worker image (the binary is installed there); PaddleOCR needs the
`ocr-benchmark` dependency group; the vision LLM needs `MEDLENS_LLM__ENABLED=true` and an API
key, and spends money (about $0.0005 per photo on the fast tier).

```bash
make ocr-benchmark PHOTOS=/path/to/photos               # Tesseract, in the worker image
make ocr-benchmark PHOTOS=/path/to/photos ENGINES=tesseract,llm   # + the vision LLM (costs)
uv run --group ocr-benchmark python scripts/ocr_benchmark.py \
  --photos /path/to/photos --labels /path/to/photos/labels.csv --engines paddle
```

The labels file is `labels.csv` in the photo folder (template:
`docs/templates/ocr-labels.csv`). It is checked first: a missing photo, a duplicate row or a
value that does not parse (NAFDAC number, batch, `MM/YYYY` expiry) stops the run with the list
of problems, since a wrong label would score a right reading as a miss.

It prints, per engine, the exact-match rate of the NAFDAC number, batch number and expiry
month, how often the brand name was found, latency (p50, p95) and cost; how many photos have
each field labelled; `enough_for_the_decision` (100 photos or more, `--min-photos`); and the
best engine per field (ties go to the engine listed first, so list the cheapest first). Record the result and
the decision in CLAUDE.md §9 ("Tesseract-first OCR", revisit when the benchmark is in).

Without real photos, `python scripts/ocr_benchmark.py` runs on the synthetic packs in
`seed/test_packs`: a smoke test of the harness, not evidence for the decision.

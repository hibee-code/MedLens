# Changelog

Changes that API clients and operators notice. Newest first. Reasons for each choice are in the
decision log (CLAUDE.md §9).

## Unreleased (2026-11-16)

### Added

- `MEDLENS_VERIFICATION__OCR_MODE`: `sparse_block` (default) or `sparse_inverted` (the
  previous OCR passes).
- `make eval-image` (the scan eval in the worker image; CI uses it), `make demo-reset`
  (local only, `CONFIRM=1`), and the seed's `--staff-password-file`.
- The demo shows the unregistered number on a registered brand, a cloned barcode and the same
  holder's barcode, reuses saved staff sessions between runs, and no longer prints the staff
  password.
- Dashboard panel: signals held back by a guard, per signal.

- Reason code `NUMBER_DOES_NOT_MATCH_BRAND` (regulators only; consumers see
  `DETAILS_DO_NOT_MATCH_REGISTER`): the printed NAFDAC number is in no registration, but the
  brand matches a registered product closely, on the brand's own words (never on the generic
  name alone). Such scans are now `SUSPICIOUS` (they were
  `UNREGISTERED`) and are linked to the product, so they appear in hotspots and in its
  manufacturer's report.
- Metric `medlens_scan_signals_held_back_total{signal, reason}` (workers): mismatches a
  confidence guard kept out of the score.

### Changed

- OCR reads brand names printed light on dark (block mode instead of the inverted image).
- A NAFDAC number read by OCR takes the lowest confidence of its own words.
- `make greenbook-import-snapshot` reports first and applies only with `CONFIRM=1`. Between a
  snapshot row and a hand-typed row of the same product, the most recently retrieved wins
  (same day: the snapshot). Both import reports list every conflicting field.

## 2026-11-14

### Changed

- **Scan reason codes: `NAME_MISMATCH` is folded into `DETAILS_DO_NOT_MATCH_REGISTER`.** A
  product name that does not match the registration is now reported to clients with the same
  coarse code as the other register mismatches. `NAME_MISMATCH` is no longer emitted. Scans
  stored before this change are shown with `DETAILS_DO_NOT_MATCH_REGISTER` as well (no
  duplicates). Regulators still see which field disagreed, in `analysis.register_check` and
  the signals. No external clients exist yet, so no transition period.
- **Register-check and cloned-barcode scoring are on** (scorer `weighted-rules-v3+barcode`).
  A pack whose ingredients, strength, dosage form, holder or manufacture date contradict its
  registration, or whose barcode belongs to another holder's product, can now be
  `SUSPICIOUS`. A mismatch counts only when it was read with confidence of at least 0.85;
  otherwise it is recorded for regulators with `counted: false`. Switches:
  `MEDLENS_VERIFICATION__REGISTER_CHECK_SCORING`, `MEDLENS_VERIFICATION__BARCODE_CHECK_SCORING`.
- **Assistant without AI (switched off or over budget):** passages below the evidence bar are
  dropped from `results`. When none is left, the answer is `insufficient_evidence` with reason
  `no_relevant_documents`.
- `docker compose up` no longer starts Flower. Use `--profile flower` (or `observability`).
- **Make targets renamed:** `make import-greenbook` is now `make greenbook-add-rows`, and
  `make greenbook-import` is now `make greenbook-import-snapshot`.

### Added

- `POST /api/v1/assistant/ask` and `/ask/stream`: `include_low_relevance` (platform admins
  only, 403 for anyone else) returns the dropped passages in `low_relevance_results`, for
  debugging.
- `GET /api/v1/products`: `gtins`, the product's barcodes.
- The hand-typed register file has a `gtin_source` column (`register` or `manufacturer`),
  required with any GTIN. Regulators see the source in the barcode check.
- Every product records where its register entry came from (`snapshot` or `manual`). A snapshot
  import updates snapshot rows only and never overwrites or removes hand-typed rows.
- Hotspot clusters list `top_unregistered_numbers`.
- `.github/dependabot.yml`: weekly updates for Docker base images, compose images and GitHub
  Actions.

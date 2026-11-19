# Changelog

Changes that API clients and operators notice. Newest first. Reasons for each choice are in the
decision log (CLAUDE.md §9).

## Unreleased (2026-11-18)

### Changed (accounts, 2026-11-19)

- **Consumers can sign in by e-mail as well as by SMS:** `POST /api/v1/auth/email-otp/request`
  and `/auth/email-otp/verify` (the account is created on first sign-in).
- **Staff sign-up is invitation-only:** `POST /auth/email/register` and `/auth/email/verify` are
  removed. Organizations register with the public form `POST /api/v1/organizations/register`
  (licence number required, then `/organizations/register/verify`); `POST /organizations` is
  for platform admins only.
- **MFA for every staff account** (any organization membership or administrative role).
- **Codes sign in consumers only:** staff accounts get no SMS or e-mail code, and an earlier
  code is refused (403).
- **Invitations go to e-mail addresses only** (`phone` is no longer accepted) and carry a link
  to `<MEDLENS_APP__FRONTEND_URL>/invitations/<token>` when that is set, else the code.
- **One-link invitations:** `GET /api/v1/invitations/{token}` previews an invitation;
  `POST /api/v1/invitations/{token}/accept-new` creates the account (name and password) and
  answers with the MFA enrolment challenge.
- **`POST /api/v1/invitations/accept` answers `{organization_id, role, signed_out}`** instead
  of the membership. An account without a password must send `password`; an account that becomes
  staff is signed out everywhere (`signed_out: true`).
- **Platform-admin invitations:** `POST/GET /api/v1/admin/invitations`,
  `DELETE /api/v1/admin/invitations/{id}`; the first platform admin with
  `make bootstrap-admin EMAIL=...` (migration 0019: invitations without an organization).
- **MFA recovery codes:** `POST /api/v1/auth/mfa/confirm` also returns ten one-time
  `recovery_codes` (shown once); `POST /auth/mfa/verify` accepts `recovery_code` instead of
  `code`; `GET /auth/mfa/recovery-codes` counts the unused ones and `POST` replaces them (with a
  current authenticator code). Migration 0020.
- **Account administration for platform admins (MFA sessions only):** `GET /api/v1/admin/users`
  (search by address, name or phone; filters by status, role and organization),
  `GET /admin/users/{id}`, and `POST /admin/users/{id}/suspend`, `/reactivate`, `/sign-out`,
  `/mfa/reset`, each with an audited reason.
- **The seed creates no users** (`--staff-password`, `--staff-password-file` and
  `--reset-staff-logins` removed): `make bootstrap-admin`, then invitations. `make demo` does
  the onboarding itself; `make demo-reset ACCOUNTS=1 CONFIRM=1` erases the demo's
  `@medlens.test` accounts (and those of earlier seeds).
- **E-mail is never sent to reserved test domains** (`.test`, `.example`, `.invalid`,
  `.localhost`): the SMTP adapter refuses them without connecting.

### Added (e-mail channel)

- Staff sign-up by e-mail: `POST /api/v1/auth/email/register` and `/auth/email/verify` (a
  6-digit code e-mailed to the address). Password sign-in now needs a verified address
  (existing e-mail accounts were marked verified by migration 0018) and answers 403 otherwise.
- Password reset: `POST /api/v1/auth/password/forgot` and `/auth/password/reset` (ends every
  session).
- Adding an address to an account: `POST /api/v1/users/me/email` and `/users/me/email/verify`;
  `GET /users/me` shows `email_verified`.
- Invitations to an e-mail address are e-mailed with their token, and only a verified address
  can accept them.
- Settings `MEDLENS_EMAIL__*` (console or SMTP) and `MEDLENS_AUTH__EMAIL_CODE_TTL_SECONDS`.


### Changed

- **LLM provider: Google Gemini replaces Anthropic Claude** (free tier by default).
  `MEDLENS_LLM__ANTHROPIC_API_KEY` and `MEDLENS_LLM__REFUSAL_FALLBACK` are removed: remove them
  from your `.env` (settings refuse unknown variables). New: `MEDLENS_LLM__GEMINI_API_KEY` and
  `MEDLENS_LLM__FALLBACK_MODELS` (per tier, the backup models tried in order). Defaults:
  `gemini-3.5-flash-lite` (fast), `gemini-3.8-flash` (balanced), fallbacks `gemini-3.8-flash`
  and `gemini-4-hydrogen`, all priced at 0. Every primary and fallback model needs a price
  (checked at startup).
- Rate limits, safety blocks, empty answers and repeated timeouts move a call to the next model;
  when every model fails, the feature's non-AI fallback answers, as before. The `llm_calls`
  ledger records the model that answered.

- With `MEDLENS_LLM__ENABLED=true`, the API and workers refuse to start without a Gemini key.
- An overloaded model (HTTP 503) is skipped at once for the tier's next model; requests leave
  thinking headroom on top of each route's answer limit.

### Added

- Metric `medlens_llm_fallbacks_total{route, reason}` (reasons include `overloaded`).

## 2026-11-16 (unreleased)

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

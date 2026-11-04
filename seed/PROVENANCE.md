# Provenance of the registered-products data

| | Committed sample | Full snapshot |
|---|---|---|
| File | `seed/products.csv` | `greenbook-2026-11-03.csv.gz` (not in Git) |
| Contents | 50 products | 8,680 products (all categories) |
| Integrity | reviewed in Git | SHA-256 in `seed/greenbook-snapshot.json`, checked on every import |
| Used by | `make seed`, tests | `make greenbook-import` (every environment) |

## Source

* **Publisher:** National Agency for Food and Drug Administration and Control (NAFDAC).
* **Register:** NAFDAC Greenbook, the public register of registered products,
  <https://greenbook.nafdac.gov.ng/>.
* **Access:** the JSON endpoint the site's own product table calls
  (`GET /?start=…&length=…` with `X-Requested-With: XMLHttpRequest`), at one request per second
  with an identifying User-Agent. No login, key or rate-limit bypass.
* **Retrieved:** 2026-11-03. 8,974 register rows: 19 had a missing or unrecognised NAFDAC number
  (5 of them written like `4/1/9086`, a form we do not map) and 275 repeated a number already
  present; 8,680 products remain. Categories: drugs, medical devices, vaccines and biologics,
  herbals and nutraceuticals, veterinary.

## Terms

On 2026-11-03 the site published no terms of use, licence or data-reuse policy. Its footer reads
"© NAFDAC 2025. All rights reserved." We have no permission to redistribute the register, so:

* Git holds only the 50-product sample, as development and test reference data.
* The full snapshot goes to **private** object storage, readable by MedLens deployments only;
  never a public release or bucket.
* The data is attributed to NAFDAC and is never presented as an official NAFDAC service.
* Revisit when NAFDAC publishes terms or an API, or when a data-sharing agreement is in place
  (see the decision log in `CLAUDE.md`).

## What is kept, and how it is cleaned

Columns: NAFDAC number, product (brand) name, active ingredient (generic name), strength, dosage
form, pack size, registration holder, category, registration status, registration expiry, the
Greenbook's product id and the fetch date. No personal data: registration holders are companies.

The register is treated as untrusted input (`src/medlens/seed/greenbook.py`):

* NAFDAC numbers must pass MedLens's own normaliser; rows that fail are dropped.
* HTML entities are decoded and whitespace normalised. Display markers (`*`, `#`, `$`, `$1`,
  `$2`, trailing `_`) and internal review notes ("(check dosage form)", "(updates needed)") are
  removed.
* Status is derived as of the fetch date: an expiry date in the past means `expired`, otherwise
  Active means `active` and Inactive means `suspended`.
* A NAFDAC number listed more than once keeps one entry: active first, then the latest expiry,
  then the oldest Greenbook id.

The sample takes, for each of a fixed list of widely sold and often counterfeited medicines, the
registered drug whose name matches most closely (`SAMPLE_BRANDS`).

## Refreshing

```bash
uv run python -m medlens.seed.greenbook export        # data/greenbook/greenbook-<date>.csv.gz
# upload the file to private object storage, then set "location" in seed/greenbook-snapshot.json
uv run python -m medlens.seed.greenbook sample --snapshot data/greenbook/greenbook-<date>.csv.gz
make greenbook-import                                 # in each environment
```

Commit the manifest, the sample and an updated date in this file together.

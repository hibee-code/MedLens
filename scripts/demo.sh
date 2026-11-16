#!/usr/bin/env bash
# MedLens demo (`make demo`): starts the stack, migrates, seeds, signs in as each kind of user
# and walks through a genuine scan, a fake scan, a hotspot query and an assistant question,
# printing every response.
#
#   DEMO_PASSWORD   staff password for the seed users (default: a random one, kept in
#                   .demo/password; never printed)
#   SKIP_UP=1       use a stack that is already running
#
# Local development only: it refuses to run against staging or production settings. Staff
# sessions (refresh tokens), authenticator secrets and the password are kept in .demo/
# (gitignored, mode 700): reruns refresh the saved sessions instead of signing in again, so
# they stay under the sign-in rate limit (which is not changed). `make demo-reset CONFIRM=1`
# clears the demo's scans between runs.
set -euo pipefail
cd "$(dirname "$0")/.."

API="${API_URL:-http://127.0.0.1:${API_HOST_PORT:-8000}}"
STATE=.demo
SESSIONS="$STATE/sessions"
mkdir -p "$SESSIONS"
chmod 700 "$STATE" "$SESSIONS"
if [[ -n "${DEMO_PASSWORD:-}" ]]; then
  (umask 177 && printf '%s' "$DEMO_PASSWORD" >"$STATE/password")
elif [[ -f "$STATE/password" ]]; then
  DEMO_PASSWORD="$(cat "$STATE/password")"
else
  DEMO_PASSWORD="demo-$(uv run python -c 'import secrets; print(secrets.token_urlsafe(18))')"
  (umask 177 && printf '%s' "$DEMO_PASSWORD" >"$STATE/password")
fi

step() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
# curl that waits out rate limits (429 + Retry-After) and fails on other errors.
call() {
  local headers body status
  headers="$(mktemp)"
  for _ in 1 2 3 4; do
    body="$(curl -sS -D "$headers" -w '\n%{http_code}' "$@")"
    status="${body##*$'\n'}"
    body="${body%$'\n'*}"
    if [[ "$status" == 429 ]]; then
      local wait
      wait="$(grep -i '^retry-after:' "$headers" | tr -dc '0-9')"
      echo "rate limited: waiting ${wait:-5}s" >&2
      sleep "${wait:-5}"
      continue
    fi
    rm -f "$headers"
    if [[ "$status" -ge 400 ]]; then
      echo "HTTP $status: $body" >&2
      return 1
    fi
    printf '%s' "$body"
    return 0
  done
  rm -f "$headers"
  return 1
}
json() { uv run --quiet python -c "import json,sys; d=json.load(sys.stdin); print($1)"; }
show() { uv run --quiet python -m json.tool; }

if [[ "${SKIP_UP:-0}" != "1" ]]; then
  step "Starting the stack (docker compose)"
  make up
fi
save_session() {  # name, token response -> keeps the refresh token (single use, rotated)
  (umask 177 && json "d['refresh_token']" <<<"$2" >"$SESSIONS/$1")
}

saved_session() {  # name -> a fresh access token from the saved session, or fails
  local file="$SESSIONS/$1" response
  [[ -s "$file" ]] || return 1
  response="$(curl -sS -f -X POST "$API/api/v1/auth/refresh" \
    -H 'Content-Type: application/json' -d "{\"refresh_token\": \"$(cat "$file")\"}" 2>/dev/null)" \
    || { rm -f "$file"; return 1; }
  save_session "$1" "$response"
  json "d['access_token']" <<<"$response"
}

step "Migrating and seeding"
make migrate
# Saved staff sessions are refreshed first: a sign-in (password + MFA) counts against the
# sign-in rate limit, a refresh does not.
PHARMACIST="$(saved_session pharmacist || true)"
MANUFACTURER="$(saved_session manufacturer || true)"
REGULATOR="$(saved_session regulator || true)"
SEED=(uv run python -m medlens.seed --staff-password-file "$STATE/password")
if [[ -n "$PHARMACIST" && -n "$MANUFACTURER" && -n "$REGULATOR" ]]; then
  "${SEED[@]}"  # keep the staff logins: their saved sessions are still valid
else
  "${SEED[@]}" --reset-staff-logins
  rm -f "$STATE"/*.totp "$SESSIONS"/*  # MFA was just switched off: enrol and sign in again
  PHARMACIST="" MANUFACTURER="" REGULATOR=""
fi
# The synthetic barcodes of the test packs (seed/test_packs/gtins.csv): local only, for the
# cloned-barcode and same-holder steps.
uv run python -m medlens.seed.demo_gtins seed/test_packs/gtins.csv
make embeddings  # the workers embed the seeded documents while the demo signs in and scans
echo "Staff password for the seed users: in $STATE/password"

password_login() {  # email -> token response (enrols MFA for roles that require it)
  local email="$1" response status mfa_token secret code
  response="$(call -X POST "$API/api/v1/auth/login" -H 'Content-Type: application/json' \
    -d "{\"email\": \"$email\", \"password\": \"$DEMO_PASSWORD\"}")"
  status="$(json "d['status']" <<<"$response")"
  if [[ "$status" == "authenticated" ]]; then
    printf '%s' "$response"
    return
  fi
  mfa_token="$(json "d['mfa_token']" <<<"$response")"
  if [[ "$status" == "mfa_enrolment_required" ]]; then
    secret="$(call -X POST "$API/api/v1/auth/mfa/enrol" \
      -H "Authorization: Bearer $mfa_token" | json "d['secret']")"
    (umask 177 && printf '%s' "$secret" >"$STATE/$email.totp")
    code="$(uv run --quiet python -c "import pyotp; print(pyotp.TOTP('$secret').now())")"
    call -X POST "$API/api/v1/auth/mfa/confirm" -H "Authorization: Bearer $mfa_token" \
      -H 'Content-Type: application/json' -d "{\"code\": \"$code\"}"
    return
  fi
  secret="$(cat "$STATE/$email.totp")"
  code="$(uv run --quiet python -c "import pyotp; print(pyotp.TOTP('$secret').now())")"
  call -X POST "$API/api/v1/auth/mfa/verify" -H 'Content-Type: application/json' \
    -d "{\"mfa_token\": \"$mfa_token\", \"code\": \"$code\"}"
}

staff_session() {  # name, email -> access token: the saved session, else a sign-in (saved)
  local response
  response="$(password_login "$2")"
  save_session "$1" "$response"
  json "d['access_token']" <<<"$response"
}

consumer_login() {  # phone -> access token (the console SMS provider logs the code)
  local phone="$1" code
  call -X POST "$API/api/v1/auth/otp/request" -H 'Content-Type: application/json' \
    -d "{\"phone\": \"$phone\"}" >/dev/null
  sleep 1
  # The log line has the code but not the number (privacy): take the newest code.
  code="$(docker compose logs api --no-color --since 2m 2>/dev/null | grep sms.console_message \
    | sed -nE 's/.*code is ([0-9]{6}).*/\1/p' | tail -1)"
  call -X POST "$API/api/v1/auth/otp/verify" -H 'Content-Type: application/json' \
    -d "{\"phone\": \"$phone\", \"code\": \"$code\"}" | json "d['access_token']"
}

scan() {  # image, claimed number, latitude, longitude, [token] -> response body
  local auth=()
  [[ -n "${5:-}" ]] && auth=(-H "Authorization: Bearer $5")
  call -X POST "$API/api/v1/scans" "${auth[@]}" \
    -H "X-Device-Id: ${DEVICE:-demo-$(basename "$1" .jpg)-$RANDOM}" \
    -F "image=@$1;type=image/jpeg" -F "nafdac_number=$2" -F "state=NG-LA" \
    -F "latitude=$3" -F "longitude=$4"
}

poll() {  # scan id, token (or device:ID for a guest) -> the scan's JSON once finished
  local body who
  if [[ "$2" == device:* ]]; then who="X-Device-Id: ${2#device:}"; else who="Authorization: Bearer $2"; fi
  for _ in $(seq 1 30); do
    body="$(call "$API/api/v1/scans/$1" -H "$who")"
    case "$(json "d['status']" <<<"$body")" in received | processing) sleep 1 ;; *) break ;; esac
  done
  printf '%s' "$body"
}

result() { poll "$1" "$2" | show; }  # scan id, token -> the scan once finished, printed

step "Signing in: consumer (SMS code), pharmacist, manufacturer and regulator (password + MFA)"
# A new consumer each run: the same photo from the same person within 24 hours returns the
# earlier scan (duplicate detection), which would hide what the current code decides.
CONSUMER="$(consumer_login "+234803$(printf '%07d' $((RANDOM * 32768 + RANDOM)) | tail -c 7)")"
if [[ -n "$REGULATOR" ]]; then
  echo "staff: reusing the saved sessions (refreshed, no sign-in)"
else
  PHARMACIST="$(staff_session pharmacist pharmacy.staff@medlens.test)"
  MANUFACTURER="$(staff_session manufacturer manufacturer.staff@medlens.test)"
  REGULATOR="$(staff_session regulator regulator.admin@medlens.test)"
fi
# Another consumer for the register-check pack: it looks almost like genuine.jpg, and a
# near-identical photo from the same person gets the earlier answer.
SECOND_CONSUMER="$(consumer_login "+234805$(printf '%07d' $((RANDOM * 32768 + RANDOM)) | tail -c 7)")"
echo "five sessions ready"

step "A genuine pack (Emzor Paracetamol Syrup, declared batch, GS1 DataMatrix)"
GENUINE="$(scan seed/test_packs/genuine.jpg 04-0289 6.5244 3.3792 "$CONSUMER" | json "d['id']")"
result "$GENUINE" "$CONSUMER"

step "The register check: the same product printed with the wrong strength (250 mg/5 ml)"
# Registered as 125 mg/5 mL in the NAFDAC Greenbook copy. Read with confidence >= 0.85, the
# mismatch makes the pack SUSPICIOUS. Consumers see the verdict, one coarse reason
# (DETAILS_DO_NOT_MATCH_REGISTER) and where the registration data comes from; regulators also
# see the field-by-field register check, each field's confidence and whether it counted.
WRONG="$(scan seed/test_packs/strength_mismatch.jpg 04-0289 6.5244 3.3792 "$SECOND_CONSUMER" \
  | json "d['id']")"
echo "As the consumer sees it:"
result "$WRONG" "$SECOND_CONSUMER"
echo "The register check, as the regulator sees it:"
call "$API/api/v1/scans/$WRONG" -H "Authorization: Bearer $REGULATOR" \
  | json "json.dumps(d['analysis']['register_check'], indent=2)"

step "A registered brand carrying a NAFDAC number that is in no registration"
# brand_wrong_number.jpg: "Coartem 20/120 Tablets" with the made-up number B4-9931. The brand
# matches a registered product closely (0.9 or more, on the brand's own words), so the scan is
# SUSPICIOUS and linked to Coartem (hotspots, its manufacturer's report), not UNREGISTERED.
# A guest with its own device: the photo looks almost like genuine.jpg, and a near-identical
# photo from the same person gets the earlier answer.
GUEST="demo-brand-$RANDOM$RANDOM"
BRAND="$(DEVICE="$GUEST" scan seed/test_packs/brand_wrong_number.jpg B4-9931 6.5244 3.3792 \
  | json "d['id']")"
echo "As the consumer sees it:"
poll "$BRAND" "device:$GUEST" | json "json.dumps({k: d[k] for k in ('final_verdict', 'reason_codes', 'product')}, indent=2)"
echo "As the regulator sees it:"
poll "$BRAND" "$REGULATOR" \
  | json "json.dumps({'reason_codes': d['reason_codes'], 'match': d['analysis']['signals']['match']}, indent=2)"

step "A cloned barcode: Augmentin printed, with the barcode of Emzor's syrup (another holder)"
# The barcode numbers come from seed/test_packs/gtins.csv (synthetic, loaded above). Guests:
# these photos look almost alike, and a near-identical photo from the same person gets the
# earlier answer.
CLONED="$(scan seed/test_packs/cloned_barcode.jpg 04-8655 6.5244 3.3792 | json "d['id']")"
poll "$CLONED" "$REGULATOR" | json "json.dumps({'final_verdict': d['final_verdict'], \
  'reason_codes': d['reason_codes'], 'risk_score': d['analysis']['risk_score'], \
  'barcode': d['analysis']['register_check']['barcode']}, indent=2)"

step "The same holder's barcode: Augmentin printed, with Zinnat's barcode (both GSK)"
# Likely a re-registration, not a clone: a weak signal (0.25), still LIKELY_GENUINE.
SAME_HOLDER="$(scan seed/test_packs/barcode_same_holder.jpg 04-8655 6.5244 3.3792 | json "d['id']")"
poll "$SAME_HOLDER" "$REGULATOR" | json "json.dumps({'final_verdict': d['final_verdict'], \
  'reason_codes': d['reason_codes'], 'risk_score': d['analysis']['risk_score'], \
  'barcode': d['analysis']['register_check']['barcode']}, indent=2)"

step "Refreshing the register by hand: the report of make greenbook-add-rows (nothing applied)"
# Where the file and a snapshot describe a product, the most recently retrieved row wins; the
# report lists every conflicting field before CONFIRM=1.
make --no-print-directory greenbook-add-rows FILE=seed/greenbook_manual.csv 2>&1 | head -12

step "Fake packs (NAFDAC numbers not in the register), scanned around Lagos"
FAKE=""
for pack in unregistered unregistered_2 unregistered_3 injection; do
  number="$(grep "^$pack.jpg," seed/test_packs/expected.csv | cut -d, -f3)"
  id="$(scan "seed/test_packs/$pack.jpg" "$number" "6.52$((RANDOM % 9))" "3.37$((RANDOM % 9))" \
    | json "d['id']")"
  FAKE="${FAKE:-$id}"
done
echo "The first fake scan, as the regulator sees it (with the evidence):"
result "$FAKE" "$REGULATOR"

step "Hotspots (regulator): clusters of flagged scans in the last day"
sleep 2
call "$API/api/v1/hotspots?days=1&min_points=3" -H "Authorization: Bearer $REGULATOR" | show

step "Review cases opened for the fake scans (regulator)"
call "$API/api/v1/cases?limit=3" -H "Authorization: Bearer $REGULATOR" | show

step "Five people in Lagos scan packs of an expired Emzor batch (the demo manufacturer's product)"
# The manufacturer report suppresses cells under 5 scans (re-identification), so five flagged
# scans of one product, state and week make one visible row. Guest scans, each from its own
# device: the same photo from the same person would return the earlier scan.
for _ in 1 2 3 4 5; do
  id="$(scan seed/test_packs/expired_batch.jpg 04-0289 "6.52$((RANDOM % 9))" "3.37$((RANDOM % 9))" \
    | json "d['id']")"
  result "$id" "$REGULATOR" >/dev/null  # wait until it is scored
done
echo "five expired-batch scans recorded"

step "The manufacturer's aggregated report (no individual scans; small cells suppressed)"
call "$API/api/v1/manufacturer/scans" -H "Authorization: Bearer $MANUFACTURER" | show

step "Assistant (pharmacist): grounded answer, or search results while AI is switched off"
call -X POST "$API/api/v1/assistant/ask" -H "Authorization: Bearer $PHARMACIST" \
  -H 'Content-Type: application/json' \
  -d '{"question": "Has an alert been published about counterfeit Augmentin 1 G?"}' | show

step "Done. API docs: $API/docs"

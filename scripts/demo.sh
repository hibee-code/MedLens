#!/usr/bin/env bash
# MedLens demo (`make demo`): starts the stack, migrates, seeds, onboards the people (the first
# platform admin with `make bootstrap-admin`, then staff by invitation into the seeded
# organizations, each setting up MFA) and walks through a genuine scan, a fake scan, a hotspot
# query and an assistant question, printing every response. Consumers scan as guests (a device
# id each), so no sign-in code is sent.
#
#   DEMO_PASSWORD   password of the demo's staff accounts (default: a random one, kept in
#                   .demo/password; never printed)
#   SKIP_UP=1       use a stack that is already running
#
# Local development only: it refuses to run against staging or production settings. Staff
# sessions (refresh tokens), authenticator secrets, recovery codes and the password are kept in
# .demo/ (gitignored, mode 700): reruns refresh the saved sessions instead of signing in again,
# so they stay under the sign-in rate limit (which is not changed). `make demo-reset CONFIRM=1`
# clears the demo's scans between runs; with ACCOUNTS=1 it also erases the demo's accounts
# (addresses @medlens.test, which are never e-mailed: reserved test domain).
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
json() {  # python expression over d (the JSON on stdin), [arguments as sys.argv[1:]]
  local expression="$1"
  shift
  uv run --quiet python -c "import json,sys; d=json.load(sys.stdin); print($expression)" "$@"
}
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

step "Migrating and seeding (organizations and products; no users)"
make migrate
uv run python -m medlens.seed
# The synthetic barcodes of the test packs (seed/test_packs/gtins.csv): local only, for the
# cloned-barcode and same-holder steps.
uv run python -m medlens.seed.demo_gtins seed/test_packs/gtins.csv
make embeddings  # the workers embed the seeded documents while the demo signs in and scans

totp_now() { uv run --quiet python -c "import pyotp, sys; print(pyotp.TOTP(sys.argv[1]).now())" "$1"; }

finish_mfa() {  # email, sign-in answer -> token response (sets up MFA on the first sign-in)
  local email="$1" response="$2" status mfa_token secret confirmed
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
    confirmed="$(call -X POST "$API/api/v1/auth/mfa/confirm" \
      -H "Authorization: Bearer $mfa_token" -H 'Content-Type: application/json' \
      -d "{\"code\": \"$(totp_now "$secret")\"}")"
    (umask 177 && json "'\\n'.join(d['recovery_codes'])" <<<"$confirmed" >"$STATE/$email.recovery")
    printf '%s' "$confirmed"
    return
  fi
  call -X POST "$API/api/v1/auth/mfa/verify" -H 'Content-Type: application/json' \
    -d "{\"mfa_token\": \"$mfa_token\", \"code\": \"$(totp_now "$(cat "$STATE/$email.totp")")\"}"
}

password_login() {  # email -> token response
  local response
  response="$(call -X POST "$API/api/v1/auth/login" -H 'Content-Type: application/json' \
    -d "{\"email\": \"$1\", \"password\": \"$DEMO_PASSWORD\"}")"
  finish_mfa "$1" "$response"
}

accept_new() {  # email, invitation token, full name -> token response (account created)
  local response
  response="$(call -X POST "$API/api/v1/invitations/$2/accept-new" \
    -H 'Content-Type: application/json' \
    -d "{\"full_name\": \"$3\", \"password\": \"$DEMO_PASSWORD\"}")"
  finish_mfa "$1" "$response"
}

session_token() {  # name, token response -> access token (the session is saved)
  save_session "$1" "$2"
  json "d['access_token']" <<<"$2"
}

step "Onboarding: the first platform admin (make bootstrap-admin), then staff by invitation"
ADMIN_EMAIL=demo.admin@medlens.test
if ADMIN="$(saved_session admin)"; then
  echo "admin: reusing the saved session"
elif [[ -f "$STATE/$ADMIN_EMAIL.totp" ]]; then
  ADMIN="$(session_token admin "$(password_login "$ADMIN_EMAIL")")"
else
  if ! output="$(make --no-print-directory bootstrap-admin EMAIL="$ADMIN_EMAIL" 2>&1)"; then
    printf '%s\n' "$output" >&2
    echo "A platform admin already exists that this demo cannot sign in as (an earlier seed's" \
      "account, or a removed .demo/). Erase the demo accounts with" \
      "'make demo-reset ACCOUNTS=1 CONFIRM=1', then run the demo again." >&2
    exit 1
  fi
  invitation="$(sed -nE 's/^Invitation token \(local only\): (.+)$/\1/p' <<<"$output")"
  ADMIN="$(session_token admin "$(accept_new "$ADMIN_EMAIL" "$invitation" "Demo Admin")")"
  echo "admin: invited by the bootstrap command, account created, MFA set up"
fi

org_id() {  # type, exact name -> the seeded organization's id
  call "$API/api/v1/organizations?type=$1&limit=100" -H "Authorization: Bearer $ADMIN" \
    | json "next(o['id'] for o in d['items'] if o['name'] == sys.argv[1])" "$2"
}

join() {  # name, email, organization type, organization name, role, full name -> access token
  local name="$1" email="$2" organization token response
  if token="$(saved_session "$name")"; then
    printf '%s' "$token"
    return
  fi
  if [[ -f "$STATE/$email.totp" ]]; then  # joined in an earlier run: sign in
    response="$(password_login "$email")"
  else
    organization="$(org_id "$3" "$4")"
    token="$(call -X POST "$API/api/v1/organizations/$organization/invitations" \
      -H "Authorization: Bearer $ADMIN" -H 'Content-Type: application/json' \
      -d "{\"email\": \"$email\", \"role\": \"$5\"}" | json "d['token']")"
    response="$(accept_new "$email" "$token" "$6")"
    echo "$name: invited as $5 into $4, account created, MFA set up" >&2
  fi
  session_token "$name" "$response"
}

PHARMACIST="$(join pharmacist demo.pharmacist@medlens.test pharmacy \
  "MedLens Test Pharmacy (seed)" pharmacy_staff "Demo Pharmacist")"
MANUFACTURER="$(join manufacturer demo.manufacturer@medlens.test manufacturer \
  "Emzor Pharmaceutical Industries Limited" manufacturer_staff "Demo Manufacturer")"
REGULATOR="$(join regulator demo.regulator@medlens.test regulator \
  "National Agency for Food and Drug Administration and Control (NAFDAC)" regulator_admin \
  "Demo Regulator")"
echo "staff sessions ready (password in $STATE/password; authenticator secrets and recovery" \
  "codes in $STATE/)"
# Consumers scan as guests, each with its own device id: the same photo from the same person
# within 24 hours returns the earlier scan (duplicate detection), which would hide what the
# current code decides.
CONSUMER="demo-consumer-$RANDOM$RANDOM"
SECOND_CONSUMER="demo-consumer-$RANDOM$RANDOM"

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

step "A genuine pack (Emzor Paracetamol Syrup, declared batch, GS1 DataMatrix)"
GENUINE="$(DEVICE="$CONSUMER" scan seed/test_packs/genuine.jpg 04-0289 6.5244 3.3792 \
  | json "d['id']")"
result "$GENUINE" "device:$CONSUMER"

step "The register check: the same product printed with the wrong strength (250 mg/5 ml)"
# Registered as 125 mg/5 mL in the NAFDAC Greenbook copy. Read with confidence >= 0.85, the
# mismatch makes the pack SUSPICIOUS. Consumers see the verdict, one coarse reason
# (DETAILS_DO_NOT_MATCH_REGISTER) and where the registration data comes from; regulators also
# see the field-by-field register check, each field's confidence and whether it counted.
WRONG="$(DEVICE="$SECOND_CONSUMER" scan seed/test_packs/strength_mismatch.jpg 04-0289 6.5244 \
  3.3792 | json "d['id']")"
echo "As the consumer sees it:"
result "$WRONG" "device:$SECOND_CONSUMER"
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

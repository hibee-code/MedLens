#!/usr/bin/env bash
# MedLens demo (`make demo`): starts the stack, migrates, seeds, signs in as each kind of user
# and walks through a genuine scan, a fake scan, a hotspot query and an assistant question,
# printing every response.
#
#   DEMO_PASSWORD   staff password for the seed users (default: a random one, printed)
#   SKIP_UP=1       use a stack that is already running
#
# Local development only: it refuses to run against staging or production settings, and the
# regulator's authenticator secret is kept in .demo/ (gitignored) so reruns can sign in.
set -euo pipefail
cd "$(dirname "$0")/.."

API="${API_URL:-http://127.0.0.1:${API_HOST_PORT:-8000}}"
STATE=.demo
mkdir -p "$STATE"
chmod 700 "$STATE"
if [[ -z "${DEMO_PASSWORD:-}" ]]; then
  if [[ -f "$STATE/password" ]]; then
    DEMO_PASSWORD="$(cat "$STATE/password")"
  else
    DEMO_PASSWORD="demo-$(uv run python -c 'import secrets; print(secrets.token_urlsafe(18))')"
    (umask 177 && printf '%s' "$DEMO_PASSWORD" >"$STATE/password")
  fi
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
step "Migrating and seeding"
make migrate
make seed SEED_ARGS="--staff-password $DEMO_PASSWORD --reset-staff-logins"
rm -f "$STATE"/*.totp  # MFA was just switched off: enrol again
make embeddings  # the workers embed the seeded documents while the demo signs in and scans
echo "Staff password for the seed users: $DEMO_PASSWORD (kept in $STATE/password)"

password_login() {  # email -> access token (enrols MFA for roles that require it)
  local email="$1" response status mfa_token secret code
  response="$(call -X POST "$API/api/v1/auth/login" -H 'Content-Type: application/json' \
    -d "{\"email\": \"$email\", \"password\": \"$DEMO_PASSWORD\"}")"
  status="$(json "d['status']" <<<"$response")"
  if [[ "$status" == "authenticated" ]]; then
    json "d['access_token']" <<<"$response"
    return
  fi
  mfa_token="$(json "d['mfa_token']" <<<"$response")"
  if [[ "$status" == "mfa_enrolment_required" ]]; then
    secret="$(call -X POST "$API/api/v1/auth/mfa/enrol" \
      -H "Authorization: Bearer $mfa_token" | json "d['secret']")"
    (umask 177 && printf '%s' "$secret" >"$STATE/$email.totp")
    code="$(uv run --quiet python -c "import pyotp; print(pyotp.TOTP('$secret').now())")"
    call -X POST "$API/api/v1/auth/mfa/confirm" -H "Authorization: Bearer $mfa_token" \
      -H 'Content-Type: application/json' -d "{\"code\": \"$code\"}" | json "d['access_token']"
    return
  fi
  secret="$(cat "$STATE/$email.totp")"
  code="$(uv run --quiet python -c "import pyotp; print(pyotp.TOTP('$secret').now())")"
  call -X POST "$API/api/v1/auth/mfa/verify" -H 'Content-Type: application/json' \
    -d "{\"mfa_token\": \"$mfa_token\", \"code\": \"$code\"}" | json "d['access_token']"
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
    -H "X-Device-Id: demo-$(basename "$1" .jpg)-$RANDOM" \
    -F "image=@$1;type=image/jpeg" -F "nafdac_number=$2" -F "state=NG-LA" \
    -F "latitude=$3" -F "longitude=$4"
}

result() {  # scan id, token -> the scan once finished (polls)
  local body
  for _ in $(seq 1 30); do
    body="$(call "$API/api/v1/scans/$1" -H "Authorization: Bearer $2")"
    case "$(json "d['status']" <<<"$body")" in received | processing) sleep 1 ;; *) break ;; esac
  done
  show <<<"$body"
}

step "Signing in: consumer (SMS code), pharmacist, manufacturer and regulator (password + MFA)"
# A new consumer each run: the same photo from the same person within 24 hours returns the
# earlier scan (duplicate detection), which would hide what the current code decides.
CONSUMER="$(consumer_login "+234803$(printf '%07d' $((RANDOM * 32768 + RANDOM)) | tail -c 7)")"
PHARMACIST="$(password_login pharmacy.staff@medlens.test)"
MANUFACTURER="$(password_login manufacturer.staff@medlens.test)"
REGULATOR="$(password_login regulator.admin@medlens.test)"
echo "four sessions ready"

step "A genuine pack (Emzor Paracetamol Syrup, declared batch, GS1 DataMatrix)"
GENUINE="$(scan seed/test_packs/genuine.jpg 04-0289 6.5244 3.3792 "$CONSUMER" | json "d['id']")"
result "$GENUINE" "$CONSUMER"

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

step "The manufacturer's aggregated report (no individual scans; small cells suppressed)"
call "$API/api/v1/manufacturer/scans" -H "Authorization: Bearer $MANUFACTURER" | show

step "Assistant (pharmacist): grounded answer, or search results while AI is switched off"
call -X POST "$API/api/v1/assistant/ask" -H "Authorization: Bearer $PHARMACIST" \
  -H 'Content-Type: application/json' \
  -d '{"question": "Has an alert been published about counterfeit Augmentin 1 G?"}' | show

step "Done. API docs: $API/docs"

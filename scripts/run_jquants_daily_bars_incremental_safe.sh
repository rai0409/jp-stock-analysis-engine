#!/usr/bin/env bash
set -u

PROJECT_DIR="${PROJECT_DIR:-/home/rai/projects/jp-stock-analysis-engine}"
STORE_DIR="${STORE_DIR:-/mnt/d/jquants_price_store}"
UNIVERSE_FILE="${UNIVERSE_FILE:-/mnt/d/jquants_price_store/topix1000_usable_tickers.csv}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_DIR/.cache/jquants}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
FETCH_BIN="${FETCH_BIN:-$PROJECT_DIR/.venv/bin/fetch-jquants-daily-bars-incremental}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
SAFE_STATE_FILE="${SAFE_STATE_FILE:-$STORE_DIR/daily_bars_safe_state.json}"
PRICE_FILE="${PRICE_FILE:-$STORE_DIR/prices_daily_bars.csv}"
TODAY="${JQUANTS_DAILY_BARS_SAFE_TODAY:-$(date +%F)}"
PROBE_TTL_HOURS="${JQUANTS_DAILY_BARS_SAFE_PROBE_TTL_HOURS:-20}"
FORCE_CURRENT_WINDOW=0

if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$PROJECT_DIR/.env"
  set +a
fi

for arg in "$@"; do
  case "$arg" in
    --force-current-window)
      FORCE_CURRENT_WINDOW=1
      ;;
    *)
      echo "unexpected argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$LOG_DIR"

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

if [[ ! -x "$FETCH_BIN" ]]; then
  log "daily_bars_safe status=error reason=missing_fetch_cli path=$FETCH_BIN" >&2
  exit 2
fi
if [[ ! -f "$UNIVERSE_FILE" ]]; then
  log "daily_bars_safe status=error reason=missing_universe_file path=$UNIVERSE_FILE" >&2
  exit 2
fi
if [[ ! -f "$PRICE_FILE" ]]; then
  log "daily_bars_safe status=error reason=missing_price_file path=$PRICE_FILE" >&2
  exit 2
fi

window_json="$("$PYTHON_BIN" - "$PRICE_FILE" "$TODAY" "$FORCE_CURRENT_WINDOW" <<'PY'
import csv
import json
import sys
from datetime import date, timedelta

price_file, today_text, force_text = sys.argv[1:4]
today = date.fromisoformat(today_text)
force = force_text == "1"
date_max = None
try:
    with open(price_file, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "date" not in (reader.fieldnames or []):
            raise SystemExit("missing date column")
        for row in reader:
            raw = (row.get("date") or "").strip()
            if not raw:
                continue
            day = date.fromisoformat(raw)
            date_max = day if date_max is None or day > date_max else date_max
except Exception as exc:
    print(json.dumps({"status": "error", "reason": f"invalid_price_csv: {exc}"}))
    raise SystemExit(0)

if date_max is None:
    print(json.dumps({"status": "error", "reason": "empty_price_csv"}))
    raise SystemExit(0)

def next_weekday(day):
    cur = day + timedelta(days=1)
    while cur.weekday() >= 5:
        cur += timedelta(days=1)
    return cur

if force:
    start = today - timedelta(days=1)
    end = today
else:
    start = next_weekday(date_max)
    end = start

if start > today:
    print(json.dumps({
        "status": "no_candidate",
        "date_max": date_max.isoformat(),
        "today": today.isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
    }))
else:
    print(json.dumps({
        "status": "fetch",
        "date_max": date_max.isoformat(),
        "today": today.isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "force_current_window": force,
    }))
PY
)"

window_status="$("$PYTHON_BIN" - "$window_json" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["status"])
PY
)"

if [[ "$window_status" == "error" ]]; then
  reason="$("$PYTHON_BIN" - "$window_json" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["reason"])
PY
)"
  log "daily_bars_safe status=error reason=\"$reason\"" >&2
  exit 2
fi

if [[ "$window_status" == "no_candidate" ]]; then
  log "daily_bars_safe status=no_candidate payload=$window_json"
  exit 0
fi

start_date="$("$PYTHON_BIN" - "$window_json" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["start"])
PY
)"
end_date="$("$PYTHON_BIN" - "$window_json" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["end"])
PY
)"
date_max="$("$PYTHON_BIN" - "$window_json" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["date_max"])
PY
)"

if [[ "$FORCE_CURRENT_WINDOW" != "1" ]]; then
  ttl_status="$("$PYTHON_BIN" - "$SAFE_STATE_FILE" "$start_date" "$date_max" "$PROBE_TTL_HOURS" <<'PY'
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

path = Path(sys.argv[1])
candidate = sys.argv[2]
date_max = sys.argv[3]
ttl_hours = float(sys.argv[4])
if not path.exists():
    print("probe")
    raise SystemExit
try:
    state = json.loads(path.read_text(encoding="utf-8"))
    generated = datetime.fromisoformat(state.get("generated_at", "").replace("Z", "+00:00"))
except Exception:
    print("probe")
    raise SystemExit
if (
    state.get("status") == "coverage_wait"
    and state.get("candidate_start_date") == candidate
    and state.get("date_max") == date_max
    and datetime.now(timezone.utc) - generated < timedelta(hours=ttl_hours)
):
    print("skip")
else:
    print("probe")
PY
)"
  if [[ "$ttl_status" == "skip" ]]; then
    log "daily_bars_safe status=coverage_wait reason=recent_coverage_wait candidate_start_date=$start_date date_max=$date_max ttl_hours=$PROBE_TTL_HOURS"
    exit 0
  fi
fi

tmp_output="$(mktemp)"
log "daily_bars_safe status=fetch_start start_date=$start_date end_date=$end_date date_max=$date_max force_current_window=$FORCE_CURRENT_WINDOW"
"$FETCH_BIN" \
  --universe-file "$UNIVERSE_FILE" \
  --store-dir "$STORE_DIR" \
  --start-date "$start_date" \
  --end-date "$end_date" \
  --cache-dir "$CACHE_DIR" \
  --allow-network \
  --sleep-seconds 90 \
  --max-retries 2 >"$tmp_output" 2>&1
rc=$?
cat "$tmp_output"

if [[ "$rc" == "0" ]]; then
  "$PYTHON_BIN" - "$SAFE_STATE_FILE" "$start_date" "$end_date" "$date_max" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(json.dumps({
    "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "status": "success",
    "candidate_start_date": sys.argv[2],
    "candidate_end_date": sys.argv[3],
    "date_max": sys.argv[4],
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  rm -f "$tmp_output"
  exit 0
fi

classification="$("$PYTHON_BIN" - "$STORE_DIR/daily_bars_fetch_state.json" "$tmp_output" <<'PY'
import json
import re
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
text = output_path.read_text(encoding="utf-8", errors="replace")
failed = {}
if state_path.exists():
    try:
        failed = json.loads(state_path.read_text(encoding="utf-8")).get("failed_dates", {}) or {}
    except Exception:
        failed = {}
reason_text = " ".join(str(v) for v in failed.values()) + " " + text
lower = reason_text.lower()
auth_patterns = ("api key", "unauthorized", "authentication", "forbidden", "401", "403")
coverage_patterns = (
    "invalid_argument",
    "invalidargument",
    "invalid argument",
    "outside",
    "not available",
    "coverage",
    "range",
    "plan",
    "subscription",
)
if any(p in lower for p in auth_patterns):
    print("unexpected_failure")
elif failed and any(p in lower for p in coverage_patterns):
    print("coverage_wait")
else:
    print("unexpected_failure")
PY
)"

if [[ "$classification" == "coverage_wait" && "$FORCE_CURRENT_WINDOW" != "1" ]]; then
  "$PYTHON_BIN" - "$SAFE_STATE_FILE" "$start_date" "$end_date" "$date_max" "$tmp_output" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
state_file = Path(sys.argv[1])
output_text = Path(sys.argv[5]).read_text(encoding="utf-8", errors="replace")
state_file.write_text(json.dumps({
    "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "status": "coverage_wait",
    "candidate_start_date": sys.argv[2],
    "candidate_end_date": sys.argv[3],
    "date_max": sys.argv[4],
    "note": "J-Quants daily bars coverage appears not to include the next missing date yet.",
    "last_cli_output": output_text[-2000:],
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  log "daily_bars_safe status=coverage_wait candidate_start_date=$start_date candidate_end_date=$end_date date_max=$date_max exit=0"
  rm -f "$tmp_output"
  exit 0
fi

log "daily_bars_safe status=error reason=unexpected_fetch_failure rc=$rc start_date=$start_date end_date=$end_date" >&2
rm -f "$tmp_output"
exit 2

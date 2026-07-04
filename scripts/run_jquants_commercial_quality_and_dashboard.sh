#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/rai/projects/jp-stock-analysis-engine"
QUALITY_CLI="$PROJECT_DIR/.venv/bin/build-jquants-commercial-quality-dataset"
DASHBOARD_CLI="$PROJECT_DIR/.venv/bin/build-jquants-commercial-daily-dashboard"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

stage="initialization"

log() {
  printf '[quality-dashboard] %s %s\n' "$1" "$2"
}

on_error() {
  local rc=$?
  printf '[quality-dashboard] failed stage=%s exit_code=%s timestamp=%s\n' "$stage" "$rc" "$(date -Is)" >&2
  exit "$rc"
}

trap on_error ERR

cd "$PROJECT_DIR"
mkdir -p logs

log "start" "timestamp=$(date -Is)"

if [[ ! -x "$QUALITY_CLI" ]]; then
  stage="quality dataset preflight"
  printf '[quality-dashboard] missing quality dataset CLI path=%s\n' "$QUALITY_CLI" >&2
  exit 2
fi

if [[ ! -x "$DASHBOARD_CLI" ]]; then
  stage="daily dashboard preflight"
  printf '[quality-dashboard] missing daily dashboard CLI path=%s\n' "$DASHBOARD_CLI" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  stage="python preflight"
  printf '[quality-dashboard] missing python path=%s\n' "$PYTHON_BIN" >&2
  exit 2
fi

stage="quality dataset"
log "running" "quality dataset"
"$QUALITY_CLI"
log "quality dataset" "finished"

stage="daily dashboard"
log "running" "daily dashboard"
"$DASHBOARD_CLI"
log "daily dashboard" "finished"

stage="output summary"
log "output" "summary"
"$PYTHON_BIN" - <<'PY'
import csv
import json
from collections import Counter
from pathlib import Path

base = Path("analysis/jquants_fin_summary")
csv_files = [
    base / "commercial_daily_dashboard_summary.csv",
    base / "commercial_daily_sector_summary.csv",
    base / "commercial_daily_market_segment_summary.csv",
    base / "commercial_daily_topix_size_summary.csv",
]
json_files = [
    base / "commercial_daily_dashboard_summary.json",
    base / "commercial_daily_latest_snapshot.json",
    base / "commercial_daily_dashboard_audit.json",
]

missing = [str(path) for path in [*csv_files, *json_files] if not path.exists()]
if missing:
    for path in missing:
        print(f"[quality-dashboard] missing output file path={path}")
    raise SystemExit(3)

for path in csv_files:
    with path.open(encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    print(f"[quality-dashboard] csv rows path={path} rows={row_count}")

for path in json_files:
    print(f"[quality-dashboard] json exists path={path}")

snapshot_path = base / "commercial_daily_latest_snapshot.json"
snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
latest_summary = {}
with (base / "commercial_daily_dashboard_summary.csv").open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        if row.get("DiscDate") == snapshot.get("latest_disc_date"):
            latest_summary = row
            break
classification_counts = snapshot.get("total_counts_by_classification") or {}
total_dashboard_rows = snapshot.get("total_dashboard_rows")
if total_dashboard_rows is None:
    total_dashboard_rows = sum(int(value) for value in classification_counts.values())
total_unique_tickers = snapshot.get("total_unique_tickers")
latest_date_counts = snapshot.get("latest_date_counts") or {}
latest_date_total_rows = snapshot.get("latest_date_total_rows", latest_summary.get("all_dashboard_rows"))
latest_date_unique_tickers = snapshot.get("latest_date_unique_tickers", latest_summary.get("unique_tickers"))
print(
    "[quality-dashboard] latest snapshot "
    f"latest_disc_date={snapshot.get('latest_disc_date')} "
    f"latest_price_date={snapshot.get('latest_price_date')} "
    f"latest_financial_summary_disc_date={snapshot.get('latest_financial_summary_disc_date')} "
    f"total_dashboard_rows={total_dashboard_rows} "
    f"total_unique_tickers={total_unique_tickers} "
    f"latest_date_total_rows={latest_date_total_rows} "
    f"latest_date_unique_tickers={latest_date_unique_tickers}"
)

if not classification_counts:
    classification_counts = Counter()
    for key, value in latest_summary.items():
        if key.endswith("_rows") and key not in {"all_dashboard_rows", "total_classified_rows"}:
            classification_counts[key.removesuffix("_rows")] = int(value or 0)
print(
    "[quality-dashboard] classification counts "
    + json.dumps(classification_counts, ensure_ascii=False, sort_keys=True)
)
print(
    "[quality-dashboard] latest date counts "
    + json.dumps(latest_date_counts, ensure_ascii=False, sort_keys=True)
)
PY

trap - ERR
log "finished" "timestamp=$(date -Is)"

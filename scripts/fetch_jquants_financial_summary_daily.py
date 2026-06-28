import os
import re
import json
import time
import shutil
import urllib.request
import urllib.error
from urllib.parse import urlencode
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd


PROJECT_DIR = Path(
    os.getenv(
        "JP_STOCK_ANALYSIS_PROJECT_DIR",
        str(Path(__file__).resolve().parents[1]),
    )
)
LOCAL_STORE = PROJECT_DIR / "data/jquants_price_store"
UNIVERSE_FILE = Path(
    os.getenv(
        "JP_STOCK_ANALYSIS_UNIVERSE_FILE",
        str(LOCAL_STORE / "topix1000_usable_tickers.csv"),
    )
)

JST = timezone(timedelta(hours=9))

RAW_NAME = "financial_summary_raw.csv"
COVERAGE_NAME = "financial_summary_coverage.csv"
STATE_NAME = "financial_summary_auto_state.json"
RUN_LOG_NAME = "financial_summary_auto_runs.jsonl"

def get_external_candidates() -> list[str]:
    raw = os.getenv("JQUANTS_EXTERNAL_STORE_DIRS") or os.getenv("JQUANTS_EXTERNAL_STORE_DIR", "")
    return [item.strip() for item in raw.split(os.pathsep) if item.strip()]


EXTERNAL_CANDIDATES = get_external_candidates()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test = path / ".write_test"
        test.write_text("ok")
        test.unlink()
        return True
    except Exception:
        return False


def choose_store_dir() -> tuple[Path, str]:
    for raw in EXTERNAL_CANDIDATES:
        if not raw:
            continue
        p = Path(raw).expanduser()
        if not p.parent.exists():
            continue
        if is_writable_dir(p):
            return p, "external"

    LOCAL_STORE.mkdir(parents=True, exist_ok=True)
    return LOCAL_STORE, "local"


def ensure_external_has_existing_data(store: Path, store_type: str) -> None:
    if store_type != "external":
        return

    for name in [RAW_NAME, COVERAGE_NAME]:
        src = LOCAL_STORE / name
        dst = store / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def normalize_code_to_ticker(code):
    s = str(code).strip()
    if s.endswith("0") and len(s) == 5 and s[:4].isdigit():
        return s[:4]
    if s.endswith("0") and len(s) == 5 and s[:3].isdigit() and s[3].isalpha():
        return s[:4]
    return s


def extract_records(data):
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("data"), list):
        return data["data"]
    for value in data.values():
        if isinstance(value, list):
            return value
    return []


def parse_subscription_window(message: str):
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})", message)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def fetch_one_date(target_date: str, api_key: str, max_attempts: int = 3):
    headers = {
        "x-api-key": api_key,
        "User-Agent": "jp-stock-analysis-engine-fin-summary-auto-backfill",
    }
    url = "https://api.jquants.com/v2/fins/summary?" + urlencode({"date": target_date})
    req = urllib.request.Request(url, headers=headers)

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=90) as res:
                data = json.loads(res.read().decode("utf-8"))
            return {
                "ok": True,
                "records": extract_records(data),
                "attempt": attempt,
                "reason": "OK",
                "error": None,
            }

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            cov_start, cov_end = parse_subscription_window(body)

            if e.code == 400 and cov_start and cov_end:
                return {
                    "ok": False,
                    "records": [],
                    "attempt": attempt,
                    "reason": "OUT_OF_SUBSCRIPTION_DATE_RANGE",
                    "error": {
                        "type": "HTTPError",
                        "code": e.code,
                        "body": body[:1000],
                        "subscription_start": cov_start,
                        "subscription_end": cov_end,
                    },
                }

            if e.code == 403:
                return {
                    "ok": False,
                    "records": [],
                    "attempt": attempt,
                    "reason": "PLAN_NOT_AVAILABLE",
                    "error": {
                        "type": "HTTPError",
                        "code": e.code,
                        "body": body[:1000],
                    },
                }

            return {
                "ok": False,
                "records": [],
                "attempt": attempt,
                "reason": "HTTP_ERROR",
                "error": {
                    "type": "HTTPError",
                    "code": e.code,
                    "body": body[:1000],
                },
            }

        except Exception as e:
            last_error = {
                "type": type(e).__name__,
                "error": repr(e),
            }
            if attempt < max_attempts:
                time.sleep(30 * attempt)

    return {
        "ok": False,
        "records": [],
        "attempt": max_attempts,
        "reason": "NETWORK_OR_DNS_RETRY_EXHAUSTED",
        "error": last_error,
    }


def get_latest_disc_date(raw_path: Path):
    if not raw_path.exists():
        return None
    try:
        df = pd.read_csv(raw_path, dtype=str)
    except Exception:
        return None
    if len(df) == 0 or "DiscDate" not in df.columns:
        return None

    s = df["DiscDate"].dropna().astype(str)
    s = s[s.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    if len(s) == 0:
        return None
    return s.max()


def build_target_dates(raw_path: Path, state_path: Path):
    today = datetime.now(JST).date()

    pending = []
    last_successful_query_date = None

    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())

            previous_failed = state.get("failed_dates_carryover", {})
            for d, info in previous_failed.items():
                reason = info.get("reason") or info.get("type") or ""
                if reason in {"OUT_OF_SUBSCRIPTION_DATE_RANGE", "PLAN_NOT_AVAILABLE"}:
                    continue
                pending.append(d)

            last_successful_query_date = state.get("last_successful_query_date")
        except Exception:
            pass

    latest_disc_date = get_latest_disc_date(raw_path)

    candidates = []
    for d in [latest_disc_date, last_successful_query_date]:
        if d:
            try:
                candidates.append(datetime.fromisoformat(d).date())
            except Exception:
                pass

    if candidates:
        start = max(candidates) + timedelta(days=1)
    else:
        start = datetime.fromisoformat("2024-04-05").date()

    incremental = []
    d = start
    while d <= today:
        incremental.append(d.isoformat())
        d += timedelta(days=1)

    seen = set()
    result = []
    for d in pending + incremental:
        if d not in seen:
            seen.add(d)
            result.append(d)

    return result


def main():
    load_env_file(PROJECT_DIR / ".env")

    api_key = os.getenv("JQUANTS_API_KEY")
    if not api_key:
        raise SystemExit("JQUANTS_API_KEY: MISSING")

    if not UNIVERSE_FILE.exists():
        raise SystemExit(f"universe file not found: {UNIVERSE_FILE}")

    store, store_type = choose_store_dir()
    ensure_external_has_existing_data(store, store_type)

    raw_out = store / RAW_NAME
    coverage_out = store / COVERAGE_NAME
    state_out = store / STATE_NAME
    run_log_out = store / RUN_LOG_NAME

    universe = pd.read_csv(UNIVERSE_FILE, dtype=str)
    universe["ticker"] = universe["ticker"].astype(str)
    usable = set(universe["ticker"])

    target_dates = build_target_dates(raw_out, state_out)

    max_dates_per_run = int(os.getenv("JQUANTS_FIN_SUMMARY_MAX_DATES_PER_RUN", "20"))
    target_dates = target_dates[:max_dates_per_run]

    all_rows = []
    failed_dates = {}
    skipped_dates = {}
    empty_dates = []
    run_results = {}
    subscription_start = None
    subscription_end = None

    print("store:", store)
    print("store_type:", store_type)
    print("target_dates:", target_dates)

    for target_date in target_dates:
        print("fetch", target_date)

        result = fetch_one_date(target_date, api_key, max_attempts=3)
        run_results[target_date] = {
            "ok": result["ok"],
            "attempt": result["attempt"],
            "reason": result["reason"],
            "raw_count": len(result["records"]),
            "error": result["error"],
        }

        if not result["ok"]:
            err = result["error"] or {}
            reason = result["reason"]

            if reason == "OUT_OF_SUBSCRIPTION_DATE_RANGE":
                subscription_start = err.get("subscription_start")
                subscription_end = err.get("subscription_end")
                skipped_dates[target_date] = {
                    "reason": reason,
                    "detail": err,
                }
                print("  skipped:", reason, subscription_start, subscription_end)
                break

            failed_dates[target_date] = {
                "reason": reason,
                "detail": err,
            }
            print("  failed:", reason, err)
            time.sleep(20)
            continue

        records = result["records"]

        if not records:
            empty_dates.append(target_date)
            print("  records: 0")
            time.sleep(20)
            continue

        usable_added = 0
        for r in records:
            r = dict(r)
            r["query_date"] = target_date
            r["ticker"] = normalize_code_to_ticker(r.get("Code", ""))
            if r["ticker"] in usable:
                all_rows.append(r)
                usable_added += 1

        print("  raw_records:", len(records), "usable_added:", usable_added)
        time.sleep(20)

    new_df = pd.DataFrame(all_rows)

    if raw_out.exists():
        old_df = pd.read_csv(raw_out, dtype=str)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    if len(df):
        dedupe_keys = [c for c in ["DiscNo", "Code", "DiscDate", "DocType"] if c in df.columns]
        if dedupe_keys:
            df = df.drop_duplicates(dedupe_keys)
        df.to_csv(raw_out, index=False)
    else:
        pd.DataFrame().to_csv(raw_out, index=False)

    got = set(df["ticker"].astype(str)) if len(df) and "ticker" in df.columns else set()

    coverage = universe.copy()
    coverage["has_any_financial_summary"] = coverage["ticker"].isin(got)
    coverage.to_csv(coverage_out, index=False)

    failed_dates_carryover = {}
    for d, info in failed_dates.items():
        reason = info.get("reason")
        if reason not in {"OUT_OF_SUBSCRIPTION_DATE_RANGE", "PLAN_NOT_AVAILABLE"}:
            failed_dates_carryover[d] = info

    successful_query_dates = [
        d for d, r in run_results.items()
        if r.get("ok") is True
    ]

    state = {
        "endpoint": "/v2/fins/summary",
        "query_mode": "date",
        "last_run_at_jst": datetime.now(JST).isoformat(),
        "store_type": store_type,
        "store_dir": str(store),
        "target_dates_last_run": target_dates,
        "successful_query_dates_last_run": successful_query_dates,
        "last_successful_query_date": max(successful_query_dates) if successful_query_dates else None,
        "rows_added_last_run_before_dedup": int(len(new_df)),
        "rows_saved_total": int(len(df)),
        "unique_tickers_saved": int(len(got)),
        "coverage_missing_count": int(len(usable - got)),
        "latest_disc_date_saved": get_latest_disc_date(raw_out),
        "failed_dates_last_run": failed_dates,
        "failed_dates_count_last_run": int(len(failed_dates)),
        "failed_dates_carryover": failed_dates_carryover,
        "failed_dates_carryover_count": int(len(failed_dates_carryover)),
        "skipped_dates_last_run": skipped_dates,
        "skipped_dates_count_last_run": int(len(skipped_dates)),
        "empty_dates_last_run": empty_dates,
        "run_results": run_results,
        "subscription_start_detected": subscription_start,
        "subscription_end_detected": subscription_end,
        "output": str(raw_out),
        "coverage_output": str(coverage_out),
        "api_key_status": "PRESENT",
        "secret_included": False,
        "note": "Only raw J-Quants fields are saved. No PER/PBR/ROE/margins/growth/yield are calculated.",
    }

    state_out.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    with run_log_out.open("a") as f:
        f.write(json.dumps(state, ensure_ascii=False) + "\n")

    print("\n== done ==")
    print("store_type:", store_type)
    print("store_dir:", store)
    print("rows_added_last_run_before_dedup:", len(new_df))
    print("rows_saved_total:", len(df))
    print("unique_tickers_saved:", len(got))
    print("coverage_missing_count:", len(usable - got))
    print("latest_disc_date_saved:", state["latest_disc_date_saved"])
    print("failed_dates_count_last_run:", len(failed_dates))
    print("failed_dates_carryover_count:", len(failed_dates_carryover))
    print("skipped_dates_count_last_run:", len(skipped_dates))
    print("state:", state_out)


if __name__ == "__main__":
    main()

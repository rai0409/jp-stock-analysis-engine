#!/usr/bin/env python3
"""Oldest-first backfill for the J-Quants Financial Summary store.

The daily script (fetch_jquants_financial_summary_daily.py) only moves forward from
its watermark. This script fills the *older* dates that are still inside the plan's
data window, oldest date first, and keeps its own progress files. It never touches
the daily script's state file or run log.

Facts this script is built on (all observable in the daily script / its run log):
  * The API reports the plan window in its out-of-range error
    (OUT_OF_SUBSCRIPTION_DATE_RANGE, "<start> ~ <end>"). A date older than <start> can
    no longer be fetched, so the script jumps forward to <start> and keeps going.
  * A date newer than <end> is not available yet; the script stops there.
  * Dates that return no records leave no trace in financial_summary_raw.csv, so
    progress is recorded in financial_summary_backfill_state.json.

Safety:
  * It takes the SAME single-writer lock as the daily job. While either one runs, the
    other refuses to start (exit code 75), even if the systemd timer is active.
  * It refuses to run when the daily state / raw CSV look rolled back or truncated
    (same check as the daily job), and when the drive or a file is unreadable.
  * New rows are appended after the existing rows; existing rows are never changed.
  * If the API returns a field that is not already a column of the raw CSV, the run
    stops before writing anything (no silent schema change).
  * Raw / coverage / state are written atomically after every date, so an interrupted
    run can be resumed.

Files (all inside the store directory chosen by the daily script's own rules):
  * financial_summary_raw.csv / financial_summary_coverage.csv   (shared, same format)
  * financial_summary_backfill_state.json / financial_summary_backfill_runs.jsonl (new)
  * financial_summary_raw.csv.bak_before_backfill_<timestamp>    (one-time backup)

Usage:
    python scripts/backfill_jquants_financial_summary.py --to 2026-07-15 --dry-run
    python scripts/backfill_jquants_financial_summary.py --to 2026-07-15
    python scripts/backfill_jquants_financial_summary.py            # resume
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

DAILY_SCRIPT = Path(__file__).with_name("fetch_jquants_financial_summary_daily.py")
STATE_NAME = "financial_summary_backfill_state.json"
RUN_LOG_NAME = "financial_summary_backfill_runs.jsonl"
DEFAULT_FROM = "2024-04-05"  # the daily script's own default start date
DEFAULT_MAX_DATES = 20  # the daily script's own per-run cap
SLEEP_SECONDS = 20  # the daily script's own pause between requests
MAX_CONSECUTIVE_FAILURES = 3
DEDUPE_KEYS = ("DiscNo", "Code", "DiscDate", "DocType")
STOP_ON_REASONS = frozenset({"PLAN_NOT_AVAILABLE"})


class BackfillError(SystemExit):
    """Clean exit with a message."""


def load_daily():
    spec = importlib.util.spec_from_file_location(
        "fetch_jquants_financial_summary_daily", DAILY_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_iso(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise BackfillError(f"{label}: not a YYYY-MM-DD date: {value!r}") from None


def merge_rows(existing: pd.DataFrame, rows: list[dict], daily) -> pd.DataFrame:
    """Same merge as the daily script, but refuses to add columns."""
    new_df = pd.DataFrame(rows)
    if len(existing.columns) == 0:
        df = new_df
    else:
        extra = [c for c in new_df.columns if c not in existing.columns]
        if extra:
            raise daily.StoreInputError(
                "the API returned fields that are not columns of the raw CSV: "
                f"{extra}. Nothing was written.",
                kind="SCHEMA_CHANGE",
            )
        df = pd.concat([existing, new_df], ignore_index=True)
    keys = [c for c in DEDUPE_KEYS if c in df.columns]
    if keys:
        df = df.drop_duplicates(list(keys))
    return df


def load_state(path: Path, daily) -> dict | None:
    if daily.stat_or_none(path, "STATE_UNREADABLE") is None:
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise daily.oserror_to_store_error(exc, path, "STATE_UNREADABLE") from exc
    except ValueError as exc:
        raise daily.StoreInputError(
            f"{path}: not valid JSON ({exc}). It was not modified.", kind="STATE_UNREADABLE"
        ) from exc
    if not isinstance(state, dict):
        raise daily.StoreInputError(f"{path}: expected a JSON object", kind="STATE_UNREADABLE")
    return state


def describe_raw(df: pd.DataFrame) -> dict:
    info = {"rows": int(len(df)), "columns": int(len(df.columns))}
    if len(df) and "query_date" in df.columns:
        q = df["query_date"].dropna()
        info["min_query_date"] = q.min() if len(q) else None
        info["max_query_date"] = q.max() if len(q) else None
    return info


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Oldest-first backfill for the J-Quants Financial Summary store."
    )
    ap.add_argument(
        "--to",
        dest="to_date",
        help="last date to fetch (YYYY-MM-DD); required on the first run",
    )
    ap.add_argument(
        "--from",
        dest="from_date",
        help=(
            f"first date to try (default {DEFAULT_FROM}); "
            "dates older than the plan window are skipped automatically"
        ),
    )
    ap.add_argument(
        "--max-dates", type=int, default=DEFAULT_MAX_DATES, help="requests per run (default 20)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and facts; no API call, no write",
    )
    ap.add_argument(
        "--allow-progress-regression",
        action="store_true",
        help="continue even if the daily state / raw CSV look rolled back (use with care)",
    )
    return ap


def run(argv=None, daily=None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_dates < 1:
        raise BackfillError("--max-dates must be >= 1")
    daily = daily or load_daily()
    if not hasattr(daily, "StoreLock"):
        raise BackfillError(
            "fetch_jquants_financial_summary_daily.py is an older version; install the "
            "updated daily script together with this one."
        )
    try:
        return _run(args, daily)
    except daily.StoreError as exc:
        print(f"{exc.kind}: {exc}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from None


def _run(args, daily) -> int:
    daily.load_env_file(daily.PROJECT_DIR / ".env")
    api_key = os.getenv("JQUANTS_API_KEY")
    if not api_key and not args.dry_run:
        raise BackfillError("JQUANTS_API_KEY: MISSING")

    universe_file = daily.get_universe_file()
    daily.check_store_reachable(universe_file.parent)
    if daily.stat_or_none(universe_file, "UNIVERSE_UNREADABLE") is None:
        raise BackfillError(f"universe file not found: {universe_file}")

    store, store_type = daily.choose_store_dir()
    daily.check_store_reachable(store)
    daily.ensure_external_has_existing_data(store, store_type)

    lock = daily.StoreLock("backfill")
    lock_status = "acquired"
    try:
        lock.acquire()
    except daily.BusyError as exc:
        if not args.dry_run:
            raise
        lock_status = f"HELD by another process ({lock.holder()})"
        print(f"BUSY (dry-run continues read-only): {exc}", file=sys.stderr)
    try:
        return _run_locked(args, daily, api_key, universe_file, store, store_type, lock_status)
    finally:
        lock.release()


def _run_locked(args, daily, api_key, universe_file, store, store_type, lock_status) -> int:
    raw_path = store / daily.RAW_NAME
    coverage_path = store / daily.COVERAGE_NAME
    state_path = store / STATE_NAME
    log_path = store / RUN_LOG_NAME

    universe = daily.load_universe(universe_file)
    usable = set(universe["ticker"])

    inputs = daily.load_store_inputs(
        raw_path, store / daily.STATE_NAME, store / daily.RUN_LOG_NAME
    )
    existing = inputs.raw
    expected_fingerprint = inputs.raw_fingerprint

    problems = daily.find_progress_problems(inputs)
    allowed = args.allow_progress_regression or os.getenv(daily.ALLOW_REGRESSION_ENV) == "1"
    if problems and not allowed and not args.dry_run:
        raise daily.ProgressRegressionError(
            "the daily state / raw CSV look rolled back or truncated, so backfilling now could "
            "overwrite newer data: " + "; ".join(problems) + ". Nothing was changed."
        )

    state = load_state(state_path, daily)

    # ---- resolve the plan -------------------------------------------------
    if state is None:
        if not args.to_date:
            raise BackfillError("--to is required on the first run (e.g. --to 2026-07-15)")
        range_to = parse_iso(args.to_date, "--to")
        range_from = parse_iso(args.from_date or DEFAULT_FROM, "--from")
        if range_from > range_to:
            raise BackfillError("--from must not be later than --to")
        state = {
            "mode": "backfill_oldest_first",
            "range_from": range_from.isoformat(),
            "range_to": range_to.isoformat(),
            "next_date": range_from.isoformat(),
            "done": False,
            "window_start_detected": None,
            "window_end_detected": None,
            "failed_dates": {},
        }
        is_new_state = True
    else:
        is_new_state = False
        for key, given in (("range_to", args.to_date), ("range_from", args.from_date)):
            if given and given != state.get(key):
                raise BackfillError(
                    f"--{key[6:]} {given} differs from the saved plan ({state.get(key)}); "
                    f"delete {state_path.name} to start a new plan"
                )
        range_to = parse_iso(state["range_to"], "state.range_to")

    cursor = parse_iso(state["next_date"], "state.next_date")
    failed: dict = dict(state.get("failed_dates") or {})

    raw_facts = describe_raw(existing)
    remaining = max((range_to - cursor).days + 1, 0)
    print("store:", store, f"({store_type})")
    print("store_mount:", daily.describe_mount(store))
    print("raw:", raw_facts)
    print("lock:", lock_status)
    verdict = "OK"
    if problems:
        verdict = ("OVERRIDDEN: " if allowed else "PROBLEMS: ") + "; ".join(problems)
    print("progress_check:", verdict)
    print(
        f"plan: {state['range_from']} .. {state['range_to']}  next_date={cursor}  "
        f"pending_failed={len(failed)}  remaining_dates={remaining}"
    )
    if args.dry_run:
        n = min(args.max_dates, len(failed) + remaining)
        print(
            f"dry-run: this run would make up to {n} requests "
            f"(~{n * (SLEEP_SECONDS + 1) // 60} min); no API call, nothing written."
        )
        return 0 if (allowed or not problems) else daily.ProgressRegressionError.exit_code

    if is_new_state and daily.stat_or_none(raw_path) is not None:
        stamp = datetime.now(daily.JST).strftime("%Y%m%d_%H%M%S")
        shutil.copy2(raw_path, raw_path.with_name(f"{raw_path.name}.bak_before_backfill_{stamp}"))

    pending = sorted(failed)  # previously failed dates: one retry each per run, before new dates
    attempted: list[str] = []
    successes: list[str] = []
    empty_dates: list[str] = []
    skipped_before_window: list[str] = []
    rows_added = 0
    stop_reason = "range_finished"
    consecutive_failures = 0
    requests_made = 0
    df = existing

    def snapshot() -> dict:
        saved = dict(state)
        saved.update(
            {
                "next_date": cursor.isoformat(),
                "done": cursor > range_to and not failed,
                "failed_dates": failed,
                "last_run_at_jst": datetime.now(daily.JST).isoformat(),
                "store_type": store_type,
                "store_dir": str(store),
                "output": str(raw_path),
                "last_run_dates": attempted,
                "last_run_successful_dates": successes,
                "last_run_empty_dates": empty_dates,
                "last_run_skipped_before_window": skipped_before_window,
                "last_run_stop_reason": stop_reason,
                "rows_added_last_run_before_dedup": rows_added,
                "rows_saved_total": int(len(df)),
                "secret_included": False,
            }
        )
        return saved

    def save_state() -> None:
        daily.atomic_write_text(state_path, json.dumps(snapshot(), ensure_ascii=False, indent=2))

    try:
        while requests_made < args.max_dates:
            from_range = False
            if pending:
                target = pending.pop(0)
            elif cursor <= range_to:
                target = cursor.isoformat()
                from_range = True
                cursor += timedelta(days=1)
            else:
                stop_reason = "range_finished"
                break

            if requests_made:
                time.sleep(SLEEP_SECONDS)
            print("fetch", target)
            result = daily.fetch_one_date(target, api_key, max_attempts=3)
            requests_made += 1
            attempted.append(target)

            if result["ok"]:
                new_rows = []
                for rec in result["records"]:
                    row = dict(rec)
                    row["query_date"] = target
                    row["ticker"] = daily.normalize_code_to_ticker(row.get("Code", ""))
                    if row["ticker"] in usable:
                        new_rows.append(row)
                if new_rows:
                    merged = merge_rows(df, new_rows, daily)  # may stop the run before any write
                    if daily.file_fingerprint(raw_path) != expected_fingerprint:
                        raise daily.StoreChangedError(
                            f"{raw_path} changed while the backfill was running. "
                            "Nothing more was written."
                        )
                    daily.atomic_write_csv(merged, raw_path)
                    expected_fingerprint = daily.file_fingerprint(raw_path)
                    got = set(merged["ticker"].astype(str)) if "ticker" in merged.columns else set()
                    cov = universe.copy()
                    cov["has_any_financial_summary"] = cov["ticker"].isin(got)
                    daily.atomic_write_csv(cov, coverage_path)
                    df = merged
                    rows_added += len(new_rows)
                else:
                    empty_dates.append(target)
                failed.pop(target, None)
                successes.append(target)
                consecutive_failures = 0
                print("  records:", len(result["records"]), "saved:", len(new_rows))
            else:
                reason = result["reason"]
                err = result["error"] or {}
                if reason == "OUT_OF_SUBSCRIPTION_DATE_RANGE":
                    w_start = err.get("subscription_start")
                    w_end = err.get("subscription_end")
                    state["window_start_detected"] = w_start
                    state["window_end_detected"] = w_end
                    print("  out of plan window:", w_start, "~", w_end)
                    if w_start and target < w_start:
                        # Too old: gone for good. Jump to the first date in the window.
                        skipped_before_window.append(target)
                        failed = {d: v for d, v in failed.items() if d >= w_start}
                        pending = [d for d in pending if d >= w_start]
                        cursor = max(cursor, parse_iso(w_start, "window start"))
                        save_state()
                        continue
                    if from_range:
                        cursor = parse_iso(target, "target")  # not consumed
                    stop_reason = "beyond_window_end"
                    break
                if reason in STOP_ON_REASONS:
                    if from_range:
                        cursor = parse_iso(target, "target")
                    stop_reason = f"stopped_on_{reason}"
                    break
                failed[target] = {"reason": reason, "detail": err}
                consecutive_failures += 1
                print("  failed:", reason, err)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    stop_reason = "consecutive_failures"
                    break
            save_state()
        else:
            stop_reason = "max_dates_reached"
    except BaseException as exc:
        stop_reason = f"aborted: {type(exc).__name__}"
        # The state file is deliberately not advanced for the date being processed.
        try:
            daily.append_line(log_path, json.dumps(snapshot(), ensure_ascii=False) + "\n")
        except daily.StoreError:
            pass  # the original error is the one worth reporting
        raise

    final = snapshot()
    daily.atomic_write_text(state_path, json.dumps(final, ensure_ascii=False, indent=2))
    daily.append_line(log_path, json.dumps(final, ensure_ascii=False) + "\n")

    print("\n== done ==")
    print("stop_reason:", stop_reason)
    print(
        "requests:", requests_made, " successes:", len(successes), " failed_pending:", len(failed)
    )
    print("rows_added_before_dedup:", rows_added, " rows_saved_total:", len(df))
    print("next_date:", final["next_date"], " done:", final["done"])
    print("window:", final["window_start_detected"], "~", final["window_end_detected"])
    return 0


def main() -> int:
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline tests for scripts/backfill_jquants_financial_summary.py.

No network, no real API, no real data directory: the API call, ``time.sleep`` and every
path are replaced or redirected to ``tmp_path``.
"""

from __future__ import annotations

import contextlib
import csv
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
DAILY_PATH = SCRIPTS / "fetch_jquants_financial_summary_daily.py"
BACKFILL_PATH = SCRIPTS / "backfill_jquants_financial_summary.py"
API_KEY = "test-key-not-real"

MANAGED_ENV = (
    "JP_STOCK_ANALYSIS_PROJECT_DIR",
    "JP_STOCK_ANALYSIS_UNIVERSE_FILE",
    "JQUANTS_EXTERNAL_STORE_DIRS",
    "JQUANTS_EXTERNAL_STORE_DIR",
    "JQUANTS_API_KEY",
    "JQUANTS_FIN_SUMMARY_MAX_DATES_PER_RUN",
)
RAW = "financial_summary_raw.csv"
COVERAGE = "financial_summary_coverage.csv"
DAILY_STATE = "financial_summary_auto_state.json"
DAILY_RUNS = "financial_summary_auto_runs.jsonl"
BF_STATE = "financial_summary_backfill_state.json"
BF_RUNS = "financial_summary_backfill_runs.jsonl"
RAW_HEADER = ["DiscDate", "DiscTime", "Code", "DiscNo", "DocType", "Sales", "query_date", "ticker"]


def ok(records=()):
    return {"ok": True, "records": list(records), "attempt": 1, "reason": "OK", "error": None}


def fail(reason="HTTP_ERROR", **error):
    return {
        "ok": False,
        "records": [],
        "attempt": 3,
        "reason": reason,
        "error": error or {"code": 500},
    }


def out_of_window(start, end):
    return {
        "ok": False,
        "records": [],
        "attempt": 1,
        "reason": "OUT_OF_SUBSCRIPTION_DATE_RANGE",
        "error": {
            "type": "HTTPError",
            "code": 400,
            "subscription_start": start,
            "subscription_end": end,
        },
    }


def record(code="72030", disc_date="2026-01-05", disc_no="1", **extra):
    row = {
        "DiscDate": disc_date,
        "DiscTime": "15:00:00",
        "Code": code,
        "DiscNo": disc_no,
        "DocType": "1QFinancialStatements_Consolidated_JP",
        "Sales": "1000",
    }
    row.update(extra)
    return row


class FakeApi:
    def __init__(self):
        self.script = {}  # date -> result | list of results (consumed in order) | callable
        self.rule = None  # optional callable(date) -> result for dates not in script
        self.calls = []
        self.sleeps = []

    def __call__(self, target_date, api_key, max_attempts=3):
        assert api_key == API_KEY
        self.calls.append(target_date)
        item = self.script.get(target_date)
        if isinstance(item, list):
            item = item.pop(0) if len(item) > 1 else item[0]
        if callable(item):
            return item()
        if item is not None:
            return item
        if self.rule:
            return self.rule(target_date)
        return ok()

    def sleep(self, seconds):
        self.sleeps.append(seconds)


@dataclass
class Env:
    daily: object
    bf: object
    api: FakeApi
    tmp: Path
    project: Path
    store: Path

    def run(self, *argv):
        return self.bf.run(list(argv), daily=self.daily)

    def bf_state(self):
        return json.loads((self.store / BF_STATE).read_text())

    def read_raw(self):
        with (self.store / RAW).open(newline="", encoding="utf-8") as f:
            return list(csv.reader(f))

    def write_raw(self, rows, header=None):
        with (self.store / RAW).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header or RAW_HEADER)
            w.writerows(rows)

    def snapshot(self):
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(self.store.iterdir())
            if p.is_file()
        }


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forbid_network(*args, **kwargs):  # pragma: no cover - only runs on regression
    raise AssertionError("network access attempted in an offline test")


@pytest.fixture
def make_env(tmp_path, monkeypatch):
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ))
        for key in MANAGED_ENV:
            os.environ.pop(key, None)
        os.environ["JQUANTS_API_KEY"] = API_KEY
        monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)

        def factory(name="project"):
            project = tmp_path / name
            store = project / "data" / "jquants_price_store"
            store.mkdir(parents=True)
            (store / "topix1000_usable_tickers.csv").write_text(
                "ticker,name\n7203,Toyota\n6758,Sony\n", encoding="utf-8"
            )
            os.environ["JP_STOCK_ANALYSIS_PROJECT_DIR"] = str(project)
            daily = _load(DAILY_PATH, f"daily_under_test_{name}")
            bf = _load(BACKFILL_PATH, f"backfill_under_test_{name}")
            api = FakeApi()
            monkeypatch.setattr(daily, "fetch_one_date", api)
            monkeypatch.setattr(daily.time, "sleep", api.sleep)
            return Env(daily, bf, api, tmp_path, project, store)

        yield factory


@pytest.fixture
def env(make_env):
    return make_env()


# --------------------------------------------------------------------------
# order, cap, resume
# --------------------------------------------------------------------------
def test_dates_are_fetched_oldest_first_and_resume_from_saved_cursor(env):
    env.run("--from", "2026-01-01", "--to", "2026-01-10", "--max-dates", "4")
    assert env.api.calls == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    state = env.bf_state()
    assert state["next_date"] == "2026-01-05"
    assert state["done"] is False
    assert state["last_run_stop_reason"] == "max_dates_reached"
    assert env.api.sleeps == [20, 20, 20]  # paused between requests, never really slept

    env.api.calls.clear()
    env.run("--max-dates", "4")  # no range arguments needed to resume
    assert env.api.calls == ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]

    env.api.calls.clear()
    env.run("--max-dates", "4")
    assert env.api.calls == ["2026-01-09", "2026-01-10"]
    state = env.bf_state()
    assert state["done"] is True
    assert state["last_run_stop_reason"] == "range_finished"

    env.api.calls.clear()
    env.run()
    assert env.api.calls == []  # finished plans make no request


def test_no_date_is_requested_twice_in_a_run(env):
    env.run("--from", "2026-01-01", "--to", "2026-01-20", "--max-dates", "20")
    assert len(env.api.calls) == len(set(env.api.calls)) == 20


# --------------------------------------------------------------------------
# plan window (facts reported by the API)
# --------------------------------------------------------------------------
def test_dates_older_than_the_window_are_skipped_and_the_run_continues(env):
    env.api.rule = lambda d: out_of_window("2026-01-04", "2026-06-01") if d < "2026-01-04" else ok()
    env.run("--from", "2026-01-01", "--to", "2026-01-08", "--max-dates", "5")
    assert env.api.calls == ["2026-01-01", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"]
    state = env.bf_state()
    assert state["last_run_skipped_before_window"] == ["2026-01-01"]
    assert state["window_start_detected"] == "2026-01-04"
    assert state["window_end_detected"] == "2026-06-01"
    assert state["next_date"] == "2026-01-08"
    assert state["failed_dates"] == {}


def test_a_date_beyond_the_window_end_stops_without_consuming_it(env):
    env.api.rule = lambda d: out_of_window("2025-01-01", "2026-01-03") if d > "2026-01-03" else ok()
    env.run("--from", "2026-01-01", "--to", "2026-01-10")
    assert env.api.calls == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    state = env.bf_state()
    assert state["last_run_stop_reason"] == "beyond_window_end"
    assert state["next_date"] == "2026-01-04"  # tried again on the next run
    assert state["done"] is False


# --------------------------------------------------------------------------
# failures
# --------------------------------------------------------------------------
def test_failed_date_is_kept_then_retried_first_on_the_next_run(env):
    env.api.script["2026-01-02"] = [
        fail("HTTP_ERROR", code=503),
        ok([record(disc_date="2026-01-02")]),
    ]
    env.run("--from", "2026-01-01", "--to", "2026-01-04")
    assert env.api.calls == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    state = env.bf_state()
    assert list(state["failed_dates"]) == ["2026-01-02"]
    assert state["done"] is False  # a pending failure keeps the plan open

    env.api.calls.clear()
    env.run()
    assert env.api.calls == ["2026-01-02"]
    state = env.bf_state()
    assert state["failed_dates"] == {}
    assert state["done"] is True
    assert [r[-2] for r in env.read_raw()[1:]] == ["2026-01-02"]


def test_a_failing_date_is_retried_once_per_run_not_in_a_loop(env):
    env.api.script["2026-01-01"] = fail("HTTP_ERROR", code=500)
    env.run("--from", "2026-01-01", "--to", "2026-01-03")
    env.api.calls.clear()
    env.run("--max-dates", "3")
    assert env.api.calls == ["2026-01-01"]  # retried once; nothing else left in the range


def test_three_consecutive_failures_stop_the_run(env):
    env.api.rule = lambda d: fail("HTTP_ERROR", code=500)
    env.run("--from", "2026-01-01", "--to", "2026-01-10")
    assert env.api.calls == ["2026-01-01", "2026-01-02", "2026-01-03"]
    state = env.bf_state()
    assert state["last_run_stop_reason"] == "consecutive_failures"
    assert list(state["failed_dates"]) == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert state["next_date"] == "2026-01-04"


def test_plan_not_available_stops_immediately_and_keeps_the_date(env):
    env.api.rule = lambda d: fail("PLAN_NOT_AVAILABLE", code=403)
    env.run("--from", "2026-01-01", "--to", "2026-01-10")
    assert env.api.calls == ["2026-01-01"]
    state = env.bf_state()
    assert state["last_run_stop_reason"] == "stopped_on_PLAN_NOT_AVAILABLE"
    assert state["next_date"] == "2026-01-01"


# --------------------------------------------------------------------------
# saved data: same format as the daily script, existing rows untouched
# --------------------------------------------------------------------------
def test_saved_rows_are_identical_to_what_the_daily_script_saves(make_env):
    records = [
        record(code="72030", disc_no="1"),
        record(code="67580", disc_no="2"),
        record(code="99990", disc_no="3"),  # not in the universe: dropped by both
    ]
    daily_env = make_env("daily_project")
    daily_env.api.script["2026-01-05"] = ok(records)
    (daily_env.store / DAILY_STATE).write_text(
        json.dumps({"last_successful_query_date": "2026-01-04", "failed_dates_carryover": {}})
    )
    # start = watermark + 1 day = 2026-01-05; a cap of 1 keeps today's real date out of it
    with mock.patch.dict(os.environ, {"JQUANTS_FIN_SUMMARY_MAX_DATES_PER_RUN": "1"}):
        daily_env.daily.main()
    assert daily_env.api.calls == ["2026-01-05"]

    bf_env = make_env("backfill_project")
    bf_env.api.script["2026-01-05"] = ok(records)
    bf_env.run("--from", "2026-01-05", "--to", "2026-01-05")

    assert (bf_env.store / RAW).read_bytes() == (daily_env.store / RAW).read_bytes()
    assert (bf_env.store / COVERAGE).read_bytes() == (daily_env.store / COVERAGE).read_bytes()
    rows = bf_env.read_raw()
    assert rows[0][-2:] == ["query_date", "ticker"]
    assert [r[-1] for r in rows[1:]] == ["7203", "6758"]


def test_existing_rows_are_unchanged_new_rows_are_appended_and_duplicates_skipped(env):
    existing = [
        ["2026-07-16", "15:00:00", "72030", "10", "FY", "1", "2026-07-16", "7203"],
        ["2026-07-17", "15:00:00", "67580", "11", "FY", "2", "2026-07-17", "6758"],
    ]
    env.write_raw(existing)
    before_lines = (env.store / RAW).read_text().splitlines()

    env.api.script["2026-01-05"] = ok(
        [
            record(code="72030", disc_date="2026-01-05", disc_no="20"),
            # same dedupe key as an existing row -> must not be added again
            record(code="72030", disc_date="2026-07-16", disc_no="10", DocType="FY", Sales="1"),
        ]
    )
    env.run("--from", "2026-01-05", "--to", "2026-01-05")

    after_lines = (env.store / RAW).read_text().splitlines()
    assert after_lines[: len(before_lines)] == before_lines  # header + old rows, byte for byte
    assert len(after_lines) == len(before_lines) + 1
    assert after_lines[-1].startswith("2026-01-05,15:00:00,72030,20,")
    assert env.bf_state()["rows_saved_total"] == 3


def test_a_field_that_is_not_a_raw_column_stops_the_run_before_any_write(env, capsys):
    env.write_raw([["2026-07-16", "15:00:00", "72030", "10", "FY", "1", "2026-07-16", "7203"]])
    before = env.snapshot()
    env.api.script["2026-01-05"] = ok([record(NewField="x")])
    with pytest.raises(SystemExit) as excinfo:
        env.run("--from", "2026-01-05", "--to", "2026-01-06")
    assert excinfo.value.code == 4  # input problem, not "busy" or "drive down"
    err = capsys.readouterr().err
    assert "SCHEMA_CHANGE" in err and "NewField" in err
    assert (env.store / RAW).read_bytes() is not None
    after = env.snapshot()
    assert after[RAW] == before[RAW]  # raw untouched
    assert BF_STATE not in after  # the date was not marked as done
    assert not list(env.store.glob("*.tmp"))
    # the abort is recorded in the run log
    last = json.loads((env.store / BF_RUNS).read_text().splitlines()[-1])
    assert last["last_run_stop_reason"].startswith("aborted")


# --------------------------------------------------------------------------
# isolation from the daily job's artifacts
# --------------------------------------------------------------------------
def test_daily_state_and_run_log_are_never_touched(env):
    (env.store / DAILY_STATE).write_text(
        json.dumps({"last_successful_query_date": "2026-08-04", "failed_dates_carryover": {}})
    )
    (env.store / DAILY_RUNS).write_text('{"x": 1}\n')
    before = env.snapshot()
    env.api.script["2026-01-05"] = ok([record()])
    env.run("--from", "2026-01-05", "--to", "2026-01-06")
    after = env.snapshot()
    assert after[DAILY_STATE] == before[DAILY_STATE]
    assert after[DAILY_RUNS] == before[DAILY_RUNS]
    assert {BF_STATE, BF_RUNS, RAW, COVERAGE} <= set(after)
    assert not list(env.store.glob("*.tmp"))


def test_one_time_backup_of_the_raw_file_is_made_only_when_the_plan_starts(env):
    env.write_raw([["2026-07-16", "15:00:00", "72030", "10", "FY", "1", "2026-07-16", "7203"]])
    original = (env.store / RAW).read_bytes()
    env.run("--from", "2026-01-01", "--to", "2026-01-06", "--max-dates", "2")
    backups = list(env.store.glob(f"{RAW}.bak_before_backfill_*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    env.run("--max-dates", "2")
    assert len(list(env.store.glob(f"{RAW}.bak_before_backfill_*"))) == 1


# --------------------------------------------------------------------------
# facts before acting, argument handling, interruption, locking
# --------------------------------------------------------------------------
def test_dry_run_prints_the_facts_and_changes_nothing(env, capsys):
    env.write_raw([["2026-07-16", "15:00:00", "72030", "10", "FY", "1", "2026-07-16", "7203"]])
    os.environ.pop("JQUANTS_API_KEY")  # a dry run needs no key
    env.api.rule = lambda d: pytest.fail("dry-run must not call the API")
    before = env.snapshot()
    assert env.run("--to", "2026-07-15", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "'min_query_date': '2026-07-16'" in out
    assert "remaining_dates=" in out
    assert "dry-run" in out
    assert env.snapshot() == before
    assert env.api.calls == []


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "--to is required"),
        (["--to", "2026-13-40"], "not a YYYY-MM-DD date"),
        (["--from", "2026-02-01", "--to", "2026-01-01"], "must not be later"),
        (["--to", "2026-01-01", "--max-dates", "0"], "--max-dates must be >= 1"),
    ],
)
def test_invalid_arguments_are_clean_exits_without_requests(env, argv, message):
    with pytest.raises(SystemExit) as excinfo:
        env.run(*argv)
    assert message in str(excinfo.value)
    assert env.api.calls == []
    assert not (env.store / BF_STATE).exists()


def test_changing_the_saved_plan_requires_deleting_the_state(env):
    env.run("--from", "2026-01-01", "--to", "2026-01-03", "--max-dates", "1")
    with pytest.raises(SystemExit) as excinfo:
        env.run("--to", "2026-02-01")
    assert "differs from the saved plan" in str(excinfo.value)
    assert env.bf_state()["range_to"] == "2026-01-03"


def test_an_interrupted_run_keeps_finished_dates_and_resumes_after_them(env):
    def boom():
        raise KeyboardInterrupt

    env.api.script["2026-01-01"] = ok([record(disc_date="2026-01-01", disc_no="1")])
    env.api.script["2026-01-02"] = boom
    with pytest.raises(KeyboardInterrupt):
        env.run("--from", "2026-01-01", "--to", "2026-01-04")
    state = env.bf_state()
    assert state["next_date"] == "2026-01-02"  # 01-02 was not recorded as done
    assert [r[-2] for r in env.read_raw()[1:]] == ["2026-01-01"]
    assert not list(env.store.glob("*.tmp"))

    env.api.script["2026-01-02"] = ok([record(disc_date="2026-01-02", disc_no="2")])
    env.api.calls.clear()
    env.run()
    assert env.api.calls == ["2026-01-02", "2026-01-03", "2026-01-04"]
    assert [r[-2] for r in env.read_raw()[1:]] == ["2026-01-01", "2026-01-02"]


def test_a_second_writer_cannot_run_while_the_store_lock_is_held(env, capsys):
    lock_path = env.daily.lock_file_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit) as excinfo:
            env.run("--from", "2026-01-01", "--to", "2026-01-02")
    assert excinfo.value.code == 75
    assert "BUSY" in capsys.readouterr().err
    assert env.api.calls == []
    assert not (env.store / BF_STATE).exists()


def test_outputs_never_contain_the_api_key(env, capsys):
    env.api.script["2026-01-05"] = ok([record()])
    env.run("--from", "2026-01-05", "--to", "2026-01-05")
    assert API_KEY not in capsys.readouterr().out
    for p in env.store.iterdir():
        if p.is_file():
            assert API_KEY not in p.read_text(encoding="utf-8")
    assert env.bf_state()["secret_included"] is False


# ==========================================================================
# Recovery work: shared lock, refusal on damaged stores, mount failures
# ==========================================================================
def real_like_daily_store(env):
    """Raw CSV + daily state + run log of a healthy store (state at 2026-08-04)."""
    env.write_raw(
        [["2026-08-04", "15:00:00", "72030", "2026080401", "FY", "1", "2026-08-04", "7203"]]
    )
    ts = "2026-11-04T08:19:21.084263+09:00"
    (env.store / DAILY_STATE).write_text(
        json.dumps(
            {
                "last_run_at_jst": ts,
                "last_successful_query_date": "2026-08-04",
                "latest_disc_date_saved": "2026-08-04",
                "rows_saved_total": 1,
                "failed_dates_carryover": {},
            }
        )
    )
    (env.store / DAILY_RUNS).write_text(
        json.dumps({"last_run_at_jst": ts, "last_successful_query_date": "2026-08-04"}) + "\n"
    )


def test_the_daily_job_cannot_start_while_the_backfill_runs_and_vice_versa(env, capsys):
    # backfill running -> daily refuses
    seen = {}

    def during_backfill(d):
        # at this moment the backfill holds the lock: a daily process must be turned away
        try:
            env.daily.StoreLock("daily").acquire()
            seen["daily_got_lock"] = True
        except env.daily.BusyError as exc:
            seen["daily_busy"] = str(exc)
        return ok()

    env.api.rule = during_backfill
    env.run("--from", "2026-01-01", "--to", "2026-01-01")
    assert "role=backfill" in seen["daily_busy"] and "daily_got_lock" not in seen

    # daily running -> backfill refuses (and the lock was released by the first run)
    holder = env.daily.StoreLock("daily")
    holder.acquire()
    try:
        env.api.calls.clear()
        with pytest.raises(SystemExit) as excinfo:
            env.run("--from", "2026-02-01", "--to", "2026-02-02")
    finally:
        holder.release()
    assert excinfo.value.code == 75
    assert "role=daily" in capsys.readouterr().err
    assert env.api.calls == []


def test_a_rolled_back_daily_store_is_refused_unless_explicitly_allowed(env, capsys):
    real_like_daily_store(env)
    env.write_raw(
        [["2026-07-15", "15:00:00", "72030", "2026070101", "FY", "1", "2026-07-15", "7203"]]
    )
    (env.store / DAILY_STATE).write_text(
        json.dumps(
            {
                "last_run_at_jst": "2026-10-07T08:00:00+09:00",
                "last_successful_query_date": "2026-07-15",
                "latest_disc_date_saved": "2026-07-15",
                "rows_saved_total": 1,
            }
        )
    )
    before = env.snapshot()
    with pytest.raises(SystemExit) as excinfo:
        env.run("--from", "2026-01-01", "--to", "2026-01-03")
    assert excinfo.value.code == 5
    assert "PROGRESS_REGRESSION" in capsys.readouterr().err
    assert env.api.calls == []
    assert env.snapshot() == before

    env.run("--from", "2026-01-01", "--to", "2026-01-03", "--allow-progress-regression")
    assert env.api.calls == ["2026-01-01", "2026-01-02", "2026-01-03"]


def test_an_unusable_daily_state_stops_the_backfill_without_changes(env, capsys):
    real_like_daily_store(env)
    (env.store / DAILY_STATE).write_text("{broken")
    before = env.snapshot()
    with pytest.raises(SystemExit) as excinfo:
        env.run("--from", "2026-01-01", "--to", "2026-01-03")
    assert excinfo.value.code == 4
    assert "STATE_UNREADABLE" in capsys.readouterr().err
    assert env.api.calls == [] and env.snapshot() == before


def test_an_unusable_backfill_state_stops_the_run_without_changes(env, capsys):
    (env.store / BF_STATE).write_text("{broken")
    with pytest.raises(SystemExit) as excinfo:
        env.run("--to", "2026-01-03")
    assert excinfo.value.code == 4
    assert env.api.calls == []
    assert (env.store / BF_STATE).read_text() == "{broken"  # not reset


def test_a_dangling_store_symlink_stops_the_backfill_and_creates_nothing(env, tmp_path, capsys):
    shutil.rmtree(env.store)
    unmounted = tmp_path / "not_mounted" / "jquants_price_store"
    env.store.symlink_to(unmounted)
    with pytest.raises(SystemExit) as excinfo:
        env.run("--from", "2026-01-01", "--to", "2026-01-02")
    assert excinfo.value.code == 3
    assert "STORE_UNAVAILABLE" in capsys.readouterr().err
    assert not unmounted.parent.exists()
    assert env.api.calls == []


def test_drive_errors_during_the_run_are_reported_and_leave_the_raw_file_alone(
    env, monkeypatch, capsys
):
    real_like_daily_store(env)
    before_raw = (env.store / RAW).read_bytes()
    env.api.script["2026-01-05"] = ok([record(disc_date="2026-01-05")])
    real_replace = os.replace

    def flaky_replace(src, dst, *a, **k):
        if str(dst).endswith(RAW):
            raise OSError(errno.ENODEV, os.strerror(errno.ENODEV))
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(SystemExit) as excinfo:
        env.run("--from", "2026-01-05", "--to", "2026-01-05")
    assert excinfo.value.code == 3
    assert "STORE_UNAVAILABLE" in capsys.readouterr().err
    assert (env.store / RAW).read_bytes() == before_raw
    assert not list(env.store.glob("*.tmp"))
    assert not (env.store / BF_STATE).exists()  # the date was not recorded as done


def test_a_writer_that_ignores_the_lock_is_not_overwritten_by_the_backfill(env, capsys):
    real_like_daily_store(env)

    def intruder():
        with (env.store / RAW).open("a") as f:
            f.write("2026080501,72030,2026-08-05,FY,2026-08-05,7203\n")
        return ok([record(disc_date="2026-01-05")])

    env.api.script["2026-01-05"] = intruder
    with pytest.raises(SystemExit) as excinfo:
        env.run("--from", "2026-01-05", "--to", "2026-01-05")
    assert excinfo.value.code == 6
    assert "2026080501" in (env.store / RAW).read_text()
    assert "STORE_CHANGED_DURING_RUN" in capsys.readouterr().err


def test_an_old_daily_script_is_rejected_with_a_clear_message(env):
    old_daily = types.SimpleNamespace()  # no StoreLock: the pre-recovery script
    with pytest.raises(SystemExit) as excinfo:
        env.bf.run(["--to", "2026-01-01"], daily=old_daily)
    assert "older version" in str(excinfo.value)


def test_dry_run_reports_lock_progress_check_and_mount(env, capsys):
    real_like_daily_store(env)
    holder = env.daily.StoreLock("daily")
    holder.acquire()
    try:
        assert env.run("--to", "2026-07-15", "--dry-run") == 0
    finally:
        holder.release()
    out = capsys.readouterr().out
    assert "HELD by another process" in out
    assert "progress_check: OK" in out
    assert "store_mount:" in out
    assert not (env.store / BF_STATE).exists()

"""Offline regression tests for scripts/fetch_jquants_financial_summary_daily.py.

No network, no real J-Quants API, no real data directories: the API call, the
clock, ``time.sleep`` and every file path are replaced or redirected to
``tmp_path``.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import importlib.util
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "fetch_jquants_financial_summary_daily.py"
)
TODAY = date(2026, 11, 17)
API_KEY = "test-key-not-real"

# Environment variables the script reads; cleared so the host environment
# (and any real .env) can never leak into a test.
MANAGED_ENV = (
    "JP_STOCK_ANALYSIS_PROJECT_DIR",
    "JP_STOCK_ANALYSIS_UNIVERSE_FILE",
    "JQUANTS_EXTERNAL_STORE_DIRS",
    "JQUANTS_EXTERNAL_STORE_DIR",
    "JQUANTS_API_KEY",
    "JQUANTS_FIN_SUMMARY_MAX_DATES_PER_RUN",
)

# Contract: literal file names and state keys (order included) of the existing
# output format. Intentionally written out instead of imported from the script.
RAW_FILE = "financial_summary_raw.csv"
COVERAGE_FILE = "financial_summary_coverage.csv"
STATE_FILE = "financial_summary_auto_state.json"
RUNS_FILE = "financial_summary_auto_runs.jsonl"
STATE_KEYS = [
    "endpoint",
    "query_mode",
    "last_run_at_jst",
    "store_type",
    "store_dir",
    "target_dates_last_run",
    "successful_query_dates_last_run",
    "last_successful_query_date",
    "rows_added_last_run_before_dedup",
    "rows_saved_total",
    "unique_tickers_saved",
    "coverage_missing_count",
    "latest_disc_date_saved",
    "failed_dates_last_run",
    "failed_dates_count_last_run",
    "failed_dates_carryover",
    "failed_dates_carryover_count",
    "skipped_dates_last_run",
    "skipped_dates_count_last_run",
    "empty_dates_last_run",
    "run_results",
    "subscription_start_detected",
    "subscription_end_detected",
    "output",
    "coverage_output",
    "api_key_status",
    "secret_included",
    "note",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def load_script():
    spec = importlib.util.spec_from_file_location(
        "fetch_jquants_financial_summary_daily_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ok(records=()):
    return {
        "ok": True,
        "records": list(records),
        "attempt": 1,
        "reason": "OK",
        "error": None,
    }


def fail(reason="HTTP_ERROR", **error):
    return {
        "ok": False,
        "records": [],
        "attempt": 3,
        "reason": reason,
        "error": error or {"code": 500},
    }


def record(code="72030", disc_date="2026-11-05", disc_no="1", **extra):
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
    """Stands in for fetch_one_date: scripted per date, records every call."""

    def __init__(self):
        self.script = {}
        self.calls = []
        self.sleeps = []

    def __call__(self, target_date, api_key, max_attempts=3):
        assert api_key == API_KEY
        self.calls.append(target_date)
        return self.script.get(target_date, ok())

    def sleep(self, seconds):
        self.sleeps.append(seconds)


@dataclass
class Env:
    mod: object
    api: FakeApi
    tmp: Path
    project: Path
    local_store: Path

    def run(self):
        self.mod.main()

    def state(self, store=None):
        return json.loads(((store or self.local_store) / STATE_FILE).read_text())

    def write_state(self, store=None, **state):
        store = store or self.local_store
        store.mkdir(parents=True, exist_ok=True)
        (store / STATE_FILE).write_text(json.dumps(state), encoding="utf-8")

    def write_raw(self, rows, store=None, header=None):
        store = store or self.local_store
        header = header or [
            "DiscNo",
            "Code",
            "DiscDate",
            "DocType",
            "query_date",
            "ticker",
        ]
        with (store / RAW_FILE).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def read_raw(self, store=None):
        with ((store or self.local_store) / RAW_FILE).open(newline="", encoding="utf-8") as f:
            return list(csv.reader(f))


def failed_entry(reason="HTTP_ERROR", marker="old"):
    return {"reason": reason, "detail": {"marker": marker}}


def _forbid_network(*args, **kwargs):  # pragma: no cover - only runs on regression
    raise AssertionError("network access attempted in an offline test")


@pytest.fixture
def env(tmp_path, monkeypatch):
    with mock.patch.dict(os.environ):
        for key in MANAGED_ENV:
            os.environ.pop(key, None)
        project = tmp_path / "project"
        local_store = project / "data" / "jquants_price_store"
        local_store.mkdir(parents=True)
        (local_store / "topix1000_usable_tickers.csv").write_text(
            "ticker,name\n7203,Toyota\n6758,Sony\n", encoding="utf-8"
        )
        os.environ["JP_STOCK_ANALYSIS_PROJECT_DIR"] = str(project)
        os.environ["JQUANTS_API_KEY"] = API_KEY

        mod = load_script()
        api = FakeApi()
        monkeypatch.setattr(mod, "fetch_one_date", api)
        monkeypatch.setattr(mod.time, "sleep", api.sleep)
        monkeypatch.setattr(mod, "get_today_jst", lambda: TODAY)
        monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)
        yield Env(mod, api, tmp_path, project, local_store)


def assert_under(base: Path, path: Path):
    assert Path(path).resolve().is_relative_to(base.resolve()), path


# --------------------------------------------------------------------------
# 1. fresh state
# --------------------------------------------------------------------------
def test_01_fresh_state_builds_dates_from_default_start(env):
    dates = env.mod.build_target_dates(
        env.local_store / RAW_FILE,
        env.local_store / STATE_FILE,
        today=date(2024, 4, 8),
    )
    assert dates == ["2024-04-05", "2024-04-06", "2024-04-07", "2024-04-08"]


def test_01_fresh_state_main_respects_cap_and_records_progress(env, monkeypatch):
    monkeypatch.setenv("JQUANTS_FIN_SUMMARY_MAX_DATES_PER_RUN", "3")
    env.run()

    expected = ["2024-04-05", "2024-04-06", "2024-04-07"]
    assert env.api.calls == expected
    assert env.api.sleeps == [20, 20, 20]  # sleeps were intercepted, not slept
    state = env.state()
    assert state["target_dates_last_run"] == expected
    assert state["successful_query_dates_last_run"] == expected
    assert state["last_successful_query_date"] == "2024-04-07"
    assert state["failed_dates_carryover"] == {}
    assert_under(env.tmp, Path(state["store_dir"]))


# --------------------------------------------------------------------------
# 2. resume from existing watermark
# --------------------------------------------------------------------------
def test_02_resumes_after_last_successful_query_date(env):
    env.write_state(last_successful_query_date="2026-11-10", failed_dates_carryover={})
    dates = env.mod.build_target_dates(env.local_store / RAW_FILE, env.local_store / STATE_FILE)
    assert dates == [f"2026-11-{d}" for d in range(11, 18)]


def test_02_start_uses_later_of_watermark_and_saved_disc_date(env):
    # Existing contract: start = max(latest saved DiscDate, watermark) + 1 day.
    env.write_state(last_successful_query_date="2026-11-10", failed_dates_carryover={})
    env.write_raw([["1", "72030", "2026-11-13", "FY", "2026-11-13", "7203"]])
    dates = env.mod.build_target_dates(env.local_store / RAW_FILE, env.local_store / STATE_FILE)
    assert dates == ["2026-11-14", "2026-11-15", "2026-11-16", "2026-11-17"]


def test_02_main_resumes_and_advances_watermark(env):
    env.write_state(last_successful_query_date="2026-11-14", failed_dates_carryover={})
    env.run()
    assert env.api.calls == ["2026-11-15", "2026-11-16", "2026-11-17"]
    assert env.state()["last_successful_query_date"] == "2026-11-17"


# --------------------------------------------------------------------------
# 3. previous failures are retried
# --------------------------------------------------------------------------
def test_03_failed_dates_are_retried_sorted_and_deduplicated(env):
    env.write_state(
        last_successful_query_date="2026-11-15",
        failed_dates_carryover={
            "2026-11-16": failed_entry(),  # also an incremental date -> once only
            "2026-11-05": failed_entry(),
            "2026-11-02": failed_entry(),
        },
    )
    dates = env.mod.build_target_dates(env.local_store / RAW_FILE, env.local_store / STATE_FILE)
    assert dates == ["2026-11-02", "2026-11-05", "2026-11-16", "2026-11-17"]
    assert len(dates) == len(set(dates))


def test_03_non_retryable_reasons_are_not_retried(env):
    env.write_state(
        last_successful_query_date="2026-11-17",
        failed_dates_carryover={
            "2026-11-01": failed_entry("OUT_OF_SUBSCRIPTION_DATE_RANGE"),
            "2026-11-02": failed_entry("PLAN_NOT_AVAILABLE"),
            "2026-11-03": {"type": "PLAN_NOT_AVAILABLE"},  # legacy "type" key
            "2026-11-04": {"type": "SomeOtherError"},
            "2026-11-05": failed_entry("HTTP_ERROR"),
        },
    )
    dates = env.mod.build_target_dates(env.local_store / RAW_FILE, env.local_store / STATE_FILE)
    assert dates == ["2026-11-04", "2026-11-05"]


# --------------------------------------------------------------------------
# 4. failed dates cut by the per-run cap are kept
# --------------------------------------------------------------------------
def october(n):
    return f"2026-10-{n:02d}"


def test_04_cap_does_not_drop_unprocessed_failed_dates(env):
    env.write_state(
        last_successful_query_date="2026-11-17",
        failed_dates_carryover={october(d): failed_entry() for d in range(1, 26)},
    )
    for d in range(1, 21):
        env.api.script[october(d)] = fail("HTTP_ERROR", code=503, marker="new")

    env.run()  # default cap = 20

    assert env.api.calls == [october(d) for d in range(1, 21)]
    assert len(env.api.calls) == len(set(env.api.calls))
    state = env.state()
    carry = state["failed_dates_carryover"]
    assert list(carry) == [october(d) for d in range(1, 26)]
    assert state["failed_dates_carryover_count"] == 25
    for d in range(1, 21):  # reprocessed and failed again: refreshed entry
        assert carry[october(d)]["detail"]["marker"] == "new"
    for d in range(21, 26):  # never reached this run: previous entry untouched
        assert carry[october(d)] == failed_entry()
    assert state["last_successful_query_date"] == "2026-11-17"


def test_04_backlog_drains_over_successive_runs(env):
    env.write_state(
        last_successful_query_date="2026-11-17",
        failed_dates_carryover={october(d): failed_entry() for d in range(1, 26)},
    )
    env.run()  # first 20 succeed
    assert list(env.state()["failed_dates_carryover"]) == [october(d) for d in range(21, 26)]
    env.api.calls.clear()
    env.run()  # remaining 5
    assert env.api.calls == [october(d) for d in range(21, 26)]
    assert env.state()["failed_dates_carryover"] == {}
    assert env.state()["failed_dates_carryover_count"] == 0


# --------------------------------------------------------------------------
# 5. retry success clears carryover
# --------------------------------------------------------------------------
def test_05_retry_success_removes_date_from_carryover(env):
    env.write_state(
        last_successful_query_date="2026-11-17",
        failed_dates_carryover={
            "2026-11-05": failed_entry(),
            "2026-11-06": failed_entry(),
        },
    )
    env.api.script["2026-11-05"] = ok([record(disc_date="2026-11-05")])
    env.api.script["2026-11-06"] = ok([])  # empty-but-successful also resolves it

    env.run()

    state = env.state()
    assert state["failed_dates_carryover"] == {}
    assert state["failed_dates_last_run"] == {}
    assert state["successful_query_dates_last_run"] == ["2026-11-05", "2026-11-06"]
    rows = env.read_raw()
    assert rows[0][-2:] == ["query_date", "ticker"]
    assert [r[rows[0].index("query_date")] for r in rows[1:]] == ["2026-11-05"]


# --------------------------------------------------------------------------
# 6. retry failure keeps carryover
# --------------------------------------------------------------------------
def test_06_retry_failure_keeps_date_and_refreshes_reason(env):
    env.write_state(
        last_successful_query_date="2026-11-17",
        failed_dates_carryover={
            "2026-11-05": failed_entry("HTTP_ERROR"),
            "2026-11-06": failed_entry("HTTP_ERROR"),
        },
    )
    env.api.script["2026-11-05"] = fail("NETWORK_OR_DNS_RETRY_EXHAUSTED", type="URLError")
    env.api.script["2026-11-06"] = ok()

    env.run()

    state = env.state()
    assert list(state["failed_dates_carryover"]) == ["2026-11-05"]
    assert (
        state["failed_dates_carryover"]["2026-11-05"]["reason"] == "NETWORK_OR_DNS_RETRY_EXHAUSTED"
    )
    assert state["failed_dates_last_run"].keys() == {"2026-11-05"}


def test_06_non_retryable_failure_is_not_carried_over(env):
    env.write_state(
        last_successful_query_date="2026-11-17",
        failed_dates_carryover={"2026-11-05": failed_entry("HTTP_ERROR")},
    )
    env.api.script["2026-11-05"] = fail("PLAN_NOT_AVAILABLE", code=403)
    env.run()
    state = env.state()
    assert state["failed_dates_carryover"] == {}
    assert state["failed_dates_last_run"].keys() == {"2026-11-05"}


# --------------------------------------------------------------------------
# 7. watermark integrity
# --------------------------------------------------------------------------
def test_07_run_without_success_keeps_previous_watermark(env):
    env.write_state(last_successful_query_date="2026-11-10", failed_dates_carryover={})
    for d in range(11, 18):
        env.api.script[f"2026-11-{d}"] = fail("HTTP_ERROR", code=500)
    env.run()
    state = env.state()
    assert state["successful_query_dates_last_run"] == []
    assert state["last_successful_query_date"] == "2026-11-10"
    assert len(state["failed_dates_carryover"]) == 7


def test_07_retrying_an_old_date_does_not_roll_back_watermark(env, monkeypatch):
    monkeypatch.setenv("JQUANTS_FIN_SUMMARY_MAX_DATES_PER_RUN", "1")
    env.write_state(
        last_successful_query_date="2026-11-10",
        failed_dates_carryover={"2026-11-01": failed_entry()},
    )
    env.run()
    assert env.api.calls == ["2026-11-01"]
    state = env.state()
    assert state["successful_query_dates_last_run"] == ["2026-11-01"]
    assert state["last_successful_query_date"] == "2026-11-10"
    assert state["failed_dates_carryover"] == {}


def test_07_no_history_and_no_success_stays_none(env, monkeypatch):
    monkeypatch.setenv("JQUANTS_FIN_SUMMARY_MAX_DATES_PER_RUN", "2")
    env.api.script["2024-04-05"] = fail()
    env.api.script["2024-04-06"] = fail()
    env.run()
    assert env.state()["last_successful_query_date"] is None


def test_07_partial_failure_does_not_skip_the_failed_date(env):
    env.write_state(last_successful_query_date="2026-11-10", failed_dates_carryover={})
    env.api.script["2026-11-12"] = fail("HTTP_ERROR", code=500)
    env.run()

    state = env.state()
    assert state["last_successful_query_date"] == "2026-11-17"
    assert list(state["failed_dates_carryover"]) == ["2026-11-12"]
    # the next run asks for the failed date again even though the watermark passed it
    next_dates = env.mod.build_target_dates(
        env.local_store / RAW_FILE, env.local_store / STATE_FILE
    )
    assert next_dates == ["2026-11-12"]


def test_07_watermark_is_independent_of_saved_disclosure_date(env):
    env.write_state(last_successful_query_date="2026-11-14", failed_dates_carryover={})
    env.api.script["2026-11-15"] = ok([record(disc_date="2026-11-12")])
    env.run()
    state = env.state()
    assert state["last_successful_query_date"] == "2026-11-17"
    assert state["latest_disc_date_saved"] == "2026-11-12"


def test_07_garbage_watermark_in_state_is_ignored(env):
    env.write_state(last_successful_query_date="not-a-date", failed_dates_carryover={})
    env.api.script["2024-04-05"] = ok()
    env.api.script["2024-04-06"] = fail()
    with mock.patch.dict(os.environ, {"JQUANTS_FIN_SUMMARY_MAX_DATES_PER_RUN": "2"}):
        env.run()
    assert env.state()["last_successful_query_date"] == "2024-04-05"


# --------------------------------------------------------------------------
# 8. normal empty responses
# --------------------------------------------------------------------------
def test_08_empty_response_is_success_not_failure(env):
    env.write_state(last_successful_query_date="2026-11-14", failed_dates_carryover={})
    env.api.script["2026-11-15"] = ok([])
    env.api.script["2026-11-16"] = fail("HTTP_ERROR", code=500)
    env.api.script["2026-11-17"] = ok([record(disc_date="2026-11-17")])
    env.run()

    state = env.state()
    assert state["empty_dates_last_run"] == ["2026-11-15"]
    assert state["failed_dates_last_run"].keys() == {"2026-11-16"}
    assert list(state["failed_dates_carryover"]) == ["2026-11-16"]
    assert state["successful_query_dates_last_run"] == ["2026-11-15", "2026-11-17"]
    assert state["run_results"]["2026-11-15"]["ok"] is True
    assert state["run_results"]["2026-11-15"]["raw_count"] == 0
    assert state["run_results"]["2026-11-16"]["ok"] is False


def test_08_empty_only_runs_are_repeatable(env, monkeypatch):
    monkeypatch.setenv("JQUANTS_FIN_SUMMARY_MAX_DATES_PER_RUN", "2")
    env.run()  # every date answers "ok, no records"
    state = env.state()
    assert state["empty_dates_last_run"] == ["2024-04-05", "2024-04-06"]
    assert state["rows_saved_total"] == 0
    assert state["last_successful_query_date"] == "2024-04-06"
    assert state["latest_disc_date_saved"] is None  # distinct from the watermark

    env.run()  # must be able to read what the first run left behind
    state = env.state()
    assert env.api.calls[-2:] == ["2024-04-07", "2024-04-08"]
    assert state["last_successful_query_date"] == "2024-04-08"
    assert state["rows_saved_total"] == 0
    with (env.local_store / COVERAGE_FILE).open(newline="", encoding="utf-8") as f:
        coverage = list(csv.DictReader(f))
    assert [r["has_any_financial_summary"] for r in coverage] == ["False", "False"]


# --------------------------------------------------------------------------
# 9. store location is resolved after .env / environment are loaded
# --------------------------------------------------------------------------
def test_09_env_file_external_store_dirs_is_honoured(env):
    external = env.tmp / "external_store"
    (env.project / ".env").write_text(
        f'# comment\nJQUANTS_EXTERNAL_STORE_DIRS="{external}"\n', encoding="utf-8"
    )
    env.run()

    state = env.state(external)
    assert state["store_type"] == "external"
    assert Path(state["store_dir"]) == external
    assert_under(env.tmp, external)
    for name in (RAW_FILE, COVERAGE_FILE, STATE_FILE, RUNS_FILE):
        assert (external / name).exists(), name
    assert not (env.local_store / STATE_FILE).exists()


def test_09_process_environment_wins_over_env_file(env):
    from_env = env.tmp / "from_process_env"
    from_file = env.tmp / "from_env_file"
    os.environ["JQUANTS_EXTERNAL_STORE_DIRS"] = str(from_env)
    (env.project / ".env").write_text(
        f"JQUANTS_EXTERNAL_STORE_DIRS={from_file}\n", encoding="utf-8"
    )
    env.run()
    assert (from_env / STATE_FILE).exists()
    assert not from_file.exists()


def test_09_dirs_has_priority_over_dir_and_dir_is_fallback(env):
    mod = env.mod
    a, b = env.tmp / "a", env.tmp / "b"
    os.environ["JQUANTS_EXTERNAL_STORE_DIR"] = str(b)
    assert mod.get_external_candidates() == [str(b)]
    os.environ["JQUANTS_EXTERNAL_STORE_DIRS"] = str(a)
    assert mod.get_external_candidates() == [str(a)]
    os.environ["JQUANTS_EXTERNAL_STORE_DIRS"] = ""
    assert mod.get_external_candidates() == [str(b)]


def test_09_candidates_are_tried_in_order_then_local_fallback(env):
    mod = env.mod
    missing_parent = env.tmp / "no_such_parent" / "store"
    usable = env.tmp / "usable_store"
    os.environ["JQUANTS_EXTERNAL_STORE_DIRS"] = os.pathsep.join(
        [str(missing_parent), f" {usable} "]
    )
    assert mod.choose_store_dir() == (usable, "external")
    assert not missing_parent.parent.exists()  # not created behind our back

    os.environ["JQUANTS_EXTERNAL_STORE_DIRS"] = str(missing_parent)
    assert mod.choose_store_dir() == (env.local_store, "local")
    os.environ.pop("JQUANTS_EXTERNAL_STORE_DIRS")
    assert mod.choose_store_dir() == (env.local_store, "local")


def test_09_existing_local_data_is_copied_to_new_external_store(env):
    external = env.tmp / "external_store"
    env.write_raw([["1", "72030", "2026-11-05", "FY", "2026-11-05", "7203"]])
    (env.local_store / COVERAGE_FILE).write_text("ticker\n7203\n", encoding="utf-8")
    os.environ["JQUANTS_EXTERNAL_STORE_DIRS"] = str(external)
    env.run()
    assert (external / RAW_FILE).read_text().splitlines()[1].startswith("1,72030")


def test_09_universe_file_and_project_dir_resolution(env):
    custom = env.tmp / "custom_universe.csv"
    custom.write_text("ticker\n9984\n", encoding="utf-8")
    (env.project / ".env").write_text(
        f"JP_STOCK_ANALYSIS_UNIVERSE_FILE={custom}\n", encoding="utf-8"
    )
    assert env.mod.PROJECT_DIR == env.project
    assert env.mod.LOCAL_STORE == env.local_store
    env.run()
    with (env.local_store / COVERAGE_FILE).open(newline="", encoding="utf-8") as f:
        assert [r["ticker"] for r in csv.DictReader(f)] == ["9984"]


def test_09_missing_universe_file_is_a_clean_exit(env):
    os.environ["JP_STOCK_ANALYSIS_UNIVERSE_FILE"] = str(env.tmp / "absent.csv")
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert "universe file not found" in str(excinfo.value)


def test_09_missing_api_key_is_a_clean_exit(env):
    os.environ.pop("JQUANTS_API_KEY")
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert str(excinfo.value) == "JQUANTS_API_KEY: MISSING"
    assert env.api.calls == []


# --------------------------------------------------------------------------
# 10. re-fetching identical data + output compatibility
# --------------------------------------------------------------------------
def test_10_refetching_same_data_does_not_duplicate_rows(env):
    env.write_state(last_successful_query_date="2026-11-04", failed_dates_carryover={})
    rec = record(code="72030", disc_date="2026-11-05", disc_no="20261105001")
    env.api.script["2026-11-05"] = ok([rec, record(code="99990", disc_no="9")])
    with mock.patch.dict(os.environ, {"JQUANTS_FIN_SUMMARY_MAX_DATES_PER_RUN": "1"}):
        env.run()
    first = env.read_raw()
    assert len(first) == 2  # header + 1 row: 99990 is not in the universe
    assert first[0] == [*rec.keys(), "query_date", "ticker"]
    assert first[1][-2:] == ["2026-11-05", "7203"]  # 72030 -> 7203

    # Force the very same date to be fetched again via the carryover.
    env.write_state(
        last_successful_query_date="2026-11-05",
        failed_dates_carryover={"2026-11-05": failed_entry()},
    )
    with mock.patch.dict(os.environ, {"JQUANTS_FIN_SUMMARY_MAX_DATES_PER_RUN": "1"}):
        env.run()

    assert env.read_raw() == first
    state = env.state()
    assert state["rows_added_last_run_before_dedup"] == 1
    assert state["rows_saved_total"] == 1
    assert state["unique_tickers_saved"] == 1
    assert state["coverage_missing_count"] == 1  # 6758 never reported


def test_10_new_rows_append_without_touching_existing_columns(env):
    env.write_raw(
        [["7", "67580", "2026-11-04", "FY", "2026-11-04", "6758"]],
        header=["DiscNo", "Code", "DiscDate", "DocType", "query_date", "ticker"],
    )
    env.write_state(last_successful_query_date="2026-11-16", failed_dates_carryover={})
    env.api.script["2026-11-17"] = ok(
        [{"DiscNo": "8", "Code": "72030", "DiscDate": "2026-11-17", "DocType": "FY", "Sales": "5"}]
    )
    env.run()
    rows = env.read_raw()
    assert rows[0] == [
        "DiscNo",
        "Code",
        "DiscDate",
        "DocType",
        "query_date",
        "ticker",
        "Sales",
    ]
    assert [r[0] for r in rows[1:]] == ["7", "8"]
    assert env.state()["latest_disc_date_saved"] == "2026-11-17"


def test_10_output_files_keep_their_contract(env):
    env.write_state(last_successful_query_date="2026-11-16", failed_dates_carryover={})
    env.api.script["2026-11-17"] = ok([record(disc_date="2026-11-17")])
    env.run()
    env.write_state(
        last_successful_query_date="2026-11-17",
        failed_dates_carryover={"2026-11-17": failed_entry()},
    )
    env.run()

    state = env.state()
    assert list(state.keys()) == STATE_KEYS
    assert state["endpoint"] == "/v2/fins/summary"
    assert state["query_mode"] == "date"
    assert state["store_type"] == "local"
    assert state["output"] == str(env.local_store / RAW_FILE)
    assert state["coverage_output"] == str(env.local_store / COVERAGE_FILE)
    assert state["api_key_status"] == "PRESENT"
    assert state["secret_included"] is False
    for key in (
        "rows_added_last_run_before_dedup",
        "rows_saved_total",
        "unique_tickers_saved",
        "coverage_missing_count",
        "failed_dates_count_last_run",
        "failed_dates_carryover_count",
        "skipped_dates_count_last_run",
    ):
        assert type(state[key]) is int, key
    assert isinstance(state["failed_dates_carryover"], dict)
    assert isinstance(state["target_dates_last_run"], list)
    assert set(state["run_results"]["2026-11-17"]) == {
        "ok",
        "attempt",
        "reason",
        "raw_count",
        "error",
    }

    lines = (env.local_store / RUNS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # appended once per run
    last = json.loads(lines[-1])
    assert list(last.keys()) == STATE_KEYS
    last["last_run_at_jst"] = state["last_run_at_jst"]
    assert last == state

    with (env.local_store / COVERAGE_FILE).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        coverage = list(reader)
        assert reader.fieldnames == ["ticker", "name", "has_any_financial_summary"]
    assert {r["ticker"]: r["has_any_financial_summary"] for r in coverage} == {
        "7203": "True",
        "6758": "False",
    }

    for name in (RAW_FILE, COVERAGE_FILE, STATE_FILE, RUNS_FILE):
        assert API_KEY not in (env.local_store / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# early stop on the subscription window
# --------------------------------------------------------------------------
def test_11_subscription_stop_keeps_unreached_failed_dates(env):
    env.write_state(
        last_successful_query_date="2026-11-10",
        failed_dates_carryover={
            "2026-11-01": failed_entry(),
            "2026-11-02": failed_entry(),
            "2026-11-03": failed_entry(),
        },
    )
    env.api.script["2026-11-02"] = fail(
        "OUT_OF_SUBSCRIPTION_DATE_RANGE",
        type="HTTPError",
        code=400,
        subscription_start="2024-08-05",
        subscription_end="2026-08-05",
    )
    env.run()

    assert env.api.calls == ["2026-11-01", "2026-11-02"]  # stopped, nothing after
    state = env.state()
    assert state["skipped_dates_last_run"].keys() == {"2026-11-02"}
    assert state["subscription_start_detected"] == "2024-08-05"
    assert state["subscription_end_detected"] == "2026-08-05"
    # 11-01 succeeded, 11-02 is non-retryable (dropped as before), 11-03 never reached.
    assert state["failed_dates_carryover"] == {"2026-11-03": failed_entry()}
    assert state["last_successful_query_date"] == "2026-11-10"


def test_11_same_date_is_never_fetched_twice_in_one_run(env):
    env.write_state(
        last_successful_query_date="2026-11-14",
        failed_dates_carryover={
            "2026-11-16": failed_entry(),
            "2026-11-15": failed_entry(),
        },
    )
    env.run()
    assert env.api.calls == ["2026-11-15", "2026-11-16", "2026-11-17"]
    assert len(env.api.calls) == len(set(env.api.calls))


# ==========================================================================
# Recovery work: fail clearly instead of silently starting over
# ==========================================================================
def store_snapshot(store: Path):
    """Hash of every file in the store (temp files included, so leftovers are caught)."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(store.iterdir())
        if p.is_file()
    }


def build_real_like_store(env):
    """The shape of the real store on 2026-11-04: 13 business days of rows (07-16 .. 08-04),
    state at 2026-08-04, and a run log whose last line is the state's own run."""
    days = [date(2026, 7, 16) + timedelta(days=i) for i in range(20)]
    business = [d for d in days if d.weekday() < 5 and d != date(2026, 7, 20)]  # 07-20 is a holiday
    rows = [
        [f"{d:%Y%m%d}01", "72030", d.isoformat(), "FY", d.isoformat(), "7203"] for d in business
    ]
    env.write_raw(rows)
    runs = [
        ("2026-10-14T08:24:59.634810+09:00", "2026-07-22"),
        ("2026-10-24T02:41:42.816104+09:00", "2026-08-01"),
        ("2026-11-04T08:19:21.084263+09:00", "2026-08-04"),
    ]
    env.write_state(
        last_run_at_jst=runs[-1][0],
        last_successful_query_date="2026-08-04",
        latest_disc_date_saved="2026-08-04",
        rows_saved_total=len(rows),
        failed_dates_carryover={},
        failed_dates_carryover_count=0,
    )
    (env.local_store / RUNS_FILE).write_text(
        "".join(
            json.dumps({"last_run_at_jst": ts, "last_successful_query_date": wm}) + "\n"
            for ts, wm in runs
        ),
        encoding="utf-8",
    )
    return rows


def exit_code_of(excinfo):
    return excinfo.value.code


# --------------------------------------------------------------------------
# the start date comes from the saved progress
# --------------------------------------------------------------------------
def test_real_like_store_resumes_on_2026_08_05(env):
    rows = build_real_like_store(env)
    assert len(rows) == 13
    dates = env.mod.build_target_dates(env.local_store / RAW_FILE, env.local_store / STATE_FILE)
    assert dates[0] == "2026-08-05"

    env.run()
    assert env.api.calls[0] == "2026-08-05"
    assert env.api.calls[-1] == "2026-08-24"
    assert len(env.api.calls) == 20
    state = env.state()
    assert state["last_successful_query_date"] == "2026-08-24"
    assert state["latest_disc_date_saved"] == "2026-08-04"  # unchanged: no new disclosures
    assert len(env.read_raw()) - 1 == 13  # existing rows kept


# --------------------------------------------------------------------------
# a rolled-back or truncated store is refused, not silently re-fetched
# --------------------------------------------------------------------------
def roll_back_everything_but_the_log(env):
    """What the 11-04 and 11-21 runs behaved like: inputs anchored at 2026-07-15 while the
    run log already records progress up to 2026-08-04."""
    env.write_raw([["2026070101", "72030", "2026-07-15", "FY", "2026-07-15", "7203"]])
    env.write_state(
        last_run_at_jst="2026-10-07T08:00:00+09:00",
        last_successful_query_date="2026-07-15",
        latest_disc_date_saved="2026-07-15",
        rows_saved_total=1,
        failed_dates_carryover={},
    )


def test_inputs_rolled_back_behind_the_run_log_are_refused(env, capsys):
    build_real_like_store(env)
    roll_back_everything_but_the_log(env)
    before = store_snapshot(env.local_store)

    with pytest.raises(SystemExit) as excinfo:
        env.run()

    assert exit_code_of(excinfo) == 5
    err = capsys.readouterr().err
    assert "PROGRESS_REGRESSION" in err and "older than the last run-log entry" in err
    assert env.api.calls == []  # 2026-07-16..2026-08-04 is not fetched again
    assert store_snapshot(env.local_store) == before


def test_the_override_is_explicit_and_then_refetches_from_the_rolled_back_anchor(env, capsys):
    build_real_like_store(env)
    roll_back_everything_but_the_log(env)
    assert env.mod.main(["--allow-progress-regression"]) == 0
    assert env.api.calls[0] == "2026-07-16"
    assert "progress check overridden" in capsys.readouterr().err


def test_the_override_can_also_come_from_the_environment(env, monkeypatch):
    build_real_like_store(env)
    roll_back_everything_but_the_log(env)
    monkeypatch.setenv("JQUANTS_FIN_SUMMARY_ALLOW_PROGRESS_REGRESSION", "1")
    env.run()
    assert env.api.calls[0] == "2026-07-16"


def truncate_raw_to_header_and_two_rows(env, rows):
    env.write_raw(rows[:2])


def empty_the_raw_file(env, rows):
    (env.local_store / RAW_FILE).write_text("")


def delete_the_raw_file(env, rows):
    (env.local_store / RAW_FILE).unlink()


def keep_only_old_disclosures(env, rows):
    env.write_raw([[r[0], r[1], "2026-07-16", r[3], r[4], r[5]] for r in rows])


def state_behind_the_log(env, rows):
    state = env.state()
    state["last_successful_query_date"] = "2026-07-22"
    state["latest_disc_date_saved"] = "2026-07-22"
    env.write_state(**state)


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        (truncate_raw_to_header_and_two_rows, "raw CSV has 2 rows but the state recorded 13"),
        (empty_the_raw_file, "raw CSV has 0 rows but the state recorded 13"),
        (delete_the_raw_file, "raw CSV has 0 rows but the state recorded 13"),
        (keep_only_old_disclosures, "newest disclosure in the raw CSV (2026-07-16) is older"),
        (state_behind_the_log, "is behind the last run-log entry"),
    ],
)
def test_each_kind_of_damage_is_detected_before_any_request(env, capsys, damage, expected):
    rows = build_real_like_store(env)
    damage(env, rows)
    before = store_snapshot(env.local_store)
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert exit_code_of(excinfo) == 5
    assert expected in capsys.readouterr().err
    assert env.api.calls == []
    assert store_snapshot(env.local_store) == before


def test_a_healthy_store_and_a_fresh_store_both_pass_the_check(env):
    build_real_like_store(env)
    assert (
        env.mod.find_progress_problems(
            env.mod.load_store_inputs(
                env.local_store / RAW_FILE,
                env.local_store / STATE_FILE,
                env.local_store / RUNS_FILE,
            )
        )
        == []
    )
    fresh = env.mod.load_store_inputs(
        env.local_store / "none.csv", env.local_store / "none.json", env.local_store / "none.jsonl"
    )
    assert env.mod.find_progress_problems(fresh) == []


# --------------------------------------------------------------------------
# unreadable state / raw: an error, never "start from scratch"
# --------------------------------------------------------------------------
@pytest.mark.parametrize("content", ["{not json", "", "[1, 2, 3]", "null", '"text"'])
def test_unusable_state_stops_the_run_and_is_not_reset(env, capsys, content):
    build_real_like_store(env)
    (env.local_store / STATE_FILE).write_text(content)
    before = store_snapshot(env.local_store)
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert exit_code_of(excinfo) == 4
    assert "STATE_UNREADABLE" in capsys.readouterr().err
    assert env.api.calls == []
    assert store_snapshot(env.local_store) == before


def test_a_state_path_that_is_a_directory_is_an_input_error(env, capsys):
    (env.local_store / STATE_FILE).mkdir()
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert exit_code_of(excinfo) == 4
    assert env.api.calls == []


@pytest.mark.parametrize(
    "content",
    [b"\xff\xfe\x00garbage\x00", b"a,b\n1,2\n3,4,5\n"],
    ids=["not-utf8", "ragged-rows"],
)
def test_unparsable_raw_csv_stops_the_run_and_is_not_replaced(env, capsys, content):
    build_real_like_store(env)
    (env.local_store / RAW_FILE).write_bytes(content)
    before = store_snapshot(env.local_store)
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert exit_code_of(excinfo) == 4
    assert "RAW_UNREADABLE" in capsys.readouterr().err
    assert env.api.calls == []
    assert store_snapshot(env.local_store) == before


# --------------------------------------------------------------------------
# a failing drive is reported as such (EIO / ENODEV), distinct from a missing file
# --------------------------------------------------------------------------
@pytest.mark.parametrize("code", [errno.EIO, errno.ENODEV])
def test_drive_errors_on_the_state_file_are_reported_as_store_unavailable(
    env, monkeypatch, capsys, code
):
    build_real_like_store(env)
    real_stat = os.stat

    def flaky_stat(path, *args, **kwargs):
        if str(path).endswith(STATE_FILE):
            raise OSError(code, os.strerror(code))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", flaky_stat)
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert exit_code_of(excinfo) == 3
    err = capsys.readouterr().err
    assert "STORE_UNAVAILABLE" in err and f"errno {code}" in err
    assert env.api.calls == []


@pytest.mark.parametrize("code", [errno.EIO, errno.ENODEV])
def test_drive_errors_while_reading_the_raw_csv_are_reported_as_store_unavailable(
    env, monkeypatch, capsys, code
):
    build_real_like_store(env)
    real_read_csv = env.mod.pd.read_csv

    def flaky_read_csv(path, *args, **kwargs):
        if str(path).endswith(RAW_FILE):
            raise OSError(code, os.strerror(code))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(env.mod.pd, "read_csv", flaky_read_csv)
    before = store_snapshot(env.local_store)
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert exit_code_of(excinfo) == 3
    assert "STORE_UNAVAILABLE" in capsys.readouterr().err
    assert env.api.calls == []
    assert store_snapshot(env.local_store) == before


def test_a_missing_state_file_is_still_just_a_fresh_start(env):
    env.mod.main()  # no state, no raw: first run, as before
    assert env.api.calls[0] == "2024-04-05"


def test_a_dangling_store_symlink_is_reported_and_nothing_is_created(env, tmp_path, capsys):
    shutil.rmtree(env.local_store)
    unmounted = tmp_path / "not_mounted" / "jquants_price_store"
    env.local_store.symlink_to(unmounted)
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert exit_code_of(excinfo) == 3
    err = capsys.readouterr().err
    assert "STORE_UNAVAILABLE" in err and "is a symlink to" in err
    assert not unmounted.exists() and not unmounted.parent.exists()  # no junk on the wrong disk
    assert env.api.calls == []


def test_a_working_store_symlink_to_another_drive_behaves_like_a_directory(env, tmp_path):
    """The real layout: data/jquants_price_store -> /mnt/d/jquants_price_store."""
    drive = tmp_path / "mnt_d" / "jquants_price_store"
    drive.parent.mkdir()
    shutil.move(str(env.local_store), str(drive))
    env.local_store.symlink_to(drive)
    env.run()
    assert (drive / STATE_FILE).exists() and (drive / RAW_FILE).exists() is not None
    assert json.loads((drive / STATE_FILE).read_text())["store_type"] == "local"


# --------------------------------------------------------------------------
# the universe file: missing is not a mount problem, and a bad one is not a crash
# --------------------------------------------------------------------------
def test_a_missing_universe_keeps_the_old_message_and_points_to_the_recovery(env, capsys):
    build_real_like_store(env)
    (env.local_store / "topix1000_usable_tickers.csv").unlink()
    before = store_snapshot(env.local_store)
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert str(excinfo.value).startswith("universe file not found: ")  # the text in your logs
    assert "recover_jquants_universe.py" in capsys.readouterr().err
    assert env.api.calls == []
    assert store_snapshot(env.local_store) == before


@pytest.mark.parametrize(
    "content",
    ["code,name\n7203,Toyota\n", "", "ticker,name\n", "ticker,name\n,Nameless\n"],
    ids=["no-ticker-column", "empty-file", "header-only", "empty-ticker"],
)
def test_an_invalid_universe_is_an_input_error(env, capsys, content):
    (env.local_store / "topix1000_usable_tickers.csv").write_text(content)
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert exit_code_of(excinfo) == 4
    assert "UNIVERSE_INVALID" in capsys.readouterr().err
    assert env.api.calls == []


def test_an_unreadable_drive_while_looking_for_the_universe_is_not_called_missing(
    env, monkeypatch, capsys
):
    real_stat = os.stat

    def flaky_stat(path, *args, **kwargs):
        if str(path).endswith("topix1000_usable_tickers.csv"):
            raise OSError(errno.EIO, "Input/output error")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", flaky_stat)
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert exit_code_of(excinfo) == 3
    assert "universe file not found" not in capsys.readouterr().err


# --------------------------------------------------------------------------
# one writer at a time, shared with the backfill
# --------------------------------------------------------------------------
def test_lock_file_lives_in_the_project_cache_not_on_the_data_drive(env):
    assert env.mod.lock_file_path() == (
        env.project / ".cache" / "jquants" / "financial_summary_store.lock"
    )


def test_a_held_lock_stops_the_daily_run_before_it_touches_anything(env, capsys):
    build_real_like_store(env)
    before = store_snapshot(env.local_store)
    backfill_side = env.mod.StoreLock("backfill")
    backfill_side.acquire()
    try:
        with pytest.raises(SystemExit) as excinfo:
            env.run()
    finally:
        backfill_side.release()
    assert exit_code_of(excinfo) == 75
    err = capsys.readouterr().err
    assert "BUSY" in err and "role=backfill" in err
    assert env.api.calls == []
    assert store_snapshot(env.local_store) == before


def test_the_lock_is_released_when_a_run_ends_even_if_it_fails(env):
    env.write_state(last_successful_query_date="2026-11-16", failed_dates_carryover={})
    (env.local_store / STATE_FILE).write_text("{broken")
    with pytest.raises(SystemExit):
        env.run()
    other = env.mod.StoreLock("probe")
    other.acquire()  # would raise BusyError if the failed run had kept the lock
    other.release()


def test_two_runs_in_a_row_work(env):
    env.write_state(last_successful_query_date="2026-11-15", failed_dates_carryover={})
    env.run()
    env.run()


def test_the_lock_file_can_be_redirected_by_environment(env, tmp_path, monkeypatch):
    monkeypatch.setenv("JQUANTS_FIN_SUMMARY_LOCK_FILE", str(tmp_path / "custom.lock"))
    assert env.mod.lock_file_path() == tmp_path / "custom.lock"


# --------------------------------------------------------------------------
# writes are all-or-nothing
# --------------------------------------------------------------------------
def fail_replace_for(monkeypatch, env, name, code=errno.EIO):
    real_replace = os.replace

    def flaky_replace(src, dst, *args, **kwargs):
        if str(dst).endswith(name):
            raise OSError(code, os.strerror(code))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", flaky_replace)


def test_a_failed_raw_write_leaves_the_old_files_and_no_temp_file(env, monkeypatch, capsys):
    build_real_like_store(env)
    env.api.script["2026-08-05"] = ok([record(disc_date="2026-08-05", disc_no="new1")])
    before = store_snapshot(env.local_store)
    fail_replace_for(monkeypatch, env, RAW_FILE)
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert exit_code_of(excinfo) == 3
    assert "STORE_UNAVAILABLE" in capsys.readouterr().err
    assert store_snapshot(env.local_store) == before  # raw, coverage, state, log, and no *.tmp


def test_a_failed_state_write_is_recovered_by_the_next_run_without_duplicates(env, monkeypatch):
    build_real_like_store(env)
    env.api.script["2026-08-05"] = ok([record(disc_date="2026-08-05", disc_no="new1")])
    with monkeypatch.context() as m:
        fail_replace_for(m, env, STATE_FILE)
        with pytest.raises(SystemExit):
            env.run()
    state_after_failure = env.state()
    assert state_after_failure["last_successful_query_date"] == "2026-08-04"  # progress not claimed

    env.api.calls.clear()
    env.run()  # raw is ahead of the state: allowed, and the same date is simply fetched again
    table = env.read_raw()
    query_dates = [r[table[0].index("query_date")] for r in table[1:]]
    assert query_dates.count("2026-08-05") == 1  # no duplicate
    # start = max(newest saved disclosure 08-05, state watermark 08-04) + 1: nothing is re-fetched
    assert env.api.calls[0] == "2026-08-06"
    assert env.state()["last_successful_query_date"] == "2026-08-25"


def test_read_back_verification_keeps_the_old_file_when_the_new_one_is_wrong(env, monkeypatch):
    import pandas as pd

    target = env.local_store / RAW_FILE
    target.write_text("keep,me\n1,2\n")
    monkeypatch.setattr(env.mod.pd, "read_csv", lambda *a, **k: pd.DataFrame({"x": ["1"]}))
    df = pd.DataFrame({"a": ["1", "2"], "b": ["3", "4"]})
    with pytest.raises(env.mod.StoreInputError) as excinfo:
        env.mod.atomic_write_csv(df, target)
    assert excinfo.value.kind == "WRITE_VERIFY_FAILED"
    assert target.read_text() == "keep,me\n1,2\n"
    assert not list(env.local_store.glob("*.tmp"))


def test_a_writer_that_ignores_the_lock_is_not_overwritten(env, monkeypatch, capsys):
    rows = build_real_like_store(env)
    extra = "2026080501,72030,2026-08-05,FY,2026-08-05,7203\n"

    def fetch(target_date, api_key, max_attempts=3):
        if target_date == "2026-08-05":  # another process appends a row while we are fetching
            with (env.local_store / RAW_FILE).open("a") as f:
                f.write(extra)
            return ok([record(disc_date="2026-08-05", disc_no="mine")])
        return ok()

    monkeypatch.setattr(env.mod, "fetch_one_date", fetch)
    state_before = (env.local_store / STATE_FILE).read_bytes()
    with pytest.raises(SystemExit) as excinfo:
        env.run()
    assert exit_code_of(excinfo) == 6
    assert "STORE_CHANGED_DURING_RUN" in capsys.readouterr().err
    assert extra in (env.local_store / RAW_FILE).read_text()  # their row is still there
    assert (env.local_store / STATE_FILE).read_bytes() == state_before
    assert len(rows) == 13


# --------------------------------------------------------------------------
# --dry-run: see what this process sees, change nothing
# --------------------------------------------------------------------------
def test_dry_run_shows_the_anchor_and_the_dates_without_touching_anything(env, capsys):
    build_real_like_store(env)
    os.environ.pop("JQUANTS_API_KEY")  # a dry run needs no key
    env.api.rule = lambda d: pytest.fail("dry-run must not call the API")
    before = store_snapshot(env.local_store)
    assert env.mod.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "anchor (next start = anchor + 1 day): 2026-08-04" in out
    assert "target_dates (cap 20): 2026-08-05 .. 2026-08-24 (20 dates)" in out
    assert "progress_check: OK" in out
    assert "store_mount:" in out
    assert store_snapshot(env.local_store) == before
    assert env.api.calls == []


def test_dry_run_reports_a_rolled_back_store_and_returns_the_failure_code(env, capsys):
    build_real_like_store(env)
    roll_back_everything_but_the_log(env)
    assert env.mod.main(["--dry-run"]) == 5
    out = capsys.readouterr().out
    assert "PROBLEMS (run would stop)" in out
    assert "anchor (next start = anchor + 1 day): 2026-07-15" in out
    assert env.api.calls == []


def test_dry_run_still_works_while_another_process_holds_the_lock(env, capsys):
    build_real_like_store(env)
    holder = env.mod.StoreLock("backfill")
    holder.acquire()
    try:
        assert env.mod.main(["--dry-run"]) == 0
    finally:
        holder.release()
    assert "HELD by another process" in capsys.readouterr().out


def test_describe_mount_names_the_path_and_a_filesystem(env):
    text = env.mod.describe_mount(env.local_store)
    assert str(env.local_store.resolve()) in text and " on " in text and "(" in text

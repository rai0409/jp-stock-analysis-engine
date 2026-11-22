"""Offline tests for scripts/recover_jquants_universe.py.

Everything happens under tmp_path. The restored file is also fed to the repository's real
universe loaders, so the claim "this satisfies the consumers' contract" is checked, not assumed.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from jp_stock_analysis.data.incremental_prices import load_universe_file
from jp_stock_analysis.validation.jquants_listed_master import _read_universe

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recover_jquants_universe.py"
UNIVERSE_COLUMNS = ["ticker", "name", "date", "new_index_category"]
COVERAGE_COLUMNS = [*UNIVERSE_COLUMNS, "has_any_financial_summary"]


def load_module():
    spec = importlib.util.spec_from_file_location("recover_universe_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_tickers(n=974):
    """Distinct 4-character codes: digits, plus letter-suffixed ones such as 130A."""
    letters = 14 if n >= 28 else n // 2
    tickers = [str(1301 + i) for i in range(n - letters)]
    tickers += [f"{130 + i}A" for i in range(letters)]
    assert len(set(tickers)) == n
    return tickers


def universe_rows(n=974):
    return [
        [t, f"Company {t}", "2026-06-30", "TOPIX Mid400" if i % 2 else "TOPIX Core30"]
        for i, t in enumerate(make_tickers(n))
    ]


def write_coverage_like_daily_script(path: Path, rows, n_true=367):
    """Exactly what the daily script does: universe.copy() + one flag column, via pandas."""
    universe = pd.DataFrame(rows, columns=UNIVERSE_COLUMNS, dtype=str)
    coverage = universe.copy()
    coverage["has_any_financial_summary"] = [i < n_true for i in range(len(universe))]
    coverage.to_csv(path, index=False)
    return coverage


@pytest.fixture
def store(tmp_path):
    d = tmp_path / "store"
    d.mkdir()
    return d


@pytest.fixture
def mod():
    return load_module()


def write_state(store, n_true=367, total=974):
    (store / "financial_summary_auto_state.json").write_text(
        json.dumps({"unique_tickers_saved": n_true, "coverage_missing_count": total - n_true})
    )


def run(mod, store, *args):
    return mod.run(["--store-dir", str(store), *args])


def snapshot(store):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(store.iterdir())}


# --------------------------------------------------------------------------
# the happy path, proven against the real consumers
# --------------------------------------------------------------------------
def test_restores_a_file_the_real_consumers_accept(mod, store):
    rows = universe_rows()
    write_coverage_like_daily_script(store / "financial_summary_coverage.csv", rows)
    write_state(store)

    assert run(mod, store, "--expect-rows", "974") == 0

    out = store / "topix1000_usable_tickers.csv"
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "ticker,name,date,new_index_category"  # exact columns and order
    assert not text.startswith("\ufeff")
    assert "\r" not in text
    # the repository's own loaders
    assert len(_read_universe(out)) == 974
    assert load_universe_file(out) == [r[0] for r in rows]  # same tickers, same order
    assert load_universe_file(out)[-1] == "143A"  # letter-suffixed codes survive


def test_round_trip_reproduces_the_coverage_file_byte_for_byte(mod, store):
    """universe -> coverage (daily script) -> universe (recovery) -> coverage must be identical."""
    source = store / "financial_summary_coverage.csv"
    write_coverage_like_daily_script(source, universe_rows())
    write_state(store)
    original_bytes = source.read_bytes()
    assert run(mod, store) == 0

    universe = pd.read_csv(store / "topix1000_usable_tickers.csv", dtype=str)
    universe["ticker"] = universe["ticker"].astype(str)
    again = universe.copy()
    again["has_any_financial_summary"] = [i < 367 for i in range(len(universe))]
    again.to_csv(store / "again.csv", index=False)
    assert (store / "again.csv").read_bytes() == original_bytes
    assert source.read_bytes() == original_bytes  # the source was only read


def test_provenance_file_is_written_and_says_what_is_not_proven(mod, store, capsys):
    write_coverage_like_daily_script(store / "financial_summary_coverage.csv", universe_rows())
    write_state(store)
    assert run(mod, store) == 0
    (sidecar,) = list(store.glob("topix1000_usable_tickers.csv.recovery_*.json"))
    info = json.loads(sidecar.read_text())
    assert info["kind"] == "universe_recovered_from_coverage"
    assert info["rows"] == 974
    assert info["byte_identity"].startswith("NOT PROVEN")
    assert (
        info["source_sha256"]
        == hashlib.sha256((store / "financial_summary_coverage.csv").read_bytes()).hexdigest()
    )
    assert "state_check" in info and info["state_check"].startswith("consistent")


# --------------------------------------------------------------------------
# never overwrites, dry-run writes nothing
# --------------------------------------------------------------------------
def test_dry_run_checks_everything_and_writes_nothing(mod, store):
    write_coverage_like_daily_script(store / "financial_summary_coverage.csv", universe_rows())
    write_state(store)
    before = snapshot(store)
    assert run(mod, store, "--dry-run") == 0
    assert snapshot(store) == before


def test_an_existing_different_file_is_never_overwritten(mod, store, capsys):
    write_coverage_like_daily_script(store / "financial_summary_coverage.csv", universe_rows())
    write_state(store)
    existing = store / "topix1000_usable_tickers.csv"
    existing.write_text("ticker\n9999\n")
    before = snapshot(store)
    assert run(mod, store) == 3
    assert "REFUSED" in capsys.readouterr().err
    assert snapshot(store) == before


def test_an_identical_existing_file_is_reported_as_already_restored(mod, store, capsys):
    write_coverage_like_daily_script(store / "financial_summary_coverage.csv", universe_rows())
    write_state(store)
    assert run(mod, store) == 0
    before = snapshot(store)
    assert run(mod, store) == 0
    assert "already restored" in capsys.readouterr().out
    assert snapshot(store) == before  # not even a second provenance file


def test_create_new_file_is_create_only_and_cleans_up_its_own_partial_file(
    mod, tmp_path, monkeypatch
):
    target = tmp_path / "x.csv"
    target.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        mod.create_new_file(target, b"new")
    assert target.read_bytes() == b"keep me"

    fresh = tmp_path / "y.csv"
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(errno.EIO, "boom")))
    with pytest.raises(OSError):
        mod.create_new_file(fresh, b"data")
    assert not fresh.exists()  # no half-written file left behind


# --------------------------------------------------------------------------
# validation: every defect stops the run before anything is written
# --------------------------------------------------------------------------
def csv_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


GOOD = [
    ["7203", "Toyota", "2026-06-30", "TOPIX Core30", "True"],
    ["6758", "Sony", "2026-06-30", "TOPIX Core30", "False"],
]


def corrupt(rows, index, column, value):
    out = [list(r) for r in rows]
    out[index][column] = value
    return out


@pytest.mark.parametrize(
    ("header", "rows", "message"),
    [
        (
            COVERAGE_COLUMNS,
            GOOD + [["7203", "Dup", "2026-06-30", "X", "True"]],
            "duplicate tickers",
        ),
        (COVERAGE_COLUMNS, corrupt(GOOD, 0, 0, "72030"), "not matching"),
        (COVERAGE_COLUMNS, corrupt(GOOD, 0, 0, "130a"), "not matching"),
        (COVERAGE_COLUMNS, corrupt(GOOD, 0, 0, " 7203"), "whitespace"),
        (COVERAGE_COLUMNS, corrupt(GOOD, 0, 0, ""), "whitespace"),
        (COVERAGE_COLUMNS, corrupt(GOOD, 1, 1, ""), "empty values"),
        (COVERAGE_COLUMNS, corrupt(GOOD, 1, 2, ""), "empty values"),
        (COVERAGE_COLUMNS, corrupt(GOOD, 1, 3, ""), "empty values"),
        (COVERAGE_COLUMNS, corrupt(GOOD, 0, 2, "2026/06/30"), "dates not YYYY-MM-DD"),
        (COVERAGE_COLUMNS, corrupt(GOOD, 0, 2, "2026-13-45"), "dates not YYYY-MM-DD"),
        (COVERAGE_COLUMNS, corrupt(GOOD, 0, 4, "yes"), "not True/False"),
        (COVERAGE_COLUMNS, GOOD + [["1301", "Short row"]], "wrong number of fields"),
        (COVERAGE_COLUMNS[:4], [r[:4] for r in GOOD], "refusing to guess"),
        (["ticker", "name", "has_any_financial_summary"], [["7203", "T", "True"]], "refusing"),
        ([*COVERAGE_COLUMNS[:4][::-1], "has_any_financial_summary"], GOOD, "refusing"),
        (COVERAGE_COLUMNS, [], "has a header but no rows"),
    ],
)
def test_defective_input_writes_nothing(mod, store, capsys, header, rows, message):
    (store / "financial_summary_coverage.csv").write_text(csv_text(header, rows))
    before = snapshot(store)
    assert run(mod, store, "--skip-state-check") == 2
    assert message in capsys.readouterr().err
    assert snapshot(store) == before


def test_empty_or_missing_or_unreadable_coverage_writes_nothing(mod, store, capsys):
    assert run(mod, store) == 2  # missing
    assert "file not found" in capsys.readouterr().err
    (store / "financial_summary_coverage.csv").write_text("")
    assert run(mod, store) == 2  # empty
    (store / "financial_summary_coverage.csv").unlink()
    (store / "financial_summary_coverage.csv").mkdir()  # a directory where the file should be
    assert run(mod, store) == 4  # I/O problem, reported as such
    assert not (store / "topix1000_usable_tickers.csv").exists()


def test_row_count_expectation_is_enforced(mod, store, capsys):
    write_coverage_like_daily_script(store / "financial_summary_coverage.csv", universe_rows(973))
    assert run(mod, store, "--skip-state-check", "--expect-rows", "974") == 2
    assert "973 rows, expected 974" in capsys.readouterr().err
    assert not (store / "topix1000_usable_tickers.csv").exists()


def test_missing_fields_need_an_explicit_opt_in(mod, store):
    rows = GOOD + [["1301", "", "2026-06-30", "X", "False"]]
    (store / "financial_summary_coverage.csv").write_text(csv_text(COVERAGE_COLUMNS, rows))
    assert run(mod, store, "--skip-state-check") == 2
    assert run(mod, store, "--skip-state-check", "--allow-missing-fields") == 0


# --------------------------------------------------------------------------
# evidence: the daily state, the original's hash, and other files
# --------------------------------------------------------------------------
def test_coverage_that_does_not_match_the_daily_state_is_rejected(mod, store, capsys):
    write_coverage_like_daily_script(store / "financial_summary_coverage.csv", universe_rows())
    write_state(store, n_true=100, total=974)  # the state describes a different coverage file
    assert run(mod, store) == 2
    assert "MISMATCH" in capsys.readouterr().err
    assert run(mod, store, "--skip-state-check") == 0


def test_expected_sha256_proves_identity_or_blocks_the_write(mod, store, capsys):
    rows = universe_rows(5)
    write_coverage_like_daily_script(store / "financial_summary_coverage.csv", rows, n_true=2)
    write_state(store, n_true=2, total=5)
    original = pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)
    original_bytes = original.to_csv(
        index=False
    ).encode()  # what a pandas-written original looks like

    wrong = "0" * 64
    assert run(mod, store, "--expect-sha256", wrong) == 2
    assert "nothing was written" in capsys.readouterr().err
    assert not (store / "topix1000_usable_tickers.csv").exists()

    assert run(mod, store, "--expect-sha256", hashlib.sha256(original_bytes).hexdigest()) == 0
    assert (store / "topix1000_usable_tickers.csv").read_bytes() == original_bytes
    (sidecar,) = list(store.glob("*.recovery_*.json"))
    assert json.loads(sidecar.read_text())["byte_identity"].startswith("PROVEN")


def test_cross_check_against_another_ticker_file(mod, store, tmp_path, capsys):
    rows = universe_rows(5)
    write_coverage_like_daily_script(store / "financial_summary_coverage.csv", rows, n_true=2)
    write_state(store, n_true=2, total=5)
    prices = tmp_path / "prices.csv"
    subset = [r[0] for r in rows[:3]]
    prices.write_text("date,ticker,close\n" + "\n".join(f"2026-01-05,{t},1" for t in subset) + "\n")
    assert run(mod, store, "--dry-run", "--cross-check-file", str(prices), "--require-subset") == 0
    out = capsys.readouterr().out
    assert '"in_file_not_in_rebuild": 0' in out and '"in_rebuild_not_in_file": 2' in out

    prices.write_text("date,ticker,close\n2026-01-05,9999,1\n")
    assert run(mod, store, "--dry-run", "--cross-check-file", str(prices), "--require-subset") == 2
    assert "not in the rebuild" in capsys.readouterr().err


# --------------------------------------------------------------------------
# the environment variable the daily script honours also decides the output path
# --------------------------------------------------------------------------
def test_output_path_follows_the_universe_file_environment_variable(
    mod, store, tmp_path, monkeypatch
):
    write_coverage_like_daily_script(
        store / "financial_summary_coverage.csv", universe_rows(5), n_true=2
    )
    write_state(store, n_true=2, total=5)
    target = tmp_path / "elsewhere" / "universe.csv"
    monkeypatch.setenv("JP_STOCK_ANALYSIS_UNIVERSE_FILE", str(target))
    assert run(mod, store) == 0
    assert target.exists()
    assert not (store / "topix1000_usable_tickers.csv").exists()

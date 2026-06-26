"""No-network tests for production J-Quants daily bars pipeline."""

from __future__ import annotations

import csv
import json
from datetime import date

from jp_stock_analysis.cli import main
from jp_stock_analysis.errors import ProviderError
from jp_stock_analysis.validation.jquants_daily_bars import (
    OUTPUT_COLUMNS,
    fetch_jquants_daily_bars_incremental,
    load_daily_bars_csv,
    map_daily_bar_row,
    merge_daily_bars,
    normalize_jquants_code_to_ticker,
    validate_daily_bars_quality,
    write_daily_bars_csv,
)


class _RowsProvider:
    api_version = "v2"

    def __init__(self, rows_by_date=None, failures=None):
        self.rows_by_date = rows_by_date or {}
        self.failures = failures or {}

    def fetch_daily_bar_rows_by_date(self, target_date, *, allow_network=False):
        day = target_date.isoformat() if hasattr(target_date, "isoformat") else str(target_date)
        if day in self.failures:
            raise self.failures[day]
        return list(self.rows_by_date.get(day, []))


def _row(code="13010", day="2024-12-13", adj_close=4070.0):
    return {
        "Date": day,
        "Code": code,
        "O": 4065.0,
        "H": 4105.0,
        "L": 4055.0,
        "C": 4070.0,
        "UL": "0",
        "LL": "0",
        "Vo": 22700.0,
        "Va": 92458500.0,
        "AdjFactor": 1.0,
        "AdjO": 4065.0,
        "AdjH": 4105.0,
        "AdjL": 4055.0,
        "AdjC": adj_close,
        "AdjVo": 22700.0,
    }


def _universe(path, rows=("1301", "7203")):
    path.write_text("ticker\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_v2_short_field_mapping_and_schema_are_stable():
    mapped = map_daily_bar_row(_row())
    assert mapped["ticker"] == "1301"
    assert mapped["date"] == "2024-12-13"
    assert mapped["adj_close"] == 4070.0
    assert mapped["turnover_value"] == 92458500.0
    assert mapped["volume"] == 22700.0
    assert mapped["upper_limit_flag"] == "0"
    assert list(mapped.keys()) == OUTPUT_COLUMNS
    assert json.loads(str(mapped["source_fields_json"]))["AdjC"] == 4070.0


def test_code_normalization_rules():
    assert normalize_jquants_code_to_ticker("13010") == "1301"
    assert normalize_jquants_code_to_ticker("72030") == "7203"
    assert normalize_jquants_code_to_ticker("12345") == "12345"
    assert normalize_jquants_code_to_ticker("167A") == "167A"


def test_merge_avoids_duplicate_ticker_date_rows(tmp_path):
    first = map_daily_bar_row(_row(adj_close=100.0))
    second = map_daily_bar_row(_row(adj_close=101.0))
    merged = merge_daily_bars([first], [second])
    assert len(merged) == 1
    assert merged[0]["adj_close"] == 101.0
    out = tmp_path / "bars.csv"
    write_daily_bars_csv(merged, out)
    assert list(csv.DictReader(out.open(encoding="utf-8")))  # schema-readable


def test_failed_date_is_recorded_in_state_and_no_secret_written(tmp_path, monkeypatch):
    monkeypatch.setenv("JQUANTS_API_KEY", "TOPSECRET")
    universe = _universe(tmp_path / "u.csv", rows=("1301",))
    provider = _RowsProvider(
        {"2024-12-13": [_row()]},
        failures={"2024-12-12": ProviderError("HTTP 500 without secret")},
    )
    result = fetch_jquants_daily_bars_incremental(
        provider,
        universe_file=universe,
        store_dir=tmp_path / "store",
        start_date=date(2024, 12, 12),
        end_date=date(2024, 12, 13),
        allow_network=True,
        sleep_seconds=0,
        sleep_fn=lambda _: None,
    )
    assert result.failed_dates == {"2024-12-12": "HTTP 500 without secret"}
    state_text = (tmp_path / "store" / "daily_bars_fetch_state.json").read_text(encoding="utf-8")
    csv_text = (tmp_path / "store" / "prices_daily_bars.csv").read_text(encoding="utf-8")
    assert "TOPSECRET" not in state_text
    assert "TOPSECRET" not in csv_text
    assert json.loads(state_text)["api_key_status"] == "PRESENT"
    assert json.loads(state_text)["secret_included"] is False


def test_quality_report_detects_duplicates_nulls_and_adjusted_close_consistency(tmp_path):
    universe = _universe(tmp_path / "u.csv", rows=("1301", "7203"))
    rows = [
        map_daily_bar_row(_row()),
        map_daily_bar_row(_row(adj_close=4071.0)),
        map_daily_bar_row(_row(code="72030", adj_close=2000.0)),
    ]
    adjusted = tmp_path / "adjusted.csv"
    adjusted.write_text(
        "ticker,date,close\n1301,2024-12-13,4070\n7203,2024-12-13,2000\n",
        encoding="utf-8",
    )
    report, coverage = validate_daily_bars_quality(
        rows,
        universe_file=universe,
        input_file=tmp_path / "bars.csv",
        adjusted_close_file=adjusted,
    )
    assert report["duplicate_ticker_date_rows"] == 1
    assert coverage["adj_close"]["null_count"] == 0
    assert report["adjusted_close_consistency"]["mismatch_row_count"] == 1


def test_verify_daily_bars_cli_writes_reports(tmp_path):
    universe = _universe(tmp_path / "u.csv", rows=("1301",))
    store = tmp_path / "store"
    write_daily_bars_csv([map_daily_bar_row(_row())], store / "prices_daily_bars.csv")
    adjusted = tmp_path / "adjusted.csv"
    adjusted.write_text("ticker,date,close\n1301,2024-12-13,4070\n", encoding="utf-8")
    rc = main(
        [
            "verify-jquants-daily-bars",
            "--store-dir",
            str(store),
            "--universe-file",
            str(universe),
            "--adjusted-close-file",
            str(adjusted),
        ]
    )
    assert rc == 0
    assert (store / "daily_bars_quality_report.json").exists()
    assert (store / "daily_bars_field_coverage_report.json").exists()


def test_daily_bars_output_can_be_loaded(tmp_path):
    out = tmp_path / "bars.csv"
    write_daily_bars_csv([map_daily_bar_row(_row())], out)
    assert load_daily_bars_csv(out)[0]["ticker"] == "1301"

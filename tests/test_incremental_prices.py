"""Tests for incremental J-Quants price store. No network."""

from __future__ import annotations

import csv
import json
from datetime import date

import pytest

from jp_stock_analysis.cli import main
from jp_stock_analysis.data.incremental_prices import (
    STORE_CSV,
    load_universe_file,
    update_incremental_price_store,
    verify_price_store,
)
from jp_stock_analysis.errors import DataValidationError, ProviderError
from jp_stock_analysis.schemas import PriceBar


class _DateProvider:
    api_version = "v2"

    def __init__(self, bars_by_date=None, failures=None):
        self.bars_by_date = bars_by_date or {}
        self.failures = {k: list(v) for k, v in (failures or {}).items()}
        self.calls = []

    def endpoint_url(self, dataset):
        return f"https://api.jquants.com/v2/{dataset}"

    def fetch_daily_bars_by_date(self, target_date, *, allow_network=False):
        day = target_date.isoformat() if hasattr(target_date, "isoformat") else str(target_date)
        self.calls.append((day, allow_network))
        if self.failures.get(day):
            item = self.failures[day].pop(0)
            if isinstance(item, Exception):
                raise item
        return list(self.bars_by_date.get(day, []))


def _bar(ticker, day, close, adjusted_close):
    return PriceBar(
        ticker=ticker,
        date=date.fromisoformat(day),
        close=close,
        adjusted_close=adjusted_close,
    )


def _universe(path, rows=("7203", "9984")):
    path.write_text("ticker\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_load_universe_csv_and_txt_normalizes(tmp_path):
    csv_path = tmp_path / "u.csv"
    csv_path.write_text("Code\n72030\n84\n72030\n", encoding="utf-8")
    assert load_universe_file(csv_path) == ["7203", "0084"]
    txt_path = tmp_path / "u.txt"
    txt_path.write_text("# c\n99840\n7203\n", encoding="utf-8")
    assert load_universe_file(txt_path) == ["9984", "7203"]


def test_incremental_append_filters_universe_and_writes_adjusted_close(tmp_path):
    universe = _universe(tmp_path / "u.csv")
    provider = _DateProvider(
        {
            "2026-03-23": [
                _bar("7203", "2026-03-23", 100, 99),
                _bar("9984", "2026-03-23", 200, 198),
                _bar("1111", "2026-03-23", 300, 297),
            ]
        }
    )
    result = update_incremental_price_store(
        provider,
        universe_file=universe,
        store_dir=tmp_path / "store",
        start_date="2026-03-23",
        end_date="2026-03-23",
        allow_network=True,
        sleep_fn=lambda _: None,
    )
    assert result.rows_added == 2
    rows = _rows(tmp_path / "store" / STORE_CSV)
    assert rows == [
        {"ticker": "7203", "date": "2026-03-23", "close": "99"},
        {"ticker": "9984", "date": "2026-03-23", "close": "198"},
    ]
    state = json.loads((tmp_path / "store" / "fetch_state.json").read_text(encoding="utf-8"))
    assert state["price_field"] == "adjusted_close"
    assert state["raw_close_fallback"] is False


def test_append_dedupes_and_resume_skips_complete_dates(tmp_path):
    universe = _universe(tmp_path / "u.csv")
    provider = _DateProvider(
        {
            "2026-03-23": [
                _bar("7203", "2026-03-23", 100, 99),
                _bar("9984", "2026-03-23", 200, 198),
            ],
            "2026-03-24": [
                _bar("7203", "2026-03-24", 101, 100),
                _bar("9984", "2026-03-24", 201, 200),
            ],
        }
    )
    update_incremental_price_store(
        provider,
        universe_file=universe,
        store_dir=tmp_path / "store",
        start_date="2026-03-23",
        end_date="2026-03-24",
        allow_network=True,
        sleep_fn=lambda _: None,
    )
    provider.calls.clear()
    result = update_incremental_price_store(
        provider,
        universe_file=universe,
        store_dir=tmp_path / "store",
        start_date="2026-03-23",
        end_date="2026-03-24",
        allow_network=True,
        sleep_fn=lambda _: None,
    )
    assert result.rows_added == 0
    assert provider.calls == []
    assert len(_rows(tmp_path / "store" / STORE_CSV)) == 4


def test_partial_existing_date_is_refetched_and_deduped(tmp_path):
    universe = _universe(tmp_path / "u.csv")
    store = tmp_path / "store"
    store.mkdir()
    (store / STORE_CSV).write_text("ticker,date,close\n7203,2026-03-23,1\n", encoding="utf-8")
    provider = _DateProvider(
        {
            "2026-03-23": [
                _bar("7203", "2026-03-23", 100, 99),
                _bar("9984", "2026-03-23", 200, 198),
            ]
        }
    )
    update_incremental_price_store(
        provider,
        universe_file=universe,
        store_dir=store,
        start_date="2026-03-23",
        end_date="2026-03-23",
        allow_network=True,
        sleep_fn=lambda _: None,
    )
    rows = _rows(store / STORE_CSV)
    assert rows == [
        {"ticker": "7203", "date": "2026-03-23", "close": "99"},
        {"ticker": "9984", "date": "2026-03-23", "close": "198"},
    ]


def test_429_retry_backoff_with_mocked_sleep(tmp_path):
    universe = _universe(tmp_path / "u.csv", rows=("7203",))
    provider = _DateProvider(
        {"2026-03-23": [_bar("7203", "2026-03-23", 100, 99)]},
        failures={"2026-03-23": [ProviderError("J-Quants request failed (HTTP 429)")]}
    )
    sleeps = []
    update_incremental_price_store(
        provider,
        universe_file=universe,
        store_dir=tmp_path / "store",
        start_date="2026-03-23",
        end_date="2026-03-23",
        allow_network=True,
        sleep_seconds=3,
        backoff_multiplier=2,
        sleep_fn=sleeps.append,
    )
    assert sleeps == [3]
    state = json.loads((tmp_path / "store" / "fetch_state.json").read_text(encoding="utf-8"))
    assert state["retry_events"][0]["reason"] == "rate_limited"


def test_missing_adjusted_close_fails_without_raw_fallback(tmp_path):
    universe = _universe(tmp_path / "u.csv", rows=("7203",))
    provider = _DateProvider(
        {"2026-03-23": [_bar("7203", "2026-03-23", 100, None)]}
    )
    update_incremental_price_store(
        provider,
        universe_file=universe,
        store_dir=tmp_path / "store",
        start_date="2026-03-23",
        end_date="2026-03-23",
        allow_network=True,
        continue_on_date_error=True,
        sleep_fn=lambda _: None,
    )
    state = json.loads((tmp_path / "store" / "fetch_state.json").read_text(encoding="utf-8"))
    assert "adjusted_close requested but missing" in state["failed_dates"]["2026-03-23"]
    assert _rows(tmp_path / "store" / STORE_CSV) == []


def test_refuses_to_mix_raw_and_adjusted_store(tmp_path):
    universe = _universe(tmp_path / "u.csv", rows=("7203",))
    provider = _DateProvider({"2026-03-23": [_bar("7203", "2026-03-23", 100, 99)]})
    update_incremental_price_store(
        provider,
        universe_file=universe,
        store_dir=tmp_path / "store",
        start_date="2026-03-23",
        end_date="2026-03-23",
        price_field="close",
        allow_network=True,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(DataValidationError, match="refusing to mix"):
        update_incremental_price_store(
            provider,
            universe_file=universe,
            store_dir=tmp_path / "store",
            start_date="2026-03-23",
            end_date="2026-03-23",
            price_field="adjusted_close",
            allow_network=True,
            sleep_fn=lambda _: None,
        )


def test_verify_price_store_reports_latest_eligibility(tmp_path):
    universe = _universe(tmp_path / "u.csv", rows=("7203",))
    provider = _DateProvider(
        {
            f"2026-03-{day:02d}": [_bar("7203", f"2026-03-{day:02d}", 100 + day, 90 + day)]
            for day in range(2, 25)
            if date(2026, 3, day).weekday() < 5
        }
    )
    update_incremental_price_store(
        provider,
        universe_file=universe,
        store_dir=tmp_path / "store",
        start_date="2026-03-02",
        end_date="2026-03-24",
        allow_network=True,
        sleep_fn=lambda _: None,
    )
    report = verify_price_store(tmp_path / "store", universe_file=universe)
    assert report["duplicate_ticker_date_rows"] == 0
    assert report["latest_eligible_decision_dates"]["h5"] == "2026-03-16"
    assert report["latest_eligible_decision_dates"]["h20"] is None
    assert (tmp_path / "store" / "coverage_report.json").exists()


def test_incremental_cli_smoke_cache_only_missing_returns_nonzero(tmp_path, capsys):
    universe = _universe(tmp_path / "u.csv", rows=("7203",))
    rc = main(
        [
            "fetch-jquants-prices-incremental",
            "--universe-file", str(universe),
            "--store-dir", str(tmp_path / "store"),
            "--start-date", "2026-03-23",
            "--end-date", "2026-03-23",
            "--price-field", "adjusted_close",
        ]
    )
    assert rc == 2
    assert "failed_dates=1" in capsys.readouterr().out


def test_existing_fetch_jquants_prices_cli_still_works(tmp_path):
    from conftest import FIXTURES_DIR

    out = tmp_path / "adj.csv"
    rc = main(
        [
            "fetch-jquants-prices",
            "--tickers", "7203",
            "--out", str(out),
            "--cache-dir", str(FIXTURES_DIR / "jquants_cache"),
            "--price-field", "adjusted_close",
        ]
    )
    assert rc == 0
    assert out.read_text(encoding="utf-8").splitlines()[0] == "ticker,date,close"

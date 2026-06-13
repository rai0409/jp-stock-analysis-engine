"""Tests for the J-Quants price CSV export wrapper. Fully offline.

Network is never exercised: tests use the synthetic cache fixture under
tests/fixtures/jquants_cache/ (cache-only provider) and inline fake providers.
No real prices, no live HTTP.
"""

from __future__ import annotations

from datetime import date

import pytest
from conftest import FIXTURES_DIR

from jp_stock_analysis.cli import main
from jp_stock_analysis.errors import DataValidationError, ProviderError
from jp_stock_analysis.providers.jquants import JQuantsProvider
from jp_stock_analysis.schemas import PriceBar
from jp_stock_analysis.validation.jquants_prices import export_jquants_prices_csv

CACHE_DIR = FIXTURES_DIR / "jquants_cache"


class _FakeProvider:
    """Returns canned bars per ticker; records calls. No network."""

    def __init__(self, bars_by_ticker):
        self._bars = bars_by_ticker
        self.calls = []

    def get_prices(self, ticker, from_date=None, to_date=None):
        self.calls.append((ticker, from_date, to_date))
        return list(self._bars.get(ticker, []))


def _bar(ticker, day, close, adjusted_close=None):
    return PriceBar(ticker=ticker, date=day, close=close, adjusted_close=adjusted_close)


def _read(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_export_from_cache_fixture_offline(tmp_path):
    provider = JQuantsProvider(cache_dir=CACHE_DIR, live=False)  # cache-only: no network
    out = tmp_path / "raw.csv"
    result = export_jquants_prices_csv(provider, ["7203"], out)
    lines = _read(out)
    assert lines[0] == "ticker,date,close"
    assert lines[1] == "7203,2025-01-06,2000"  # raw close from the fixture
    assert result.rows_per_ticker["7203"] == 80
    assert result.total_rows_written == 80
    assert any("raw close" in w for w in result.warnings)


def test_export_sorted_by_ticker_then_date(tmp_path):
    provider = _FakeProvider(
        {
            "4107": [_bar("4107", date(2026, 3, 31), 5060), _bar("4107", date(2026, 3, 28), 5050)],
            "3928": [_bar("3928", date(2026, 3, 28), 1010)],
        }
    )
    out = tmp_path / "raw.csv"
    export_jquants_prices_csv(provider, ["4107", "3928"], out)
    assert _read(out) == [
        "ticker,date,close",
        "3928,2026-03-28,1010",
        "4107,2026-03-28,5050",
        "4107,2026-03-31,5060",
    ]


def test_adjusted_close_written_into_close_column(tmp_path):
    """--price-field adjusted_close puts AdjC values into the 'close' column."""
    provider = _FakeProvider(
        {
            "4107": [
                _bar("4107", date(2025, 12, 26), 43050, adjusted_close=4783.3),
                _bar("4107", date(2025, 12, 29), 4985, adjusted_close=4985.0),
            ]
        }
    )
    out = tmp_path / "adj.csv"
    result = export_jquants_prices_csv(
        provider, ["4107"], out, price_field="adjusted_close"
    )
    assert result.price_field == "adjusted_close"
    # column header is still 'close'; values are the adjusted closes, not raw
    assert _read(out) == [
        "ticker,date,close",
        "4107,2025-12-26,4783.3",
        "4107,2025-12-29,4985",
    ]
    assert any("adjusted close" in w for w in result.warnings)


def test_adjusted_close_missing_fails_clearly_no_fallback(tmp_path):
    """If any row lacks adjusted close, fail clearly — never silently fall back."""
    provider = _FakeProvider(
        {
            "3928": [
                _bar("3928", date(2026, 3, 18), 1000, adjusted_close=1000.0),
                _bar("3928", date(2026, 3, 19), 1010, adjusted_close=None),
            ]
        }
    )
    out = tmp_path / "adj.csv"
    with pytest.raises(DataValidationError, match="adjusted_close requested but missing"):
        export_jquants_prices_csv(provider, ["3928"], out, price_field="adjusted_close")
    assert not out.exists()  # no partial file


def test_default_price_field_is_raw_close(tmp_path):
    provider = _FakeProvider(
        {"3928": [_bar("3928", date(2026, 3, 19), 1010, adjusted_close=999.0)]}
    )
    out = tmp_path / "raw.csv"
    result = export_jquants_prices_csv(provider, ["3928"], out)  # no price_field
    assert result.price_field == "close"
    assert _read(out)[1] == "3928,2026-03-19,1010"  # raw close, not 999


def test_invalid_price_field_raises(tmp_path):
    provider = _FakeProvider({"3928": [_bar("3928", date(2026, 3, 19), 1010)]})
    with pytest.raises(DataValidationError, match="invalid price_field"):
        export_jquants_prices_csv(
            provider, ["3928"], tmp_path / "x.csv", price_field="bogus"
        )


def test_cli_adjusted_close_smoke_from_cache(tmp_path, capsys):
    """The cache fixture carries AdjustmentClose; --price-field adjusted_close works."""
    out = tmp_path / "adj.csv"
    code = main(
        [
            "fetch-jquants-prices",
            "--tickers", "7203",
            "--out", str(out),
            "--cache-dir", str(CACHE_DIR),
            "--price-field", "adjusted_close",
        ]
    )
    assert code == 0
    assert "(adjusted_close)" in capsys.readouterr().out
    lines = _read(out)
    assert lines[0] == "ticker,date,close"
    # fixture's AdjustmentClose equals close, so first adjusted value is 2000
    assert lines[1] == "7203,2025-01-06,2000"


def test_date_range_passed_through_to_provider(tmp_path):
    provider = _FakeProvider({"3928": [_bar("3928", date(2026, 3, 28), 1010)]})
    out = tmp_path / "raw.csv"
    export_jquants_prices_csv(
        provider, ["3928"], out, from_date=date(2026, 3, 28), to_date=date(2026, 6, 30)
    )
    assert provider.calls == [("3928", date(2026, 3, 28), date(2026, 6, 30))]


def test_empty_ticker_raises_blocked_not_silent(tmp_path):
    provider = _FakeProvider({"3928": [_bar("3928", date(2026, 3, 28), 1010)]})
    out = tmp_path / "raw.csv"
    with pytest.raises(DataValidationError, match="no price rows returned for ticker"):
        export_jquants_prices_csv(provider, ["3928", "4107"], out)
    assert not out.exists()  # no partial/empty file written


def test_no_tickers_raises(tmp_path):
    with pytest.raises(DataValidationError, match="no tickers requested"):
        export_jquants_prices_csv(_FakeProvider({}), [], tmp_path / "raw.csv")


def test_cache_only_missing_cache_propagates_provider_error(tmp_path):
    provider = JQuantsProvider(cache_dir=tmp_path / "empty_cache", live=False)
    with pytest.raises(ProviderError, match="no cached J-Quants"):
        export_jquants_prices_csv(provider, ["9999"], tmp_path / "raw.csv")


def test_cli_smoke_cache_only(tmp_path, capsys):
    out = tmp_path / "raw.csv"
    code = main(
        [
            "fetch-jquants-prices",
            "--tickers",
            "7203",
            "--out",
            str(out),
            "--cache-dir",
            str(CACHE_DIR),
        ]
    )
    assert code == 0
    assert out.exists()
    assert "Exported 80 rows" in capsys.readouterr().out


def test_cli_missing_cache_returns_nonzero(tmp_path, capsys):
    out = tmp_path / "raw.csv"
    code = main(
        [
            "fetch-jquants-prices",
            "--tickers",
            "3928,4107,4264",
            "--out",
            str(out),
            "--cache-dir",
            str(tmp_path / "empty_cache"),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "no cached J-Quants" in err
    assert not out.exists()


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["fetch-jquants-prices", "--help"])
    assert exc.value.code == 0
    assert "--allow-network" in capsys.readouterr().out

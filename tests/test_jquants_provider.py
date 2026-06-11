"""Tests for the cache-first J-Quants provider. Fully offline.

Live HTTP is never exercised: tests cover the cache path, the explicit
missing-key / missing-cache error paths, and schema mapping against synthetic
cache fixtures under tests/fixtures/jquants_cache/.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from conftest import FIXTURES_DIR

from jp_stock_analysis.cli import main
from jp_stock_analysis.errors import ProviderError
from jp_stock_analysis.providers.jquants import ENV_API_KEY, JQuantsProvider

CACHE_DIR = FIXTURES_DIR / "jquants_cache"


def _provider(**kwargs) -> JQuantsProvider:
    kwargs.setdefault("cache_dir", CACHE_DIR)
    return JQuantsProvider(**kwargs)


def test_construct_without_api_key(monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    provider = _provider()
    assert provider.live is False  # constructing never needs credentials


def test_cache_miss_without_live_raises_provider_error(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    provider = JQuantsProvider(cache_dir=tmp_path / "empty")
    with pytest.raises(ProviderError, match="no cached J-Quants"):
        provider.get_prices("7203")
    with pytest.raises(ProviderError, match="no cached J-Quants"):
        provider.get_statements("7203")
    with pytest.raises(ProviderError, match="no cached J-Quants"):
        provider.get_metadata("7203")


def test_live_fetch_without_api_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    provider = JQuantsProvider(cache_dir=tmp_path / "empty", live=True)
    with pytest.raises(ProviderError, match=ENV_API_KEY):
        provider.get_prices("7203")


def test_cached_daily_quotes_map_to_price_bars(monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    bars = _provider().get_prices("7203")
    assert len(bars) == 80
    assert bars == sorted(bars, key=lambda bar: bar.date)
    first = bars[0]
    assert first.ticker == "7203"  # requested code, not the 5-digit row Code
    assert first.date == date(2025, 1, 6)
    assert first.close == 2000.0
    assert first.adjusted_close == first.close
    assert first.volume == 1_000_000.0


def test_price_date_range_filtering():
    bars = _provider().get_prices("7203", from_date="2025-02-01", to_date=date(2025, 2, 28))
    assert bars
    assert all(date(2025, 2, 1) <= bar.date <= date(2025, 2, 28) for bar in bars)
    assert len(bars) < 80


def test_cached_statements_map_to_financial_statements():
    statements = _provider().get_statements("7203")
    assert [s.fiscal_year for s in statements] == [2023, 2024]
    latest = statements[-1]
    assert latest.ticker == "7203"
    assert latest.fiscal_period == "FY"
    assert latest.revenue == 4.0e12
    assert latest.eps == 213.0
    assert latest.bps == 2000.0
    assert latest.shares_outstanding == 1.5e9
    assert latest.operating_cash_flow == 5.0e11
    # J-Quants statements carry no capex column: must stay None, never fabricated
    assert latest.capital_expenditure is None
    assert latest.source_metadata["source"] == "jquants"


def test_cached_listed_info_maps_to_company_metadata():
    company = _provider().get_metadata("7203")
    assert company is not None
    assert company.company_name == "サンプル自動車株式会社"
    assert company.sector == "輸送用機器"
    assert company.market == "プライム"


def test_invalid_cache_file_raises_provider_error(tmp_path):
    cache = tmp_path / "cache"
    (cache / "daily_quotes").mkdir(parents=True)
    (cache / "daily_quotes" / "7203.json").write_text("{\"not\": \"a list\"}", encoding="utf-8")
    with pytest.raises(ProviderError, match="invalid J-Quants cache"):
        JQuantsProvider(cache_dir=cache).get_prices("7203")


def test_live_fetch_uses_injected_http_and_writes_cache(tmp_path, monkeypatch):
    """Fetch logic verified with an injected fake transport — no real network."""
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    pages = [
        {"daily_quotes": [{"Date": "2025-01-06", "Code": "72030", "Close": 100.0}],
         "pagination_key": "next"},
        {"daily_quotes": [{"Date": "2025-01-07", "Code": "72030", "Close": 101.0}]},
    ]
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_http_get(url: str, headers: dict[str, str]) -> dict:
        calls.append((url, headers))
        return pages[len(calls) - 1]

    provider = JQuantsProvider(
        cache_dir=tmp_path / "cache", live=True, api_key="test-key", http_get=fake_http_get
    )
    bars = provider.get_prices("7203")
    assert [bar.close for bar in bars] == [100.0, 101.0]
    assert len(calls) == 2  # pagination followed
    assert all(headers == {"x-api-key": "test-key"} for _, headers in calls)
    assert "pagination_key=next" in calls[1][0]

    # cache was written and is readable offline afterwards
    cache_file = provider.cache_path("daily_quotes", "7203")
    assert cache_file.exists()
    assert len(json.loads(cache_file.read_text(encoding="utf-8"))) == 2
    offline = JQuantsProvider(cache_dir=tmp_path / "cache")  # live=False, no key
    assert len(offline.get_prices("7203")) == 2


def test_cli_jquants_cache_smoke(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    argv = [
        "analyze",
        "--provider", "jquants-cache",
        "--jquants-cache-dir", str(CACHE_DIR),
        "--jquants-code", "7203",
        "--output-dir", str(tmp_path),
    ]
    assert main(argv) == 0
    assert (tmp_path / "screening.csv").exists()
    assert (tmp_path / "7203.md").exists()
    payload = json.loads((tmp_path / "screening.json").read_text(encoding="utf-8"))
    assert payload["signal_mode"] == "analysis_only"  # default mode unchanged
    entry = payload["results"][0]
    assert entry["ticker"] == "7203"
    assert entry["company_name"] == "サンプル自動車株式会社"
    assert entry["fundamentals"]["fiscal_year"] == 2024
    assert entry["momentum"]["observations"] == 80
    assert "signal" not in entry and "screening_label" not in entry


def test_cli_jquants_live_without_key_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    argv = [
        "analyze",
        "--provider", "jquants-live",
        "--jquants-cache-dir", str(tmp_path / "empty_cache"),
        "--jquants-code", "9999",
        "--output-dir", str(tmp_path / "out"),
    ]
    assert main(argv) == 1
    assert ENV_API_KEY in capsys.readouterr().err
    assert not (tmp_path / "out").exists()  # nothing written on failure


def test_cli_jquants_requires_code(tmp_path):
    argv = [
        "analyze",
        "--provider", "jquants-cache",
        "--output-dir", str(tmp_path),
    ]
    with pytest.raises(SystemExit):
        main(argv)

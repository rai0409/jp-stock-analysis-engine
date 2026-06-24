"""Tests for the cache-first J-Quants provider. Fully offline.

Live HTTP is never exercised: tests cover the cache path, the explicit
missing-key / missing-cache error paths, and schema mapping against synthetic
cache fixtures under tests/fixtures/jquants_cache/.
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import date

import pytest
from conftest import FIXTURES_DIR

from jp_stock_analysis.cli import main
from jp_stock_analysis.errors import ProviderError
from jp_stock_analysis.providers.jquants import (
    ENV_API_KEY,
    ENV_API_VERSION,
    ENV_BASE_URL,
    JQuantsProvider,
    _map_listed_info,
)

CACHE_DIR = FIXTURES_DIR / "jquants_cache"

_ENDPOINT_ENV_VARS = (
    ENV_BASE_URL,
    ENV_API_VERSION,
    "JQUANTS_DAILY_QUOTES_PATH",
    "JQUANTS_STATEMENTS_PATH",
    "JQUANTS_LISTED_INFO_PATH",
)


@pytest.fixture(autouse=True)
def _clean_endpoint_env(monkeypatch):
    """Endpoint config must come from the test, not the developer's shell."""
    for name in _ENDPOINT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _raise_http_error(body: str, code: int = 403):
    def transport(url: str, headers: dict[str, str]) -> dict:
        raise urllib.error.HTTPError(
            url, code, "Forbidden", None, io.BytesIO(body.encode("utf-8"))
        )

    return transport


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


def test_listed_info_mapping_preserves_useful_source_metadata():
    company = _map_listed_info(
        {
            "Code": "167A",
            "CoName": "サンプル株式会社",
            "CoNameEn": "Sample Inc.",
            "S17Nm": "情報通信・サービスその他",
            "S33Nm": "情報・通信業",
            "MktNm": "グロース",
        },
        "167A",
    )

    assert company.source_metadata == {
        "source": "jquants",
        "raw_code": "167A",
        "company_name_en": "Sample Inc.",
        "sector_17": "情報通信・サービスその他",
        "sector_33": "情報・通信業",
        "market": "グロース",
    }


def test_invalid_cache_file_raises_provider_error(tmp_path):
    cache = tmp_path / "cache"
    (cache / "daily_quotes").mkdir(parents=True)
    (cache / "daily_quotes" / "7203.json").write_text("{\"not\": \"a list\"}", encoding="utf-8")
    with pytest.raises(ProviderError, match="invalid J-Quants cache"):
        JQuantsProvider(cache_dir=cache).get_prices("7203")


def test_live_fetch_uses_injected_http_and_writes_cache(tmp_path, monkeypatch):
    """Fetch logic verified with an injected fake transport — no real network.

    Uses the V2 response shape: rows under ``data`` with abbreviated field names.
    """
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    pages = [
        {"data": [{"Date": "2025-01-06", "Code": "72030", "C": 100.0}],
         "pagination_key": "next"},
        {"data": [{"Date": "2025-01-07", "Code": "72030", "C": 101.0}]},
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


def test_default_endpoint_urls():
    """Defaults resolve to the verified V2 routes."""
    provider = _provider()
    assert (
        provider.endpoint_url("daily_quotes")
        == "https://api.jquants.com/v2/equities/bars/daily"
    )
    assert provider.endpoint_url("statements") == "https://api.jquants.com/v2/fins/summary"
    assert provider.endpoint_url("listed_info") == "https://api.jquants.com/v2/equities/master"


def test_v2_daily_bars_fields_map_to_price_bars(tmp_path, monkeypatch):
    """The verified V2 /equities/bars/daily field names map correctly."""
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    row = {
        "Date": "2026-03-19", "Code": "39280",
        "O": 1100.0, "H": 1150.0, "L": 1090.0, "C": 1125.0,
        "AdjC": 1125.0, "Vo": 250000, "Va": 281250000,
    }

    def transport(url: str, headers: dict[str, str]) -> dict:
        return {"data": [row]}

    provider = JQuantsProvider(
        cache_dir=tmp_path / "c", live=True, api_key="k", http_get=transport
    )
    bars = provider.get_prices("3928")
    assert len(bars) == 1
    bar = bars[0]
    assert bar.ticker == "3928"  # requested code preserved, not the 5-digit row Code
    assert bar.date == date(2026, 3, 19)
    assert (bar.open, bar.high, bar.low, bar.close) == (1100.0, 1150.0, 1090.0, 1125.0)
    assert bar.adjusted_close == 1125.0
    assert bar.volume == 250000.0


def test_date_based_daily_bars_fetch_maps_codes_and_writes_cache(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    calls = []

    def transport(url: str, headers: dict[str, str]) -> dict:
        calls.append((url, headers))
        return {
            "data": [
                {"Date": "2026-03-24", "Code": "72030", "C": 100.0, "AdjC": 99.5},
                {"Date": "2026-03-24", "Code": "99840", "C": 200.0, "AdjC": 199.5},
            ]
        }

    provider = JQuantsProvider(
        cache_dir=tmp_path / "cache", live=True, api_key="k", http_get=transport
    )
    bars = provider.fetch_daily_bars_by_date("2026-03-24", allow_network=True)
    assert [(bar.ticker, bar.adjusted_close) for bar in bars] == [
        ("7203", 99.5),
        ("9984", 199.5),
    ]
    assert "date=2026-03-24" in calls[0][0]
    assert "code=" not in calls[0][0]
    assert provider.date_cache_path("daily_quotes", "2026-03-24").exists()

    offline = JQuantsProvider(cache_dir=tmp_path / "cache")
    assert len(offline.fetch_daily_bars_by_date("2026-03-24")) == 2


def test_endpoint_does_not_exist_maps_to_helpful_error(tmp_path):
    probe_body = (
        '{"message": "The requested endpoint does not exist. Please check the URL, '
        'HTTP method, and API version:https://jpx-jquants.com/spec/"}'
    )
    provider = JQuantsProvider(
        cache_dir=tmp_path / "empty",
        live=True,
        api_key="secret-key-123",
        http_get=_raise_http_error(probe_body),
    )
    with pytest.raises(ProviderError) as err:
        provider.get_prices("7203")
    message = str(err.value)
    assert "endpoint not found" in message
    assert "version or path" in message
    assert ENV_API_VERSION in message  # tells the user how to fix it
    assert "JQUANTS_DAILY_QUOTES_PATH" in message
    assert "jpx-jquants.com/spec" in message
    assert "secret-key-123" not in message  # never leak the key


def test_malformed_authorization_maps_to_helpful_error(tmp_path):
    body = '{"message": "Invalid Authorization header: Bearer token is malformed"}'
    provider = JQuantsProvider(
        cache_dir=tmp_path / "empty",
        live=True,
        api_key="secret-key-123",
        http_get=_raise_http_error(body),
    )
    with pytest.raises(ProviderError) as err:
        provider.get_prices("7203")
    message = str(err.value)
    assert "not a Bearer token" in message
    assert "x-api-key" in message
    assert "secret-key-123" not in message


def test_v1_retired_maps_to_migration_error(tmp_path):
    body = (
        '{"message": "J-QuantsはV2に移行しました。", '
        '"migration_url": "https://jpx-jquants.com/ja/spec/migration-v1-v2"}'
    )
    provider = JQuantsProvider(
        cache_dir=tmp_path / "empty",
        live=True,
        api_key="secret-key-123",
        http_get=_raise_http_error(body, code=410),
    )
    with pytest.raises(ProviderError) as err:
        provider.get_prices("7203")
    message = str(err.value)
    assert "V1 has been retired" in message
    assert "/equities/bars/daily" in message  # points at the correct V2 path
    assert "secret-key-123" not in message


def test_plan_coverage_limit_maps_to_helpful_error(tmp_path):
    body = (
        '{"message": "Your subscription covers the following dates: 2024-03-21 ~ '
        '2026-03-21. If you want more data, please check other plans:'
        'https://jpx-jquants.com/#dataset"}'
    )
    provider = JQuantsProvider(
        cache_dir=tmp_path / "empty",
        live=True,
        api_key="secret-key-123",
        http_get=_raise_http_error(body, code=400),
    )
    with pytest.raises(ProviderError) as err:
        provider.get_prices("3928", from_date="2026-03-28")
    message = str(err.value)
    assert "plan/date-coverage limit" in message
    assert "subscription" in message.lower()
    assert "secret-key-123" not in message


def test_other_http_errors_report_status_and_body(tmp_path):
    provider = JQuantsProvider(
        cache_dir=tmp_path / "empty",
        live=True,
        api_key="k",
        http_get=_raise_http_error('{"message": "rate limit"}', code=429),
    )
    with pytest.raises(ProviderError, match="HTTP 429"):
        provider.get_prices("7203")


def test_env_endpoint_overrides_change_request_url(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_BASE_URL, "https://alt.example")
    monkeypatch.setenv(ENV_API_VERSION, "v9")
    monkeypatch.setenv("JQUANTS_DAILY_QUOTES_PATH", "/markets/daily")
    calls: list[str] = []

    def transport(url: str, headers: dict[str, str]) -> dict:
        calls.append(url)
        return {"data": []}

    provider = JQuantsProvider(cache_dir=tmp_path / "c", live=True, api_key="k",
                               http_get=transport)
    provider.get_prices("7203")
    assert calls and calls[0].startswith("https://alt.example/v9/markets/daily?")
    # constructor arguments win over the environment
    explicit = JQuantsProvider(cache_dir=tmp_path / "c2", api_version="v1")
    assert explicit.endpoint_url("daily_quotes").startswith("https://alt.example/v1/")


def test_cache_first_never_touches_endpoint_config(monkeypatch):
    monkeypatch.setenv(ENV_API_VERSION, "v999-does-not-exist")

    def transport(url: str, headers: dict[str, str]) -> dict:
        raise AssertionError("cache hit must not reach the network")

    provider = JQuantsProvider(cache_dir=CACHE_DIR, http_get=transport)
    assert len(provider.get_prices("7203")) == 80  # served from cache fixture


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

"""Tests for local providers and import-safe stubs. No network access."""

from __future__ import annotations

from datetime import date

import pytest

from jp_stock_analysis.errors import DataValidationError, ProviderError
from jp_stock_analysis.providers.edinet_stub import EDINETProvider
from jp_stock_analysis.providers.local_csv import (
    load_company_metadata_csv,
    load_disclosure_texts,
    load_fundamentals_csv,
    load_prices_csv,
)
from jp_stock_analysis.providers.local_json import read_json, write_json
from jp_stock_analysis.providers.news_stub import NewsProvider
from jp_stock_analysis.providers.tdnet_stub import TDnetProvider


def test_load_prices_groups_by_ticker_sorted_by_date(fixtures_dir):
    prices = load_prices_csv(fixtures_dir / "prices_sample.csv")
    assert set(prices) == {"6758", "7203", "9984"}
    bars = prices["7203"]
    assert len(bars) == 130
    assert bars == sorted(bars, key=lambda bar: bar.date)
    assert all(bar.close > 0 for bar in bars)


def test_load_prices_normalizes_column_aliases(tmp_path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "code,日付,終値,adj_close,vol\n7203,2025/01/06,100,101,500\n", encoding="utf-8"
    )
    bar = load_prices_csv(csv_path)["7203"][0]
    assert bar.date == date(2025, 1, 6)
    assert bar.close == 100.0
    assert bar.adjusted_close == 101.0
    assert bar.volume == 500.0


def test_load_prices_requires_ticker_date_close(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("ticker,date\n7203,2025-01-06\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_prices_csv(csv_path)


def test_load_prices_missing_file_raises(tmp_path):
    with pytest.raises(DataValidationError):
        load_prices_csv(tmp_path / "nope.csv")


def test_load_fundamentals_sorted_with_optional_fields(fixtures_dir):
    statements = load_fundamentals_csv(fixtures_dir / "fundamentals_sample.csv")
    assert [s.fiscal_year for s in statements["7203"]] == [2023, 2024]
    assert statements["9984"][-1].eps == -40.0


def test_load_fundamentals_optional_columns_default_to_none(tmp_path):
    csv_path = tmp_path / "fundamentals.csv"
    csv_path.write_text("ticker,fiscal_year,revenue\n7203,2024,1000\n", encoding="utf-8")
    statement = load_fundamentals_csv(csv_path)["7203"][0]
    assert statement.revenue == 1000.0
    assert statement.eps is None
    assert statement.equity is None


def test_load_company_metadata(fixtures_dir):
    metadata = load_company_metadata_csv(fixtures_dir / "company_metadata_sample.csv")
    assert metadata["7203"].company_name == "サンプル自動車株式会社"
    assert metadata["9984"].sector == "情報・通信業"


def test_load_disclosure_texts(fixtures_dir):
    documents = load_disclosure_texts(fixtures_dir / "disclosures")
    assert set(documents) == {"6758", "7203", "9984"}
    assert "増収" in documents["7203"].text
    with pytest.raises(DataValidationError):
        load_disclosure_texts(fixtures_dir / "no_such_dir")


def test_local_json_roundtrip_and_errors(tmp_path):
    payload = {"ticker": "7203", "値": 1.5}
    path = write_json(tmp_path / "data.json", payload)
    assert read_json(path) == payload
    with pytest.raises(DataValidationError):
        read_json(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DataValidationError):
        read_json(bad)


def test_stub_providers_are_import_safe_but_unusable():
    with pytest.raises(ProviderError):
        EDINETProvider().get_disclosures("7203")
    with pytest.raises(ProviderError):
        TDnetProvider().get_disclosures("7203")
    with pytest.raises(ProviderError):
        NewsProvider().get_disclosures("7203")

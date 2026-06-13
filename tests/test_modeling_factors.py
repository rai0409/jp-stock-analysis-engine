"""Tests for factor feature engineering. Deterministic, offline."""

from __future__ import annotations

from datetime import date, timedelta

from jp_stock_analysis.modeling.factors import (
    compute_factors,
    sector_zscore,
    winsorize,
    zscore,
)
from jp_stock_analysis.schemas import DisclosureDocument, FinancialStatement, PriceBar


def _statement(**overrides) -> FinancialStatement:
    base = {
        "ticker": "1301",
        "fiscal_year": 2025,
        "revenue": 1000.0,
        "operating_income": 150.0,
        "net_income": 100.0,
        "equity": 1000.0,
        "total_assets": 2000.0,
        "shares_outstanding": 10.0,
    }
    base.update(overrides)
    return FinancialStatement(**base)


def _price_series(values: list[float]) -> list[PriceBar]:
    start = date(2025, 1, 1)
    bars = []
    day = start
    for value in values:
        while day.weekday() >= 5:
            day += timedelta(days=1)
        bars.append(PriceBar(ticker="1301", date=day, close=value, adjusted_close=value))
        day += timedelta(days=1)
    return bars


def test_quality_factors_are_ratios_in_percent():
    result = compute_factors(_statement(), None, None)
    f = result.features
    assert f["roe"] == 10.0  # 100 / 1000
    assert f["roa"] == 5.0  # 100 / 2000
    assert f["operating_margin"] == 15.0  # 150 / 1000
    assert f["equity_ratio"] == 50.0  # 1000 / 2000
    assert f["leverage"] == 2.0  # 2000 / 1000


def test_value_factors_need_price_and_shares():
    no_price = compute_factors(_statement(), None, None)
    assert no_price.features["earnings_yield"] is None
    assert no_price.features["book_to_market"] is None

    bars = _price_series([50.0])
    priced = compute_factors(_statement(), None, bars)
    # market cap = 50 * 10 = 500; earnings_yield = 100/500*100 = 20%
    assert priced.features["earnings_yield"] == 20.0
    assert priced.features["book_to_market"] == 2.0  # 1000 / 500
    assert priced.features["sales_to_price"] == 2.0  # 1000 / 500


def test_growth_factors_need_prior_statement():
    current = _statement(revenue=1100.0, net_income=120.0)
    prior = _statement(fiscal_year=2024, revenue=1000.0, net_income=100.0)
    none_prior = compute_factors(current, None, None)
    assert none_prior.features["revenue_growth_yoy"] is None

    with_prior = compute_factors(current, prior, None)
    assert with_prior.features["revenue_growth_yoy"] == 10.0
    assert with_prior.features["net_income_growth_yoy"] == 20.0


def test_momentum_and_drawdown_from_prices():
    series = [100.0] * 21
    series[-1] = 110.0  # +10% vs 20 rows ago
    bars = _price_series(series)
    result = compute_factors(_statement(), None, bars)
    assert round(result.features["momentum_20d"], 6) == 10.0
    # a monotone rise then the spike: drawdown is 0 (never below running peak)
    assert result.features["max_drawdown"] == 0.0


def test_divide_by_zero_yields_none_never_crashes():
    result = compute_factors(_statement(equity=0.0, total_assets=0.0), None, None)
    assert result.features["roe"] is None
    assert result.features["leverage"] is None
    assert result.features["equity_ratio"] is None


def test_missing_statement_marks_all_fundamental_factors_missing():
    result = compute_factors(None, None, None)
    assert result.features["roe"] is None
    assert "roe" in result.missing_factors
    assert result.available_count >= 1  # narrative_available is always present


def test_narrative_placeholder_no_text_is_not_fabricated():
    result = compute_factors(_statement(), None, None, narrative=None)
    assert result.features["narrative_available"] == 0.0
    assert result.features["risk_keyword_count"] is None  # not zero — not attempted
    assert result.features["sentiment_placeholder"] is None


def test_narrative_keyword_count_when_text_present():
    doc = DisclosureDocument(ticker="1301", text="為替リスクと減損リスクがあります")
    result = compute_factors(_statement(), None, None, narrative=doc)
    assert result.features["narrative_available"] == 1.0
    assert result.features["risk_keyword_count"] >= 2.0  # 為替 / リスク / 減損
    assert result.features["sentiment_placeholder"] == 0.0


def test_zscore_handles_missing_and_zero_variance():
    assert zscore([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]  # zero variance
    out = zscore([1.0, None, 3.0])
    assert out[1] is None
    assert round(out[0], 4) == -round(out[2], 4)  # symmetric around the mean


def test_winsorize_clips_outliers_keeps_none():
    values = [1.0, 2.0, 3.0, 4.0, 100.0, None]
    clipped = winsorize(values, 0.0, 0.8)
    assert clipped[-1] is None
    assert max(v for v in clipped if v is not None) < 100.0


def test_sector_zscore_normalizes_within_sector():
    values = [1.0, 3.0, 10.0, 30.0]
    sectors = ["a", "a", "b", "b"]
    out = sector_zscore(values, sectors)
    # within each 2-member sector, the lower value z-scores negative
    assert out[0] < 0 < out[1]
    assert out[2] < 0 < out[3]


def test_sector_zscore_ungrouped_sector_is_neutralized():
    out = sector_zscore([5.0, 6.0], [None, None])
    assert out == [0.0, 0.0]

"""Tests for fundamental analysis."""

from __future__ import annotations

import pytest
from conftest import make_statement

from jp_stock_analysis.analysis.fundamentals import (
    analyze_fundamentals,
    analyze_fundamentals_by_ticker,
    pct_change,
    safe_divide,
)


def test_safe_divide_handles_zero_and_none():
    assert safe_divide(10, 2) == 5
    assert safe_divide(10, 0) is None
    assert safe_divide(None, 2) is None
    assert safe_divide(10, None) is None


def test_pct_change_handles_zero_and_none():
    assert pct_change(110, 100) == pytest.approx(10.0)
    assert pct_change(90, -100) == pytest.approx(190.0)  # divides by abs(previous)
    assert pct_change(110, 0) is None
    assert pct_change(None, 100) is None


def test_margins_roe_and_growth():
    previous = make_statement(
        fiscal_year=2023, revenue=1000.0, operating_income=110.0, net_income=80.0, eps=80.0
    )
    current = make_statement(
        fiscal_year=2024,
        revenue=1100.0,
        operating_income=121.0,
        net_income=88.0,
        eps=88.0,
        equity=800.0,
        total_assets=2000.0,
    )
    metrics = analyze_fundamentals(current, previous)
    assert metrics.revenue_growth_yoy == pytest.approx(10.0)
    assert metrics.operating_income_growth_yoy == pytest.approx(10.0)
    assert metrics.eps_growth_yoy == pytest.approx(10.0)
    assert metrics.operating_margin == pytest.approx(11.0)
    assert metrics.net_margin == pytest.approx(8.0)
    assert metrics.roe == pytest.approx(11.0)
    assert metrics.roa == pytest.approx(4.4)
    assert metrics.equity_ratio == pytest.approx(40.0)
    assert metrics.fcf_margin == pytest.approx(80.0 / 1100.0 * 100.0)  # (120 - 40) / 1100
    assert metrics.confidence_score == 100.0


def test_missing_previous_year_disables_growth_with_warning():
    metrics = analyze_fundamentals(make_statement())
    assert metrics.revenue_growth_yoy is None
    assert metrics.eps_growth_yoy is None
    assert any("previous" in warning for warning in metrics.warnings)
    assert metrics.confidence_score < 100.0


def test_negative_eps_is_allowed_but_warned():
    metrics = analyze_fundamentals(make_statement(eps=-25.0, net_income=-25.0))
    assert metrics.latest_eps == -25.0
    assert any("negative EPS" in warning for warning in metrics.warnings)
    # payout ratio must not be fabricated from negative EPS
    assert metrics.dividend_payout_ratio is None


def test_fiscal_period_mismatch_warns():
    previous = make_statement(fiscal_year=2023, fiscal_period="Q2")
    metrics = analyze_fundamentals(make_statement(), previous)
    assert any("fiscal period mismatch" in warning for warning in metrics.warnings)


def test_zero_revenue_yields_none_margins_not_crash():
    metrics = analyze_fundamentals(make_statement(revenue=0.0))
    assert metrics.operating_margin is None
    assert metrics.net_margin is None


def test_analyze_by_ticker_uses_latest_two_years():
    statements = {
        "7203": [
            make_statement(fiscal_year=2022, revenue=900.0),
            make_statement(fiscal_year=2024, revenue=1100.0),
            make_statement(fiscal_year=2023, revenue=1000.0),
        ]
    }
    by_ticker = analyze_fundamentals_by_ticker(statements)
    assert by_ticker["7203"].fiscal_year == 2024
    assert by_ticker["7203"].revenue_growth_yoy == pytest.approx(10.0)

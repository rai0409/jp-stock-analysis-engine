"""Tests for valuation analysis."""

from __future__ import annotations

import pytest
from conftest import make_statement

from jp_stock_analysis.analysis.valuation import analyze_valuation, classify_valuation
from jp_stock_analysis.schemas import ValuationMetrics


def test_core_valuation_metrics():
    statement = make_statement(
        eps=100.0, bps=1000.0, dividends_per_share=30.0, shares_outstanding=2.0, revenue=500.0
    )
    metrics = analyze_valuation(statement, market_price=1500.0, eps_growth_yoy=10.0)
    assert metrics.per == pytest.approx(15.0)
    assert metrics.pbr == pytest.approx(1.5)
    assert metrics.market_cap == pytest.approx(3000.0)
    assert metrics.psr == pytest.approx(6.0)
    assert metrics.dividend_yield == pytest.approx(2.0)
    assert metrics.peg == pytest.approx(1.5)
    assert metrics.valuation_classification == "fair"


def test_negative_eps_makes_per_unavailable_with_warning():
    statement = make_statement(eps=-50.0)
    metrics = analyze_valuation(statement, market_price=1000.0)
    assert metrics.per is None
    assert metrics.peg is None
    assert any("negative" in warning for warning in metrics.warnings)
    # PBR still computable, so classification is not "unavailable"
    assert metrics.pbr is not None


def test_missing_market_price_disables_everything():
    metrics = analyze_valuation(make_statement(), market_price=None)
    assert metrics.per is None
    assert metrics.pbr is None
    assert metrics.valuation_classification == "unavailable"
    assert metrics.confidence_score == 0.0
    assert metrics.warnings


def test_peg_requires_positive_eps_growth():
    statement = make_statement(eps=100.0)
    metrics = analyze_valuation(statement, market_price=1500.0, eps_growth_yoy=-5.0)
    assert metrics.peg is None
    assert any("PEG" in warning for warning in metrics.warnings)


def test_classification_bands():
    def metrics_for(per=None, pbr=None):
        return ValuationMetrics(ticker="7203", per=per, pbr=pbr)

    assert classify_valuation(metrics_for(per=10.0, pbr=1.5)) == "cheap"
    assert classify_valuation(metrics_for(per=35.0, pbr=2.0)) == "expensive"
    assert classify_valuation(metrics_for(per=20.0, pbr=2.0)) == "fair"
    assert classify_valuation(metrics_for()) == "unavailable"
    # conflicting signals resolve to fair
    assert classify_valuation(metrics_for(per=10.0, pbr=5.0)) == "fair"

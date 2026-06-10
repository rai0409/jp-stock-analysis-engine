"""Tests for risk analysis."""

from __future__ import annotations

from jp_stock_analysis.analysis.disclosure_nlp import RuleBasedDisclosureAnalyzer
from jp_stock_analysis.analysis.risk import analyze_risks
from jp_stock_analysis.schemas import (
    DisclosureDocument,
    FundamentalMetrics,
    MomentumMetrics,
    ValuationMetrics,
)


def _flag_ids(risk_metrics):
    return {flag.risk_id for flag in risk_metrics.flags}


def test_negative_eps_and_declining_revenue_flags():
    fundamentals = FundamentalMetrics(
        ticker="9984",
        latest_eps=-40.0,
        revenue_growth_yoy=-12.0,
        operating_income_growth_yoy=-30.0,
        equity_ratio=18.0,
        confidence_score=80.0,
    )
    risks = analyze_risks(fundamentals, None, None, None)
    ids = _flag_ids(risks)
    assert {"negative_eps", "declining_revenue", "declining_operating_income",
            "low_equity_ratio"} <= ids
    assert all(flag.evidence for flag in risks.flags)


def test_high_valuation_weak_growth_flag():
    fundamentals = FundamentalMetrics(
        ticker="7203", revenue_growth_yoy=1.0, eps_growth_yoy=0.5, confidence_score=80.0
    )
    valuation = ValuationMetrics(
        ticker="7203", per=40.0, valuation_classification="expensive", confidence_score=80.0
    )
    risks = analyze_risks(fundamentals, valuation, None, None)
    assert "high_valuation_weak_growth" in _flag_ids(risks)


def test_large_drawdown_and_volatility_flags():
    momentum = MomentumMetrics(
        ticker="9984",
        observations=300,
        max_drawdown=-45.0,
        volatility_annualized=65.0,
        confidence_score=100.0,
    )
    risks = analyze_risks(None, None, momentum, None)
    ids = _flag_ids(risks)
    assert "large_drawdown" in ids
    assert "high_volatility" in ids
    drawdown_flag = next(f for f in risks.flags if f.risk_id == "large_drawdown")
    assert drawdown_flag.severity == "high"


def test_negative_disclosure_tone_and_uncertainty_flags():
    text = (
        "減収かつ減益となりました。需要が減少しました。下方修正を行いました。"
        "先行きは不透明です。今後の業績は変動する可能性があります。懸念があります。"
    )
    disclosure = RuleBasedDisclosureAnalyzer().analyze(
        DisclosureDocument(ticker="9984", text=text)
    )
    risks = analyze_risks(None, None, None, disclosure)
    ids = _flag_ids(risks)
    assert "negative_disclosure_tone" in ids
    assert "many_uncertainty_mentions" in ids


def test_critical_disclosure_escalates_severity():
    disclosure = RuleBasedDisclosureAnalyzer().analyze(
        DisclosureDocument(
            ticker="9984", text="継続企業の前提に関する重要事象等が存在しております。"
        )
    )
    risks = analyze_risks(None, None, None, disclosure)
    tone_flag = next(f for f in risks.flags if f.risk_id == "negative_disclosure_tone")
    assert tone_flag.severity == "critical"


def test_missing_inputs_produce_insufficient_data_flag():
    risks = analyze_risks(None, None, None, None)
    assert "insufficient_data" in _flag_ids(risks)
    assert risks.warnings
    assert risks.confidence_score == 20.0


def test_risk_score_bounded_0_to_100():
    fundamentals = FundamentalMetrics(
        ticker="9984",
        latest_eps=-40.0,
        revenue_growth_yoy=-20.0,
        operating_income_growth_yoy=-50.0,
        equity_ratio=10.0,
        confidence_score=80.0,
    )
    momentum = MomentumMetrics(
        ticker="9984",
        observations=300,
        max_drawdown=-60.0,
        volatility_annualized=80.0,
        confidence_score=100.0,
    )
    risks = analyze_risks(fundamentals, None, momentum, None)
    assert 0.0 <= risks.risk_score <= 100.0
    assert risks.risk_score == 100.0  # many high flags cap out

    clean = analyze_risks(
        FundamentalMetrics(
            ticker="7203",
            latest_eps=200.0,
            revenue_growth_yoy=10.0,
            operating_income_growth_yoy=12.0,
            equity_ratio=55.0,
            confidence_score=100.0,
        ),
        ValuationMetrics(ticker="7203", per=12.0, valuation_classification="fair",
                         confidence_score=100.0),
        MomentumMetrics(ticker="7203", observations=300, max_drawdown=-5.0,
                        volatility_annualized=15.0, confidence_score=100.0),
        RuleBasedDisclosureAnalyzer().analyze(
            DisclosureDocument(ticker="7203", text="増収および増益となりました。需要が堅調です。")
        ),
    )
    assert clean.risk_score == 0.0

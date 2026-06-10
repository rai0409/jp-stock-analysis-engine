"""Tests for integrated scoring."""

from __future__ import annotations

from jp_stock_analysis.analysis.disclosure_nlp import RuleBasedDisclosureAnalyzer
from jp_stock_analysis.analysis.scoring import score_stock
from jp_stock_analysis.config import AnalysisConfig
from jp_stock_analysis.schemas import (
    DisclosureDocument,
    FundamentalMetrics,
    MomentumMetrics,
    RiskMetrics,
    ValuationMetrics,
)

CONFIG = AnalysisConfig()


def _fundamentals() -> FundamentalMetrics:
    return FundamentalMetrics(
        ticker="7203",
        latest_eps=213.0,
        revenue_growth_yoy=11.0,
        operating_income_growth_yoy=22.0,
        eps_growth_yoy=18.0,
        operating_margin=11.0,
        net_margin=8.0,
        roe=10.7,
        roa=5.3,
        equity_ratio=50.0,
        fcf_margin=7.5,
        confidence_score=100.0,
    )


def _valuation() -> ValuationMetrics:
    return ValuationMetrics(
        ticker="7203",
        per=11.9,
        pbr=1.3,
        dividend_yield=3.0,
        valuation_classification="cheap",
        confidence_score=100.0,
    )


def _momentum() -> MomentumMetrics:
    return MomentumMetrics(
        ticker="7203",
        observations=300,
        return_3m=12.0,
        return_6m=26.0,
        return_12m=30.0,
        max_drawdown=-5.0,
        volatility_annualized=18.0,
        confidence_score=100.0,
    )


def _disclosure():
    return RuleBasedDisclosureAnalyzer().analyze(
        DisclosureDocument(
            ticker="7203", text="増収および増益となりました。需要が堅調に推移しました。"
        )
    )


def _risks(score: float = 0.0) -> RiskMetrics:
    return RiskMetrics(ticker="7203", risk_score=score, confidence_score=100.0)


def test_final_score_is_reproducible():
    first = score_stock(_fundamentals(), _valuation(), _momentum(), _disclosure(), _risks(), CONFIG)
    second = score_stock(
        _fundamentals(), _valuation(), _momentum(), _disclosure(), _risks(), CONFIG
    )
    assert first == second
    assert first.final_score is not None
    assert 0.0 <= first.final_score <= 100.0
    assert first.reasons["final_score"]


def test_strong_inputs_score_high():
    score = score_stock(_fundamentals(), _valuation(), _momentum(), _disclosure(), _risks(), CONFIG)
    assert score.final_score is not None and score.final_score >= 70.0
    assert score.quality_score is not None and score.quality_score >= 60.0
    assert score.confidence_score >= 80.0


def test_risk_score_lowers_final_score():
    low_risk = score_stock(
        _fundamentals(), _valuation(), _momentum(), _disclosure(), _risks(0.0), CONFIG
    )
    high_risk = score_stock(
        _fundamentals(), _valuation(), _momentum(), _disclosure(), _risks(50.0), CONFIG
    )
    assert low_risk.final_score is not None and high_risk.final_score is not None
    assert high_risk.final_score < low_risk.final_score
    # penalty = risk_score * risk_adjustment weight
    assert low_risk.final_score - high_risk.final_score == 5.0


def test_missing_components_lower_confidence_and_warn():
    full = score_stock(_fundamentals(), _valuation(), _momentum(), _disclosure(), _risks(), CONFIG)
    partial = score_stock(_fundamentals(), None, None, None, _risks(), CONFIG)
    assert partial.confidence_score < full.confidence_score
    assert partial.valuation_score is None
    assert partial.momentum_score is None
    assert partial.disclosure_score is None
    assert any("sub-scores unavailable" in warning for warning in partial.warnings)


def test_no_data_yields_no_final_score():
    score = score_stock(None, None, None, None, None, CONFIG)
    assert score.final_score is None
    assert score.confidence_score == 0.0
    assert any("final score unavailable" in warning for warning in score.warnings)


def test_scores_are_never_fabricated_from_missing_metrics():
    empty_fundamentals = FundamentalMetrics(ticker="7203", confidence_score=0.0)
    score = score_stock(empty_fundamentals, None, None, None, _risks(), CONFIG)
    assert score.quality_score is None
    assert score.growth_score is None

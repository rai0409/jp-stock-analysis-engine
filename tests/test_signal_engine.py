"""Tests for the opt-in trade-signal engine."""

from __future__ import annotations

from datetime import date

from conftest import make_result, make_score

from jp_stock_analysis.analysis.signal_engine import generate_signal, generate_signals
from jp_stock_analysis.config import DEFAULT_DISCLAIMER, AnalysisConfig
from jp_stock_analysis.schemas import RiskFlag, StockAnalysisResult

ANALYSIS_ONLY = AnalysisConfig()
SCREENING = AnalysisConfig(signal_mode="screening")
TRADE_SIGNAL = AnalysisConfig(signal_mode="trade_signal")


def _strong_result(ticker: str = "7203"):
    return make_result(
        ticker,
        signal_mode="trade_signal",
        score=make_score(
            ticker=ticker,
            final_score=85.0,
            confidence_score=85.0,
            quality_score=72.0,
            growth_score=78.0,
            valuation_score=80.0,
            momentum_score=75.0,
            disclosure_score=68.0,
            risk_score=10.0,
        ),
    )


def test_analysis_only_mode_produces_no_signal():
    assert generate_signal(_strong_result(), ANALYSIS_ONLY) is None


def test_screening_mode_produces_no_signal():
    assert generate_signal(_strong_result(), SCREENING) is None


def test_buy_signal_includes_thresholds_evidence_and_disclaimer():
    signal = generate_signal(_strong_result(), TRADE_SIGNAL)
    assert signal is not None
    assert signal.label == "buy_signal"
    assert signal.disclaimer == DEFAULT_DISCLAIMER
    assert signal.thresholds_used["buy_signal_threshold"] == 78.0
    assert signal.evidence
    assert len(signal.supporting_factors) >= 2
    assert signal.rationale


def test_valuation_alone_never_creates_buy_signal():
    result = make_result(
        "7203",
        signal_mode="trade_signal",
        score=make_score(
            final_score=90.0,
            confidence_score=85.0,
            quality_score=None,
            growth_score=None,
            momentum_score=None,
            disclosure_score=None,
            valuation_score=95.0,
            risk_score=5.0,
        ),
    )
    signal = generate_signal(result, TRADE_SIGNAL)
    assert signal is not None
    assert signal.label != "buy_signal"


def test_low_confidence_yields_insufficient_data():
    result = make_result(
        "7203",
        signal_mode="trade_signal",
        score=make_score(final_score=85.0, confidence_score=30.0),
    )
    signal = generate_signal(result, TRADE_SIGNAL)
    assert signal is not None and signal.label == "insufficient_data"


def test_weak_final_score_yields_sell_signal():
    result = make_result(
        "9984",
        signal_mode="trade_signal",
        score=make_score(ticker="9984", final_score=20.0, confidence_score=80.0, risk_score=80.0),
    )
    signal = generate_signal(result, TRADE_SIGNAL)
    assert signal is not None and signal.label == "sell_signal"


def test_critical_risk_with_decent_score_yields_avoid_signal():
    critical = RiskFlag(
        risk_id="negative_disclosure_tone",
        severity="critical",
        explanation="going-concern language",
        confidence=85.0,
    )
    result = make_result(
        "9984",
        signal_mode="trade_signal",
        score=make_score(ticker="9984", final_score=70.0, confidence_score=80.0, risk_score=50.0),
        risk_flags=[critical],
    )
    signal = generate_signal(result, TRADE_SIGNAL)
    assert signal is not None
    assert signal.label == "avoid_signal"
    assert signal.blocking_risks


def test_high_risk_score_blocks_buy():
    result = make_result(
        "7203",
        signal_mode="trade_signal",
        score=make_score(final_score=85.0, confidence_score=85.0, risk_score=60.0),
    )
    signal = generate_signal(result, TRADE_SIGNAL)
    assert signal is not None
    assert signal.label != "buy_signal"


def test_neutral_profile_yields_hold_signal():
    result = make_result(
        "6758",
        signal_mode="trade_signal",
        score=make_score(
            ticker="6758",
            final_score=55.0,
            confidence_score=75.0,
            quality_score=55.0,
            growth_score=50.0,
            momentum_score=50.0,
            disclosure_score=50.0,
            risk_score=20.0,
        ),
    )
    signal = generate_signal(result, TRADE_SIGNAL)
    assert signal is not None and signal.label == "hold_signal"


def test_missing_risk_assessment_blocks_buy():
    result = StockAnalysisResult(
        ticker="7203",
        analysis_date=date(2025, 6, 30),
        signal_mode="trade_signal",
        score=make_score(final_score=85.0, confidence_score=85.0, risk_score=None),
        risks=None,
    )
    signal = generate_signal(result, TRADE_SIGNAL)
    assert signal is not None
    assert signal.label != "buy_signal"
    assert "risk assessment unavailable" in signal.blocking_risks


def test_generate_signals_attaches_to_results():
    results = [_strong_result("7203"), _strong_result("6758")]
    signals = generate_signals(results, TRADE_SIGNAL)
    assert len(signals) == 2
    assert all(result.signal is not None for result in results)

    plain = [make_result("7203")]
    assert generate_signals(plain, ANALYSIS_ONLY) == []
    assert plain[0].signal is None

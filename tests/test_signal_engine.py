"""Tests for the opt-in trade-signal engine."""

from __future__ import annotations

from datetime import date

import pytest
from conftest import make_result, make_score

from jp_stock_analysis.analysis.signal_engine import generate_signal, generate_signals
from jp_stock_analysis.config import DEFAULT_DISCLAIMER, AnalysisConfig, SignalThresholds
from jp_stock_analysis.schemas import RiskFlag, SectorRelativeMetrics, StockAnalysisResult


def _sector_factors(signal) -> list[str]:
    return [f for f in signal.supporting_factors if f.startswith("sector_relative_score=")]


def _sector_relative(
    ticker: str = "7203",
    score: float = 95.0,
    peer_count: int = 5,
    confidence: float = 100.0,
) -> SectorRelativeMetrics:
    return SectorRelativeMetrics(
        ticker=ticker,
        sector="輸送用機器",
        peer_count=peer_count,
        sector_relative_score=score,
        confidence_score=confidence,
    )

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


def test_sector_relative_plus_valuation_cannot_buy():
    """High valuation + high sector-relative with no core factors must not buy."""
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
    result.sector_relative = _sector_relative(score=95.0, peer_count=5)
    signal = generate_signal(result, TRADE_SIGNAL)
    assert signal is not None
    assert signal.label != "buy_signal"
    # the factor may appear, but only as labelled evidence
    sector_factors = [
        f for f in signal.supporting_factors if f.startswith("sector_relative_score=")
    ]
    assert len(sector_factors) == 1
    assert "supporting evidence only" in sector_factors[0]


def test_eligible_sector_factor_is_appended_to_buy_signal():
    result = _strong_result()
    result.sector_relative = _sector_relative(score=92.9, peer_count=5)
    signal = generate_signal(result, TRADE_SIGNAL)
    assert signal is not None
    assert signal.label == "buy_signal"  # decided by core factors, unchanged
    core = [f for f in signal.supporting_factors if not f.startswith("sector_relative_score=")]
    assert len(core) >= 2
    assert any("sector_relative_score=92.9" in f for f in signal.supporting_factors)


def test_sector_factor_requires_peers_score_and_confidence():
    cases = [
        _sector_relative(score=95.0, peer_count=2),  # too few peers
        _sector_relative(score=60.0, peer_count=5),  # score below threshold
        _sector_relative(score=95.0, peer_count=5, confidence=30.0),  # low confidence
        SectorRelativeMetrics(  # score unavailable
            ticker="7203", sector="輸送用機器", peer_count=5, confidence_score=100.0
        ),
    ]
    for relative in cases:
        result = _strong_result()
        result.sector_relative = relative
        signal = generate_signal(result, TRADE_SIGNAL)
        assert signal is not None
        assert not any(
            f.startswith("sector_relative_score=") for f in signal.supporting_factors
        ), relative


def test_sector_support_defaults_match_previous_constants():
    thresholds = AnalysisConfig().thresholds
    assert thresholds.sector_support_score_threshold == 70.0
    assert thresholds.sector_support_min_peers == 4
    assert thresholds.sector_support_min_confidence == 50.0


def test_custom_sector_support_thresholds_change_evidence_only():
    def signal_with(thresholds: SignalThresholds | None = None):
        config = AnalysisConfig(signal_mode="trade_signal")
        if thresholds is not None:
            config = AnalysisConfig(signal_mode="trade_signal", thresholds=thresholds)
        result = _strong_result()
        result.sector_relative = _sector_relative(score=80.0, peer_count=5, confidence=80.0)
        return generate_signal(result, config)

    baseline = signal_with()
    assert baseline.label == "buy_signal"
    assert len(_sector_factors(baseline)) == 1

    for custom in (
        SignalThresholds(sector_support_score_threshold=90.0),  # 80 < 90
        SignalThresholds(sector_support_min_peers=6),  # 5 < 6
        SignalThresholds(sector_support_min_confidence=90.0),  # 80 < 90
    ):
        tightened = signal_with(custom)
        assert _sector_factors(tightened) == []
        assert tightened.label == baseline.label  # eligibility never moves the label
        assert tightened.thresholds_used == baseline.thresholds_used

    loosened = signal_with(SignalThresholds(sector_support_score_threshold=50.0))
    assert len(_sector_factors(loosened)) == 1
    assert "(>= 50," in _sector_factors(loosened)[0]
    assert loosened.label == baseline.label


def test_sector_support_thresholds_not_in_thresholds_used():
    result = _strong_result()
    result.sector_relative = _sector_relative()
    signal = generate_signal(result, TRADE_SIGNAL)
    assert all("sector" not in key for key in signal.thresholds_used)


def test_sector_support_config_validation():
    with pytest.raises(ValueError):
        SignalThresholds(sector_support_score_threshold=150.0)
    with pytest.raises(ValueError):
        SignalThresholds(sector_support_min_confidence=-1.0)
    with pytest.raises(ValueError):
        SignalThresholds(sector_support_min_peers=0)
    # the other thresholds keep their existing validation
    with pytest.raises(ValueError):
        SignalThresholds(buy_signal_threshold=101.0)


def test_sector_relative_never_creates_signal_outside_trade_signal_mode():
    result = _strong_result()
    result.sector_relative = _sector_relative()
    assert generate_signal(result, ANALYSIS_ONLY) is None
    assert generate_signal(result, SCREENING) is None


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

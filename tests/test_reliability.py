"""Tests for the confidence-aware screening reliability guard.

Covers the real product-risk case: a ticker with only 2 price bars (no
fundamentals, no valuation, no disclosure text) gets a very high final_score
from the single momentum sub-score, and must never be ranked or presented as
a strong screening candidate.
"""

from __future__ import annotations

import csv
import json
from datetime import date

from conftest import make_result, make_score

from jp_stock_analysis.analysis.momentum import analyze_momentum
from jp_stock_analysis.analysis.reliability import assess_reliability
from jp_stock_analysis.analysis.risk import analyze_risks
from jp_stock_analysis.analysis.scoring import score_stock
from jp_stock_analysis.analysis.screening import screen_stocks
from jp_stock_analysis.analysis.signal_engine import generate_signal
from jp_stock_analysis.cli import main
from jp_stock_analysis.config import DEFAULT_DISCLAIMER, AnalysisConfig
from jp_stock_analysis.reports.csv_report import write_screening_csv
from jp_stock_analysis.reports.markdown_report import render_markdown_report
from jp_stock_analysis.schemas import (
    DisclosureAnalysisResult,
    FundamentalMetrics,
    MomentumMetrics,
    PriceBar,
    StockAnalysisResult,
    ValuationMetrics,
)

CONFIG = AnalysisConfig()


def make_two_bar_result(ticker: str = "9991") -> StockAnalysisResult:
    """Reproduce the linked-export case through the real analyzers."""
    bars = [
        PriceBar(ticker=ticker, date=date(2026, 6, 3), close=1000.0),
        PriceBar(ticker=ticker, date=date(2026, 6, 4), close=1010.0),
    ]
    momentum = analyze_momentum(bars)
    risks = analyze_risks(None, None, momentum, None)
    score = score_stock(None, None, momentum, None, risks, CONFIG)
    return StockAnalysisResult(
        ticker=ticker,
        analysis_date=date(2026, 6, 4),
        signal_mode="analysis_only",
        momentum=momentum,
        risks=risks,
        score=score,
        confidence_score=score.confidence_score,
    )


def make_full_coverage_result(
    ticker: str = "7203", final: float = 70.0, confidence: float = 80.0
) -> StockAnalysisResult:
    """Synthetic result with all five components present and confident."""
    result = make_result(
        ticker,
        score=make_score(ticker=ticker, final_score=final, confidence_score=confidence),
    )
    result.fundamentals = FundamentalMetrics(ticker=ticker, confidence_score=80.0)
    result.valuation = ValuationMetrics(ticker=ticker, confidence_score=80.0)
    result.momentum = MomentumMetrics(ticker=ticker, observations=300, confidence_score=90.0)
    result.disclosure = DisclosureAnalysisResult(ticker=ticker, confidence_score=70.0)
    return result


def test_two_bar_high_final_score_is_marked_unreliable():
    result = make_two_bar_result()
    score = result.score
    # the dangerous combination this guard exists for
    assert score.final_score is not None and score.final_score >= 80.0
    assert score.confidence_score < 30.0

    assessment = assess_reliability(result, CONFIG.thresholds)
    assert assessment.screening_eligible is False
    assert assessment.reliability_grade == "low"
    assert assessment.available_subscores == 1  # momentum only
    assert assessment.data_coverage_score <= 40.0
    # reliability-adjusted score collapses far below the raw final score
    assert assessment.screening_score is not None
    assert assessment.screening_score < score.final_score / 2
    assert assessment.screening_score < 20.0
    joined = " ".join(assessment.warnings)
    assert "confidence" in joined
    assert "sub-scores" in joined
    assert "must not be interpreted as a strong candidate" in joined


def test_full_coverage_result_remains_eligible():
    result = make_full_coverage_result()
    assessment = assess_reliability(result, CONFIG.thresholds)
    assert assessment.screening_eligible is True
    assert assessment.reliability_grade == "high"
    assert assessment.data_coverage_score == 100.0
    assert assessment.screening_score == 56.0  # 70 * 0.8 * 1.0
    assert assessment.warnings == []


def test_eligibility_boundaries_are_inclusive():
    at_minimum = make_full_coverage_result(confidence=30.0)
    assert assess_reliability(at_minimum, CONFIG.thresholds).screening_eligible is True
    assert assess_reliability(at_minimum, CONFIG.thresholds).reliability_grade == "medium"
    below_minimum = make_full_coverage_result(confidence=29.9)
    assert assess_reliability(below_minimum, CONFIG.thresholds).screening_eligible is False
    assert assess_reliability(below_minimum, CONFIG.thresholds).reliability_grade == "low"


def test_no_score_is_never_eligible():
    result = make_result("0000", score=make_score(ticker="0000", final_score=None))
    assessment = assess_reliability(result, CONFIG.thresholds)
    assert assessment.screening_eligible is False
    assert assessment.screening_score is None
    assert assessment.reliability_grade == "low"


def test_ranking_prefers_eligible_over_low_confidence_high_final():
    low_data = make_two_bar_result("9991")  # final ~98, unreliable
    solid = make_full_coverage_result("7203", final=65.0, confidence=80.0)
    screening = screen_stocks([low_data, solid], CONFIG)

    assert [entry.ticker for entry in screening] == ["7203", "9991"]
    assert [entry.rank for entry in screening] == [1, 2]
    top, bottom = screening
    assert top.screening_eligible is True
    assert bottom.screening_eligible is False
    assert bottom.reliability_grade == "low"
    # final_score transparency is preserved even though ranking demotes it
    assert bottom.final_score > top.final_score
    assert bottom.screening_score < top.screening_score
    assert any("screening reliability" in w for w in bottom.warnings)


def test_screening_csv_contains_reliability_fields(tmp_path):
    results = [make_two_bar_result("9991"), make_full_coverage_result("7203")]
    screening = screen_stocks(results, CONFIG)
    path = write_screening_csv(results, screening, tmp_path / "screening.csv")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for column in (
        "final_score",
        "confidence_score",
        "data_coverage_score",
        "screening_score",
        "screening_eligible",
        "reliability_grade",
    ):
        assert column in rows[0], column
    by_ticker = {row["ticker"]: row for row in rows}
    assert by_ticker["7203"]["screening_eligible"] == "true"
    assert by_ticker["9991"]["screening_eligible"] == "false"
    assert by_ticker["9991"]["reliability_grade"] == "low"
    assert float(by_ticker["9991"]["screening_score"]) < float(
        by_ticker["9991"]["final_score"]
    )


def test_markdown_flags_low_reliability_and_keeps_final_score():
    result = make_two_bar_result("9991")
    text = render_markdown_report(result, CONFIG)
    assert "Reliability grade: `low`" in text
    assert "Screening eligible: no" in text
    assert "**Low reliability:**" in text
    assert "must NOT be read as a strong candidate" in text
    assert "- Final score: " in text  # raw score still visible
    assert DEFAULT_DISCLAIMER in text


def test_markdown_high_reliability_has_no_low_warning():
    text = render_markdown_report(make_full_coverage_result(), CONFIG)
    assert "Reliability grade: `high`" in text
    assert "Screening eligible: yes" in text
    assert "**Low reliability:**" not in text


def test_low_reliability_result_gets_insufficient_data_signal():
    """trade_signal mode stays protected: low confidence never yields buy."""
    result = make_two_bar_result("9991")
    signal = generate_signal(result, AnalysisConfig(signal_mode="trade_signal"))
    assert signal.label == "insufficient_data"


def test_cli_smoke_emits_reliability_fields_in_all_outputs(fixtures_dir, tmp_path):
    argv = [
        "analyze",
        "--prices", str(fixtures_dir / "prices_sample.csv"),
        "--fundamentals", str(fixtures_dir / "fundamentals_sample.csv"),
        "--metadata", str(fixtures_dir / "company_metadata_sample.csv"),
        "--disclosures", str(fixtures_dir / "disclosures"),
        "--output-dir", str(tmp_path),
    ]
    assert main(argv) == 0

    payload = json.loads((tmp_path / "screening.json").read_text(encoding="utf-8"))
    assert payload["disclaimer"] == DEFAULT_DISCLAIMER
    assert payload["signal_mode"] == "analysis_only"
    for entry in payload["screening"]:
        for field in (
            "final_score",
            "confidence_score",
            "data_coverage_score",
            "screening_score",
            "screening_eligible",
            "reliability_grade",
        ):
            assert field in entry, field
        # full fixtures: every ticker has all five components
        assert entry["data_coverage_score"] == 100.0
        assert entry["screening_eligible"] is True
        # analysis_only behavior unchanged
        assert "screening_label" not in entry

    with (tmp_path / "screening.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["reliability_grade"] in {"high", "medium", "low"} for row in rows)

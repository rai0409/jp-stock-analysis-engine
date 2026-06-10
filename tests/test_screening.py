"""Tests for screening labels and ranking."""

from __future__ import annotations

from conftest import make_result, make_score

from jp_stock_analysis.analysis.screening import assign_screening_label, screen_stocks
from jp_stock_analysis.config import AnalysisConfig

ANALYSIS_ONLY = AnalysisConfig()  # default mode
SCREENING = AnalysisConfig(signal_mode="screening")


def test_default_config_is_analysis_only():
    assert ANALYSIS_ONLY.signal_mode == "analysis_only"


def test_analysis_only_mode_assigns_no_labels():
    results = [make_result("7203"), make_result("6758")]
    screening = screen_stocks(results, ANALYSIS_ONLY)
    assert all(entry.screening_label is None for entry in screening)


def test_screening_mode_assigns_labels_by_threshold():
    cases = {
        "A100": (85.0, "strong_candidate"),
        "B200": (70.0, "candidate"),
        "C300": (55.0, "watchlist"),
        "D400": (42.0, "watchlist"),  # borderline zone is conservatively watchlist
        "E500": (30.0, "avoid_candidate"),
    }
    results = [
        make_result(ticker, score=make_score(ticker=ticker, final_score=final))
        for ticker, (final, _) in cases.items()
    ]
    screening = screen_stocks(results, SCREENING)
    labels = {entry.ticker: entry.screening_label for entry in screening}
    for ticker, (_, expected) in cases.items():
        assert labels[ticker] == expected, ticker


def test_low_confidence_yields_insufficient_data():
    score = make_score(final_score=90.0, confidence_score=30.0)
    assert assign_screening_label(score, SCREENING) == "insufficient_data"
    assert assign_screening_label(None, SCREENING) == "insufficient_data"
    no_final = make_score(final_score=None)
    assert assign_screening_label(no_final, SCREENING) == "insufficient_data"


def test_ranking_is_descending_by_final_score():
    results = [
        make_result("LOW", score=make_score(ticker="LOW", final_score=40.0)),
        make_result("HIGH", score=make_score(ticker="HIGH", final_score=90.0)),
        make_result("MID", score=make_score(ticker="MID", final_score=60.0)),
        make_result("NONE", score=make_score(ticker="NONE", final_score=None)),
    ]
    screening = screen_stocks(results, SCREENING)
    assert [entry.ticker for entry in screening] == ["HIGH", "MID", "LOW", "NONE"]
    assert [entry.rank for entry in screening] == [1, 2, 3, 4]


def test_screening_never_outputs_trade_signal_labels():
    results = [make_result("7203", score=make_score(final_score=95.0))]
    screening = screen_stocks(results, SCREENING)
    valid = {"strong_candidate", "candidate", "watchlist", "avoid_candidate", "insufficient_data"}
    assert all(entry.screening_label in valid for entry in screening)

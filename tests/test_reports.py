"""Tests for JSON/CSV/Markdown report writers."""

from __future__ import annotations

import csv
import json

from conftest import make_result, make_score

from jp_stock_analysis.analysis.screening import screen_stocks
from jp_stock_analysis.analysis.signal_engine import generate_signals
from jp_stock_analysis.config import DEFAULT_DISCLAIMER, AnalysisConfig
from jp_stock_analysis.reports.csv_report import write_screening_csv
from jp_stock_analysis.reports.json_report import write_json_report
from jp_stock_analysis.reports.markdown_report import (
    render_markdown_report,
    write_markdown_report,
)

REQUIRED_MARKDOWN_SECTIONS = [
    "## Executive Summary",
    "## Data Coverage",
    "## Fundamental Metrics",
    "## Valuation Metrics",
    "## Momentum Metrics",
    "## Disclosure Analysis",
    "## Risk Flags",
    "## Integrated Score",
    "## Evidence and Warnings",
    "## Limitations",
    "## Disclaimer",
]


def _results(mode: str):
    return [
        make_result("7203", signal_mode=mode,
                    score=make_score(ticker="7203", final_score=82.0)),
        make_result("9984", signal_mode=mode,
                    score=make_score(ticker="9984", final_score=30.0, risk_score=80.0)),
    ]


def test_json_report_written_with_disclaimer(tmp_path):
    config = AnalysisConfig()
    results = _results("analysis_only")
    screening = screen_stocks(results, config)
    path = write_json_report(results, screening, tmp_path / "screening.json", config)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["disclaimer"] == DEFAULT_DISCLAIMER
    assert payload["signal_mode"] == "analysis_only"
    assert payload["result_count"] == 2
    # analysis_only: no labels, no signals anywhere
    for entry in payload["results"] + payload["screening"]:
        assert "signal" not in entry
        assert "screening_label" not in entry


def test_json_report_includes_signal_only_in_trade_signal_mode(tmp_path):
    config = AnalysisConfig(signal_mode="trade_signal")
    results = _results("trade_signal")
    generate_signals(results, config)
    screening = screen_stocks(results, config)
    path = write_json_report(results, screening, tmp_path / "screening.json", config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload["results"]:
        assert "signal" in entry
        assert entry["signal"]["disclaimer"] == DEFAULT_DISCLAIMER


def test_json_report_includes_labels_in_screening_mode(tmp_path):
    config = AnalysisConfig(signal_mode="screening")
    results = _results("screening")
    screening = screen_stocks(results, config)
    labels = {entry.ticker: entry.screening_label for entry in screening}
    for result in results:
        result.screening_label = labels[result.ticker]
    path = write_json_report(results, screening, tmp_path / "screening.json", config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload["results"]:
        assert entry["screening_label"]
        assert "signal" not in entry
    for entry in payload["screening"]:
        assert entry["screening_label"]


def test_csv_report_columns_and_label_handling(tmp_path):
    analysis_config = AnalysisConfig()
    results = _results("analysis_only")
    screening = screen_stocks(results, analysis_config)
    path = write_screening_csv(results, screening, tmp_path / "screening.csv")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["rank"] == "1"
    assert rows[0]["ticker"] == "7203"  # highest score first
    assert "screening_label" not in rows[0]
    assert "trade_signal" not in rows[0]
    assert "warnings_count" in rows[0]

    screening_config = AnalysisConfig(signal_mode="screening")
    labelled = screen_stocks(_results("screening"), screening_config)
    path2 = write_screening_csv(_results("screening"), labelled, tmp_path / "screening2.csv")
    with path2.open(encoding="utf-8") as handle:
        rows2 = list(csv.DictReader(handle))
    assert rows2[0]["screening_label"] == "strong_candidate"


def test_markdown_report_sections_and_disclaimer(tmp_path):
    config = AnalysisConfig()
    result = _results("analysis_only")[0]
    text = render_markdown_report(result, config)
    for section in REQUIRED_MARKDOWN_SECTIONS:
        assert section in text, section
    assert DEFAULT_DISCLAIMER in text
    # analysis_only must not contain screening or signal sections
    assert "## Screening" not in text
    assert "## Research Signal" not in text

    path = write_markdown_report(result, tmp_path, config)
    assert path.name == "7203.md"
    assert path.exists()


def test_markdown_trade_signal_mode_includes_signal_section():
    config = AnalysisConfig(signal_mode="trade_signal")
    results = _results("trade_signal")
    generate_signals(results, config)
    screening = screen_stocks(results, config)
    labels = {entry.ticker: entry.screening_label for entry in screening}
    for result in results:
        result.screening_label = labels[result.ticker]
    text = render_markdown_report(results[0], config)
    assert "## Screening" in text
    assert "## Research Signal" in text
    assert "Label: `" in text
    assert DEFAULT_DISCLAIMER in text

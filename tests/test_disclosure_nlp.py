"""Tests for deterministic Japanese disclosure analysis. No LLM, no network."""

from __future__ import annotations

from jp_stock_analysis.analysis.disclosure_nlp import (
    NoOpLLMDisclosureAnalyzer,
    RuleBasedDisclosureAnalyzer,
)
from jp_stock_analysis.schemas import DisclosureDocument

_SAMPLE_TEXT = (
    "当期は販売が好調に推移し、増収となりました。"
    "一方で、コスト上昇が利益を圧迫しました。"
    "通期の業績予想については、上方修正を行いました。"
    "海外展開に向けた成長投資を継続します。"
    "為替変動および競争激化が重要なリスクであります。"
    "先行きは不透明であり、業績は変動する可能性があります。"
)


def _analyze(text: str, ticker: str = "7203"):
    return RuleBasedDisclosureAnalyzer().analyze(DisclosureDocument(ticker=ticker, text=text))


def test_detects_positive_negative_risk_uncertainty_and_guidance():
    result = _analyze(_SAMPLE_TEXT)
    categories = {finding.category for finding in result.findings}
    assert "positive_factor" in categories
    assert "negative_factor" in categories
    assert "risk_factor" in categories
    assert "uncertainty" in categories
    assert "guidance_revision" in categories
    assert "growth_driver" in categories
    assert result.positive_count >= 1
    assert result.negative_count >= 1
    assert result.risk_count >= 2
    assert result.uncertainty_count >= 1


def test_every_finding_has_evidence_and_rule_id():
    result = _analyze(_SAMPLE_TEXT)
    assert result.findings
    for finding in result.findings:
        assert finding.evidence_text.strip()
        assert finding.rule_id
        assert finding.confidence > 0
        # evidence must come from the source text, never be invented
        assert finding.evidence_text in _SAMPLE_TEXT


def test_going_concern_is_critical():
    result = _analyze("継続企業の前提に関する重要事象等が存在しております。")
    critical = [f for f in result.findings if f.severity == "critical"]
    assert critical and critical[0].rule_id == "RISK-003"


def test_tone_score_sign_tracks_content():
    positive = _analyze("増収および増益を達成し、需要が堅調に推移しました。利益率が改善しました。")
    negative = _analyze("減収かつ減益となりました。需要が減少し、下方修正を行いました。")
    assert positive.tone_score > 0
    assert negative.tone_score < 0


def test_deterministic_output():
    assert _analyze(_SAMPLE_TEXT) == _analyze(_SAMPLE_TEXT)


def test_empty_text_yields_no_findings_zero_confidence():
    result = _analyze("")
    assert result.findings == []
    assert result.confidence_score == 0.0
    assert result.warnings


def test_unmatched_text_warns_low_confidence():
    result = _analyze("本日は晴天なり。")
    assert result.findings == []
    assert result.confidence_score <= 30.0
    assert result.warnings


def test_rule_based_analyzer_propagates_document_metadata():
    document = DisclosureDocument(
        ticker="7203", text=_SAMPLE_TEXT, document_type="tanshin", fiscal_year=2024
    )
    result = RuleBasedDisclosureAnalyzer().analyze(document)
    assert result.document_type == "tanshin"
    assert result.fiscal_year == 2024

    # the empty-text early return must preserve provenance too
    empty = RuleBasedDisclosureAnalyzer().analyze(
        DisclosureDocument(ticker="7203", text="", document_type="tanshin", fiscal_year=2024)
    )
    assert empty.document_type == "tanshin"
    assert empty.fiscal_year == 2024


def test_noop_llm_analyzer_propagates_document_metadata():
    document = DisclosureDocument(
        ticker="7203", text=_SAMPLE_TEXT, document_type="yuho", fiscal_year=2023
    )
    result = NoOpLLMDisclosureAnalyzer().analyze(document)
    assert result.document_type == "yuho"
    assert result.fiscal_year == 2023


def test_metadata_defaults_to_none_when_absent():
    result = _analyze(_SAMPLE_TEXT)
    assert result.document_type is None
    assert result.fiscal_year is None


def test_noop_llm_analyzer_returns_no_findings():
    result = NoOpLLMDisclosureAnalyzer().analyze(
        DisclosureDocument(ticker="7203", text=_SAMPLE_TEXT)
    )
    assert result.findings == []
    assert result.confidence_score == 0.0
    assert any("LLM" in warning for warning in result.warnings)

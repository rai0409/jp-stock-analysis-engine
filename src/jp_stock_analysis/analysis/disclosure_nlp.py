"""Deterministic rule-based Japanese disclosure text analysis.

The default analyzer is keyword/pattern based: no LLM, no network, fully
reproducible. Every finding carries ``evidence_text`` (the matched sentence),
``category``, ``severity``, ``confidence``, and ``rule_id``. The analyzer
never infers claims beyond the matched text.

``tone_score`` is the clamped sum of per-rule tone weights (positive rules
add, negative/risk/uncertainty rules subtract) and feeds the disclosure
sub-score in scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jp_stock_analysis.schemas import (
    DisclosureAnalysisResult,
    DisclosureDocument,
    DisclosureFinding,
    FindingCategory,
    RiskSeverity,
)

_MAX_EVIDENCE_CHARS = 200
_SENTENCE_SPLIT = re.compile(r"[。\n]+")


def _document_source_metadata(document: DisclosureDocument) -> dict[str, str]:
    """Provenance for results: provider metadata wins over the bare source path."""
    return {"source": document.source or "unknown", **document.source_metadata}


@runtime_checkable
class DisclosureNLPProvider(Protocol):
    """Anything that turns a disclosure document into an analysis result."""

    def analyze(self, document: DisclosureDocument) -> DisclosureAnalysisResult: ...


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    category: FindingCategory
    keywords: tuple[str, ...]
    severity: RiskSeverity
    confidence: float
    summary: str
    tone_weight: float


_RULES: tuple[_Rule, ...] = (
    # Positive factors
    _Rule("POS-001", "positive_factor", ("増収",), "low", 80, "Revenue increase mentioned", 8),
    _Rule("POS-002", "positive_factor", ("増益",), "low", 80, "Profit increase mentioned", 8),
    _Rule("POS-003", "positive_factor", ("需要が堅調",), "low", 75, "Solid demand mentioned", 8),
    _Rule("POS-004", "positive_factor", ("受注が増加",), "low", 75, "Order growth mentioned", 8),
    _Rule(
        "POS-005",
        "positive_factor",
        ("価格改定が寄与",),
        "low",
        70,
        "Price revisions contributed to results",
        6,
    ),
    _Rule(
        "POS-006", "positive_factor", ("利益率が改善",), "low", 75,
        "Margin improvement mentioned", 8,
    ),
    # Negative factors
    _Rule("NEG-001", "negative_factor", ("減収",), "medium", 80, "Revenue decline mentioned", -8),
    _Rule("NEG-002", "negative_factor", ("減益",), "medium", 80, "Profit decline mentioned", -8),
    _Rule(
        "NEG-003", "negative_factor", ("需要が減少",), "medium", 75, "Demand decline mentioned", -8
    ),
    _Rule("NEG-004", "negative_factor", ("コスト上昇",), "low", 70, "Cost increases mentioned", -6),
    _Rule(
        "NEG-005",
        "negative_factor",
        ("原材料価格の高騰",),
        "medium",
        75,
        "Raw material price surge mentioned",
        -6,
    ),
    _Rule(
        "NEG-006", "negative_factor", ("為替の影響",), "low", 60,
        "Foreign-exchange impact noted", -4,
    ),
    # Risk factors
    _Rule(
        "RISK-001", "risk_factor", ("事業等のリスク",), "low", 70,
        "Business risk section present", -2,
    ),
    _Rule(
        "RISK-002", "risk_factor", ("重要なリスク",), "medium", 75,
        "Material risk highlighted", -4,
    ),
    _Rule(
        "RISK-003",
        "risk_factor",
        ("継続企業の前提",),
        "critical",
        90,
        "Going-concern language present",
        -20,
    ),
    _Rule("RISK-004", "risk_factor", ("競争激化",), "medium", 75, "Intensifying competition", -4),
    _Rule("RISK-005", "risk_factor", ("規制変更",), "medium", 70, "Regulatory change risk", -3),
    _Rule(
        "RISK-006", "risk_factor", ("サプライチェーン",), "low", 65,
        "Supply-chain risk mentioned", -2,
    ),
    _Rule("RISK-007", "risk_factor", ("金利上昇",), "medium", 70, "Rising interest rate risk", -3),
    _Rule("RISK-008", "risk_factor", ("為替変動",), "low", 65, "Currency fluctuation risk", -2),
    # Growth drivers
    _Rule("GRW-001", "growth_driver", ("成長投資",), "low", 70, "Growth investment mentioned", 4),
    _Rule("GRW-002", "growth_driver", ("新規事業",), "low", 70, "New business initiatives", 4),
    _Rule("GRW-003", "growth_driver", ("海外展開",), "low", 70, "Overseas expansion", 4),
    _Rule("GRW-004", "growth_driver", ("DX",), "low", 60, "Digital transformation initiatives", 3),
    _Rule("GRW-005", "growth_driver", ("研究開発",), "low", 65, "R&D investment mentioned", 3),
    _Rule("GRW-006", "growth_driver", ("設備投資",), "low", 65, "Capital investment mentioned", 3),
    # Management outlook
    _Rule("OUT-001", "management_outlook", ("見通し",), "low", 55, "Outlook statement", 0),
    _Rule("OUT-002", "management_outlook", ("予想",), "low", 50, "Forecast statement", 0),
    _Rule("OUT-003", "management_outlook", ("計画",), "low", 50, "Plan statement", 0),
    _Rule(
        "OUT-004", "management_outlook", ("中期経営計画",), "low", 65, "Mid-term management plan", 2
    ),
    _Rule("OUT-005", "management_outlook", ("通期",), "low", 50, "Full-year guidance reference", 0),
    # Business environment
    _Rule(
        "ENV-001",
        "business_environment",
        ("事業環境", "経営環境", "市場環境"),
        "low",
        55,
        "Business environment commentary",
        0,
    ),
    # Guidance revisions
    _Rule(
        "GUID-001",
        "guidance_revision",
        ("上方修正",),
        "low",
        85,
        "Upward guidance revision",
        12,
    ),
    _Rule(
        "GUID-002",
        "guidance_revision",
        ("下方修正",),
        "high",
        85,
        "Downward guidance revision",
        -12,
    ),
    _Rule(
        "GUID-003",
        "guidance_revision",
        ("業績予想を修正",),
        "medium",
        70,
        "Earnings forecast revised (direction unspecified)",
        -2,
    ),
    _Rule(
        "GUID-004",
        "guidance_revision",
        ("配当予想を修正",),
        "medium",
        70,
        "Dividend forecast revised (direction unspecified)",
        -2,
    ),
    # One-time factors
    _Rule("ONE-001", "one_time_factor", ("特別利益",), "low", 75, "Extraordinary gain", 0),
    _Rule("ONE-002", "one_time_factor", ("特別損失",), "medium", 75, "Extraordinary loss", -3),
    _Rule("ONE-003", "one_time_factor", ("一過性",), "low", 65, "One-off factor noted", 0),
    _Rule("ONE-004", "one_time_factor", ("減損損失",), "medium", 80, "Impairment loss", -6),
    _Rule(
        "ONE-005", "one_time_factor", ("固定資産売却益",), "low", 75,
        "Gain on sale of fixed assets", 0,
    ),
    # Uncertainty language
    _Rule("UNC-001", "uncertainty", ("不透明",), "low", 70, "Uncertain environment language", -3),
    _Rule("UNC-002", "uncertainty", ("未確定",), "low", 70, "Undetermined items language", -3),
    _Rule(
        "UNC-003", "uncertainty", ("可能性があります",), "low", 60,
        "Possibility-hedged language", -2,
    ),
    _Rule("UNC-004", "uncertainty", ("懸念",), "low", 70, "Concern language", -3),
    _Rule(
        "UNC-005", "uncertainty", ("変動する可能性",), "low", 65,
        "Volatility-possibility language", -3,
    ),
)

_COUNTED_CATEGORIES: dict[str, str] = {
    "positive_factor": "positive_count",
    "negative_factor": "negative_count",
    "risk_factor": "risk_count",
    "uncertainty": "uncertainty_count",
}


class RuleBasedDisclosureAnalyzer:
    """Default deterministic analyzer using Japanese keyword rules."""

    def analyze(self, document: DisclosureDocument) -> DisclosureAnalysisResult:
        text = document.text or ""
        if not text.strip():
            return DisclosureAnalysisResult(
                ticker=document.ticker,
                document_type=document.document_type,
                fiscal_year=document.fiscal_year,
                analyzer="rule_based",
                warnings=[*document.warnings, "empty disclosure text: no findings"],
                confidence_score=0.0,
                source_metadata=_document_source_metadata(document),
            )

        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        findings: list[DisclosureFinding] = []
        tone = 0.0
        counts = {"positive_count": 0, "negative_count": 0, "risk_count": 0, "uncertainty_count": 0}

        for rule in _RULES:
            for sentence in sentences:
                if any(keyword in sentence for keyword in rule.keywords):
                    findings.append(
                        DisclosureFinding(
                            category=rule.category,
                            summary=rule.summary,
                            evidence_text=sentence[:_MAX_EVIDENCE_CHARS],
                            severity=rule.severity,
                            confidence=rule.confidence,
                            rule_id=rule.rule_id,
                        )
                    )
                    tone += rule.tone_weight
                    counter = _COUNTED_CATEGORIES.get(rule.category)
                    if counter:
                        counts[counter] += 1

        warnings: list[str] = list(document.warnings)
        if not findings:
            warnings.append("no rule matched the disclosure text: coverage limited")
            confidence = 30.0
        else:
            confidence = min(90.0, 50.0 + 5.0 * len(findings))
        if len(text) < 100:
            warnings.append("very short disclosure text: confidence reduced")
            confidence = max(10.0, confidence - 20.0)

        return DisclosureAnalysisResult(
            ticker=document.ticker,
            document_type=document.document_type,
            fiscal_year=document.fiscal_year,
            analyzer="rule_based",
            findings=findings,
            tone_score=max(-100.0, min(100.0, tone)),
            warnings=warnings,
            confidence_score=round(confidence, 1),
            source_metadata=_document_source_metadata(document),
            **counts,
        )


class NoOpLLMDisclosureAnalyzer:
    """Placeholder for a future opt-in LLM analyzer.

    Returns no findings and zero confidence so downstream scoring degrades
    gracefully. Never calls a network or an LLM.
    """

    def analyze(self, document: DisclosureDocument) -> DisclosureAnalysisResult:
        return DisclosureAnalysisResult(
            ticker=document.ticker,
            document_type=document.document_type,
            fiscal_year=document.fiscal_year,
            analyzer="noop_llm",
            warnings=["LLM-based disclosure analysis is not enabled; no findings produced"],
            confidence_score=0.0,
        )

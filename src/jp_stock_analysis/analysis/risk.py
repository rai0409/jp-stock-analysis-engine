"""Risk analysis: evidence-backed risk flags and a 0-100 risk score.

0 = low risk, 100 = high risk. Missing analysis inputs produce an
``insufficient_data`` flag and reduced confidence, never fabricated values.
"""

from __future__ import annotations

from jp_stock_analysis.schemas import (
    DisclosureAnalysisResult,
    FundamentalMetrics,
    MomentumMetrics,
    RiskFlag,
    RiskMetrics,
    RiskSeverity,
    ValuationMetrics,
)

_SEVERITY_WEIGHTS: dict[RiskSeverity, float] = {
    "low": 5.0,
    "medium": 15.0,
    "high": 30.0,
    "critical": 50.0,
}


def _flag(
    risk_id: str,
    severity: RiskSeverity,
    explanation: str,
    evidence: list[str],
    confidence: float,
) -> RiskFlag:
    return RiskFlag(
        risk_id=risk_id,
        severity=severity,
        explanation=explanation,
        evidence=evidence,
        confidence=confidence,
    )


def _fundamental_flags(fundamentals: FundamentalMetrics) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    eps = fundamentals.latest_eps
    if eps is not None and eps < 0:
        flags.append(
            _flag(
                "negative_eps",
                "high",
                "Latest earnings per share is negative.",
                [f"latest_eps={eps:.2f}"],
                90,
            )
        )
    revenue_growth = fundamentals.revenue_growth_yoy
    if revenue_growth is not None and revenue_growth < 0:
        severity: RiskSeverity = "high" if revenue_growth <= -10 else "medium"
        flags.append(
            _flag(
                "declining_revenue",
                severity,
                "Revenue declined year over year.",
                [f"revenue_growth_yoy={revenue_growth:.1f}%"],
                85,
            )
        )
    operating_growth = fundamentals.operating_income_growth_yoy
    if operating_growth is not None and operating_growth < 0:
        severity = "high" if operating_growth <= -15 else "medium"
        flags.append(
            _flag(
                "declining_operating_income",
                severity,
                "Operating income declined year over year.",
                [f"operating_income_growth_yoy={operating_growth:.1f}%"],
                85,
            )
        )
    equity_ratio = fundamentals.equity_ratio
    if equity_ratio is not None and equity_ratio < 30:
        severity = "high" if equity_ratio < 20 else "medium"
        flags.append(
            _flag(
                "low_equity_ratio",
                severity,
                "Equity ratio is low; balance-sheet resilience is limited.",
                [f"equity_ratio={equity_ratio:.1f}%"],
                85,
            )
        )
    return flags


def _valuation_flags(
    valuation: ValuationMetrics, fundamentals: FundamentalMetrics | None
) -> list[RiskFlag]:
    if valuation.valuation_classification != "expensive" or fundamentals is None:
        return []
    revenue_growth = fundamentals.revenue_growth_yoy
    eps_growth = fundamentals.eps_growth_yoy
    weak_revenue = revenue_growth is None or revenue_growth < 5
    weak_eps = eps_growth is None or eps_growth < 5
    if weak_revenue and weak_eps:
        evidence = [f"valuation_classification={valuation.valuation_classification}"]
        if valuation.per is not None:
            evidence.append(f"per={valuation.per:.1f}")
        if valuation.pbr is not None:
            evidence.append(f"pbr={valuation.pbr:.2f}")
        evidence.append(f"revenue_growth_yoy={revenue_growth}")
        evidence.append(f"eps_growth_yoy={eps_growth}")
        return [
            _flag(
                "high_valuation_weak_growth",
                "medium",
                "Valuation is expensive while growth is weak or unknown.",
                evidence,
                75,
            )
        ]
    return []


def _momentum_flags(momentum: MomentumMetrics) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    vol = momentum.volatility_annualized
    if vol is not None and vol >= 40:
        severity: RiskSeverity = "high" if vol >= 60 else "medium"
        flags.append(
            _flag(
                "high_volatility",
                severity,
                "Annualized price volatility is elevated.",
                [f"volatility_annualized={vol:.1f}%"],
                80,
            )
        )
    drawdown = momentum.max_drawdown
    if drawdown is not None and drawdown <= -25:
        severity = "high" if drawdown <= -40 else "medium"
        flags.append(
            _flag(
                "large_drawdown",
                severity,
                "Price experienced a large peak-to-trough decline.",
                [f"max_drawdown={drawdown:.1f}%"],
                80,
            )
        )
    return flags


def _disclosure_flags(disclosure: DisclosureAnalysisResult) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    critical_findings = [f for f in disclosure.findings if f.severity == "critical"]
    if critical_findings:
        flags.append(
            _flag(
                "negative_disclosure_tone",
                "critical",
                "Disclosure contains critical risk language (e.g. going-concern).",
                [f.evidence_text for f in critical_findings[:3]],
                85,
            )
        )
    elif disclosure.tone_score <= -10 and disclosure.negative_count >= 1:
        negative_evidence = [
            f.evidence_text for f in disclosure.findings if f.category == "negative_factor"
        ][:3]
        flags.append(
            _flag(
                "negative_disclosure_tone",
                "medium",
                "Disclosure tone is negative.",
                [f"tone_score={disclosure.tone_score:.0f}", *negative_evidence],
                70,
            )
        )
    if disclosure.uncertainty_count >= 3:
        severity: RiskSeverity = "medium" if disclosure.uncertainty_count >= 5 else "low"
        flags.append(
            _flag(
                "many_uncertainty_mentions",
                severity,
                "Disclosure contains repeated uncertainty language.",
                [f"uncertainty_count={disclosure.uncertainty_count}"],
                70,
            )
        )
    return flags


def analyze_risks(
    fundamentals: FundamentalMetrics | None,
    valuation: ValuationMetrics | None,
    momentum: MomentumMetrics | None,
    disclosure: DisclosureAnalysisResult | None,
) -> RiskMetrics:
    """Aggregate risk flags across all analysis dimensions."""
    inputs = (
        ("fundamentals", fundamentals),
        ("valuation", valuation),
        ("momentum", momentum),
        ("disclosure", disclosure),
    )
    ticker = next((value.ticker for _, value in inputs if value is not None), "UNKNOWN")

    flags: list[RiskFlag] = []
    if fundamentals is not None:
        flags.extend(_fundamental_flags(fundamentals))
    if valuation is not None:
        flags.extend(_valuation_flags(valuation, fundamentals))
    if momentum is not None:
        flags.extend(_momentum_flags(momentum))
    if disclosure is not None:
        flags.extend(_disclosure_flags(disclosure))

    warnings: list[str] = []
    missing = [name for name, value in inputs if value is None]
    if missing:
        flags.append(
            _flag(
                "insufficient_data",
                "medium",
                "Risk assessment is based on partial data.",
                [f"missing input: {name}" for name in missing],
                100,
            )
        )
        warnings.append("risk assessment missing inputs: " + ", ".join(missing))

    risk_score = min(100.0, sum(_SEVERITY_WEIGHTS[flag.severity] for flag in flags))
    confidence = max(20.0, 100.0 - 20.0 * len(missing))
    return RiskMetrics(
        ticker=ticker,
        flags=flags,
        risk_score=round(risk_score, 1),
        warnings=warnings,
        confidence_score=round(confidence, 1),
    )

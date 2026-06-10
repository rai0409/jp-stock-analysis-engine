"""Integrated scoring: 0-100 sub-scores and a risk-adjusted final score.

Conservative, reproducible, and explainable. A sub-score is ``None`` (never
a fabricated neutral value) when its underlying metrics are unavailable; the
final score is the weighted average of available sub-scores minus a risk
penalty. Missing components reduce ``confidence_score``.
"""

from __future__ import annotations

from jp_stock_analysis.config import AnalysisConfig
from jp_stock_analysis.schemas import (
    DisclosureAnalysisResult,
    FundamentalMetrics,
    MomentumMetrics,
    RiskMetrics,
    ScoreBreakdown,
    ValuationMetrics,
)

_COMPONENT_COUNT = 5  # fundamentals, valuation, momentum, disclosure, risks


def _scale(value: float | None, low: float, high: float) -> float | None:
    """Linearly map ``value`` from [low, high] onto [0, 100], clamped."""
    if value is None or high == low:
        return None
    position = (value - low) / (high - low)
    return max(0.0, min(1.0, position)) * 100.0


def _invert(score: float | None) -> float | None:
    return None if score is None else 100.0 - score


def _mean(components: list[float | None]) -> float | None:
    available = [value for value in components if value is not None]
    if not available:
        return None
    return sum(available) / len(available)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def _quality_score(fundamentals: FundamentalMetrics | None) -> tuple[float | None, str]:
    if fundamentals is None:
        return None, "unavailable: no fundamentals data"
    score = _mean(
        [
            _scale(fundamentals.operating_margin, 0, 20),
            _scale(fundamentals.roe, 0, 15),
            _scale(fundamentals.equity_ratio, 10, 60),
            _scale(fundamentals.fcf_margin, -5, 15),
        ]
    )
    if score is None:
        return None, "unavailable: operating margin, ROE, equity ratio, FCF margin all missing"
    reason = (
        f"operating_margin={fundamentals.operating_margin}, roe={fundamentals.roe}, "
        f"equity_ratio={fundamentals.equity_ratio}, fcf_margin={fundamentals.fcf_margin}"
    )
    return score, reason


def _growth_score(fundamentals: FundamentalMetrics | None) -> tuple[float | None, str]:
    if fundamentals is None:
        return None, "unavailable: no fundamentals data"
    score = _mean(
        [
            _scale(fundamentals.revenue_growth_yoy, -10, 20),
            _scale(fundamentals.operating_income_growth_yoy, -15, 30),
            _scale(fundamentals.eps_growth_yoy, -15, 30),
        ]
    )
    if score is None:
        return None, "unavailable: no growth metrics (previous-year data missing?)"
    reason = (
        f"revenue_growth_yoy={fundamentals.revenue_growth_yoy}, "
        f"operating_income_growth_yoy={fundamentals.operating_income_growth_yoy}, "
        f"eps_growth_yoy={fundamentals.eps_growth_yoy}"
    )
    return score, reason


def _valuation_score(valuation: ValuationMetrics | None) -> tuple[float | None, str]:
    if valuation is None:
        return None, "unavailable: no valuation data"
    score = _mean(
        [
            _invert(_scale(valuation.per, 8, 35)),
            _invert(_scale(valuation.pbr, 0.8, 4.0)),
            _scale(valuation.dividend_yield, 0, 4),
        ]
    )
    if score is None:
        return None, "unavailable: PER, PBR, and dividend yield all missing"
    reason = (
        f"per={valuation.per}, pbr={valuation.pbr}, "
        f"dividend_yield={valuation.dividend_yield}, "
        f"classification={valuation.valuation_classification}"
    )
    return score, reason


def _momentum_score(momentum: MomentumMetrics | None) -> tuple[float | None, str]:
    if momentum is None:
        return None, "unavailable: no price history"
    score = _mean(
        [
            _scale(momentum.return_3m, -20, 20),
            _scale(momentum.return_6m, -25, 25),
            _scale(momentum.return_12m, -30, 30),
            _scale(momentum.max_drawdown, -50, 0),
        ]
    )
    if score is None:
        return None, f"unavailable: insufficient history ({momentum.observations} bars)"
    reason = (
        f"return_3m={momentum.return_3m}, return_6m={momentum.return_6m}, "
        f"return_12m={momentum.return_12m}, max_drawdown={momentum.max_drawdown}"
    )
    return score, reason


def _disclosure_score(disclosure: DisclosureAnalysisResult | None) -> tuple[float | None, str]:
    if disclosure is None:
        return None, "unavailable: no disclosure document"
    if not disclosure.findings:
        return None, "unavailable: no findings extracted from disclosure text"
    score = max(0.0, min(100.0, 50.0 + disclosure.tone_score))
    reason = (
        f"tone_score={disclosure.tone_score:.0f}, positive={disclosure.positive_count}, "
        f"negative={disclosure.negative_count}, risk={disclosure.risk_count}, "
        f"uncertainty={disclosure.uncertainty_count}"
    )
    return score, reason


def score_stock(
    fundamentals: FundamentalMetrics | None,
    valuation: ValuationMetrics | None,
    momentum: MomentumMetrics | None,
    disclosure: DisclosureAnalysisResult | None,
    risks: RiskMetrics | None,
    config: AnalysisConfig,
) -> ScoreBreakdown:
    """Combine component analyses into one explainable score breakdown."""
    components = (fundamentals, valuation, momentum, disclosure, risks)
    ticker = next((c.ticker for c in components if c is not None), "UNKNOWN")
    reasons: dict[str, str] = {}
    warnings: list[str] = []

    quality, reasons["quality_score"] = _quality_score(fundamentals)
    growth, reasons["growth_score"] = _growth_score(fundamentals)
    valuation_score, reasons["valuation_score"] = _valuation_score(valuation)
    momentum_score, reasons["momentum_score"] = _momentum_score(momentum)
    disclosure_score, reasons["disclosure_score"] = _disclosure_score(disclosure)

    risk_score = risks.risk_score if risks is not None else None
    reasons["risk_score"] = (
        f"{len(risks.flags)} flag(s); risk_score={risks.risk_score}"
        if risks is not None
        else "unavailable: no risk assessment"
    )

    weights = config.weights
    weighted = [
        (quality, weights.quality_score),
        (growth, weights.growth_score),
        (valuation_score, weights.valuation_score),
        (momentum_score, weights.momentum_score),
        (disclosure_score, weights.disclosure_score),
    ]
    available = [(score, weight) for score, weight in weighted if score is not None and weight > 0]

    if available:
        total_weight = sum(weight for _, weight in available)
        base = sum(score * weight for score, weight in available) / total_weight
        penalty = (risk_score or 0.0) * weights.risk_adjustment
        final = max(0.0, min(100.0, base - penalty))
        reasons["final_score"] = (
            f"weighted base {base:.1f} over {len(available)}/5 sub-scores, "
            f"risk penalty {penalty:.1f}"
        )
        if risk_score is None:
            warnings.append("risk assessment unavailable: no risk penalty applied")
    else:
        final = None
        reasons["final_score"] = "unavailable: no sub-score could be computed"
        warnings.append("final score unavailable: all sub-scores missing")

    missing_subscores = [
        name
        for name, value in (
            ("quality_score", quality),
            ("growth_score", growth),
            ("valuation_score", valuation_score),
            ("momentum_score", momentum_score),
            ("disclosure_score", disclosure_score),
        )
        if value is None
    ]
    if missing_subscores:
        warnings.append("sub-scores unavailable: " + ", ".join(missing_subscores))

    component_confidences = [c.confidence_score if c is not None else 0.0 for c in components]
    confidence = sum(component_confidences) / _COMPONENT_COUNT

    return ScoreBreakdown(
        ticker=ticker,
        quality_score=_round(quality),
        growth_score=_round(growth),
        valuation_score=_round(valuation_score),
        momentum_score=_round(momentum_score),
        disclosure_score=_round(disclosure_score),
        risk_score=_round(risk_score),
        final_score=_round(final),
        reasons=reasons,
        warnings=warnings,
        confidence_score=round(confidence, 1),
    )

"""Research trade-signal engine. Explicit opt-in only.

Signals are produced ONLY when ``config.signal_mode == "trade_signal"``;
``analysis_only`` and ``screening`` modes always return ``None``.

A ``buy_signal`` requires multiple confirming dimensions: high final score,
sufficient confidence, acceptable risk score, no critical risk flag, and at
least two supporting factors from quality/growth/momentum/disclosure. A high
valuation score alone can therefore never produce a buy signal.

No position sizing, broker execution, or portfolio optimization.
"""

from __future__ import annotations

from jp_stock_analysis.config import AnalysisConfig
from jp_stock_analysis.schemas import SignalResult, StockAnalysisResult, TradeSignalLabel

_SUPPORT_THRESHOLD = 60.0


def _supporting_factors(result: StockAnalysisResult) -> list[str]:
    """Non-valuation positive confirmations. Valuation is deliberately excluded."""
    score = result.score
    if score is None:
        return []
    factors: list[str] = []
    for name, value in (
        ("quality_score", score.quality_score),
        ("growth_score", score.growth_score),
        ("momentum_score", score.momentum_score),
        ("disclosure_score", score.disclosure_score),
    ):
        if value is not None and value >= _SUPPORT_THRESHOLD:
            factors.append(f"{name}={value} (>= {_SUPPORT_THRESHOLD:.0f})")
    if result.disclosure is not None and result.disclosure.positive_count >= 2:
        factors.append(
            f"disclosure positive findings: {result.disclosure.positive_count}"
        )
    return factors


def _evidence(result: StockAnalysisResult) -> list[str]:
    evidence: list[str] = []
    score = result.score
    if score is not None:
        evidence.append(
            f"scores: final={score.final_score}, quality={score.quality_score}, "
            f"growth={score.growth_score}, valuation={score.valuation_score}, "
            f"momentum={score.momentum_score}, disclosure={score.disclosure_score}, "
            f"risk={score.risk_score}, confidence={score.confidence_score}"
        )
    if result.disclosure is not None:
        for finding in result.disclosure.findings[:3]:
            evidence.append(f"disclosure [{finding.rule_id}]: {finding.evidence_text}")
    if result.risks is not None:
        for flag in result.risks.flags[:3]:
            evidence.append(f"risk [{flag.risk_id}/{flag.severity}]: {flag.explanation}")
    return evidence


def _blocking_risks(result: StockAnalysisResult) -> list[str]:
    if result.risks is None:
        return ["risk assessment unavailable"]
    return [
        f"{flag.risk_id} ({flag.severity}): {flag.explanation}"
        for flag in result.risks.flags
        if flag.severity in ("high", "critical")
    ]


def generate_signal(result: StockAnalysisResult, config: AnalysisConfig) -> SignalResult | None:
    """Generate one research signal; ``None`` unless mode is ``trade_signal``."""
    if config.signal_mode != "trade_signal":
        return None

    thresholds = config.thresholds
    thresholds_used = {
        "buy_signal_threshold": thresholds.buy_signal_threshold,
        "sell_signal_threshold": thresholds.sell_signal_threshold,
        "min_confidence_for_signal": thresholds.min_confidence_for_signal,
        "max_risk_score_for_buy_signal": thresholds.max_risk_score_for_buy_signal,
        "candidate_threshold": thresholds.candidate_threshold,
    }

    score = result.score
    final = score.final_score if score else None
    confidence = score.confidence_score if score else 0.0
    risk_score = result.risks.risk_score if result.risks is not None else None
    has_critical = result.risks is not None and any(
        flag.severity == "critical" for flag in result.risks.flags
    )
    supporting = _supporting_factors(result)
    blocking = _blocking_risks(result)

    label: TradeSignalLabel
    if final is None or confidence < thresholds.min_confidence_for_signal:
        label = "insufficient_data"
        rationale = (
            f"Confidence {confidence:.1f} is below the minimum "
            f"{thresholds.min_confidence_for_signal:.0f} or the final score is unavailable; "
            "no directional research signal is justified."
        )
    elif final <= thresholds.sell_signal_threshold or (
        has_critical and final < thresholds.candidate_threshold
    ):
        label = "sell_signal"
        rationale = (
            f"Final score {final:.1f} is at or below the sell threshold "
            f"{thresholds.sell_signal_threshold:.0f}, or a critical risk flag combined with a "
            "weak score indicates severe deterioration."
        )
    elif has_critical or (risk_score is not None and risk_score >= 70):
        label = "avoid_signal"
        rationale = (
            f"Risk profile is unacceptable (risk_score={risk_score}, "
            f"critical_flag={has_critical}) despite final score {final:.1f}."
        )
    elif (
        final >= thresholds.buy_signal_threshold
        and confidence >= thresholds.min_confidence_for_signal
        and risk_score is not None
        and risk_score <= thresholds.max_risk_score_for_buy_signal
        and not has_critical
        and len(supporting) >= 2
    ):
        label = "buy_signal"
        rationale = (
            f"Final score {final:.1f} >= {thresholds.buy_signal_threshold:.0f} with "
            f"confidence {confidence:.1f}, risk_score {risk_score:.1f} <= "
            f"{thresholds.max_risk_score_for_buy_signal:.0f}, no critical risk flag, and "
            f"{len(supporting)} non-valuation supporting factors."
        )
    elif final >= thresholds.candidate_threshold:
        label = "watch_signal"
        rationale = (
            f"Final score {final:.1f} is promising but buy confirmation is incomplete "
            f"(supporting factors: {len(supporting)}, risk_score: {risk_score})."
        )
    elif risk_score is not None and risk_score >= 60:
        label = "avoid_signal"
        rationale = (
            f"Final score {final:.1f} is unremarkable and risk_score {risk_score:.1f} "
            "is elevated; structurally weak profile."
        )
    else:
        label = "hold_signal"
        rationale = (
            f"Final score {final:.1f} is mixed/neutral: no buy confirmation and no sell "
            "or avoid trigger."
        )

    return SignalResult(
        ticker=result.ticker,
        label=label,
        confidence=round(confidence, 1),
        rationale=rationale,
        evidence=_evidence(result),
        blocking_risks=blocking,
        supporting_factors=supporting,
        thresholds_used=thresholds_used,
        disclaimer=config.disclaimer,
    )


def generate_signals(
    results: list[StockAnalysisResult], config: AnalysisConfig
) -> list[SignalResult]:
    """Generate and attach signals for all results; empty unless trade_signal mode."""
    signals: list[SignalResult] = []
    for result in results:
        signal = generate_signal(result, config)
        result.signal = signal
        if signal is not None:
            signals.append(signal)
    return signals

"""Valuation analysis: PER, PBR, PSR, dividend yield, PEG, classification.

Valuation output is descriptive only. Per project rules, valuation alone must
never produce a buy/sell signal; the signal engine enforces multi-dimension
confirmation.
"""

from __future__ import annotations

from jp_stock_analysis.schemas import (
    FinancialStatement,
    ValuationClassification,
    ValuationMetrics,
)

_CHEAP_PER = 12.0
_EXPENSIVE_PER = 30.0
_CHEAP_PBR = 1.0
_EXPENSIVE_PBR = 4.0

_METRIC_FIELDS = ("per", "pbr", "psr", "dividend_yield", "peg", "market_cap")


def classify_valuation(metrics: ValuationMetrics) -> ValuationClassification:
    """Classify valuation as cheap / fair / expensive / unavailable."""
    per, pbr = metrics.per, metrics.pbr
    if per is None and pbr is None:
        return "unavailable"
    cheap = (per is not None and per < _CHEAP_PER) or (pbr is not None and pbr < _CHEAP_PBR)
    expensive = (per is not None and per > _EXPENSIVE_PER) or (
        pbr is not None and pbr > _EXPENSIVE_PBR
    )
    if cheap and not expensive:
        return "cheap"
    if expensive and not cheap:
        return "expensive"
    return "fair"


def analyze_valuation(
    statement: FinancialStatement,
    market_price: float | None,
    eps_growth_yoy: float | None = None,
) -> ValuationMetrics:
    """Derive valuation metrics from one statement and the current market price."""
    warnings: list[str] = []
    per = pbr = psr = dividend_yield = peg = market_cap = None

    if market_price is None or market_price <= 0:
        warnings.append("market price unavailable: valuation metrics cannot be computed")
        market_price = None
    else:
        if statement.eps is None:
            warnings.append("EPS missing: PER unavailable")
        elif statement.eps <= 0:
            warnings.append("EPS is zero or negative: PER unavailable")
        else:
            per = market_price / statement.eps

        if statement.bps is not None and statement.bps > 0:
            pbr = market_price / statement.bps
        else:
            warnings.append("BPS missing or non-positive: PBR unavailable")

        if statement.shares_outstanding is not None and statement.shares_outstanding > 0:
            market_cap = market_price * statement.shares_outstanding
        else:
            warnings.append("shares outstanding missing: market cap and PSR unavailable")

        if market_cap is not None:
            if statement.revenue is not None and statement.revenue > 0:
                psr = market_cap / statement.revenue
            else:
                warnings.append("revenue missing or non-positive: PSR unavailable")

        if statement.dividends_per_share is not None:
            dividend_yield = statement.dividends_per_share / market_price * 100.0
        else:
            warnings.append("dividends per share missing: dividend yield unavailable")

        if per is not None:
            if eps_growth_yoy is None:
                warnings.append("EPS growth unavailable: PEG unavailable")
            elif eps_growth_yoy <= 0:
                warnings.append("EPS growth not positive: PEG unavailable")
            else:
                peg = per / eps_growth_yoy

    metrics = ValuationMetrics(
        ticker=statement.ticker,
        market_price=market_price,
        per=per,
        pbr=pbr,
        psr=psr,
        dividend_yield=dividend_yield,
        peg=peg,
        market_cap=market_cap,
        warnings=warnings,
        source_metadata=dict(statement.source_metadata),
    )
    metrics.valuation_classification = classify_valuation(metrics)
    available = sum(1 for field in _METRIC_FIELDS if getattr(metrics, field) is not None)
    metrics.confidence_score = round(available / len(_METRIC_FIELDS) * 100.0, 1)
    return metrics

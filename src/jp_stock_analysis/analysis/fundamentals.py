"""Fundamental analysis: growth, margins, returns, balance-sheet health.

All ratios are percent values (10.0 means 10%). Missing inputs produce
``None`` metrics plus warnings; values are never fabricated.
"""

from __future__ import annotations

from jp_stock_analysis.schemas import FinancialStatement, FundamentalMetrics

_METRIC_FIELDS = (
    "revenue_growth_yoy",
    "operating_income_growth_yoy",
    "net_income_growth_yoy",
    "eps_growth_yoy",
    "operating_margin",
    "net_margin",
    "roe",
    "roa",
    "equity_ratio",
    "fcf_margin",
    "dividend_payout_ratio",
)


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    """Divide, returning ``None`` on missing inputs or zero denominator."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def pct_change(current: float | None, previous: float | None) -> float | None:
    """Change in percent versus the previous value; ``None`` when not computable."""
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / abs(previous) * 100.0


def _as_percent(ratio: float | None) -> float | None:
    return None if ratio is None else ratio * 100.0


def analyze_fundamentals(
    current: FinancialStatement,
    previous: FinancialStatement | None = None,
) -> FundamentalMetrics:
    """Derive fundamental metrics from the latest (and optionally prior) statement."""
    warnings: list[str] = []

    if previous is None:
        warnings.append("no previous-year statement: growth metrics unavailable")
    elif (
        current.fiscal_period
        and previous.fiscal_period
        and current.fiscal_period != previous.fiscal_period
    ):
        warnings.append(
            "fiscal period mismatch: "
            f"{current.fiscal_period!r} vs {previous.fiscal_period!r}; "
            "growth comparison may be unreliable"
        )

    revenue_growth = pct_change(current.revenue, previous.revenue) if previous else None
    operating_income_growth = (
        pct_change(current.operating_income, previous.operating_income) if previous else None
    )
    net_income_growth = pct_change(current.net_income, previous.net_income) if previous else None
    eps_growth = pct_change(current.eps, previous.eps) if previous else None

    if current.eps is not None and current.eps < 0:
        warnings.append("negative EPS in latest statement")

    free_cash_flow: float | None = None
    if current.operating_cash_flow is not None and current.capital_expenditure is not None:
        free_cash_flow = current.operating_cash_flow - current.capital_expenditure

    dividend_payout_ratio: float | None = None
    if current.dividends_per_share is not None:
        if current.eps is not None and current.eps > 0:
            dividend_payout_ratio = current.dividends_per_share / current.eps * 100.0
        else:
            warnings.append("dividend payout ratio unavailable (EPS missing, zero, or negative)")

    metrics = FundamentalMetrics(
        ticker=current.ticker,
        fiscal_year=current.fiscal_year,
        latest_eps=current.eps,
        revenue_growth_yoy=revenue_growth,
        operating_income_growth_yoy=operating_income_growth,
        net_income_growth_yoy=net_income_growth,
        eps_growth_yoy=eps_growth,
        operating_margin=_as_percent(safe_divide(current.operating_income, current.revenue)),
        net_margin=_as_percent(safe_divide(current.net_income, current.revenue)),
        roe=_as_percent(safe_divide(current.net_income, current.equity)),
        roa=_as_percent(safe_divide(current.net_income, current.total_assets)),
        equity_ratio=_as_percent(safe_divide(current.equity, current.total_assets)),
        fcf_margin=_as_percent(safe_divide(free_cash_flow, current.revenue)),
        dividend_payout_ratio=dividend_payout_ratio,
        warnings=warnings,
        source_metadata=dict(current.source_metadata),
    )

    missing = [field for field in _METRIC_FIELDS if getattr(metrics, field) is None]
    if missing:
        metrics.warnings.append("metrics unavailable: " + ", ".join(missing))
    available = len(_METRIC_FIELDS) - len(missing)
    metrics.confidence_score = round(available / len(_METRIC_FIELDS) * 100.0, 1)
    return metrics


def analyze_fundamentals_by_ticker(
    statements: dict[str, list[FinancialStatement]],
) -> dict[str, FundamentalMetrics]:
    """Analyze the latest two statements per ticker."""
    results: dict[str, FundamentalMetrics] = {}
    for ticker, ticker_statements in statements.items():
        if not ticker_statements:
            continue
        ordered = sorted(
            ticker_statements,
            key=lambda s: (s.fiscal_year is None, s.fiscal_year or 0, s.fiscal_period or ""),
        )
        previous = ordered[-2] if len(ordered) > 1 else None
        results[ticker] = analyze_fundamentals(ordered[-1], previous)
    return results

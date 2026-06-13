"""Factor feature engineering for cross-sectional stock screening research.

Pure, deterministic, explainable factor computation. Every factor is computed
from already-loaded inputs only; a missing or zero denominator yields ``None``
(never a fabricated number, never a divide-by-zero crash). Nothing here makes a
predictive or trading claim — these are descriptive features.

Factor groups (see :data:`FACTOR_GROUPS`):

- ``value``       earnings yield, book-to-market, sales-to-price (need a price)
- ``quality``     ROE, ROA, operating margin, equity ratio
- ``growth``      revenue / net-income YoY (need a prior-year statement)
- ``momentum``    20/60/120-day price momentum (need adjusted-close history)
- ``risk``        volatility, max drawdown, leverage proxy
- ``disclosure``  narrative presence / risk-keyword count / sentiment placeholder
                  (NO LLM, NO external NLP — placeholders until real extraction)

Normalisation helpers (:func:`winsorize`, :func:`zscore`, :func:`sector_zscore`)
operate cross-sectionally over a list of values aligned to one decision date.
``None`` values stay ``None`` (a missing-value indicator), are never imputed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from jp_stock_analysis.schemas import (
    CompanyMetadata,
    DisclosureDocument,
    FinancialStatement,
    PriceBar,
)

FACTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "value": ("earnings_yield", "book_to_market", "sales_to_price"),
    "quality": ("roe", "roa", "operating_margin", "equity_ratio"),
    "growth": ("revenue_growth_yoy", "net_income_growth_yoy"),
    "momentum": ("momentum_20d", "momentum_60d", "momentum_120d"),
    "risk": ("volatility", "max_drawdown", "leverage"),
    "disclosure": ("narrative_available", "risk_keyword_count", "sentiment_placeholder"),
}

ALL_FACTORS: tuple[str, ...] = tuple(f for group in FACTOR_GROUPS.values() for f in group)

# Is a higher raw value more favourable? Used by the baseline ranker and
# sector-neutral IC so lower-is-better factors are direction-inverted.
FACTOR_DIRECTION: dict[str, bool] = {
    "earnings_yield": True,
    "book_to_market": True,
    "sales_to_price": True,
    "roe": True,
    "roa": True,
    "operating_margin": True,
    "equity_ratio": True,
    "revenue_growth_yoy": True,
    "net_income_growth_yoy": True,
    "momentum_20d": True,
    "momentum_60d": True,
    "momentum_120d": True,
    "volatility": False,
    "max_drawdown": True,  # stored as a negative percent; less negative is better
    "leverage": False,
    "narrative_available": True,
    "risk_keyword_count": False,
    "sentiment_placeholder": True,
}

# Conservative Japanese disclosure risk keywords (presence-count only; no LLM).
RISK_KEYWORDS: tuple[str, ...] = (
    "減損",
    "下方修正",
    "リスク",
    "訴訟",
    "為替",
    "繰延税金",
    "継続企業",
    "重要事象",
)

_MOMENTUM_WINDOWS = {"momentum_20d": 20, "momentum_60d": 60, "momentum_120d": 120}
_TRADING_DAYS_PER_YEAR = 252


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Return ``numerator/denominator`` or ``None`` (missing or zero divisor)."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _pct(value: float | None) -> float | None:
    return None if value is None else value * 100.0


@dataclass(frozen=True)
class FactorResult:
    """Factors for one ticker at one decision date, with provenance."""

    ticker: str
    features: dict[str, float | None]
    market_price: float | None
    available_count: int
    missing_factors: tuple[str, ...]


def _adjusted_series(bars: Sequence[PriceBar]) -> list[float]:
    """Closing series in date order, preferring adjusted close when complete."""
    ordered = sorted(bars, key=lambda b: b.date)
    if ordered and all(b.adjusted_close is not None for b in ordered):
        return [float(b.adjusted_close) for b in ordered]  # type: ignore[arg-type]
    return [float(b.close) for b in ordered]


def _momentum(series: Sequence[float], window: int) -> float | None:
    if len(series) < window + 1:
        return None
    base = series[-1 - window]
    if base == 0:
        return None
    return (series[-1] / base - 1.0) * 100.0


def _volatility(series: Sequence[float]) -> float | None:
    if len(series) < 3:
        return None
    rets = [
        series[i] / series[i - 1] - 1.0
        for i in range(1, len(series))
        if series[i - 1] != 0
    ]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(_TRADING_DAYS_PER_YEAR) * 100.0


def _max_drawdown(series: Sequence[float]) -> float | None:
    if len(series) < 2:
        return None
    peak = series[0]
    worst = 0.0
    for price in series:
        peak = max(peak, price)
        if peak > 0:
            worst = min(worst, price / peak - 1.0)
    return worst * 100.0


def _narrative_features(narrative: DisclosureDocument | None) -> dict[str, float | None]:
    """Disclosure placeholder features. No LLM, no external NLP.

    ``narrative_available`` is always 0/1 (a presence indicator). The keyword
    count and sentiment placeholder are ``None`` when no narrative text exists
    yet (extraction not attempted) — never fabricated as 0.
    """
    text = (narrative.text or "").strip() if narrative else ""
    if not text:
        return {
            "narrative_available": 0.0,
            "risk_keyword_count": None,
            "sentiment_placeholder": None,
        }
    keyword_count = float(sum(text.count(keyword) for keyword in RISK_KEYWORDS))
    return {
        "narrative_available": 1.0,
        "risk_keyword_count": keyword_count,
        # neutral placeholder: a real sentiment model is explicitly out of scope.
        "sentiment_placeholder": 0.0,
    }


def compute_factors(
    statement: FinancialStatement | None,
    prior_statement: FinancialStatement | None,
    bars: Sequence[PriceBar] | None,
    metadata: CompanyMetadata | None = None,
    narrative: DisclosureDocument | None = None,
    *,
    market_price: float | None = None,
) -> FactorResult:
    """Compute every factor for one ticker; missing inputs yield ``None``.

    ``market_price`` overrides the latest close for valuation factors; when not
    given the latest bar's adjusted/close is used. Value factors need a price
    and per-share data, so they stay ``None`` when either is absent.
    """
    ticker = (
        statement.ticker
        if statement is not None
        else (metadata.ticker if metadata else "")
    )
    features: dict[str, float | None] = dict.fromkeys(ALL_FACTORS)

    ordered_bars = sorted(bars, key=lambda b: b.date) if bars else []
    series = _adjusted_series(ordered_bars) if ordered_bars else []
    price = market_price
    if price is None and series:
        price = series[-1]

    # ----- value (need a price and per-share / aggregate equity data) -----
    market_cap: float | None = None
    if statement is not None and price is not None and statement.shares_outstanding:
        market_cap = price * statement.shares_outstanding
    if statement is not None and market_cap:
        features["earnings_yield"] = _pct(_safe_div(statement.net_income, market_cap))
        features["book_to_market"] = _safe_div(statement.equity, market_cap)
        features["sales_to_price"] = _safe_div(statement.revenue, market_cap)

    # ----- quality -----
    if statement is not None:
        features["roe"] = _pct(_safe_div(statement.net_income, statement.equity))
        features["roa"] = _pct(_safe_div(statement.net_income, statement.total_assets))
        features["operating_margin"] = _pct(
            _safe_div(statement.operating_income, statement.revenue)
        )
        features["equity_ratio"] = _pct(_safe_div(statement.equity, statement.total_assets))
        features["leverage"] = _safe_div(statement.total_assets, statement.equity)

    # ----- growth (need a prior-year statement) -----
    if statement is not None and prior_statement is not None:
        features["revenue_growth_yoy"] = _pct(
            _safe_div(
                (statement.revenue or 0) - (prior_statement.revenue or 0)
                if statement.revenue is not None and prior_statement.revenue is not None
                else None,
                prior_statement.revenue,
            )
        )
        features["net_income_growth_yoy"] = _pct(
            _safe_div(
                (statement.net_income or 0) - (prior_statement.net_income or 0)
                if statement.net_income is not None
                and prior_statement.net_income is not None
                else None,
                prior_statement.net_income,
            )
        )

    # ----- momentum / risk (need price history) -----
    if series:
        for name, window in _MOMENTUM_WINDOWS.items():
            features[name] = _momentum(series, window)
        features["volatility"] = _volatility(series)
        features["max_drawdown"] = _max_drawdown(series)

    # ----- disclosure placeholders -----
    features.update(_narrative_features(narrative))

    available = sum(1 for value in features.values() if value is not None)
    missing = tuple(name for name in ALL_FACTORS if features[name] is None)
    return FactorResult(
        ticker=ticker,
        features=features,
        market_price=price,
        available_count=available,
        missing_factors=missing,
    )


# --------------------------------------------------------------------------- #
# Cross-sectional normalisation (deterministic; None stays None)
# --------------------------------------------------------------------------- #
def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted list."""
    if not sorted_values:
        raise ValueError("quantile of empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[int(pos)]
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def winsorize(
    values: Sequence[float | None], lower: float = 0.05, upper: float = 0.95
) -> list[float | None]:
    """Clip non-missing values to the [lower, upper] quantile band."""
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("require 0 <= lower < upper <= 1")
    present = sorted(v for v in values if v is not None)
    if len(present) < 2:
        return list(values)
    lo = _quantile(present, lower)
    hi = _quantile(present, upper)
    return [None if v is None else min(max(v, lo), hi) for v in values]


def zscore(values: Sequence[float | None]) -> list[float | None]:
    """Z-score non-missing values. Zero/degenerate variance -> 0.0 for present."""
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return [None if v is None else 0.0 for v in values]
    mean = sum(present) / len(present)
    var = sum((v - mean) ** 2 for v in present) / (len(present) - 1)
    std = math.sqrt(var)
    if std == 0:
        return [None if v is None else 0.0 for v in values]
    return [None if v is None else (v - mean) / std for v in values]


def sector_zscore(
    values: Sequence[float | None], sectors: Sequence[str | None]
) -> list[float | None]:
    """Z-score within each sector group; ungrouped / tiny groups -> 0.0/None."""
    if len(values) != len(sectors):
        raise ValueError("values and sectors must align")
    groups: dict[str | None, list[int]] = {}
    for index, sector in enumerate(sectors):
        groups.setdefault(sector, []).append(index)
    out: list[float | None] = [None] * len(values)
    for sector, indices in groups.items():
        group_values = [values[i] for i in indices]
        if sector is None:
            scored: list[float | None] = [None if v is None else 0.0 for v in group_values]
        else:
            scored = zscore(group_values)
        for position, index in enumerate(indices):
            out[index] = scored[position]
    return out

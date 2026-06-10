"""Momentum analysis: returns, moving averages, volatility, drawdown, volume.

Uses ``adjusted_close`` when available, otherwise ``close``. Bars are sorted
by date. Insufficient history yields ``None`` metrics with warnings and a
reduced confidence score.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from jp_stock_analysis.schemas import MomentumMetrics, PriceBar, VolumeTrend

_TRADING_DAYS_PER_YEAR = 252

_RETURN_LOOKBACKS = {
    "return_1m": 21,
    "return_3m": 63,
    "return_6m": 126,
    "return_12m": 252,
}

_MA_WINDOWS = {
    "moving_average_20d": 20,
    "moving_average_60d": 60,
    "moving_average_120d": 120,
    "moving_average_200d": 200,
}

_MIN_BARS_FOR_VOLATILITY = 21
_VOLUME_WINDOW = 20


def calculate_returns(closes: Sequence[float], lookback_days: int) -> float | None:
    """Percent return over the given trading-day lookback; ``None`` if short."""
    if len(closes) <= lookback_days or lookback_days <= 0:
        return None
    base = closes[-1 - lookback_days]
    if base == 0:
        return None
    return (closes[-1] / base - 1.0) * 100.0


def moving_average(closes: Sequence[float], window: int) -> float | None:
    """Simple moving average of the last ``window`` closes; ``None`` if short."""
    if window <= 0 or len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def volatility(closes: Sequence[float]) -> float | None:
    """Annualized daily-return volatility in percent; ``None`` if short."""
    if len(closes) < _MIN_BARS_FOR_VOLATILITY:
        return None
    daily_returns = [
        closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1] != 0
    ]
    if len(daily_returns) < 2:
        return None
    return statistics.stdev(daily_returns) * math.sqrt(_TRADING_DAYS_PER_YEAR) * 100.0


def max_drawdown(closes: Sequence[float]) -> float | None:
    """Worst peak-to-trough decline as a negative percent; ``None`` if short."""
    if len(closes) < 2:
        return None
    peak = closes[0]
    worst = 0.0
    for close in closes:
        peak = max(peak, close)
        if peak > 0:
            worst = min(worst, (close / peak - 1.0) * 100.0)
    return worst


def _volume_trend(bars: Sequence[PriceBar]) -> VolumeTrend | None:
    volumes = [bar.volume for bar in bars if bar.volume is not None]
    if len(volumes) < 2 * _VOLUME_WINDOW:
        return None
    recent = sum(volumes[-_VOLUME_WINDOW:]) / _VOLUME_WINDOW
    prior = sum(volumes[-2 * _VOLUME_WINDOW : -_VOLUME_WINDOW]) / _VOLUME_WINDOW
    if prior <= 0:
        return None
    ratio = recent / prior
    if ratio > 1.1:
        return "increasing"
    if ratio < 0.9:
        return "decreasing"
    return "flat"


def analyze_momentum(price_bars: Sequence[PriceBar]) -> MomentumMetrics:
    """Derive momentum metrics from daily price bars."""
    if not price_bars:
        return MomentumMetrics(
            ticker="UNKNOWN",
            observations=0,
            warnings=["no price history available"],
            confidence_score=0.0,
        )

    bars = sorted(price_bars, key=lambda bar: bar.date)
    closes = [
        bar.adjusted_close if bar.adjusted_close is not None else bar.close for bar in bars
    ]
    warnings: list[str] = []

    values: dict[str, float | None] = {}
    for name, lookback in _RETURN_LOOKBACKS.items():
        values[name] = calculate_returns(closes, lookback)
    for name, window in _MA_WINDOWS.items():
        values[name] = moving_average(closes, window)
    values["volatility_annualized"] = volatility(closes)
    values["max_drawdown"] = max_drawdown(closes)

    missing = [name for name, value in values.items() if value is None]
    if missing:
        warnings.append(
            f"insufficient history ({len(bars)} bars) for: " + ", ".join(sorted(missing))
        )
    if len(bars) < 60:
        warnings.append(f"short price history ({len(bars)} bars): momentum confidence reduced")

    trend = _volume_trend(bars)
    if trend is None:
        warnings.append("volume trend unavailable (needs 40 bars with volume)")

    confidence = round(min(1.0, len(bars) / _TRADING_DAYS_PER_YEAR) * 100.0, 1)
    return MomentumMetrics(
        ticker=bars[-1].ticker,
        observations=len(bars),
        volume_trend=trend,
        warnings=warnings,
        confidence_score=confidence,
        **values,
    )

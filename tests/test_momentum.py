"""Tests for momentum analysis."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from conftest import make_price_bars

from jp_stock_analysis.analysis.momentum import (
    analyze_momentum,
    calculate_returns,
    max_drawdown,
    moving_average,
    volatility,
)
from jp_stock_analysis.schemas import PriceBar


def test_calculate_returns_exact():
    closes = [100.0] + [105.0] * 20 + [110.0]  # 22 points, lookback 21 hits index 0
    assert calculate_returns(closes, 21) == pytest.approx(10.0)
    assert calculate_returns(closes, 100) is None


def test_moving_average_exact():
    closes = [float(i) for i in range(1, 31)]
    assert moving_average(closes, 10) == pytest.approx(25.5)
    assert moving_average(closes, 31) is None


def test_max_drawdown_exact():
    assert max_drawdown([100.0, 120.0, 60.0, 90.0]) == pytest.approx(-50.0)
    assert max_drawdown([100.0]) is None


def test_volatility_positive_for_moving_series():
    closes = [bar.close for bar in make_price_bars(days=100)]
    vol = volatility(closes)
    assert vol is not None and vol > 0
    assert volatility([100.0] * 10) is None  # too short


def test_full_history_produces_all_metrics():
    bars = make_price_bars(days=300, daily_drift=0.001, volume_slope=500.0)
    metrics = analyze_momentum(bars)
    assert metrics.observations == 300
    assert metrics.return_1m is not None
    assert metrics.return_12m is not None
    assert metrics.moving_average_200d is not None
    assert metrics.volatility_annualized is not None and metrics.volatility_annualized > 0
    assert metrics.max_drawdown is not None and metrics.max_drawdown <= 0
    assert metrics.volume_trend is not None
    assert metrics.confidence_score == 100.0
    # rising series must show positive long-horizon return
    assert metrics.return_12m > 0


def test_volume_trend_direction():
    rising = analyze_momentum(
        make_price_bars(days=60, volume_base=1000.0, volume_slope=1000.0)
    )
    assert rising.volume_trend == "increasing"
    falling = analyze_momentum(
        make_price_bars(days=60, volume_base=100_000.0, volume_slope=-1000.0)
    )
    assert falling.volume_trend == "decreasing"
    flat = analyze_momentum(make_price_bars(days=60, volume_slope=0.0))
    assert flat.volume_trend == "flat"


def test_insufficient_history_returns_none_with_warnings():
    metrics = analyze_momentum(make_price_bars(days=10))
    assert metrics.return_1m is None
    assert metrics.return_12m is None
    assert metrics.moving_average_200d is None
    assert metrics.warnings
    assert metrics.confidence_score < 10.0


def test_empty_history():
    metrics = analyze_momentum([])
    assert metrics.observations == 0
    assert metrics.confidence_score == 0.0
    assert metrics.warnings


def test_uses_adjusted_close_when_available():
    bars = [
        PriceBar(
            ticker="7203",
            date=date(2024, 1, 1) + timedelta(days=i),
            close=100.0,
            adjusted_close=100.0 + i,
        )
        for i in range(30)
    ]
    metrics = analyze_momentum(bars)
    # 21-day return on the adjusted series: (129 / 108 - 1) * 100
    assert metrics.return_1m == pytest.approx((129.0 / 108.0 - 1.0) * 100.0)


def test_unsorted_input_is_sorted_by_date():
    bars = make_price_bars(days=100)
    assert analyze_momentum(list(reversed(bars))) == analyze_momentum(bars)

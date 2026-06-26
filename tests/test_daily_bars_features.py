"""No-network tests for daily bars analysis-prep features."""

from __future__ import annotations

import math

import pandas as pd

from jp_stock_analysis.validation.jquants_daily_bars import (
    build_daily_bars_analysis_features,
    map_daily_bar_row,
    write_daily_bars_csv,
)


def _raw(ticker: str, day: str, close: float, open_: float, high: float, low: float, volume: int):
    return {
        "Date": day,
        "Code": f"{ticker}0" if ticker.isdigit() and len(ticker) == 4 else ticker,
        "O": open_,
        "H": high,
        "L": low,
        "C": close,
        "UL": "0",
        "LL": "0",
        "Vo": volume,
        "Va": volume * close,
        "AdjFactor": 1.0,
        "AdjO": open_,
        "AdjH": high,
        "AdjL": low,
        "AdjC": close,
        "AdjVo": volume,
    }


def test_feature_builder_computes_returns_ranges_rolling_and_coverage_join(tmp_path):
    daily = tmp_path / "bars.csv"
    rows = [
        map_daily_bar_row(_raw("1301", "2024-12-10", 100, 98, 101, 97, 1000)),
        map_daily_bar_row(_raw("7203", "2024-12-10", 200, 198, 202, 196, 2000)),
        map_daily_bar_row(_raw("1301", "2024-12-11", 110, 105, 112, 104, 1200)),
        map_daily_bar_row(_raw("7203", "2024-12-11", 210, 205, 213, 204, 2200)),
        map_daily_bar_row(_raw("1301", "2024-12-12", 121, 115, 123, 114, 1400)),
        map_daily_bar_row(_raw("7203", "2024-12-12", 231, 225, 235, 224, 2400)),
    ]
    write_daily_bars_csv(rows, daily)
    coverage = tmp_path / "coverage.csv"
    coverage.write_text(
        "ticker,sector,market,new_index_category\n"
        "1301,Fish,Prime,TOPIX1000\n"
        "7203,Auto,Prime,TOPIX1000\n",
        encoding="utf-8",
    )
    out = tmp_path / "features.csv"
    features = build_daily_bars_analysis_features(
        daily_bars_file=daily,
        coverage_file=coverage,
        output_file=out,
        lookback_days=2,
        include_partial_history=True,
    )
    row = features[(features["ticker"] == "1301") & (features["date"] == "2024-12-11")].iloc[0]
    assert math.isclose(row["ret_1d"], 0.1)
    assert math.isclose(row["intraday_return"], 110 / 105 - 1)
    assert math.isclose(row["high_low_range"], 112 / 104 - 1)
    later = features[(features["ticker"] == "1301") & (features["date"] == "2024-12-12")].iloc[0]
    assert math.isclose(later["avg_volume_20d"], 1300.0)
    assert math.isclose(later["avg_turnover_20d"], (1200 * 110 + 1400 * 121) / 2)
    assert later["sector"] == "Fish"
    assert later["market"] == "Prime"
    assert later["new_index_category"] == "TOPIX1000"
    assert out.exists()
    written = pd.read_csv(out, dtype={"ticker": str})
    assert list(written.columns) == list(features.columns)

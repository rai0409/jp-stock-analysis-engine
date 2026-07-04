from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from jp_stock_analysis.analysis.jquants_liquidity import build_liquidity_features


def _price_rows(ticker: str, avg_turnover: float) -> list[dict[str, object]]:
    rows = []
    day = date(2024, 1, 1)
    for i in range(22):
        rows.append(
            {
                "ticker": ticker,
                "date": (day + timedelta(days=i)).isoformat(),
                "adj_close": 100 + i,
                "turnover_value": avg_turnover,
                "volume": 1000 + i,
                "adj_volume": 1000 + i,
            }
        )
    return rows


def test_liquidity_mid_and_high_buckets(tmp_path):
    abnormal_path = tmp_path / "abnormal.csv"
    prices_path = tmp_path / "prices.csv"
    output_path = tmp_path / "liquidity.csv"
    summary_path = tmp_path / "summary.json"
    pd.DataFrame(
        [
            {
                "ticker": "1001",
                "reaction_base_date": "2024-01-21",
                "base_turnover_value": 300_000_000,
                "base_volume": 2000,
                "base_adj_volume": 2000,
            },
            {
                "ticker": "1002",
                "reaction_base_date": "2024-01-21",
                "base_turnover_value": 2_000_000_000,
                "base_volume": 2000,
                "base_adj_volume": 2000,
            },
        ]
    ).to_csv(abnormal_path, index=False)
    pd.DataFrame(_price_rows("1001", 100_000_000) + _price_rows("1002", 1_000_000_000)).to_csv(
        prices_path, index=False
    )

    build_liquidity_features(abnormal_path, prices_path, output_path, summary_path)
    out = pd.read_csv(output_path, dtype={"ticker": str})

    assert out[out["ticker"] == "1001"].iloc[0]["liquidity_bucket"] == "mid"
    assert out[out["ticker"] == "1001"].iloc[0]["is_liquid_enough"]
    assert out[out["ticker"] == "1002"].iloc[0]["liquidity_bucket"] == "high"
    assert out[out["ticker"] == "1002"].iloc[0]["is_liquid_enough"]

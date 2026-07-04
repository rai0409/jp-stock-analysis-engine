from __future__ import annotations

import json

import pandas as pd

from jp_stock_analysis.analysis.jquants_price_reaction_v2 import build_price_reaction_v2


def _prices(ticker: str, dates: list[str], start: float = 100.0) -> list[dict[str, object]]:
    rows = []
    for i, date in enumerate(dates):
        close = start + i
        rows.append(
            {
                "ticker": ticker,
                "date": date,
                "adj_close": close,
                "turnover_value": 1_000_000_000,
                "volume": 100_000,
                "adj_open": close - 1,
                "adj_high": close,
                "adj_low": close - 2,
                "adjustment_factor": 1,
                "open": close - 1,
                "high": close,
                "low": close - 2,
                "close": close,
                "adj_volume": 100_000,
                "upper_limit_flag": 0,
                "lower_limit_flag": 0,
                "source_fields_json": "{}",
            }
        )
    return rows


def test_price_reaction_v2_market_close_base_dates_and_stale_gap(tmp_path):
    candidates = pd.DataFrame(
        [
            {
                "DiscDate": "2024-11-01",
                "ticker": "1001",
                "first_time": "15:00:00",
                "candidate_bucket": "A",
                "commercial_signal": "earnings_upward",
            },
            {
                "DiscDate": "2024-11-06",
                "ticker": "1002",
                "first_time": "15:00:00",
                "candidate_bucket": "A",
                "commercial_signal": "earnings_upward",
            },
            {
                "DiscDate": "2024-11-06",
                "ticker": "1003",
                "first_time": "15:30:00",
                "candidate_bucket": "A",
                "commercial_signal": "earnings_downward",
            },
            {
                "DiscDate": "2024-11-06",
                "ticker": "1004",
                "first_time": "08:59:00",
                "candidate_bucket": "A",
                "commercial_signal": "earnings_upward",
            },
            {
                "DiscDate": "2024-01-02",
                "ticker": "1005",
                "first_time": "08:30:00",
                "candidate_bucket": "A",
                "commercial_signal": "earnings_upward",
            },
        ]
    )
    candidates_path = tmp_path / "candidates.csv"
    prices_path = tmp_path / "prices.csv"
    output_path = tmp_path / "out.csv"
    summary_path = tmp_path / "summary.json"
    candidates.to_csv(candidates_path, index=False)
    stale_dates = [f"2024-01-{day:02d}" for day in range(2, 22)] + ["2024-03-20"]
    prices = (
        _prices("1001", ["2024-11-01", "2024-11-05"])
        + _prices("1002", ["2024-11-06", "2024-11-07"])
        + _prices("1003", ["2024-11-06", "2024-11-07"])
        + _prices("1004", ["2024-11-06", "2024-11-07"])
        + _prices("1005", stale_dates)
    )
    pd.DataFrame(prices).to_csv(prices_path, index=False)

    build_price_reaction_v2(candidates_path, prices_path, output_path, summary_path)
    out = pd.read_csv(output_path, dtype={"ticker": str})

    pre_change = out[out["ticker"] == "1001"].iloc[0]
    assert pre_change["market_close_time_used"] == "15:00"
    assert pre_change["event_time_bucket"] == "after_close"
    assert pre_change["reaction_base_date"] == "2024-11-05"

    post_change = out[out["ticker"] == "1002"].iloc[0]
    assert post_change["market_close_time_used"] == "15:30"
    assert post_change["event_time_bucket"] == "afternoon_session"
    assert post_change["reaction_base_date"] == "2024-11-06"

    after_close = out[out["ticker"] == "1003"].iloc[0]
    assert after_close["event_time_bucket"] == "after_close"
    assert after_close["reaction_base_date"] == "2024-11-07"

    before_open = out[out["ticker"] == "1004"].iloc[0]
    assert before_open["event_time_bucket"] == "before_open"
    assert before_open["reaction_base_date"] == "2024-11-06"

    stale = out[out["ticker"] == "1005"].iloc[0]
    assert stale["ret_20d_status"] == "stale_calendar_gap"
    assert pd.isna(stale["ret_20d"])

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["output_rows"] == 5

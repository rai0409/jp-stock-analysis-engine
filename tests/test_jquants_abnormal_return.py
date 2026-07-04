from __future__ import annotations

import pandas as pd
import pytest

from jp_stock_analysis.analysis.jquants_abnormal_return import build_abnormal_return


def test_abnormal_return_is_event_return_minus_universe_return(tmp_path):
    reaction_path = tmp_path / "reaction.csv"
    prices_path = tmp_path / "prices.csv"
    output_path = tmp_path / "abnormal.csv"
    summary_path = tmp_path / "summary.json"
    pd.DataFrame(
        [
            {
                "ticker": "1001",
                "DiscDate": "2024-01-02",
                "commercial_signal": "earnings_upward",
                "reaction_base_date": "2024-01-02",
                "ret_1d": 0.10,
                "ret_1d_date": "2024-01-03",
                "ret_1d_status": "valid",
                "ret_3d_status": "insufficient_future_price",
                "ret_5d_status": "insufficient_future_price",
                "ret_20d_status": "insufficient_future_price",
            }
        ]
    ).to_csv(reaction_path, index=False)
    pd.DataFrame(
        [
            {"ticker": "1001", "date": "2024-01-02", "adj_close": 100},
            {"ticker": "1001", "date": "2024-01-03", "adj_close": 110},
            {"ticker": "1002", "date": "2024-01-02", "adj_close": 200},
            {"ticker": "1002", "date": "2024-01-03", "adj_close": 210},
        ]
    ).to_csv(prices_path, index=False)

    build_abnormal_return(reaction_path, prices_path, output_path, summary_path)
    out = pd.read_csv(output_path)

    universe_ret = (0.10 + 0.05) / 2
    assert out.loc[0, "universe_ret_1d"] == pytest.approx(universe_ret)
    assert out.loc[0, "abnormal_ret_1d"] == pytest.approx(0.10 - universe_ret)
    assert out.loc[0, "abnormal_ret_1d_status"] == "valid"
    assert out.loc[0, "abnormal_reaction_flag_1d"] == "positive_event_abnormal_positive_1d"

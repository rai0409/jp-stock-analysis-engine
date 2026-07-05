from __future__ import annotations

import pandas as pd
import pytest

from jp_stock_analysis.analysis.jquants_abnormal_return import build_abnormal_return
from jp_stock_analysis.analysis import jquants_commercial_quality_enrichment as enrichment


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
    assert out.loc[0, "abnormal_return_adjustment_basis"] in {
        "universe_adjusted_only",
        "adjustment_unavailable",
        "size_adjusted",
    }


def test_abnormal_return_v2_peer_adjustments_and_flags(tmp_path, monkeypatch):
    reaction_path = tmp_path / "reaction.csv"
    prices_path = tmp_path / "prices.csv"
    output_path = tmp_path / "abnormal.csv"
    summary_path = tmp_path / "summary.json"

    tickers = [f"10{i:02d}" for i in range(1, 9)]
    rows = []
    for ticker in tickers:
        rows.append(
            {
                "ticker": ticker,
                "DiscDate": "2024-01-02",
                "commercial_signal": "earnings_upward",
                "reaction_base_date": "2024-01-02",
                "ret_1d": 0.10 if ticker == "1001" else 0.02,
                "ret_1d_date": "2024-01-03",
                "ret_1d_status": "valid",
                "ret_3d": 0.12 if ticker == "1001" else 0.03,
                "ret_3d_date": "2024-01-04",
                "ret_3d_status": "valid",
                "ret_5d": 0.14 if ticker == "1001" else 0.04,
                "ret_5d_date": "2024-01-05",
                "ret_5d_status": "valid",
                "ret_20d_status": "insufficient_future_price",
            }
        )
    pd.DataFrame(rows).to_csv(reaction_path, index=False)

    price_rows = []
    returns = {
        "1001": [0.10, 0.12, 0.14],
        "1002": [0.04, 0.04, 0.04],
        "1003": [0.02, 0.02, 0.02],
        "1004": [0.02, 0.02, 0.02],
        "1005": [0.02, 0.02, 0.02],
        "1006": [0.02, 0.02, 0.02],
        "1007": [0.01, 0.01, 0.01],
        "1008": [0.01, 0.01, 0.01],
    }
    for ticker, rets in returns.items():
        price_rows.append({"ticker": ticker, "date": "2024-01-02", "adj_close": 100, "turnover_value": 1000})
        for date, ret in zip(["2024-01-03", "2024-01-04", "2024-01-05"], rets, strict=True):
            price_rows.append({"ticker": ticker, "date": date, "adj_close": 100 * (1 + ret), "turnover_value": 1000})
    pd.DataFrame(price_rows).to_csv(prices_path, index=False)

    metadata = pd.DataFrame(
        [
            {"ticker": "1001", "sector33_name": "sector_a", "topix_size_category": "large"},
            {"ticker": "1002", "sector33_name": "sector_a", "topix_size_category": "large"},
            {"ticker": "1003", "sector33_name": "sector_a", "topix_size_category": "large"},
            {"ticker": "1004", "sector33_name": "sector_a", "topix_size_category": "large"},
            {"ticker": "1005", "sector33_name": "sector_a", "topix_size_category": "large"},
            {"ticker": "1006", "sector33_name": "sector_a", "topix_size_category": "large"},
            {"ticker": "1007", "sector33_name": "sector_b", "topix_size_category": "small"},
            {"ticker": "1008", "sector33_name": "sector_b", "topix_size_category": "small"},
        ]
    )
    monkeypatch.setattr(
        enrichment,
        "load_metadata",
        lambda *args, **kwargs: (metadata, {"warnings": []}),
    )

    build_abnormal_return(reaction_path, prices_path, output_path, summary_path)
    out = pd.read_csv(output_path, dtype={"ticker": str})
    row = out[out["ticker"] == "1001"].iloc[0]

    # Sector/size/sector-size peers exclude target ticker: mean of 1002..1006.
    expected_peer = (0.04 + 0.02 + 0.02 + 0.02 + 0.02) / 5
    assert row["sector_peer_return_1d"] == pytest.approx(expected_peer)
    assert row["size_peer_return_1d"] == pytest.approx(expected_peer)
    assert row["sector_size_peer_return_1d"] == pytest.approx(expected_peer)
    assert row["sector_adjusted_return_1d"] == pytest.approx(0.10 - expected_peer)
    assert row["sector_size_adjusted_return_1d"] == pytest.approx(0.10 - expected_peer)
    assert int(row["sector_size_peer_count_1d"]) == 5
    assert row["abnormal_return_adjustment_basis"] == "sector_size_adjusted"
    assert row["abnormal_return_adjustment_quality"] == "acceptable_with_limitations"
    assert pd.notna(row["commercial_abnormal_reaction_score"])

    weak = out[out["ticker"] == "1007"].iloc[0]
    assert bool(weak["insufficient_sector_size_peer_flag"])
    assert pd.isna(weak["sector_size_adjusted_return_1d"])
    assert "insufficient_sector_size_peers_1d" in weak["abnormal_return_adjustment_flags"]
    assert "20d_source_unavailable" in weak["abnormal_return_adjustment_flags"]


def test_abnormal_return_v2_missing_metadata_and_zero_mad_are_safe(tmp_path, monkeypatch):
    reaction_path = tmp_path / "reaction.csv"
    prices_path = tmp_path / "prices.csv"
    output_path = tmp_path / "abnormal.csv"
    summary_path = tmp_path / "summary.json"
    pd.DataFrame(
        [
            {
                "ticker": "2001",
                "DiscDate": "2024-01-02",
                "commercial_signal": "earnings_upward",
                "reaction_base_date": "2024-01-02",
                "ret_1d": 0.01,
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
            {"ticker": "2001", "date": "2024-01-02", "adj_close": 100},
            {"ticker": "2001", "date": "2024-01-03", "adj_close": 101},
        ]
    ).to_csv(prices_path, index=False)
    monkeypatch.setattr(
        enrichment,
        "load_metadata",
        lambda *args, **kwargs: (pd.DataFrame(columns=["ticker"]), {"warnings": []}),
    )

    build_abnormal_return(reaction_path, prices_path, output_path, summary_path)
    out = pd.read_csv(output_path, dtype={"ticker": str})
    assert out.loc[0, "abnormal_return_adjustment_basis"] in {
        "universe_adjusted_only",
        "adjustment_unavailable",
    }
    assert bool(out.loc[0, "insufficient_sector_peer_flag"])
    assert pd.isna(out.loc[0, "sector_adjusted_return_1d_robust_z"])
    assert pd.notna(out.loc[0, "commercial_abnormal_reaction_score"])

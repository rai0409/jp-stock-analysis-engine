from __future__ import annotations

import pandas as pd

from jp_stock_analysis.analysis.jquants_robust_summary import build_robust_summary


def _rows(n: int, bucket: str) -> list[dict[str, object]]:
    return [
        {
            "candidate_bucket": bucket,
            "commercial_signal": "earnings_upward",
            "reaction_flag_v2": "positive_event_reacted_positive_1d",
            "abnormal_reaction_flag_1d": "positive_event_abnormal_positive_1d",
            "liquidity_bucket": "mid",
            "ret_1d": 0.01 if i % 2 == 0 else -0.01,
            "ret_3d": 0.02,
            "ret_5d": 0.03,
            "ret_20d": 0.04,
            "abnormal_ret_1d": 0.005,
            "abnormal_ret_3d": 0.006,
            "abnormal_ret_5d": 0.007,
            "abnormal_ret_20d": 0.008,
        }
        for i in range(n)
    ]


def test_sample_size_bucket_by_group_count(tmp_path):
    liquidity_path = tmp_path / "liquidity.csv"
    output_path = tmp_path / "robust.csv"
    summary_path = tmp_path / "summary.json"
    pd.DataFrame(
        _rows(4, "VERY_LOW")
        + _rows(5, "LOW")
        + _rows(20, "MEDIUM")
        + _rows(50, "USABLE")
    ).to_csv(liquidity_path, index=False)

    build_robust_summary(liquidity_path, output_path, summary_path)
    out = pd.read_csv(output_path)
    buckets = dict(zip(out["candidate_bucket"], out["sample_size_bucket"], strict=True))

    assert buckets["VERY_LOW"] == "very_low_sample"
    assert buckets["LOW"] == "low_sample"
    assert buckets["MEDIUM"] == "medium_sample"
    assert buckets["USABLE"] == "usable_sample"
    assert "ret_1d_trimmed_mean_10pct" in out.columns
    assert "abnormal_ret_20d_positive_rate" in out.columns

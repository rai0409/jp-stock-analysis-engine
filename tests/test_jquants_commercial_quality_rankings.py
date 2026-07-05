from __future__ import annotations

import json

import pandas as pd

from jp_stock_analysis.analysis.jquants_commercial_quality_rankings import (
    build_commercial_quality_rankings,
)


def _base_row(ticker: str, signal: str, abnormal: float) -> dict[str, object]:
    return {
        "DiscDate": "2024-12-01",
        "ticker": ticker,
        "first_time": "15:30:00",
        "event_time_bucket": "after_close",
        "reaction_base_date": "2024-12-02",
        "event_combo": "EARNINGS_REVISION_ONLY",
        "candidate_bucket": "A",
        "commercial_signal": signal,
        "commercial_candidate_score": 100,
        "earnings_revision_direction": "upward" if "upward" in signal else "downward",
        "dividend_revision_direction": "",
        "ret_1d": 0.01,
        "universe_ret_1d": 0.0,
        "abnormal_ret_1d": abnormal,
        "ret_3d": 0.01,
        "universe_ret_3d": 0.0,
        "abnormal_ret_3d": abnormal,
        "ret_5d": 0.01,
        "universe_ret_5d": 0.0,
        "abnormal_ret_5d": abnormal,
        "ret_20d": 0.01,
        "universe_ret_20d": 0.0,
        "abnormal_ret_20d": abnormal,
        "abnormal_return_adjustment_basis": "sector_size_adjusted",
        "abnormal_return_adjustment_quality": "acceptable_with_limitations",
        "abnormal_return_adjustment_flags": "none",
        "commercial_abnormal_reaction_score": abnormal * 10,
        "commercial_abnormal_reaction_score_components": "{\"fixture\":true}",
        "sector_adjusted_return_1d": abnormal + 0.001,
        "size_adjusted_return_1d": abnormal + 0.002,
        "sector_size_adjusted_return_1d": abnormal + 0.003,
        "sector_peer_count_1d": 5,
        "size_peer_count_1d": 5,
        "sector_size_peer_count_1d": 5,
        "insufficient_sector_peer_flag": False,
        "insufficient_size_peer_flag": False,
        "insufficient_sector_size_peer_flag": False,
        "ret_1d_status": "valid",
        "ret_5d_status": "valid",
        "ret_20d_status": "valid",
        "abnormal_ret_1d_status": "valid",
        "abnormal_ret_5d_status": "valid",
        "abnormal_ret_20d_status": "valid",
        "reaction_flag_v2": "flag",
        "abnormal_reaction_flag_1d": "flag",
        "abnormal_reaction_flag_5d": "flag",
        "abnormal_reaction_flag_20d": "flag",
        "avg_turnover_20d": 1_000_000_000,
        "median_turnover_20d": 1_000_000_000,
        "base_turnover_value": 2_000_000_000,
        "turnover_spike_ratio": 2.0,
        "avg_volume_20d": 100_000,
        "median_volume_20d": 100_000,
        "base_volume": 200_000,
        "volume_spike_ratio": 2.0,
        "liquidity_bucket": "high",
        "liquidity_status": "valid",
        "is_liquid_enough": True,
        "price_reaction_status_v2": "full_20d_available",
        "doctypes": "EarnForecastRevision",
        "discnos": "D1",
    }


def _run(tmp_path, rows):
    input_path = tmp_path / "liquidity.csv"
    out_paths = {
        "positive_reacted": tmp_path / "positive_reacted.csv",
        "positive_unreacted": tmp_path / "positive_unreacted.csv",
        "negative_reacted": tmp_path / "negative_reacted.csv",
        "negative_reversal": tmp_path / "negative_reversal.csv",
        "audit": tmp_path / "audit.csv",
        "unclassified": tmp_path / "unclassified.csv",
        "summary": tmp_path / "summary.json",
    }
    pd.DataFrame(rows).to_csv(input_path, index=False)
    build_commercial_quality_rankings(
        input_path,
        out_paths["positive_reacted"],
        out_paths["positive_unreacted"],
        out_paths["negative_reacted"],
        out_paths["negative_reversal"],
        out_paths["audit"],
        out_paths["unclassified"],
        out_paths["summary"],
    )
    return out_paths


def test_commercial_quality_rankings_classify_rank_and_audit_exclusions(tmp_path):
    rows = [
        _base_row("1001", "earnings_upward", 0.03),
        _base_row("1002", "earnings_upward", 0.06),
        _base_row("1003", "earnings_upward", -0.04),
        _base_row("1004", "earnings_downward", -0.05),
        _base_row("1005", "earnings_downward", 0.02),
        _base_row("1006", "earnings_upward", 0.01),
        _base_row("1007", "earnings_upward", 0.01),
        _base_row("1008", "earnings_upward", 0.01),
        _base_row("1009", "earnings_upward", 0.01),
        _base_row("1010", "earnings_upward", 0.01),
    ]
    rows[5]["liquidity_bucket"] = "mid"
    rows[5]["ret_20d_status"] = "insufficient_future_price"
    rows[5]["abnormal_ret_20d_status"] = "invalid_source_return"
    rows[5]["ret_20d"] = None
    rows[5]["abnormal_ret_20d"] = None
    rows[6]["liquidity_bucket"] = "low"
    rows[6]["is_liquid_enough"] = False
    rows[7]["liquidity_bucket"] = "unknown"
    rows[7]["is_liquid_enough"] = False
    rows[8]["ret_1d_status"] = "insufficient_future_price"
    rows[9]["abnormal_ret_1d_status"] = "invalid_source_return"

    paths = _run(tmp_path, rows)

    positive_reacted = pd.read_csv(paths["positive_reacted"], dtype={"ticker": str})
    assert list(positive_reacted["ticker"]) == ["1002", "1001", "1006"]
    assert list(positive_reacted["rank"]) == [1, 2, 3]
    assert set(positive_reacted["liquidity_bucket"]) == {"high", "mid"}
    for col in [
        "classification_confidence",
        "classification_reason",
        "classification_reason_components",
        "manual_review_required",
        "manual_review_reason",
        "ranking_score",
        "ranking_score_components",
        "primary_signal",
        "secondary_signal",
        "evidence_summary",
        "horizon_flags",
        "liquidity_flags",
        "duplicate_event_flag",
        "data_quality_flags",
        "product_risk_flags",
        "product_acceptability",
    ]:
        assert col in positive_reacted.columns
    assert positive_reacted["ranking_score"].notna().all()
    assert positive_reacted["ranking_score_components"].astype(str).str.len().gt(0).all()
    assert positive_reacted["ranking_score_components"].astype(str).str.contains("sector_size_adjusted_return_1d").all()
    assert positive_reacted["evidence_summary"].astype(str).str.len().gt(0).all()
    assert positive_reacted["evidence_summary"].astype(str).str.contains("adjustment_basis=").all()
    assert "ret_20d_unavailable" in positive_reacted.loc[positive_reacted["ticker"] == "1006", "horizon_flags"].iloc[0]
    assert "long_horizon_unavailable" in positive_reacted.loc[positive_reacted["ticker"] == "1006", "evidence_summary"].iloc[0]
    forbidden = r"buy|sell|hold|recommend|target price"
    assert not positive_reacted["evidence_summary"].astype(str).str.lower().str.contains(forbidden, regex=True).any()

    assert pd.read_csv(paths["positive_unreacted"], dtype={"ticker": str}).loc[
        0, "ticker"
    ] == "1003"
    assert pd.read_csv(paths["negative_reacted"], dtype={"ticker": str}).loc[0, "ticker"] == "1004"
    assert pd.read_csv(paths["negative_reversal"], dtype={"ticker": str}).loc[0, "ticker"] == "1005"

    audit = pd.read_csv(paths["audit"], dtype={"ticker": str})
    reasons = dict(zip(audit["ticker"], audit["exclusion_reasons"], strict=True))
    assert "low_liquidity" in reasons["1007"]
    assert "unknown_liquidity" in reasons["1008"]
    assert "invalid_ret_1d_status" in reasons["1009"]
    assert "invalid_abnormal_ret_1d_status" in reasons["1010"]
    assert "liquidity_flags" in audit.columns
    assert "low_liquidity" in audit.loc[audit["ticker"] == "1007", "liquidity_flags"].iloc[0]
    manual = audit[audit["manual_review_required"].astype(str).str.lower().isin(["true", "1"])]
    assert manual["manual_review_reason"].astype(str).str.len().gt(0).all()

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["eligible_rows"] == 6
    assert summary["excluded_rows"] == 4
    assert summary["ranking_counts"] == {
        "positive_reacted": 3,
        "positive_unreacted": 1,
        "negative_reacted": 1,
        "negative_reversal": 1,
    }
    assert summary["exclusion_reason_counts"]["low_liquidity"] == 1
    assert summary["exclusion_reason_counts"]["unknown_liquidity"] == 1
    assert summary["unclassified_rows"] == 0
    assert pd.read_csv(paths["unclassified"]).empty


def test_commercial_quality_rankings_tolerates_missing_optional_columns(tmp_path):
    row = {
        "DiscDate": "2024-12-01",
        "ticker": "2001",
        "commercial_signal": "earnings_upward",
        "ret_1d_status": "valid",
        "abnormal_ret_1d_status": "valid",
        "price_reaction_status_v2": "partial_1d_available",
        "abnormal_ret_1d": 0.01,
        "liquidity_bucket": "mid",
        "liquidity_status": "valid",
        "is_liquid_enough": True,
    }
    paths = _run(tmp_path, [row])

    positive_reacted = pd.read_csv(paths["positive_reacted"], dtype={"ticker": str})
    assert positive_reacted.loc[0, "ticker"] == "2001"
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert "commercial_candidate_score" in summary["missing_output_columns"]
    assert summary["ranking_counts"]["positive_reacted"] == 1
    assert summary["unclassified_rows"] == 0


def test_commercial_quality_rankings_writes_unclassified_audit(tmp_path):
    rows = [
        _base_row("3001", "mixed_revision", 0.02),
        _base_row("3002", "earnings_upward", 0.0),
        _base_row("3003", "earnings_upward", 0.03),
        _base_row("3004", "mixed_revision", 0.01),
    ]
    rows[3]["liquidity_bucket"] = "low"
    rows[3]["is_liquid_enough"] = False

    paths = _run(tmp_path, rows)

    unclassified = pd.read_csv(paths["unclassified"], dtype={"ticker": str})
    assert set(unclassified["ticker"]) == {"3001", "3002"}
    reasons = dict(zip(unclassified["ticker"], unclassified["unclassified_reason"], strict=True))
    assert reasons["3001"] == "signal_not_positive_or_negative"
    assert reasons["3002"] == "abnormal_ret_1d_zero"
    mixed_row = unclassified[unclassified["ticker"] == "3001"].iloc[0]
    assert str(mixed_row["mixed_event_flag"]).lower() in {"true", "1"}
    assert "mixed_event_context" in mixed_row["mixed_event_reason"]
    assert str(mixed_row["manual_review_required"]).lower() in {"true", "1"}
    assert len(str(mixed_row["manual_review_reason"])) > 0

    audit = pd.read_csv(paths["audit"], dtype={"ticker": str})
    assert set(audit["ticker"]) == {"3004"}
    assert "3004" not in set(unclassified["ticker"])

    positive_reacted = pd.read_csv(paths["positive_reacted"], dtype={"ticker": str})
    assert list(positive_reacted["ticker"]) == ["3003"]

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["eligible_rows"] == 3
    assert summary["excluded_rows"] == 1
    assert summary["unclassified_rows"] == 2
    assert summary["unclassified_reason_counts"] == {
        "abnormal_ret_1d_zero": 1,
        "signal_not_positive_or_negative": 1,
    }
    assert "unclassified_eligible_rows_exist" in summary["warnings"]
    assert summary["ranking_counts"]["positive_reacted"] == 1


def test_commercial_quality_rankings_flags_duplicate_mixed_events(tmp_path):
    rows = [
        _base_row("4001", "earnings_upward_and_dividend_decrease", 0.02),
        _base_row("4001", "earnings_upward_and_dividend_decrease", 0.01),
    ]
    rows[0]["dividend_revision_direction"] = "decrease"
    rows[1]["dividend_revision_direction"] = "decrease"

    paths = _run(tmp_path, rows)

    ranked = pd.read_csv(paths["positive_reacted"], dtype={"ticker": str})
    assert len(ranked) == 2
    assert ranked["duplicate_event_flag"].astype(str).str.lower().isin(["true", "1"]).all()
    assert ranked["mixed_event_flag"].astype(str).str.lower().isin(["true", "1"]).all()
    assert ranked["manual_review_required"].astype(str).str.lower().isin(["true", "1"]).all()
    assert ranked["manual_review_reason"].astype(str).str.contains("duplicate_event_context_unresolved").all()

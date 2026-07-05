from __future__ import annotations

import json

import pandas as pd

from jp_stock_analysis.analysis.jquants_commercial_quality_rankings import (
    build_commercial_quality_rankings,
    load_financial_disclosure_events,
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


def _run(tmp_path, rows, events_path=None):
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
        events_path or tmp_path / "missing_financial_disclosure_events.csv",
    )
    return out_paths


def _write_events(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


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
    rows[1]["sector_size_adjusted_return_1d"] = 0.0
    rows[1]["sector_adjusted_return_1d"] = 0.0
    rows[1]["size_adjusted_return_1d"] = 0.0
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


def test_classification_quality_v2_uses_adjusted_evidence_and_fields(tmp_path):
    rows = [_base_row("5001", "earnings_upward", 0.0)]
    rows[0]["sector_size_adjusted_return_1d"] = 0.025
    rows[0]["sector_size_adjusted_return_3d"] = 0.03
    rows[0]["sector_size_adjusted_return_5d"] = 0.035
    rows[0]["sector_size_adjusted_return_20d"] = 0.04
    paths = _run(tmp_path, rows)

    ranked = pd.read_csv(paths["positive_reacted"], dtype={"ticker": str})
    row = ranked.iloc[0]
    assert row["event_classification_detail"] == "positive_confirmed"
    assert row["event_direction_final"] == "positive"
    assert row["event_reaction_pattern"] == "immediate_confirmed"
    assert row["event_confidence_tier"] in {"high", "medium"}
    assert row["classification_basis_level"] == "sector_size_adjusted"
    assert int(row["classification_evidence_count"]) == 4
    assert str(row["auto_classification_safe_flag"]).lower() in {"true", "1"}
    components = json.loads(row["ranking_score_components"])
    assert components["adjusted_return_basis"] == "sector_size_adjusted"
    assert "event_confidence_tier" in components


def test_universe_only_evidence_cannot_be_high_confidence(tmp_path):
    rows = [_base_row("5002", "earnings_upward", 0.02)]
    rows[0]["abnormal_return_adjustment_basis"] = "universe_adjusted_only"
    rows[0]["sector_size_adjusted_return_1d"] = None
    rows[0]["sector_adjusted_return_1d"] = None
    rows[0]["size_adjusted_return_1d"] = None
    paths = _run(tmp_path, rows)

    row = pd.read_csv(paths["positive_reacted"], dtype={"ticker": str}).iloc[0]
    assert row["classification_basis_level"] == "universe_adjusted_only"
    assert row["event_confidence_tier"] != "high"
    assert "fallback_universe_only" in row["manual_review_category"]
    assert str(row["auto_classification_safe_flag"]).lower() in {"false", "0"}


def test_mixed_reaction_horizons_create_contradiction_flags(tmp_path):
    rows = [_base_row("5003", "earnings_upward", 0.02)]
    rows[0]["sector_size_adjusted_return_1d"] = 0.02
    rows[0]["sector_size_adjusted_return_5d"] = -0.03
    rows[0]["sector_size_adjusted_return_20d"] = -0.04
    paths = _run(tmp_path, rows)

    row = pd.read_csv(paths["positive_reacted"], dtype={"ticker": str}).iloc[0]
    assert row["event_reaction_pattern"] == "overreaction_or_reversal"
    assert "1d_5d_reaction_conflict" in row["contradiction_flags"]
    assert row["mixed_event_severity"] in {"medium", "high"}


def test_fundamental_guidance_and_dividend_evidence_fields(tmp_path):
    rows = [
        _base_row("6001", "earnings_upward", 0.03),
        _base_row("6002", "earnings_downward", -0.03),
        _base_row("6003", "earnings_upward_and_dividend_decrease", 0.02),
    ]
    rows[0]["change_op_pct"] = 12.0
    rows[1]["change_np_pct"] = -8.0
    rows[2]["dividend_revision_direction"] = "decrease"
    rows[2]["change_fdivann_pct"] = -15.0

    paths = _run(tmp_path, rows)
    pos = pd.read_csv(paths["positive_reacted"], dtype={"ticker": str})
    neg = pd.read_csv(paths["negative_reacted"], dtype={"ticker": str})

    up = pos[pos["ticker"] == "6001"].iloc[0]
    assert up["fundamental_event_type"] == "guidance_revision"
    assert up["fundamental_event_direction"] == "positive"
    assert up["guidance_revision_direction"] == "positive"
    assert up["fundamental_event_confidence"] in {"high", "medium"}
    assert str(up["auto_classification_safe_flag"]).lower() in {"true", "1"}
    assert "fundamental_event_direction" in json.loads(up["ranking_score_components"])
    assert "fundamental_event=" in up["evidence_summary"]

    down = neg[neg["ticker"] == "6002"].iloc[0]
    assert down["fundamental_event_direction"] == "negative"
    assert down["guidance_revision_direction"] == "negative"

    mixed = pos[pos["ticker"] == "6003"].iloc[0]
    assert str(mixed["fundamental_mixed_signal_flag"]).lower() in {"true", "1"}
    assert "dividend_revision" in mixed["fundamental_mixed_signal_reason"]
    assert mixed["dividend_revision_direction"] == "negative"


def test_insufficient_fundamental_evidence_is_explicit(tmp_path):
    row = _base_row("6004", "neutral_event", 0.0)
    row["earnings_revision_direction"] = ""
    row["dividend_revision_direction"] = ""
    row["change_sales_pct"] = None
    row["change_op_pct"] = None
    row["change_np_pct"] = None
    row["change_eps_pct"] = None
    paths = _run(tmp_path, [row])

    unclassified = pd.read_csv(paths["unclassified"], dtype={"ticker": str})
    evidence = unclassified.iloc[0]
    assert evidence["fundamental_event_direction"] == "insufficient_evidence"
    assert evidence["fundamental_event_confidence"] == "not_evaluable"
    assert "insufficient_fundamental_evidence" in evidence["fundamental_event_flags"]


def test_fundamental_price_conflict_requires_review(tmp_path):
    rows = [_base_row("6005", "earnings_upward", -0.03)]
    rows[0]["change_op_pct"] = 15.0
    rows[0]["sector_size_adjusted_return_1d"] = -0.03
    rows[0]["sector_size_adjusted_return_3d"] = -0.02
    rows[0]["sector_size_adjusted_return_5d"] = -0.02
    paths = _run(tmp_path, rows)

    row = pd.read_csv(paths["positive_unreacted"], dtype={"ticker": str}).iloc[0]
    assert "fundamental_price_reaction_conflict" in row["contradiction_flags"]
    assert str(row["manual_review_required"]).lower() in {"true", "1"}
    assert "fundamental_price_reaction_conflict" in row["manual_review_reason"]


def test_financial_disclosure_loader_missing_file_is_safe(tmp_path):
    assert load_financial_disclosure_events(tmp_path / "missing.csv").empty


def test_financial_disclosure_join_by_discno_and_direction(tmp_path):
    row = _base_row("7001", "earnings_upward", 0.03)
    row["discnos"] = "DISC1"
    events = _write_events(
        tmp_path / "events.csv",
        [
            {
                "ticker": "7001",
                "DiscDate": "2024-12-01",
                "DiscNo": "DISC1",
                "DocType": "EarnForecastRevision",
                "is_financial_statement": False,
                "is_earnings_forecast_revision": True,
                "is_dividend_forecast_revision": False,
                "change_op_pct": 12.0,
            }
        ],
    )
    paths = _run(tmp_path, [row], events)
    out = pd.read_csv(paths["positive_reacted"], dtype={"ticker": str}).iloc[0]
    assert out["financial_disclosure_join_status"] == "matched_discno"
    assert int(out["financial_disclosure_match_count"]) == 1
    assert out["earnings_forecast_revision_direction_v2"] == "positive"
    assert out["earnings_forecast_revision_strength_v2"] == "strong"
    assert "financial_disclosure" in out["fundamental_event_components"]
    assert str(out["auto_classification_safe_flag"]).lower() in {"true", "1"}


def test_financial_disclosure_join_by_doctype_and_downward_revision(tmp_path):
    row = _base_row("7002", "earnings_downward", -0.03)
    row["discnos"] = "NO_MATCH"
    row["doctypes"] = "EarnForecastRevision"
    events = _write_events(
        tmp_path / "events.csv",
        [
            {
                "ticker": "7002",
                "DiscDate": "2024-12-01",
                "DiscNo": "DISC2",
                "DocType": "EarnForecastRevision",
                "is_financial_statement": False,
                "is_earnings_forecast_revision": True,
                "is_dividend_forecast_revision": False,
                "change_np_pct": -7.0,
            }
        ],
    )
    paths = _run(tmp_path, [row], events)
    out = pd.read_csv(paths["negative_reacted"], dtype={"ticker": str}).iloc[0]
    assert out["financial_disclosure_join_status"] == "matched_doctype"
    assert out["earnings_forecast_revision_direction_v2"] == "negative"
    assert out["financial_disclosure_confidence"] == "medium"


def test_financial_disclosure_multiple_date_matches_aggregate(tmp_path):
    row = _base_row("7003", "earnings_upward", 0.02)
    row["discnos"] = "NO_MATCH"
    row["doctypes"] = "NO_MATCH"
    events = _write_events(
        tmp_path / "events.csv",
        [
            {
                "ticker": "7003",
                "DiscDate": "2024-12-01",
                "DiscNo": "A",
                "DocType": "EarnForecastRevision",
                "is_financial_statement": False,
                "is_earnings_forecast_revision": True,
                "is_dividend_forecast_revision": False,
                "change_op_pct": 4.0,
            },
            {
                "ticker": "7003",
                "DiscDate": "2024-12-01",
                "DiscNo": "B",
                "DocType": "DividendForecastRevision",
                "is_financial_statement": False,
                "is_earnings_forecast_revision": False,
                "is_dividend_forecast_revision": True,
                "change_fdivann_pct": 10.0,
            },
        ],
    )
    paths = _run(tmp_path, [row], events)
    out = pd.read_csv(paths["positive_reacted"], dtype={"ticker": str}).iloc[0]
    assert out["financial_disclosure_join_status"] == "matched_date"
    assert int(out["financial_disclosure_match_count"]) == 2
    assert "multiple_financial_disclosure_matches" in out["financial_disclosure_limitations"]
    assert out["dividend_revision_direction_v2"] == "positive"


def test_financial_disclosure_mixed_earnings_and_dividend_and_statement_context(tmp_path):
    rows = [
        _base_row("7004", "earnings_upward", 0.02),
        _base_row("7005", "neutral_event", 0.0),
    ]
    rows[0]["discnos"] = "MIX"
    rows[1]["discnos"] = "FS"
    rows[1]["earnings_revision_direction"] = ""
    events = _write_events(
        tmp_path / "events.csv",
        [
            {
                "ticker": "7004",
                "DiscDate": "2024-12-01",
                "DiscNo": "MIX",
                "DocType": "EarnForecastRevision",
                "is_financial_statement": False,
                "is_earnings_forecast_revision": True,
                "is_dividend_forecast_revision": False,
                "change_sales_pct": 6.0,
                "change_op_pct": -5.0,
            },
            {
                "ticker": "7005",
                "DiscDate": "2024-12-01",
                "DiscNo": "FS",
                "DocType": "FYFinancialStatements_Consolidated_JP",
                "is_financial_statement": True,
                "is_earnings_forecast_revision": False,
                "is_dividend_forecast_revision": False,
                "Sales_num": 1000,
                "OP_num": 100,
            },
        ],
    )
    paths = _run(tmp_path, rows, events)
    mixed = pd.read_csv(paths["positive_reacted"], dtype={"ticker": str}).iloc[0]
    assert mixed["earnings_forecast_revision_direction_v2"] == "mixed"
    assert str(mixed["fundamental_mixed_signal_flag"]).lower() in {"true", "1"}

    unclassified = pd.read_csv(paths["unclassified"], dtype={"ticker": str})
    statement = unclassified[unclassified["ticker"] == "7005"].iloc[0]
    assert statement["financial_statement_direction"] == "context_only"
    assert statement["fundamental_event_direction"] == "insufficient_evidence"
    assert statement["event_confidence_tier"] != "high"

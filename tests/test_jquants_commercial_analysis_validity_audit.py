from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from jp_stock_analysis.analysis.jquants_commercial_analysis_validity_audit import (
    OUTPUT_FILES,
    abnormal_return_v2_summary,
    build_commercial_analysis_validity_audit,
    build_critical_issues,
    build_signal_performance,
    determine_product_status,
    main,
    validate_liquidity,
    validate_ranking,
)


def _write_minimal_inputs(base):
    analysis = base / "analysis"
    analysis.mkdir()
    financial = base / "financial_summary_raw.csv"
    prices = base / "prices_daily_bars.csv"
    pd.DataFrame(
        [
            {"ticker": "1234", "DiscDate": "2025-01-10", "DocType": "EarnForecastRevision"},
            {"ticker": "1234", "DiscDate": "2025-01-10", "DocType": "EarnForecastRevision"},
            {"ticker": "5678", "DiscDate": "2025-01-11", "DocType": "DividendForecastRevision"},
        ]
    ).to_csv(financial, index=False)
    pd.DataFrame(
        [
            {"ticker": "1234", "date": "2025-01-10", "adj_close": 100, "turnover_value": 1_000_000_000, "volume": 1_000_000},
            {"ticker": "1234", "date": "2025-01-14", "adj_close": 101, "turnover_value": 1_000_000_000, "volume": 1_000_000},
        ]
    ).to_csv(prices, index=False)
    ranking_row = {
        "rank": 1,
        "ranking_label": "positive_reacted",
        "ranking_reason": "positive_signal_and_positive_abnormal_1d",
        "DiscDate": "2025-01-10",
        "ticker": "1234",
        "company_name": "Test Co",
        "commercial_signal": "earnings_upward",
        "candidate_bucket": "POSITIVE_EARNINGS",
        "commercial_candidate_score": 90,
        "abnormal_ret_1d": 0.02,
        "abnormal_ret_3d": 0.03,
        "ret_1d": 0.03,
        "ret_3d": 0.04,
        "ret_5d": 0.05,
        "universe_ret_1d": 0.01,
        "ret_1d_status": "valid",
        "ret_3d_status": "valid",
        "ret_5d_status": "valid",
        "ret_20d_status": "insufficient_future_price",
        "abnormal_ret_1d_status": "valid",
        "reaction_base_date": "2025-01-14",
        "liquidity_bucket": "mid",
        "liquidity_status": "valid",
        "is_liquid_enough": True,
        "avg_turnover_20d": 200_000_000,
        "turnover_spike_ratio": 2.0,
        "doctypes": "EarnForecastRevision",
        "event_classification_detail": "positive_confirmed",
        "event_direction_final": "positive",
        "event_reaction_pattern": "immediate_confirmed",
        "event_confidence_tier": "high",
        "classification_basis_level": "sector_size_adjusted",
        "classification_evidence_count": 3,
        "classification_limitation_count": 1,
        "mixed_event_severity": "none",
        "manual_review_category": "none",
        "manual_review_priority": "low",
        "auto_classification_safe_flag": True,
    }
    pd.DataFrame([ranking_row]).to_csv(
        analysis / "commercial_quality_positive_reacted_ranking_enriched.csv", index=False
    )
    for name in [
        "commercial_quality_positive_unreacted_ranking_enriched.csv",
        "commercial_quality_negative_reacted_ranking_enriched.csv",
        "commercial_quality_negative_reversal_ranking_enriched.csv",
        "commercial_quality_rankings_excluded_audit_enriched.csv",
        "commercial_quality_rankings_unclassified_audit_enriched.csv",
    ]:
        pd.DataFrame().to_csv(analysis / name, index=False)
    return financial, prices, analysis


def test_cli_help_works(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "financial-summary" in capsys.readouterr().out


def test_empty_input_generates_report(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    financial = tmp_path / "financial.csv"
    prices = tmp_path / "prices.csv"
    pd.DataFrame().to_csv(financial, index=False)
    pd.DataFrame().to_csv(prices, index=False)
    report = build_commercial_analysis_validity_audit(financial, prices, analysis, analysis)
    assert report["commercial_product_analysis_status"] in {
        "product_ready",
        "product_needs_corrections",
        "poc_only",
        "invalid",
    }
    assert (analysis / OUTPUT_FILES["report_json"]).exists()
    assert (analysis / OUTPUT_FILES["report_md"]).exists()


def test_critical_issue_detection():
    ranked = pd.DataFrame(
        [
            {
                "ranking_file": "positive_reacted",
                "classification": "positive_reacted",
                "ticker": "1234",
                "DiscDate": "2025-01-10",
            }
        ]
    )
    ranking_validation = pd.DataFrame(
        [
            {
                "duplicate_event_flag": True,
                "score_explainable": False,
                "insufficient_horizon_flag": False,
                "low_liquidity_flag": False,
                "rank": 1,
            }
        ]
    )
    classification = pd.DataFrame(
        [{"classification_issue": "", "ranking_file": "positive_reacted"}]
    )
    abnormal = pd.DataFrame(
        [{"sector_adjustment_detected": False, "size_adjustment_detected": False}]
    )
    issues = build_critical_issues(
        ranked,
        ranked,
        classification,
        ranking_validation,
        abnormal,
        pd.DataFrame(),
    )
    assert "duplicate_event_ranked" in set(issues["issue_type"])


def test_ranking_excluded_case_can_be_critical():
    ranked = pd.DataFrame(
        [{"ranking_file": "excluded", "classification": "excluded", "ticker": "1234", "DiscDate": "2025-01-10"}]
    )
    issues = build_critical_issues(
        ranked,
        ranked,
        pd.DataFrame(columns=["classification_issue", "ranking_file"]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert "excluded_or_unclassified_in_ranking" in set(issues["issue_type"])
    assert "critical" in set(issues["severity"])


def test_insufficient_20d_horizon_detected():
    rows = pd.DataFrame(
        [
            {
                "ranking_file": "positive_reacted",
                "ticker": "1234",
                "DiscDate": "2025-01-10",
                "rank": 1,
                "ranking_reason": "reason",
                "commercial_candidate_score": 1,
                "abnormal_ret_1d": 0.1,
                "turnover_spike_ratio": 1,
                "avg_turnover_20d": 100,
                "ret_20d_status": "insufficient_future_price",
                "liquidity_bucket": "mid",
                "is_liquid_enough": True,
            }
        ]
    )
    out = validate_ranking(rows, set())
    assert bool(out.loc[0, "insufficient_horizon_flag"]) is True


def test_duplicate_event_detected():
    rows = pd.DataFrame(
        [
            {
                "ranking_file": "positive_reacted",
                "ticker": "1234",
                "DiscDate": "2025-01-10",
                "rank": 1,
                "ranking_reason": "reason",
                "commercial_candidate_score": 1,
                "abnormal_ret_1d": 0.1,
                "ret_20d_status": "valid",
                "liquidity_bucket": "mid",
                "is_liquid_enough": True,
            }
        ]
    )
    out = validate_ranking(rows, {("1234", "2025-01-10")})
    assert bool(out.loc[0, "duplicate_event_flag"]) is True


def test_low_liquidity_flag_detected():
    rows = pd.DataFrame(
        [
            {
                "classification": "positive_reacted",
                "ticker": "1234",
                "DiscDate": "2025-01-10",
                "rank": 1,
                "liquidity_bucket": "low",
                "is_liquid_enough": False,
                "avg_turnover_20d": 10,
                "avg_volume_20d": 10,
            }
        ]
    )
    out = validate_liquidity(rows)
    assert bool(out.loc[0, "low_liquidity_in_top_rank_flag"]) is True


def test_abnormal_return_v2_summary_detects_adjusted_coverage():
    rows = pd.DataFrame(
        [
            {
                "sector_adjusted_return_1d": 0.01,
                "size_adjusted_return_1d": 0.02,
                "sector_size_adjusted_return_1d": 0.03,
                "commercial_abnormal_reaction_score": 1.2,
                "insufficient_sector_peer_flag": False,
                "insufficient_size_peer_flag": True,
                "insufficient_sector_size_peer_flag": False,
                "abnormal_return_adjustment_quality": "acceptable_with_limitations",
                "abnormal_return_adjustment_basis": "sector_size_adjusted",
            },
            {
                "sector_adjusted_return_1d": None,
                "size_adjusted_return_1d": None,
                "sector_size_adjusted_return_1d": None,
                "commercial_abnormal_reaction_score": 0.0,
                "insufficient_sector_peer_flag": True,
                "insufficient_size_peer_flag": True,
                "insufficient_sector_size_peer_flag": True,
                "abnormal_return_adjustment_quality": "insufficient_peers",
                "abnormal_return_adjustment_basis": "universe_adjusted_only",
            },
        ]
    )
    summary = abnormal_return_v2_summary(rows)
    assert summary["sector_adjusted_coverage_ratio"] == 0.5
    assert summary["size_adjusted_coverage_ratio"] == 0.5
    assert summary["sector_size_adjusted_coverage_ratio"] == 0.5
    assert summary["robust_adjusted_score_coverage_ratio"] == 1.0
    assert summary["insufficient_sector_peer_rows"] == 1
    assert summary["insufficient_size_peer_rows"] == 2
    assert summary["universe_adjusted_only_rows"] == 1


def test_classification_quality_v2_report_metrics(tmp_path):
    financial, prices, analysis = _write_minimal_inputs(tmp_path)
    report = build_commercial_analysis_validity_audit(financial, prices, analysis, analysis)
    assert report["classification_confidence_tier_counts"]["high"] == 1
    assert report["event_direction_final_counts"]["positive"] == 1
    assert report["event_reaction_pattern_counts"]["immediate_confirmed"] == 1
    assert report["manual_review_priority_counts"]["low"] == 1
    assert report["auto_classification_safe_ratio"] == 1.0
    assert report["adjusted_evidence_supported_classification_ratio"] == 1.0
    assert report["high_priority_manual_review_rows"] == 0
    assert report["universe_only_high_confidence_rows"] == 0
    assert report["classification_quality_score"] > 0


def test_universe_only_high_confidence_is_critical_issue():
    classification = pd.DataFrame(
        [
            {
                "classification_issue": "",
                "ranking_file": "positive_reacted",
                "classification_basis_level": "universe_adjusted_only",
                "event_confidence_tier": "high",
            }
        ]
    )
    issues = build_critical_issues(
        pd.DataFrame(),
        pd.DataFrame(),
        classification,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert "universe_only_high_confidence_classification" in set(issues["issue_type"])


def test_signal_bucket_summary_generated():
    rows = pd.DataFrame(
        [
            {"classification": "positive_reacted", "rank": i, "commercial_candidate_score": 100 - i, "ret_1d": 0.01, "ret_3d": 0.02, "ret_5d": 0.03, "abnormal_ret_1d": 0.01, "abnormal_ret_3d": 0.02}
            for i in range(1, 12)
        ]
    )
    out = build_signal_performance(rows)
    assert set(out["bucket"]).issuperset({"top_10", "top_25", "middle", "bottom"})
    assert out["count"].sum() == len(rows)


def test_json_has_no_nan_or_nat(tmp_path):
    financial, prices, analysis = _write_minimal_inputs(tmp_path)
    report = build_commercial_analysis_validity_audit(financial, prices, analysis, analysis)
    text = (analysis / OUTPUT_FILES["report_json"]).read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "NaT" not in text
    loaded = json.loads(text)
    assert loaded["raw_csv_modified"] is False
    assert not any(isinstance(v, float) and np.isnan(v) for v in report.values())


def test_product_status_function_all_verdicts():
    assert determine_product_status(0, 0, 90, classification_explainable_ratio=0.9, signal_direction_ok=True, score_explainable_ratio=0.9) == "product_ready"
    assert determine_product_status(0, 1, 80, classification_explainable_ratio=0.9, signal_direction_ok=True, score_explainable_ratio=0.9) == "product_needs_corrections"
    assert determine_product_status(1, 0, 70, classification_explainable_ratio=0.9, signal_direction_ok=True, score_explainable_ratio=0.9) == "poc_only"
    assert determine_product_status(3, 0, 70, classification_explainable_ratio=0.9, signal_direction_ok=True, score_explainable_ratio=0.9) == "invalid"

from __future__ import annotations

import json

import pandas as pd

from jp_stock_analysis.analysis.jquants_commercial_signal_validation import (
    OUTPUT_FILES,
    RANKING_FILES,
    build_commercial_signal_validation,
    build_event_study,
    build_event_validation_panel,
    build_liquidity_cost,
    build_quantile_spread,
    build_rank_ic,
    build_sector_size_neutral,
    build_stability,
    score_signal_validation,
)


def _ranking_rows(count: int = 12) -> list[dict]:
    rows = []
    for i in range(count):
        score = float(i + 1)
        date = f"2025-01-{10 + (i % 3):02d}"
        rows.append(
            {
                "ticker": f"{1000 + i}",
                "DiscDate": date,
                "reaction_base_date": date,
                "first_time": "09:00:00",
                "ranking_label": "positive_reacted",
                "candidate_bucket": "quality_screening",
                "ranking_score": score,
                "commercial_abnormal_reaction_score": score / 10.0,
                "event_direction_final": "positive",
                "event_classification_detail": "positive_confirmed",
                "event_confidence_tier": "medium",
                "classification_basis_level": "sector_size_adjusted",
                "product_acceptability": "acceptable_with_limitations",
                "manual_review_required": False,
                "manual_review_priority": "low",
                "mixed_event_severity": "none",
                "auto_classification_safe_flag": True,
                "abnormal_return_adjustment_basis": "sector_size_adjusted",
                "abnormal_return_adjustment_quality": "acceptable_with_limitations",
                "financial_disclosure_available": True,
                "financial_disclosure_join_status": "matched_discno",
                "financial_disclosure_confidence": "medium",
                "earnings_forecast_revision_direction_v2": "positive",
                "dividend_revision_direction_v2": "insufficient_evidence",
                "liquidity_status": "valid",
                "avg_turnover_20d": 1_500_000_000 if i % 2 == 0 else 200_000_000,
                "is_liquid_enough": True,
                "sector33_name": "Sector A" if i % 2 == 0 else "Sector B",
                "topix_size_category": "Large" if i % 2 == 0 else "Mid",
                "horizon_flags": "none",
                "product_risk_flags": "none",
                "ret_1d": score / 1000,
                "ret_3d": score / 900,
                "ret_5d": score / 800,
                "ret_20d": score / 700,
                "abnormal_ret_1d": score / 1100,
                "abnormal_ret_3d": score / 1000,
                "abnormal_ret_5d": score / 900,
                "abnormal_ret_20d": score / 800,
                "sector_adjusted_return_1d": score / 1200,
                "sector_adjusted_return_3d": score / 1100,
                "sector_adjusted_return_5d": score / 1000,
                "sector_adjusted_return_20d": score / 900,
                "sector_size_adjusted_return_1d": score / 1300,
                "sector_size_adjusted_return_3d": score / 1200,
                "sector_size_adjusted_return_5d": score / 1100,
                "sector_size_adjusted_return_20d": score / 1000,
                "ret_1d_status": "valid",
                "ret_3d_status": "valid",
                "ret_5d_status": "valid",
                "ret_20d_status": "valid",
            }
        )
    return rows


def _panel(count: int = 12) -> pd.DataFrame:
    return build_event_validation_panel(pd.DataFrame(_ranking_rows(count)), asof="2026-07-06T00:00:00+09:00")


def test_event_validation_panel_creation_and_missing_20d_status():
    rows = _ranking_rows(1)
    rows[0]["ret_20d"] = None
    rows[0]["ret_20d_status"] = "insufficient_future_price"
    panel = build_event_validation_panel(pd.DataFrame(rows), asof="2026-07-06T00:00:00+09:00")
    assert panel.loc[0, "ticker"] == "1000"
    assert panel.loc[0, "validation_asof_date"] == "2026-07-06T00:00:00+09:00"
    assert pd.isna(panel.loc[0, "fwd_ret_20d"])
    assert panel.loc[0, "fwd_ret_20d_status"] == "missing"


def test_missing_input_files_handled_safely(tmp_path):
    summary = build_commercial_signal_validation(tmp_path, tmp_path / "prices.csv", tmp_path)
    assert summary["finance_grade_signal_validation_status"] == "not_evaluable"
    assert summary["validation_panel_rows"] == 0
    for filename in OUTPUT_FILES.values():
        assert (tmp_path / filename).exists()


def test_leakage_flag_when_reaction_base_precedes_disclosure():
    rows = _ranking_rows(1)
    rows[0]["DiscDate"] = "2025-01-10"
    rows[0]["reaction_base_date"] = "2025-01-09"
    panel = build_event_validation_panel(pd.DataFrame(rows))
    assert bool(panel.loc[0, "leakage_risk_flag"]) is True
    assert "reaction_base_before_disclosure" in panel.loc[0, "validation_exclusion_reason"]


def test_rank_ic_calculation_with_deterministic_fixture():
    rank_ic = build_rank_ic(_panel(12))
    overall = rank_ic[
        (rank_ic["group_by"] == "overall")
        & (rank_ic["horizon"] == "1d")
        & (rank_ic["label"] == "fwd_abnormal_ret_1d")
        & (rank_ic["status"] == "ok")
    ]
    assert not overall.empty
    assert overall.iloc[0]["rank_ic_spearman"] > 0.99
    assert overall.iloc[0]["sample_count"] == 12


def test_insufficient_sample_returns_insufficient_sample():
    rank_ic = build_rank_ic(_panel(2))
    assert "insufficient_sample" in set(rank_ic["status"])


def test_quantile_spread_and_monotonicity_score():
    quantile = build_quantile_spread(_panel(12))
    row = quantile[(quantile["score"] == "ranking_score") & (quantile["label"] == "fwd_abnormal_ret_1d")].iloc[0]
    assert row["top_bottom_spread"] > 0
    assert row["monotonicity_score"] == 1.0


def test_event_study_aar_car_calculation():
    event_study = build_event_study(_panel(12))
    row = event_study[(event_study["bucket"] == "ranking_label") & (event_study["window"] == "0_1d")].iloc[0]
    assert row["status"] == "ok"
    assert row["aar"] > 0
    assert row["car"] == row["aar"]


def test_liquidity_cost_proxy_calculation():
    liquidity = build_liquidity_cost(_panel(20))
    assert set(liquidity["liquidity_capacity_bucket"]).issuperset({"high_capacity", "medium_capacity"})
    ok = liquidity[liquidity["status"] == "ok"]
    assert not ok.empty
    assert (ok["net_validation_spread"] < ok["gross_validation_spread"]).all()


def test_sector_size_neutral_validation_with_insufficient_sample_guard():
    neutral = build_sector_size_neutral(_panel(4))
    assert neutral["status"].astype(str).str.contains("insufficient").any()


def test_stability_analysis_outputs_month_and_group_rows():
    stability = build_stability(_panel(12))
    assert {"event_month", "sector33_name", "topix_size_category"}.issubset(set(stability["group_by"]))


def test_finance_grade_score_caps_for_leakage_risk():
    rows = _ranking_rows(80)
    for row in rows:
        row["reaction_base_date"] = "2025-01-01"
        row["DiscDate"] = "2025-01-10"
    panel = build_event_validation_panel(pd.DataFrame(rows))
    rank_ic = build_rank_ic(panel)
    quantile = build_quantile_spread(panel)
    event_study = build_event_study(panel)
    liquidity = build_liquidity_cost(panel)
    neutral = build_sector_size_neutral(panel)
    stability = build_stability(panel)
    score = score_signal_validation(panel, rank_ic, quantile, event_study, liquidity, neutral, stability)
    assert score["leakage_risk_rows"] == 80
    assert score["finance_grade_signal_validation_score"] <= 35.0


def test_finance_grade_score_caps_for_small_sample():
    panel = _panel(12)
    score = score_signal_validation(
        panel,
        build_rank_ic(panel),
        build_quantile_spread(panel),
        build_event_study(panel),
        build_liquidity_cost(panel),
        build_sector_size_neutral(panel),
        build_stability(panel),
    )
    assert score["validation_panel_rows"] == 12
    assert score["finance_grade_signal_validation_score"] <= 30.0


def test_build_writes_all_outputs_from_fixture_files(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    pd.DataFrame(_ranking_rows(12)).to_csv(
        analysis / RANKING_FILES["positive_reacted"], index=False
    )
    summary = build_commercial_signal_validation(analysis, tmp_path / "prices.csv", analysis)
    assert summary["validation_panel_rows"] == 12
    loaded = json.loads((analysis / OUTPUT_FILES["summary"]).read_text(encoding="utf-8"))
    assert loaded["rank_ic_available"] is True
    for filename in OUTPUT_FILES.values():
        assert (analysis / filename).exists()

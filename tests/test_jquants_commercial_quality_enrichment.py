from __future__ import annotations

import json

import pandas as pd

from jp_stock_analysis.analysis.jquants_commercial_quality_enrichment import (
    build_commercial_quality_enriched_rankings,
)
from jp_stock_analysis.analysis import jquants_commercial_quality_pipeline as pipeline


def _write_inputs(tmp_path):
    ranking = tmp_path / "positive.csv"
    audit = tmp_path / "audit.csv"
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "rank": 1,
                "ranking_label": "positive_reacted",
                "ranking_reason": "positive_signal_and_positive_abnormal_1d",
                "DiscDate": "2024-12-01",
                "ticker": 7203,
                "liquidity_bucket": "high",
                "event_time_bucket": "after_close",
                "commercial_signal": "earnings_upward",
                "candidate_bucket": "A",
                "abnormal_ret_1d": 0.03,
                "ret_1d": 0.04,
                "universe_ret_1d": 0.01,
                "avg_turnover_20d": 1_000_000_000,
                "turnover_spike_ratio": 2.0,
                "reaction_base_date": "2024-12-02",
            },
            {
                "rank": 2,
                "ranking_label": "positive_reacted",
                "ranking_reason": "positive_signal_and_positive_abnormal_1d",
                "DiscDate": "2024-12-01",
                "ticker": 9999,
                "liquidity_bucket": "mid",
                "event_time_bucket": "after_close",
                "commercial_signal": "earnings_upward",
                "candidate_bucket": "A",
                "abnormal_ret_1d": 0.02,
                "ret_1d": 0.03,
                "universe_ret_1d": 0.01,
                "avg_turnover_20d": 500_000_000,
                "turnover_spike_ratio": 1.5,
                "reaction_base_date": "2024-12-02",
            },
        ]
    ).to_csv(ranking, index=False)
    pd.DataFrame(
        [
            {
                "DiscDate": "2024-12-01",
                "ticker": "7203.0",
                "commercial_signal": "mixed_revision",
                "candidate_bucket": "A",
            }
        ]
    ).to_csv(audit, index=False)
    pd.DataFrame(
        [
            {
                "Code": "72030",
                "CompanyName": "Toyota Motor",
                "CompanyNameEnglish": "TOYOTA MOTOR CORPORATION",
                "MarketCodeName": "Prime",
                "MarketCode": "0111",
                "Sector33Code": "3700",
                "Sector33CodeName": "Transportation Equipment",
                "ScaleCategory": "TOPIX Core30",
            }
        ]
    ).to_csv(metadata, index=False)
    return ranking, audit, metadata


def test_enrichment_joins_metadata_orders_columns_and_marks_missing(tmp_path):
    ranking, audit, metadata = _write_inputs(tmp_path)
    summary_path = tmp_path / "summary.json"
    files = {
        "ranking": (ranking, tmp_path / "ranking_enriched.csv"),
        "audit": (audit, tmp_path / "audit_enriched.csv"),
    }

    summary = build_commercial_quality_enriched_rankings(
        metadata_path=metadata,
        summary_path=summary_path,
        input_output_files=files,
    )

    enriched = pd.read_csv(files["ranking"][1], dtype={"ticker": str})
    assert len(enriched) == 2
    assert enriched.loc[0, "ticker"] == "7203"
    assert enriched.loc[0, "company_name"] == "Toyota Motor"
    assert enriched.loc[0, "sector33_name"] == "Transportation Equipment"
    assert enriched.loc[0, "market_segment"] == "Prime"
    assert bool(enriched.loc[1, "metadata_missing"])
    assert enriched.columns[:8].tolist() == [
        "rank",
        "ranking_label",
        "ranking_reason",
        "DiscDate",
        "ticker",
        "company_name",
        "company_name_en",
        "market_segment",
    ]
    for col in [
        "classification_confidence",
        "manual_review_required",
        "ranking_score_components",
        "evidence_summary",
        "horizon_flags",
        "liquidity_flags",
        "product_risk_flags",
        "product_acceptability",
    ]:
        assert col in enriched.columns

    audit_enriched = pd.read_csv(files["audit"][1], dtype={"ticker": str})
    assert "rank" not in audit_enriched.columns
    assert audit_enriched.loc[0, "ticker"] == "7203"
    assert not bool(audit_enriched.loc[0, "metadata_missing"])

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_metadata_missing_rows"] == 1
    assert payload["total_metadata_missing_rows"] == 1
    assert payload["metadata_missing_tickers"] == ["9999"]
    assert "sector17_name" in payload["missing_metadata_columns"]


def test_enrichment_tolerates_partial_metadata_and_no_missing_rows(tmp_path):
    ranking = tmp_path / "positive.csv"
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame([{"ticker": "7203", "DiscDate": "2024-12-01"}]).to_csv(ranking, index=False)
    pd.DataFrame([{"ticker": 7203.0, "company_name": "Toyota Motor"}]).to_csv(
        metadata, index=False
    )
    files = {"ranking": (ranking, tmp_path / "ranking_enriched.csv")}

    summary = build_commercial_quality_enriched_rankings(
        metadata_path=metadata,
        summary_path=tmp_path / "summary.json",
        input_output_files=files,
    )

    enriched = pd.read_csv(files["ranking"][1], dtype={"ticker": str})
    assert len(enriched) == 1
    assert enriched.loc[0, "company_name"] == "Toyota Motor"
    assert not bool(enriched.loc[0, "metadata_missing"])
    assert summary["total_metadata_missing_rows"] == 0
    assert "sector33_name" in summary["missing_metadata_columns"]


def test_pipeline_final_summary_includes_enriched_rankings(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pipeline,
        "build_price_reaction_v2",
        lambda *args, **kwargs: {"input_rows": 1, "output_rows": 1, "price_reaction_status_v2_counts": {}},
    )
    monkeypatch.setattr(
        pipeline,
        "build_abnormal_return",
        lambda *args, **kwargs: {"input_rows": 1, "output_rows": 1, "abnormal_ret_status_counts": {}},
    )
    monkeypatch.setattr(
        pipeline,
        "build_liquidity_features",
        lambda *args, **kwargs: {"input_rows": 1, "output_rows": 1, "liquidity_bucket_counts": {}},
    )
    monkeypatch.setattr(
        pipeline,
        "build_robust_summary",
        lambda *args, **kwargs: {"rows": 1, "groups": 1},
    )
    monkeypatch.setattr(
        pipeline,
        "build_commercial_quality_rankings",
        lambda *args, **kwargs: {
            "input_rows": 1,
            "eligible_rows": 1,
            "excluded_rows": 0,
            "unclassified_rows": 0,
            "ranking_counts": {},
            "output_paths": {},
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        pipeline,
        "build_commercial_quality_enriched_rankings",
        lambda *args, **kwargs: {
            "output_files": {"ranking": "ranking_enriched.csv"},
            "total_rows": 1,
            "total_metadata_missing_rows": 0,
            "metadata_input_path": "metadata.csv",
            "warnings": [],
        },
    )

    final = pipeline.build_commercial_quality_dataset(
        tmp_path / "candidates.csv",
        tmp_path / "prices.csv",
        tmp_path / "final.json",
    )
    assert final["enriched_rankings"]["total_rows"] == 1
    assert final["enriched_rankings"]["metadata_input_path"] == "metadata.csv"

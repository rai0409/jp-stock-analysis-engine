"""Tests for the offline modeling report orchestrator."""

from __future__ import annotations

from jp_stock_analysis.modeling.dataset import build_modeling_dataset
from jp_stock_analysis.modeling.fixtures import build_synthetic_bundle
from jp_stock_analysis.modeling.ml_models import STATUS_TRAINED
from jp_stock_analysis.modeling.report import (
    build_modeling_report,
    write_modeling_report_outputs,
)


def _bundle_and_dataset():
    b = build_synthetic_bundle()
    ds = build_modeling_dataset(
        b.fundamentals,
        b.prices,
        b.metadata,
        b.narratives,
        decision_dates=b.decision_dates,
        horizons=b.horizons,
        bundle_disclosure_date=b.bundle_disclosure_date,
        is_synthetic=True,
    )
    return b, ds


def test_report_has_all_required_sections():
    b, ds = _bundle_and_dataset()
    report = build_modeling_report(ds, b.prices, bundle_disclosure_date=b.bundle_disclosure_date)
    payload = report.to_dict()
    for key in (
        "disclaimer",
        "synthetic",
        "data_coverage",
        "factor_score_distribution",
        "ranking_by_horizon",
        "walk_forward",
        "optional_backends",
        "model_comparison",
        "portfolio_long_short",
        "neutralized_rank_metrics",
        "mmc_style",
        "model_diversity",
        "no_look_ahead_status",
        "limitations",
    ):
        assert key in payload
    cov = payload["data_coverage"]
    assert "accounting_basis_distribution" in cov
    assert "exclusions" in cov
    assert "feature_coverage" in cov


def test_report_flags_synthetic_and_disclaims():
    b, ds = _bundle_and_dataset()
    payload = build_modeling_report(
        ds, b.prices, bundle_disclosure_date=b.bundle_disclosure_date
    ).to_dict()
    assert payload["synthetic"] is True
    assert "not real market evidence" in payload["synthetic_warning"]
    assert "not personalized financial advice" in payload["disclaimer"]


def test_model_comparison_baseline_trained_optional_skipped():
    b, ds = _bundle_and_dataset()
    payload = build_modeling_report(
        ds, b.prices, bundle_disclosure_date=b.bundle_disclosure_date
    ).to_dict()
    by_type = {m["model_type"]: m for m in payload["model_comparison"]}
    assert by_type["baseline_factor_ranker"]["status"] == STATUS_TRAINED
    # optional backends absent in the test env -> skipped, not failed
    assert by_type["lightgbm_ranker"]["status"] in (
        "optional_dependency_missing",
        STATUS_TRAINED,
    )


def test_report_outputs_written_and_marked_synthetic(tmp_path):
    b, ds = _bundle_and_dataset()
    report = build_modeling_report(ds, b.prices, bundle_disclosure_date=b.bundle_disclosure_date)
    paths = write_modeling_report_outputs(report, tmp_path / "out")
    assert paths["json_path"].exists()
    md = paths["markdown_path"].read_text(encoding="utf-8")
    assert "SYNTHETIC FIXTURE RESULTS" in md
    assert "No-look-ahead status" in md
    assert "Limitations" in md
    assert "JPX-style long-short spread evaluation" in md
    assert "Numerai-style neutralized rank metrics" in md
    assert "MMC-style contribution delta" in md
    assert "Model diversity, stability & explainability" in md


def test_report_model_diversity_section_populated():
    b, ds = _bundle_and_dataset()
    md = build_modeling_report(
        ds, b.prices, bundle_disclosure_date=b.bundle_disclosure_date
    ).to_dict()["model_diversity"]
    assert md["linear_models"]["ridge"]["status"] == "fitted"
    en = md["linear_models"]["elastic_net"]
    assert en["status"] in ("fitted", "not_converged")
    assert "converged" in en and "sparsity" in en
    assert set(md["ensemble"]) == {"rank_average", "weighted_blend"}
    assert "rank_ic" in md["stability"]["fold_stability"]
    assert md["feature_importance"]["coefficient"]["method"] == "coefficient"
    assert md["feature_importance"]["permutation"]["method"] == "permutation"


def test_report_portfolio_and_neutralized_sections_populated():
    b, ds = _bundle_and_dataset()
    payload = build_modeling_report(
        ds, b.prices, bundle_disclosure_date=b.bundle_disclosure_date, transaction_cost_bps=10.0
    ).to_dict()
    # one portfolio summary per horizon
    assert set(payload["portfolio_long_short"]) == {str(h) for h in ds.horizons}
    for summary in payload["portfolio_long_short"].values():
        assert "sharpe_like" in summary and "average_turnover" in summary
        assert "net_mean_spread" in summary  # transaction cost was enabled
    neutral = payload["neutralized_rank_metrics"]
    assert neutral is not None
    assert "neutralized_ic_mean" in neutral
    assert "exposure_diagnostics" in neutral
    # MMC-style needs >=2 trained models; absent optional backends -> None
    assert payload["mmc_style"] is None or "contribution_delta" in payload["mmc_style"]


def test_no_look_ahead_status_is_eligible_for_synthetic_bundle():
    b, ds = _bundle_and_dataset()
    payload = build_modeling_report(
        ds, b.prices, bundle_disclosure_date=b.bundle_disclosure_date
    ).to_dict()
    # the synthetic bundle has ample later prices -> readiness eligible
    assert payload["no_look_ahead_status"]["overall_status"] == "eligible"

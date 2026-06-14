"""Tests for the end-to-end pipeline runner. Deterministic, offline."""

from __future__ import annotations

import json
from datetime import date, timedelta

from jp_stock_analysis.modeling.dataset import build_modeling_dataset
from jp_stock_analysis.modeling.fixtures import build_synthetic_bundle
from jp_stock_analysis.modeling.pipeline import PipelineConfig, run_pipeline
from jp_stock_analysis.schemas import FinancialStatement, PriceBar

STAMP = "1970-01-01T00:00:00Z"

EXPECTED_ARTIFACTS = [
    "pipeline_summary.json",
    "pipeline_summary.md",
    "artifact_manifest.json",
    "artifact_manifest.md",
    "audit_manifest.json",
    "audit_manifest.md",
    "modeling_report.json",
    "modeling_report.md",
    "ranking/ranking_metrics.json",
    "portfolio/portfolio_metrics.json",
    "neutralization/neutralized_metrics.json",
    "stability/model_stability.json",
    "monitoring/monitoring.json",
    "constraints/constrained_portfolio.json",
    "linear/elastic_net/model_metadata.json",
]


def _synthetic_dataset():
    b = build_synthetic_bundle()
    ds = build_modeling_dataset(
        b.fundamentals, b.prices, b.metadata, b.narratives,
        decision_dates=b.decision_dates, horizons=b.horizons,
        bundle_disclosure_date=b.bundle_disclosure_date, is_synthetic=True,
    )
    return ds, b


def _run(tmp_path, *, run_id="run", config=None):
    ds, b = _synthetic_dataset()
    return run_pipeline(
        ds, b.prices, output_dir=tmp_path, run_id=run_id, fixed_timestamp=STAMP,
        disclosure_date=b.bundle_disclosure_date,
        config=config or PipelineConfig(transaction_cost_bps=10.0, max_weight_per_name=0.34),
    ), tmp_path / run_id


def test_synthetic_run_produces_expected_artifacts(tmp_path):
    _summary, run_dir = _run(tmp_path)
    for rel in EXPECTED_ARTIFACTS:
        assert (run_dir / rel).exists(), f"missing artifact {rel}"


def test_pipeline_summary_lists_each_step_with_status(tmp_path):
    summary, run_dir = _run(tmp_path)
    assert summary["step_count"] >= 14
    names = {s["name"]: s["status"] for s in summary["steps"]}
    for step in (
        "build_modeling_dataset",
        "check_forward_readiness",
        "evaluate_ranking",
        "evaluate_portfolio",
        "build_audit_manifest",
        "build_artifact_manifest",
    ):
        assert step in names
    assert summary["synthetic_vs_real"] == "synthetic"


def test_artifact_manifest_fingerprints_real_files(tmp_path):
    _summary, run_dir = _run(tmp_path)
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    by_path = {a["relative_path"]: a for a in manifest["artifacts"]}
    entry = by_path["portfolio/portfolio_metrics.json"]
    actual_bytes = (run_dir / "portfolio/portfolio_metrics.json").read_bytes()
    import hashlib

    assert entry["sha256"] == hashlib.sha256(actual_bytes).hexdigest()
    assert entry["size_bytes"] == len(actual_bytes)
    # manifest excludes itself + the summary to avoid self-reference
    assert "artifact_manifest.json" not in by_path
    assert "pipeline_summary.json" not in by_path


def test_audit_manifest_includes_real_output_fingerprints(tmp_path):
    _summary, run_dir = _run(tmp_path)
    audit = json.loads((run_dir / "audit_manifest.json").read_text(encoding="utf-8"))
    assert "modeling_report.json" in audit["output_files"]
    assert audit["synthetic_vs_real"] == "synthetic"
    assert audit["no_look_ahead_status"] == "eligible"


def test_no_secrets_in_any_manifest(tmp_path):
    _summary, run_dir = _run(tmp_path)
    for name in ("audit_manifest.json", "artifact_manifest.json", "pipeline_summary.json"):
        blob = (run_dir / name).read_text(encoding="utf-8")
        for marker in ("JQUANTS_API_KEY", "EDINET_API_KEY", "x-api-key"):
            assert marker not in blob


def test_readiness_blocked_is_surfaced_not_bypassed(tmp_path):
    # prices ending before the disclosure date -> no post-disclosure window -> BLOCKED
    start = date(2025, 1, 1)
    bars = []
    day = start
    for _ in range(40):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        bars.append(PriceBar(ticker="1301", date=day, close=100.0, adjusted_close=100.0))
        day += timedelta(days=1)
    decision = bars[-1].date
    # disclosure == decision == last price date -> obs is included (disclosure <=
    # decision) but there are zero price rows strictly after it -> readiness BLOCKED
    disclosure = bars[-1].date
    funds = {
        "1301": [
            FinancialStatement(
                ticker="1301", fiscal_year=2025, accounting_basis="consolidated",
                revenue=1000.0, operating_income=150.0, net_income=100.0,
                equity=1000.0, total_assets=2000.0, shares_outstanding=10.0,
            )
        ]
    }
    ds = build_modeling_dataset(
        funds, {"1301": bars}, {}, decision_dates=[decision], horizons=[5],
        bundle_disclosure_date=disclosure, is_synthetic=False,
    )
    summary = run_pipeline(
        ds, {"1301": bars}, output_dir=tmp_path, run_id="run", fixed_timestamp=STAMP,
        disclosure_date=disclosure,
    )
    readiness_step = next(s for s in summary["steps"] if s["name"] == "check_forward_readiness")
    assert summary["no_look_ahead_status"] == "blocked"
    assert readiness_step["status"] == "blocked"
    assert any("BLOCKED" in w for w in readiness_step["warnings"])


def test_missing_adv_does_not_fabricate_liquidity(tmp_path):
    # request a liquidity constraint with no ADV data -> liquidity_data_missing
    _summary, run_dir = _run(
        tmp_path, config=PipelineConfig(min_adv=10.0, max_participation_rate=0.1)
    )
    constraints = json.loads(
        (run_dir / "constraints/constrained_portfolio.json").read_text(encoding="utf-8")
    )
    assert constraints["status"] == "liquidity_data_missing"
    assert any("not fabricating ADV" in w for w in constraints["warnings"])


def test_optional_backend_skip_preserved(tmp_path):
    # the consolidated report's model comparison should still skip absent backends
    _summary, run_dir = _run(tmp_path)
    report = json.loads((run_dir / "modeling_report.json").read_text(encoding="utf-8"))
    statuses = {m["model_type"]: m["status"] for m in report["model_comparison"]}
    assert statuses["lightgbm_ranker"] in ("optional_dependency_missing", "trained")


def test_existing_report_sections_present(tmp_path):
    _summary, run_dir = _run(tmp_path)
    report = json.loads((run_dir / "modeling_report.json").read_text(encoding="utf-8"))
    for key in ("portfolio_long_short", "model_diversity", "commercial_validation"):
        assert key in report

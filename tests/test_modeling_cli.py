"""CLI tests for the modeling subcommands. Offline; synthetic fixtures."""

from __future__ import annotations

import json

from jp_stock_analysis.cli import main


def test_build_modeling_dataset_synthetic(tmp_path, capsys):
    out = tmp_path / "ds"
    rc = main(["build-modeling-dataset", "--synthetic", "--output-dir", str(out)])
    assert rc == 0
    assert (out / "modeling_dataset.csv").exists()
    summary = json.loads((out / "modeling_dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["is_synthetic"] is True
    assert "not real market evidence" in summary["synthetic_warning"]


def test_evaluate_factor_ranking_synthetic(tmp_path):
    out = tmp_path / "rank"
    rc = main(["evaluate-factor-ranking", "--synthetic", "--output-dir", str(out)])
    assert rc == 0
    assert (out / "ranking_metrics.json").exists()
    assert (out / "ranking_metrics.csv").exists()


def test_walk_forward_synthetic(tmp_path):
    out = tmp_path / "wf"
    rc = main(
        [
            "run-walk-forward-ranking",
            "--synthetic",
            "--min-train-periods",
            "1",
            "--test-periods",
            "1",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads((out / "walk_forward.json").read_text(encoding="utf-8"))
    assert payload["fold_count"] >= 1


def test_train_ranking_model_optional_backend_skips(tmp_path):
    out = tmp_path / "model"
    rc = main(
        [
            "train-ranking-model",
            "--synthetic",
            "--model-type",
            "lightgbm_ranker",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0  # missing optional dep is a clean skip, not an error
    result = json.loads((out / "model_result.json").read_text(encoding="utf-8"))
    assert result["status"] in ("optional_dependency_missing", "trained")


def test_train_ranking_model_baseline(tmp_path):
    out = tmp_path / "baseline"
    rc = main(
        [
            "train-ranking-model",
            "--synthetic",
            "--model-type",
            "baseline_factor_ranker",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    result = json.loads((out / "model_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "trained"


def test_modeling_report_synthetic(tmp_path):
    out = tmp_path / "report"
    rc = main(["modeling-report", "--synthetic", "--output-dir", str(out)])
    assert rc == 0
    assert (out / "modeling_report.json").exists()
    assert "SYNTHETIC" in (out / "modeling_report.md").read_text(encoding="utf-8")


def test_evaluate_portfolio_ranking_synthetic(tmp_path):
    out = tmp_path / "port"
    rc = main(
        [
            "evaluate-portfolio-ranking",
            "--synthetic",
            "--horizon",
            "20",
            "--portfolio-rank-weighted",
            "--transaction-cost-bps",
            "10",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads((out / "portfolio_metrics.json").read_text(encoding="utf-8"))
    assert payload["research_only"] is True
    assert payload["is_synthetic"] is True
    assert payload["transaction_cost"]["transaction_cost_bps"] == 10.0
    assert (out / "portfolio_metrics.csv").exists()


def test_train_linear_ranking_model_ridge(tmp_path):
    out = tmp_path / "ridge"
    rc = main(
        [
            "train-linear-ranking-model",
            "--synthetic",
            "--linear-model-type",
            "ridge",
            "--horizon",
            "20",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    meta = json.loads((out / "model_metadata.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == "ridge_v1"
    assert meta["status"] == "fitted"
    assert (out / "coefficients.csv").exists()
    assert (out / "predictions.csv").exists()


def test_train_linear_ranking_model_elastic_net_with_importance(tmp_path):
    out = tmp_path / "en"
    rc = main(
        [
            "train-linear-ranking-model",
            "--synthetic",
            "--linear-model-type",
            "elastic_net",
            "--horizon",
            "20",
            "--alpha",
            "0.05",
            "--l1-ratio",
            "0.5",
            "--feature-importance",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    meta = json.loads((out / "model_metadata.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == "elastic_net_coordinate_descent_v1"
    assert "objective_history" in meta and "converged" in meta
    imp = json.loads((out / "feature_importance.json").read_text(encoding="utf-8"))
    assert "coefficient" in imp and "permutation" in imp


def test_evaluate_model_stability_synthetic(tmp_path):
    out = tmp_path / "stab"
    rc = main(
        [
            "evaluate-model-stability",
            "--synthetic",
            "--horizon",
            "20",
            "--seed-count",
            "4",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads((out / "model_stability.json").read_text(encoding="utf-8"))
    assert payload["research_only"] is True
    assert "rank_ic" in payload["fold_stability"]
    assert payload["seed_stability"] is not None
    assert (out / "model_stability.csv").exists()


def test_evaluate_portfolio_constraints_synthetic(tmp_path):
    out = tmp_path / "con"
    rc = main(
        [
            "evaluate-portfolio-constraints",
            "--synthetic",
            "--horizon",
            "20",
            "--max-weight-per-name",
            "0.34",
            "--max-sector-weight",
            "0.6",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads((out / "constrained_portfolio.json").read_text(encoding="utf-8"))
    assert payload["constraints"]["research_only"] is True
    assert (out / "constrained_portfolio.csv").exists()


def test_build_audit_manifest_synthetic_deterministic(tmp_path):
    out1 = tmp_path / "a1"
    out2 = tmp_path / "a2"
    args = [
        "build-audit-manifest",
        "--synthetic",
        "--input",
        "tests/fixtures/modeling/fundamentals.csv",
        "--run-id",
        "run_fixed",
        "--fixed-timestamp",
        "1970-01-01T00:00:00Z",
    ]
    assert main([*args, "--output-dir", str(out1)]) == 0
    assert main([*args, "--output-dir", str(out2)]) == 0
    m1 = json.loads((out1 / "audit_manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads((out2 / "audit_manifest.json").read_text(encoding="utf-8"))
    # identical except the git_commit which reflects the live repo (still equal here)
    assert m1["run_id"] == "run_fixed"
    assert m1["input_fingerprints"] == m2["input_fingerprints"]
    assert "SECRET" not in json.dumps(m1)


def test_evaluate_model_monitoring_synthetic(tmp_path):
    out = tmp_path / "mon"
    rc = main(
        ["evaluate-model-monitoring", "--synthetic", "--horizon", "20", "--output-dir", str(out)]
    )
    assert rc == 0
    payload = json.loads((out / "monitoring.json").read_text(encoding="utf-8"))
    assert payload["research_only"] is True
    assert "long_short_spread" in payload["metrics"]


def test_run_modeling_pipeline_synthetic(tmp_path):
    out = tmp_path / "pipe"
    rc = main(
        [
            "run-modeling-pipeline",
            "--synthetic",
            "--run-id",
            "run",
            "--fixed-timestamp",
            "1970-01-01T00:00:00Z",
            "--transaction-cost-bps",
            "10",
            "--max-weight-per-name",
            "0.34",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    run_dir = out / "run"
    summary = json.loads((run_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["synthetic_vs_real"] == "synthetic"
    assert (run_dir / "artifact_manifest.json").exists()
    assert (run_dir / "audit_manifest.json").exists()
    assert (run_dir / "modeling_report.json").exists()


def test_verify_pipeline_determinism_synthetic(tmp_path):
    out = tmp_path / "verify"
    rc = main(
        [
            "verify-pipeline-determinism",
            "--synthetic",
            "--run-id-prefix",
            "det",
            "--fail-on-difference",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0  # two synthetic runs are identical
    report = json.loads((out / "determinism_report.json").read_text(encoding="utf-8"))
    assert report["overall"] == "identical"
    assert report["research_only"] is True
    assert (out / "determinism_report.md").exists()


def test_check_pipeline_regression_synthetic_no_regression(tmp_path):
    out = tmp_path / "reg"
    rc = main(
        [
            "check-pipeline-regression",
            "--synthetic",
            "--fail-on-regression",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0  # committed fixture matches a fresh synthetic run
    report = json.loads((out / "pipeline_regression_report.json").read_text(encoding="utf-8"))
    assert report["regression_detected"] is False
    assert report["research_only"] is True
    assert (out / "pipeline_regression_report.md").exists()


def test_check_pipeline_regression_fails_on_controlled_regression(tmp_path):
    # build a baseline from a non-default config, then check against the default
    # committed-style baseline path that does NOT contain that artifact set
    baseline_path = tmp_path / "baseline.json"
    rc_update = main(
        [
            "check-pipeline-regression",
            "--synthetic",
            "--update-baseline",
            "--baseline-path",
            str(baseline_path),
            "--output-dir",
            str(tmp_path / "cap"),
        ]
    )
    assert rc_update == 0
    # corrupt the baseline's recorded metric so a fresh run is a regression
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for artifact in baseline["artifacts"]:
        if artifact["relative_path"] == "ranking/ranking_metrics.json":
            artifact["canonical_sha256"] = "0" * 64  # force a mismatch
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    rc = main(
        [
            "check-pipeline-regression",
            "--synthetic",
            "--baseline-path",
            str(baseline_path),
            "--fail-on-regression",
            "--output-dir",
            str(tmp_path / "chk"),
        ]
    )
    assert rc == 2  # regression -> nonzero exit
    report = json.loads(
        (tmp_path / "chk" / "pipeline_regression_report.json").read_text(encoding="utf-8")
    )
    assert report["regression_detected"] is True


def test_check_pipeline_regression_update_baseline_writes(tmp_path):
    baseline_path = tmp_path / "golden.json"
    rc = main(
        [
            "check-pipeline-regression",
            "--synthetic",
            "--update-baseline",
            "--baseline-path",
            str(baseline_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert baseline_path.exists()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "pipeline_regression_baseline_v1"
    assert baseline["synthetic"] is True


def test_compare_pipeline_runs_writes_outputs(tmp_path):
    from jp_stock_analysis.modeling.regression_baseline import run_golden_synthetic_pipeline

    a = run_golden_synthetic_pipeline(tmp_path / "a")
    b = run_golden_synthetic_pipeline(tmp_path / "b")
    out = tmp_path / "cmp"
    rc = main(
        ["compare-pipeline-runs", "--run-a", str(a), "--run-b", str(b), "--output-dir", str(out)]
    )
    assert rc == 0
    report = json.loads((out / "run_comparison.json").read_text(encoding="utf-8"))
    assert report["comparison_status"] == "identical"
    assert report["research_only"] is True
    assert (out / "run_comparison.md").exists()


def test_promote_pipeline_baseline_blocked_without_approval(tmp_path):
    from jp_stock_analysis.modeling.regression_baseline import run_golden_synthetic_pipeline

    a = run_golden_synthetic_pipeline(tmp_path / "a")
    baseline_path = tmp_path / "baseline.json"
    out = tmp_path / "prom"
    rc = main(
        [
            "promote-pipeline-baseline",
            "--from-run",
            str(a),
            "--baseline-path",
            str(baseline_path),
            "--reviewer-note",
            "test note",
            "--require-approval",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 2  # blocked, exits nonzero
    assert not baseline_path.exists()  # baseline NOT updated
    record = json.loads((out / "baseline_promotion_record.json").read_text(encoding="utf-8"))
    assert record["status"] == "blocked_approval_required"


def test_promote_pipeline_baseline_with_approval_updates(tmp_path):
    from jp_stock_analysis.modeling.regression_baseline import run_golden_synthetic_pipeline

    a = run_golden_synthetic_pipeline(tmp_path / "a")
    baseline_path = tmp_path / "baseline.json"
    out = tmp_path / "prom"
    rc = main(
        [
            "promote-pipeline-baseline",
            "--from-run",
            str(a),
            "--baseline-path",
            str(baseline_path),
            "--reviewer-note",
            "approved reference",
            "--require-approval",
            "--approve",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    assert baseline_path.exists()
    record = json.loads((out / "baseline_promotion_record.json").read_text(encoding="utf-8"))
    assert record["status"] == "promoted"
    assert record["reviewer_note"] == "approved reference"


def test_show_baseline_history_writes_outputs(tmp_path):
    out = tmp_path / "hist"
    rc = main(
        [
            "show-baseline-history",
            "--ledger-path",
            "tests/fixtures/pipeline_baseline/baseline_history.jsonl",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    summary = json.loads((out / "baseline_history.json").read_text(encoding="utf-8"))
    assert summary["chain_status"] == "valid"
    assert summary["entry_count"] == 1
    assert (out / "baseline_history.md").exists()


def test_verify_baseline_lineage_passes_on_valid_fixture(tmp_path):
    out = tmp_path / "ver"
    rc = main(
        [
            "verify-baseline-lineage",
            "--ledger-path",
            "tests/fixtures/pipeline_baseline/baseline_history.jsonl",
            "--fail-on-invalid",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    report = json.loads(
        (out / "baseline_lineage_verification.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "valid"


def test_verify_baseline_lineage_fails_on_tampered_ledger(tmp_path):
    import shutil

    ledger = tmp_path / "ledger.jsonl"
    shutil.copy("tests/fixtures/pipeline_baseline/baseline_history.jsonl", ledger)
    entries = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    entries[0]["reviewer_note"] = "TAMPERED"
    ledger.write_text("\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n")
    rc = main(
        ["verify-baseline-lineage", "--ledger-path", str(ledger), "--fail-on-invalid"]
    )
    assert rc == 2  # tampering detected -> nonzero


def test_export_audit_bundle_writes_expected_manifest(tmp_path):
    out = tmp_path / "bundle"
    rc = main(
        [
            "export-audit-bundle",
            "--synthetic",
            "--bundle-id",
            "cli-fixed",
            "--fixed-timestamp",
            "1970-01-01T00:00:00Z",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    manifest = json.loads((out / "audit_bundle_manifest.json").read_text(encoding="utf-8"))
    paths = {entry["relative_path"] for entry in manifest["bundle_contents"]}
    assert "baseline/golden_pipeline_baseline.json" in paths
    assert "ledger/baseline_history.jsonl" in paths
    assert manifest["bundle_id"] == "cli-fixed"


def test_verify_audit_bundle_cli_passes_and_writes_outputs(tmp_path):
    bundle = tmp_path / "bundle"
    out = tmp_path / "verify"
    assert main(["export-audit-bundle", "--synthetic", "--output-dir", str(bundle)]) == 0
    rc = main(
        [
            "verify-audit-bundle",
            "--bundle-dir",
            str(bundle),
            "--output-dir",
            str(out),
            "--fail-on-invalid",
        ]
    )
    assert rc == 0
    report = json.loads((out / "audit_bundle_verification.json").read_text(encoding="utf-8"))
    assert report["status"] == "valid"
    assert (out / "audit_bundle_verification.md").exists()


def test_verify_audit_bundle_cli_fail_on_invalid_exits_nonzero(tmp_path):
    bundle = tmp_path / "bundle"
    assert main(["export-audit-bundle", "--synthetic", "--output-dir", str(bundle)]) == 0
    baseline = bundle / "baseline/golden_pipeline_baseline.json"
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["artifact_count"] = 999
    baseline.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    rc = main(["verify-audit-bundle", "--bundle-dir", str(bundle), "--fail-on-invalid"])
    assert rc == 2


def test_promote_with_ledger_appends_on_approval(tmp_path):
    import shutil

    from jp_stock_analysis.modeling.baseline_history import load_ledger
    from jp_stock_analysis.modeling.regression_baseline import run_golden_synthetic_pipeline

    a = run_golden_synthetic_pipeline(tmp_path / "a")
    ledger = tmp_path / "ledger.jsonl"
    shutil.copy("tests/fixtures/pipeline_baseline/baseline_history.jsonl", ledger)
    rc = main(
        [
            "promote-pipeline-baseline",
            "--from-run",
            str(a),
            "--baseline-path",
            str(tmp_path / "b.json"),
            "--reviewer-note",
            "ledger append",
            "--require-approval",
            "--approve",
            "--ledger-path",
            str(ledger),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert len(load_ledger(ledger)) == 2  # genesis + appended


def test_promote_without_approval_does_not_append(tmp_path):
    import shutil

    from jp_stock_analysis.modeling.baseline_history import load_ledger
    from jp_stock_analysis.modeling.regression_baseline import run_golden_synthetic_pipeline

    a = run_golden_synthetic_pipeline(tmp_path / "a")
    ledger = tmp_path / "ledger.jsonl"
    shutil.copy("tests/fixtures/pipeline_baseline/baseline_history.jsonl", ledger)
    rc = main(
        [
            "promote-pipeline-baseline",
            "--from-run",
            str(a),
            "--baseline-path",
            str(tmp_path / "b.json"),
            "--reviewer-note",
            "no approval",
            "--require-approval",
            "--ledger-path",
            str(ledger),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2  # blocked
    assert len(load_ledger(ledger)) == 1  # unchanged genesis only


def test_evaluate_neutralized_ranking_synthetic(tmp_path):
    out = tmp_path / "neut"
    rc = main(
        [
            "evaluate-neutralized-ranking",
            "--synthetic",
            "--horizon",
            "20",
            "--neutralize-exposures",
            "momentum_60d,leverage",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads((out / "neutralized_metrics.json").read_text(encoding="utf-8"))
    assert payload["research_only"] is True
    assert "neutralized_ic_mean" in payload
    assert "exposure_diagnostics" in payload


def test_file_inputs_require_decision_dates(tmp_path, capsys):
    out = tmp_path / "err"
    rc = main(
        [
            "build-modeling-dataset",
            "--prices",
            "tests/fixtures/modeling/prices.csv",
            "--fundamentals",
            "tests/fixtures/modeling/fundamentals.csv",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 1
    assert "decision-dates" in capsys.readouterr().err


def test_build_modeling_dataset_from_csv_fixtures(tmp_path):
    out = tmp_path / "ds_csv"
    rc = main(
        [
            "build-modeling-dataset",
            "--prices",
            "tests/fixtures/modeling/prices.csv",
            "--fundamentals",
            "tests/fixtures/modeling/fundamentals.csv",
            "--metadata",
            "tests/fixtures/modeling/metadata.csv",
            "--decision-dates",
            "2025-02-12,2025-03-26,2025-05-07",
            "--disclosure-date",
            "2025-02-11",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    summary = json.loads((out / "modeling_dataset_summary.json").read_text(encoding="utf-8"))
    # non_consolidated excluded by default -> matches the in-memory bundle
    assert summary["eligible_observations"] == 33

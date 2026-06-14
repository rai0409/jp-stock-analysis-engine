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

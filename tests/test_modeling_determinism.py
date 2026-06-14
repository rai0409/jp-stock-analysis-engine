"""Tests for the determinism gate. Deterministic, offline."""

from __future__ import annotations

import json

from jp_stock_analysis.modeling.dataset import build_modeling_dataset
from jp_stock_analysis.modeling.determinism import (
    VERDICT_DIFFERENT,
    VERDICT_IDENTICAL,
    VERDICT_ONLY_IN_A,
    canonicalize_json,
    compare_artifact_trees,
    write_determinism_report,
)
from jp_stock_analysis.modeling.fixtures import build_synthetic_bundle
from jp_stock_analysis.modeling.pipeline import PipelineConfig, run_pipeline

STAMP = "1970-01-01T00:00:00Z"


def _two_runs(tmp_path, config=None):
    b = build_synthetic_bundle()
    config = config or PipelineConfig(transaction_cost_bps=10.0, max_weight_per_name=0.34)
    run_dirs = []
    for suffix in ("a", "b"):
        ds = build_modeling_dataset(
            b.fundamentals, b.prices, b.metadata, b.narratives,
            decision_dates=b.decision_dates, horizons=b.horizons,
            bundle_disclosure_date=b.bundle_disclosure_date, is_synthetic=True,
        )
        parent = tmp_path / suffix
        run_pipeline(
            ds, b.prices, output_dir=parent, run_id="run", fixed_timestamp=STAMP,
            disclosure_date=b.bundle_disclosure_date, config=config,
        )
        run_dirs.append(parent / "run")
    return run_dirs


def test_two_runs_are_byte_identical_for_stable_artifacts(tmp_path):
    a, b = _two_runs(tmp_path)
    comparison = compare_artifact_trees(a, b, volatile_values=[str(a), str(b)])
    assert comparison["overall"] == VERDICT_IDENTICAL
    assert comparison["counts"].get(VERDICT_DIFFERENT, 0) == 0
    assert comparison["file_count"] > 20


def test_canonicalization_passes_when_only_volatile_fields_differ():
    a = {"run_id": "x", "created_at_utc": "2025-01-01T00:00:00Z", "ic_mean": 0.12}
    b = {"run_id": "y", "created_at_utc": "2026-06-14T00:00:00Z", "ic_mean": 0.12}
    assert canonicalize_json(a, volatile_keys=("run_id", "created_at_utc")) == canonicalize_json(
        b, volatile_keys=("run_id", "created_at_utc")
    )


def test_comparison_fails_when_numeric_metric_changes(tmp_path):
    a, b = _two_runs(tmp_path)
    # mutate a real metric in one tree -> must be reported different (not ignored)
    target = b / "ranking" / "ranking_metrics.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["horizons"][0]["ic_mean"] = (payload["horizons"][0]["ic_mean"] or 0.0) + 1.0
    target.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    comparison = compare_artifact_trees(a, b)
    assert comparison["overall"] == VERDICT_DIFFERENT
    diffs = [e for e in comparison["entries"] if e["verdict"] == VERDICT_DIFFERENT]
    assert any("ranking_metrics.json" in e["path"] for e in diffs)


def test_comparison_fails_when_artifact_missing(tmp_path):
    a, b = _two_runs(tmp_path)
    (b / "portfolio" / "portfolio_metrics.json").unlink()
    comparison = compare_artifact_trees(a, b)
    assert comparison["overall"] == VERDICT_DIFFERENT
    assert any(
        e["verdict"] == VERDICT_ONLY_IN_A and "portfolio_metrics.json" in e["path"]
        for e in comparison["entries"]
    )


def test_artifact_manifest_order_is_deterministic(tmp_path):
    a, b = _two_runs(tmp_path)
    ma = json.loads((a / "artifact_manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "artifact_manifest.json").read_text(encoding="utf-8"))
    paths_a = [x["relative_path"] for x in ma["artifacts"]]
    paths_b = [x["relative_path"] for x in mb["artifacts"]]
    assert paths_a == paths_b == sorted(paths_a)


def test_write_determinism_report(tmp_path):
    a, b = _two_runs(tmp_path)
    comparison = compare_artifact_trees(a, b, volatile_values=[str(a), str(b)])
    paths = write_determinism_report(comparison, tmp_path / "out")
    assert paths["json_path"].exists()
    payload = json.loads(paths["json_path"].read_text(encoding="utf-8"))
    assert payload["research_only"] is True
    assert "reproducibility, not model validity" in payload["disclaimer"]
    assert "Determinism Report" in paths["markdown_path"].read_text(encoding="utf-8")

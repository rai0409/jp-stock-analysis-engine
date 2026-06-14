"""Tests for pipeline run comparison + baseline promotion. Deterministic, offline."""

from __future__ import annotations

import json

from jp_stock_analysis.modeling.regression_baseline import (
    GOLDEN_TIMESTAMP,
    capture_baseline,
    run_golden_synthetic_pipeline,
    write_baseline,
)
from jp_stock_analysis.modeling.run_compare import (
    CLASS_ONLY_IN_A,
    CLASS_ONLY_IN_B,
    PROMOTION_BLOCKED_APPROVAL,
    PROMOTION_BLOCKED_NOTE,
    PROMOTION_PROMOTED,
    STATUS_CHANGED,
    STATUS_IDENTICAL,
    STATUS_MISSING_ARTIFACTS,
    STATUS_NEW_ARTIFACTS,
    compare_runs,
    promote_pipeline_baseline,
    write_run_comparison_outputs,
)

NEUTRAL_DIRECTIONS = {"increased", "decreased", "changed", "unchanged"}


def _two_runs(tmp_path):
    a = run_golden_synthetic_pipeline(tmp_path / "a")
    b = run_golden_synthetic_pipeline(tmp_path / "b")
    return a, b


def _compare(a, b, **kw):
    return compare_runs(a, b, run_id_a="golden", run_id_b="golden",
                        fixed_timestamp=GOLDEN_TIMESTAMP, **kw)


# ----------------------------- run comparison -------------------------------- #
def test_identical_runs_have_no_change(tmp_path):
    a, b = _two_runs(tmp_path)
    report = _compare(a, b)
    assert report["comparison_status"] == STATUS_IDENTICAL
    assert report["changed_artifacts"] == []
    assert set(report["counts"]) == {"unchanged"}


def test_changed_metric_reported_with_before_after_and_neutral_direction(tmp_path):
    a, b = _two_runs(tmp_path)
    target = b / "ranking" / "ranking_metrics.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["horizons"][0]["ic_mean"] = (payload["horizons"][0]["ic_mean"] or 0.0) + 0.3
    target.write_text(json.dumps(payload), encoding="utf-8")
    report = _compare(a, b)
    assert report["comparison_status"] == STATUS_CHANGED
    deltas = report["headline_metric_deltas"]
    row = next(d for d in deltas if d["metric"] == "horizons.0.ic_mean")
    assert "a" in row and "b" in row and row["a"] != row["b"]
    assert row["direction"] in NEUTRAL_DIRECTIONS


def test_no_better_worse_performance_language_in_report(tmp_path):
    a, b = _two_runs(tmp_path)
    target = b / "portfolio" / "portfolio_metrics.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["spread_series"]["sharpe_like"] = 9.99
    target.write_text(json.dumps(payload), encoding="utf-8")
    report = _compare(a, b)
    # directions are neutral; no good/bad judgement on the data (disclaimer aside)
    for row in report["headline_metric_deltas"]:
        assert row["direction"] in NEUTRAL_DIRECTIONS
    blob = json.dumps(report["headline_metric_deltas"] + report["entries"]).lower()
    for forbidden in ("better", "worse", "improv", "outperform", "degrad"):
        assert forbidden not in blob


def test_missing_artifact_is_only_in_a(tmp_path):
    a, b = _two_runs(tmp_path)
    (b / "portfolio" / "portfolio_metrics.json").unlink()
    report = _compare(a, b)
    assert report["comparison_status"] == STATUS_MISSING_ARTIFACTS
    assert "portfolio/portfolio_metrics.json" in report["only_in_a"]
    assert any(e["classification"] == CLASS_ONLY_IN_A for e in report["entries"])


def test_new_artifact_is_only_in_b(tmp_path):
    a, b = _two_runs(tmp_path)
    (b / "ranking" / "extra.json").write_text("{}\n", encoding="utf-8")
    report = _compare(a, b)
    assert report["comparison_status"] == STATUS_NEW_ARTIFACTS
    assert "ranking/extra.json" in report["only_in_b"]
    assert any(e["classification"] == CLASS_ONLY_IN_B for e in report["entries"])


def test_volatile_only_difference_is_not_substantive(tmp_path):
    # a declared-volatile field present in BOTH runs, differing only in value:
    # canonical matches (value stripped) but raw differs -> volatile_only
    a, b = _two_runs(tmp_path)
    for run, stamp in ((a, "2025-01-01T00:00:00Z"), (b, "2099-12-31T00:00:00Z")):
        target = run / "ranking" / "ranking_metrics.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["created_at_utc"] = stamp  # declared volatile field, different per run
        target.write_text(json.dumps(payload), encoding="utf-8")
    report = _compare(a, b)
    classes = {e["relative_path"]: e["classification"] for e in report["entries"]}
    assert classes["ranking/ranking_metrics.json"] == "volatile_only"
    assert report["comparison_status"] == STATUS_IDENTICAL  # no substantive change


def test_csv_numeric_column_deltas_when_changed(tmp_path):
    a, b = _two_runs(tmp_path)
    target = b / "monitoring" / "monitoring.csv"
    text = target.read_text(encoding="utf-8")
    target.write_text(text + "extra,changed,1,2,3,4,5,6\n", encoding="utf-8")
    report = _compare(a, b)
    entry = next(e for e in report["entries"] if e["relative_path"] == "monitoring/monitoring.csv")
    assert entry["classification"] == STATUS_CHANGED
    assert "row_count" in entry["delta"]


def test_comparison_outputs_research_only_and_no_abs_paths(tmp_path):
    a, b = _two_runs(tmp_path)
    report = _compare(a, b)
    paths = write_run_comparison_outputs(report, tmp_path / "out")
    assert paths["json_path"].exists()
    blob = paths["json_path"].read_text(encoding="utf-8")
    assert report["research_only"] is True
    assert "not real market evidence" in (report["synthetic_warning"] or "")
    for forbidden in (str(tmp_path), "JQUANTS_API_KEY", "EDINET_API_KEY", "x-api-key"):
        assert forbidden not in blob


# ----------------------------- promotion ------------------------------------- #
def test_promotion_requires_approval_when_required(tmp_path):
    a = run_golden_synthetic_pipeline(tmp_path / "a")
    baseline_path = tmp_path / "baseline.json"
    record, updated = promote_pipeline_baseline(
        a, baseline_path, reviewer_note="note", require_approval=True, approved=False,
        fixed_timestamp=GOLDEN_TIMESTAMP,
    )
    assert updated is False
    assert record["status"] == PROMOTION_BLOCKED_APPROVAL
    assert not baseline_path.exists()  # baseline NOT modified


def test_promotion_requires_reviewer_note(tmp_path):
    a = run_golden_synthetic_pipeline(tmp_path / "a")
    baseline_path = tmp_path / "baseline.json"
    record, updated = promote_pipeline_baseline(
        a, baseline_path, reviewer_note="", require_approval=False, approved=True,
        fixed_timestamp=GOLDEN_TIMESTAMP,
    )
    assert updated is False
    assert record["status"] == PROMOTION_BLOCKED_NOTE
    assert not baseline_path.exists()


def test_approved_promotion_writes_baseline_and_record(tmp_path):
    a = run_golden_synthetic_pipeline(tmp_path / "a")
    baseline_path = tmp_path / "baseline.json"
    record, updated = promote_pipeline_baseline(
        a, baseline_path, reviewer_note="approved reference", require_approval=True,
        approved=True, fixed_timestamp=GOLDEN_TIMESTAMP,
    )
    assert updated is True
    assert record["status"] == PROMOTION_PROMOTED
    assert baseline_path.exists()
    assert record["new_baseline_fingerprint"]
    assert record["approved"] is True


def test_promotion_record_includes_metric_deltas_vs_previous(tmp_path):
    a = run_golden_synthetic_pipeline(tmp_path / "a")
    # build a prior baseline whose recorded metric differs from the run
    prior = capture_baseline(a, run_id="golden", fixed_timestamp=GOLDEN_TIMESTAMP)
    for art in prior["artifacts"]:
        if art["relative_path"] == "ranking/ranking_metrics.json":
            art["canonical_sha256"] = "0" * 64  # force a "changed" vs the run
    prior_path = tmp_path / "prior.json"
    write_baseline(prior, prior_path)
    record, updated = promote_pipeline_baseline(
        a, tmp_path / "new.json", reviewer_note="bump", require_approval=True, approved=True,
        previous_baseline_path=prior_path, fixed_timestamp=GOLDEN_TIMESTAMP,
    )
    assert updated is True
    assert record["previous_baseline_fingerprint"] != record["new_baseline_fingerprint"]
    assert record["artifact_classification_counts"].get("changed", 0) >= 1


def test_promotion_record_no_secrets_or_abs_paths(tmp_path):
    a = run_golden_synthetic_pipeline(tmp_path / "a")
    record, _ = promote_pipeline_baseline(
        a, tmp_path / "b.json", reviewer_note="ok", require_approval=True, approved=True,
        fixed_timestamp=GOLDEN_TIMESTAMP,
    )
    blob = json.dumps(record)
    for forbidden in (str(tmp_path), "/home", "JQUANTS_API_KEY", "EDINET_API_KEY", "x-api-key"):
        assert forbidden not in blob
    assert record["baseline_path"] == "b.json"  # basename only


def test_promotion_is_deterministic(tmp_path):
    a = run_golden_synthetic_pipeline(tmp_path / "a")
    r1, _ = promote_pipeline_baseline(
        a, tmp_path / "x.json", reviewer_note="n", require_approval=True, approved=True,
        fixed_timestamp=GOLDEN_TIMESTAMP,
    )
    r2, _ = promote_pipeline_baseline(
        a, tmp_path / "y.json", reviewer_note="n", require_approval=True, approved=True,
        fixed_timestamp=GOLDEN_TIMESTAMP,
    )
    assert r1["new_baseline_fingerprint"] == r2["new_baseline_fingerprint"]

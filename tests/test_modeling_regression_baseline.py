"""Tests for the pipeline regression baseline & change detection. Deterministic."""

from __future__ import annotations

import json
from pathlib import Path

from jp_stock_analysis.modeling.regression_baseline import (
    CLASS_CHANGED,
    CLASS_MISSING,
    CLASS_NEW,
    CLASS_UNCHANGED,
    CLASS_VOLATILE_ONLY,
    GOLDEN_RUN_ID,
    GOLDEN_TIMESTAMP,
    capture_baseline,
    compare_to_baseline,
    load_baseline,
    run_golden_synthetic_pipeline,
    write_baseline,
)

COMMITTED = Path("tests/fixtures/pipeline_baseline/golden_pipeline_baseline.json")


def _golden(tmp_path, name="g"):
    return run_golden_synthetic_pipeline(tmp_path / name)


def _baseline(tmp_path):
    rd = _golden(tmp_path, "base")
    return capture_baseline(
        rd, run_id=GOLDEN_RUN_ID, fixed_timestamp=GOLDEN_TIMESTAMP, is_synthetic=True
    )


def test_golden_baseline_matches_fresh_run(tmp_path):
    baseline = _baseline(tmp_path)
    fresh = _golden(tmp_path, "fresh")
    report = compare_to_baseline(
        fresh, baseline, run_id=GOLDEN_RUN_ID, fixed_timestamp=GOLDEN_TIMESTAMP
    )
    assert report["regression_detected"] is False
    assert set(report["counts"]) == {CLASS_UNCHANGED}


def test_changed_metric_is_regression(tmp_path):
    baseline = _baseline(tmp_path)
    fresh = _golden(tmp_path, "fresh")
    target = fresh / "ranking" / "ranking_metrics.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["horizons"][0]["ic_mean"] = (payload["horizons"][0]["ic_mean"] or 0.0) + 0.5
    target.write_text(json.dumps(payload), encoding="utf-8")
    report = compare_to_baseline(
        fresh, baseline, run_id=GOLDEN_RUN_ID, fixed_timestamp=GOLDEN_TIMESTAMP
    )
    assert report["regression_detected"] is True
    changed = [e for e in report["entries"] if e["classification"] == CLASS_CHANGED]
    assert any("ranking_metrics.json" in e["relative_path"] for e in changed)
    diff = next(e for e in changed if "ranking_metrics.json" in e["relative_path"])["diff"]
    assert "headline_metrics" in diff and "horizons.0.ic_mean" in diff["headline_metrics"]


def test_different_run_id_is_not_a_regression(tmp_path):
    # the tracked metric artifacts contain no run id/timestamp, so a different run
    # id changes nothing tracked (an even-stronger property than volatile_only)
    baseline = _baseline(tmp_path)
    fresh = run_golden_synthetic_pipeline(tmp_path / "other", run_id="otherrun")
    report = compare_to_baseline(
        fresh, baseline, run_id="otherrun", fixed_timestamp=GOLDEN_TIMESTAMP
    )
    assert report["regression_detected"] is False
    assert report["counts"].get(CLASS_CHANGED, 0) == 0


def test_volatile_only_difference_is_not_regression(tmp_path):
    # a declared-volatile field (created_at_utc) present in both, differing in value:
    # canonical fingerprint is unchanged but raw bytes differ -> volatile_only
    a = tmp_path / "a"
    a.mkdir()
    (a / "m.json").write_text(
        json.dumps({"created_at_utc": "2025-01-01T00:00:00Z", "ic_mean": 0.1}),
        encoding="utf-8",
    )
    baseline = capture_baseline(a, run_id="r", fixed_timestamp="t", is_synthetic=True)
    b = tmp_path / "b"
    b.mkdir()
    (b / "m.json").write_text(
        json.dumps({"created_at_utc": "2099-12-31T00:00:00Z", "ic_mean": 0.1}),
        encoding="utf-8",
    )
    report = compare_to_baseline(b, baseline, run_id="r", fixed_timestamp="t")
    classes = {e["relative_path"]: e["classification"] for e in report["entries"]}
    assert classes["m.json"] == CLASS_VOLATILE_ONLY
    assert report["regression_detected"] is False


def test_missing_artifact_is_regression(tmp_path):
    baseline = _baseline(tmp_path)
    fresh = _golden(tmp_path, "fresh")
    (fresh / "portfolio" / "portfolio_metrics.json").unlink()
    report = compare_to_baseline(
        fresh, baseline, run_id=GOLDEN_RUN_ID, fixed_timestamp=GOLDEN_TIMESTAMP
    )
    assert report["regression_detected"] is True
    assert any(
        e["classification"] == CLASS_MISSING and "portfolio_metrics.json" in e["relative_path"]
        for e in report["entries"]
    )


def test_new_artifact_only_regression_in_strict_mode(tmp_path):
    baseline = _baseline(tmp_path)
    fresh = _golden(tmp_path, "fresh")
    (fresh / "ranking" / "extra_unexpected.json").write_text("{}\n", encoding="utf-8")
    lenient = compare_to_baseline(
        fresh, baseline, run_id=GOLDEN_RUN_ID, fixed_timestamp=GOLDEN_TIMESTAMP
    )
    strict = compare_to_baseline(
        fresh, baseline, run_id=GOLDEN_RUN_ID, fixed_timestamp=GOLDEN_TIMESTAMP,
        strict_new_artifacts=True,
    )
    assert any(e["classification"] == CLASS_NEW for e in lenient["entries"])
    assert lenient["regression_detected"] is False  # new is not a regression by default
    assert strict["regression_detected"] is True  # strict treats new as regression


def test_update_baseline_refreshes_canonical_set(tmp_path):
    baseline = _baseline(tmp_path)
    path = tmp_path / "baseline.json"
    write_baseline(baseline, path)
    reloaded = load_baseline(path)
    assert reloaded["schema_version"] == baseline["schema_version"]
    assert reloaded["artifact_count"] == baseline["artifact_count"]
    fresh = _golden(tmp_path, "fresh")
    report = compare_to_baseline(
        fresh, reloaded, run_id=GOLDEN_RUN_ID, fixed_timestamp=GOLDEN_TIMESTAMP
    )
    assert report["regression_detected"] is False


def test_artifact_order_is_deterministic(tmp_path):
    baseline = _baseline(tmp_path)
    paths = [a["relative_path"] for a in baseline["artifacts"]]
    assert paths == sorted(paths)


def test_baseline_has_no_absolute_paths_or_secrets(tmp_path):
    baseline = _baseline(tmp_path)
    blob = json.dumps(baseline)
    for forbidden in ("/tmp", "/home", "JQUANTS_API_KEY", "EDINET_API_KEY", "x-api-key"):
        assert forbidden not in blob


def test_committed_fixture_matches_fresh_synthetic_run(tmp_path):
    baseline = load_baseline(COMMITTED)
    fresh = _golden(tmp_path, "fresh")
    report = compare_to_baseline(
        fresh, baseline, run_id=GOLDEN_RUN_ID, fixed_timestamp=GOLDEN_TIMESTAMP
    )
    assert report["regression_detected"] is False, report["counts"]


def test_committed_fixture_is_safe_to_commit():
    blob = COMMITTED.read_text(encoding="utf-8")
    for forbidden in ("/tmp", "/home", "JQUANTS_API_KEY", "EDINET_API_KEY", "x-api-key"):
        assert forbidden not in blob
    baseline = json.loads(blob)
    assert baseline["synthetic"] is True
    assert "not real market evidence" in baseline["synthetic_warning"]


def test_scoped_golden_diagnostic_roundoff_keeps_exact_raw_hash(tmp_path):
    """Cross-platform last-bit noise cannot invalidate canonical golden diagnostics."""
    a = tmp_path / "golden_a"
    b = tmp_path / "golden_b"
    samples = {
        "linear/ridge/model_metadata.json": (
            {"coefficients": {"roa": -0.6595186024330528},
             "scaled_coefficients": {"roa": -0.20068511980381365}},
            {"coefficients": {"roa": -0.6595186024330505},
             "scaled_coefficients": {"roa": -0.20068511980381296}},
        ),
        "linear/elastic_net/model_metadata.json": (
            {"intercept": -0.00278914433873497,
             "objective_history": [0.03750047869956567]},
            {"intercept": -0.0027891443387349697,
             "objective_history": [0.03750047869956566]},
        ),
        "modeling_report.json": (
            {"model_diversity": {"linear_models": {
                "ridge": {"coefficients": {"roa": -0.6595186024330528},
                          "scaled_coefficients": {"roa": -0.20068511980381365}},
                "elastic_net": {"intercept": -0.00278914433873497}}},
             "neutralized_rank_metrics": {"exposure_diagnostics": {
                 "post_neutralization_exposure_corr": {"momentum_60d": -2.3728689627520734e-15},
                 "max_abs_exposure_corr_after": 2.3728689627520734e-15}}},
            {"model_diversity": {"linear_models": {
                "ridge": {"coefficients": {"roa": -0.6595186024330505},
                          "scaled_coefficients": {"roa": -0.20068511980381296}},
                "elastic_net": {"intercept": -0.0027891443387349697}}},
             "neutralized_rank_metrics": {"exposure_diagnostics": {
                 "post_neutralization_exposure_corr": {"momentum_60d": 6.819352798519816e-16},
                 "max_abs_exposure_corr_after": 6.819352798519816e-16}}},
        ),
        "neutralization/neutralized_metrics.json": (
            {"exposure_diagnostics": {
                "post_neutralization_exposure_corr": {"momentum_60d": -2.3728689627520734e-15},
                "max_abs_exposure_corr_after": 2.3728689627520734e-15}},
            {"exposure_diagnostics": {
                "post_neutralization_exposure_corr": {"momentum_60d": 6.819352798519816e-16},
                "max_abs_exposure_corr_after": 6.819352798519816e-16}},
        ),
    }
    for rel, (left, right) in samples.items():
        for parent, value in ((a, left), (b, right)):
            file = parent / rel
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    old = capture_baseline(a)
    new = capture_baseline(b)
    old_by = {e["relative_path"]: e for e in old["artifacts"]}
    new_by = {e["relative_path"]: e for e in new["artifacts"]}
    for rel in samples:
        assert old_by[rel]["raw_sha256"] != new_by[rel]["raw_sha256"]
        assert old_by[rel]["canonical_sha256"] == new_by[rel]["canonical_sha256"]
    report = compare_to_baseline(b, old)
    assert report["regression_detected"] is False
    assert report["counts"] == {CLASS_VOLATILE_ONLY: 4}


def test_scoped_golden_diagnostic_real_changes_still_fail(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    examples = {
        "linear/ridge/model_metadata.json": (
            {"coefficients": {"x": 0.1}},
            {"coefficients": {"x": 0.10000001}},
        ),
        "ranking/ranking_metrics.json": (
            {"x": 0.1},
            {"x": 0.100000000000001},
        ),
        "neutralization/neutralized_metrics.json": (
            {"exposure_diagnostics": {
                "post_neutralization_exposure_corr": {"x": 0.0}}, "status": "ok"},
            {"exposure_diagnostics": {
                "post_neutralization_exposure_corr": {"x": 0.0}}, "status": "changed"},
        ),
    }
    for rel, (left, right) in examples.items():
        for parent, payload in ((a, left), (b, right)):
            file = parent / rel
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    report = compare_to_baseline(b, capture_baseline(a))
    assert report["regression_detected"] is True
    assert report["counts"][CLASS_CHANGED] == len(examples)

"""Tests for the audit manifest. Deterministic, offline, secret-safe."""

from __future__ import annotations

import json

from jp_stock_analysis.modeling.audit import (
    build_audit_manifest,
    fingerprint_file,
    fingerprint_records,
    scrub_secrets,
    write_audit_manifest_outputs,
)


def _manifest(rows, command=None):
    return build_audit_manifest(
        command=command or {"cmd": "x"},
        model_versions=["ridge_v1", "elastic_net_coordinate_descent_v1"],
        feature_columns=["roe", "leverage"],
        target_columns=["forward_return_h20"],
        horizons=[5, 20, 60],
        is_synthetic=True,
        input_fingerprints=[fingerprint_records("f", rows)],
        stable=True,
    )


def test_identical_inputs_produce_identical_manifest():
    rows = [{"a": 1}, {"a": 2}]
    assert _manifest(rows) == _manifest(rows)


def test_changed_input_changes_fingerprint_and_run_id():
    a = _manifest([{"a": 1}, {"a": 2}])
    b = _manifest([{"a": 1}, {"a": 999}])
    assert a["input_fingerprints"][0]["sha256"] != b["input_fingerprints"][0]["sha256"]
    assert a["run_id"] != b["run_id"]


def test_manifest_includes_required_fields():
    m = _manifest([{"a": 1}])
    for key in (
        "model_versions",
        "synthetic_vs_real",
        "input_fingerprints",
        "command",
        "warnings",
        "feature_columns",
        "horizons",
    ):
        assert key in m
    assert m["synthetic_vs_real"] == "synthetic"


def test_secrets_are_scrubbed_and_never_serialized():
    m = _manifest(
        [{"a": 1}],
        command={"jquants_api_key": "TOPSECRET", "edinet_token": "X", "ok": 1},
    )
    assert m["command"]["jquants_api_key"] == "***REDACTED***"
    assert m["command"]["edinet_token"] == "***REDACTED***"
    assert "TOPSECRET" not in json.dumps(m)


def test_scrub_secrets_nested():
    out = scrub_secrets({"outer": {"api_key": "x", "value": 3}})
    assert out["outer"]["api_key"] == "***REDACTED***"
    assert out["outer"]["value"] == 3


def test_fingerprint_file_uses_basename_not_absolute_path(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    fp = fingerprint_file(f)
    assert fp["name"] == "data.csv"
    assert "absolute_path" not in fp  # default omits non-deterministic paths
    assert fp["row_count"] == 2
    assert fp["columns"] == ["a", "b"]


def test_missing_file_reports_status(tmp_path):
    fp = fingerprint_file(tmp_path / "nope.csv")
    assert fp["status"] == "missing"


def test_markdown_manifest_written(tmp_path):
    paths = write_audit_manifest_outputs(_manifest([{"a": 1}]), tmp_path / "out")
    assert paths["json_path"].exists()
    md = paths["markdown_path"].read_text(encoding="utf-8")
    assert "Audit Manifest" in md
    assert "SYNTHETIC" in md

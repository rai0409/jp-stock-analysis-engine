"""Tests for offline audit bundle export/verification."""

from __future__ import annotations

import json
from pathlib import Path

from jp_stock_analysis.modeling.audit import _canonical, _sha256
from jp_stock_analysis.modeling.audit_bundle import (
    MANIFEST_NAME,
    export_audit_bundle,
    verify_audit_bundle,
)
from jp_stock_analysis.modeling.baseline_history import (
    append_baseline_history_entry,
    load_ledger,
)
from jp_stock_analysis.modeling.regression_baseline import GOLDEN_TIMESTAMP

BASELINE = Path("tests/fixtures/pipeline_baseline/golden_pipeline_baseline.json")
LEDGER = Path("tests/fixtures/pipeline_baseline/baseline_history.jsonl")


def _baseline_fingerprint(path: Path = BASELINE) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _promotion_record(tmp_path: Path, fingerprint: str | None = None) -> Path:
    entry = load_ledger(LEDGER)[0]
    record = {
        "status": "promoted",
        "research_only": True,
        "synthetic": True,
        "new_baseline_fingerprint": fingerprint or _baseline_fingerprint(),
        "appended_entry_hash": entry["entry_hash"],
        "reviewer_note": "fixture promotion",
        "warnings": [],
    }
    path = tmp_path / "baseline_promotion_record.json"
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _export(tmp_path: Path, **kwargs):
    return export_audit_bundle(
        tmp_path / "bundle",
        baseline_path=kwargs.pop("baseline_path", BASELINE),
        ledger_path=kwargs.pop("ledger_path", LEDGER),
        synthetic=True,
        fixed_timestamp=GOLDEN_TIMESTAMP,
        bundle_id="bundle-fixed",
        **kwargs,
    )


def _load_manifest(bundle_dir: Path) -> dict:
    return json.loads((bundle_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def test_export_manifest_covers_expected_files_and_promotion_record(tmp_path):
    record = _promotion_record(tmp_path)
    manifest = _export(tmp_path, promotion_record_path=record)
    paths = {entry["relative_path"] for entry in manifest["bundle_contents"]}
    assert "baseline/golden_pipeline_baseline.json" in paths
    assert "ledger/baseline_history.jsonl" in paths
    assert "promotion_records/baseline_promotion_record.json" in paths
    assert manifest["research_only"] is True
    assert manifest["synthetic"] is True


def test_verify_passes_on_fresh_export(tmp_path):
    _export(tmp_path)
    report = verify_audit_bundle(tmp_path / "bundle")
    assert report["status"] == "valid"
    assert report["issues"] == []


def test_tampered_bundled_baseline_is_detected(tmp_path):
    _export(tmp_path)
    baseline = tmp_path / "bundle" / "baseline/golden_pipeline_baseline.json"
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["artifact_count"] = 999
    baseline.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    report = verify_audit_bundle(tmp_path / "bundle")
    assert report["status"] == "invalid"
    assert any("baseline fingerprint mismatch" in issue for issue in report["issues"])


def test_tampered_ledger_is_detected(tmp_path):
    _export(tmp_path)
    ledger = tmp_path / "bundle" / "ledger/baseline_history.jsonl"
    entries = load_ledger(ledger)
    entries[0]["reviewer_note"] = "tampered"
    ledger.write_text("\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n")
    report = verify_audit_bundle(tmp_path / "bundle")
    assert report["status"] == "invalid"
    assert any("entry_hash mismatch" in issue for issue in report["issues"])


def test_broken_ledger_chain_is_detected(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    record = {
        "status": "promoted",
        "new_baseline_fingerprint": _baseline_fingerprint(),
        "previous_baseline_fingerprint": None,
        "source_run_id": "golden",
        "reviewer_note": "first",
        "approved": True,
        "approval_required": True,
        "synthetic": True,
        "baseline_path": "golden.json",
        "artifact_classification_counts": {},
        "headline_metric_deltas": [],
        "warnings": [],
    }
    append_baseline_history_entry(ledger, record, created_at_utc=GOLDEN_TIMESTAMP)
    append_baseline_history_entry(
        ledger, {**record, "reviewer_note": "second"}, created_at_utc=GOLDEN_TIMESTAMP
    )
    entries = load_ledger(ledger)
    entries[1]["parent_hash"] = "0" * 64
    from jp_stock_analysis.modeling.baseline_history import _content_hash

    entries[1]["entry_hash"] = _content_hash(entries[1])
    ledger.write_text("\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n")
    _export(tmp_path, ledger_path=ledger)
    report = verify_audit_bundle(tmp_path / "bundle")
    assert report["status"] == "invalid"
    assert any("parent_hash does not match" in issue for issue in report["issues"])


def test_manifest_file_fingerprint_mismatch_is_detected(tmp_path):
    _export(tmp_path)
    manifest_path = tmp_path / "bundle" / MANIFEST_NAME
    manifest = _load_manifest(tmp_path / "bundle")
    manifest["bundle_contents"][0]["canonical_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = verify_audit_bundle(tmp_path / "bundle")
    assert report["status"] == "invalid"
    assert any("manifest/content mismatch" in issue for issue in report["issues"])


def test_baseline_vs_ledger_head_mismatch_is_detected(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    record = {
        "status": "promoted",
        "new_baseline_fingerprint": "0" * 64,
        "previous_baseline_fingerprint": None,
        "source_run_id": "golden",
        "reviewer_note": "mismatch",
        "approved": True,
        "approval_required": True,
        "synthetic": True,
        "baseline_path": "golden.json",
        "artifact_classification_counts": {},
        "headline_metric_deltas": [],
        "warnings": [],
    }
    append_baseline_history_entry(ledger, record, created_at_utc=GOLDEN_TIMESTAMP)
    _export(tmp_path, ledger_path=ledger)
    report = verify_audit_bundle(tmp_path / "bundle")
    assert report["status"] == "invalid"
    assert any("does not match ledger head" in issue for issue in report["issues"])


def test_missing_bundled_file_is_detected(tmp_path):
    _export(tmp_path)
    (tmp_path / "bundle" / "baseline/golden_pipeline_baseline.json").unlink()
    report = verify_audit_bundle(tmp_path / "bundle")
    assert report["status"] == "invalid"
    assert any("missing bundled file" in issue for issue in report["issues"])


def test_overall_bundle_fingerprint_is_deterministic(tmp_path):
    m1 = export_audit_bundle(
        tmp_path / "a",
        baseline_path=BASELINE,
        ledger_path=LEDGER,
        synthetic=True,
        fixed_timestamp=GOLDEN_TIMESTAMP,
        bundle_id="fixed",
    )
    m2 = export_audit_bundle(
        tmp_path / "b",
        baseline_path=BASELINE,
        ledger_path=LEDGER,
        synthetic=True,
        fixed_timestamp=GOLDEN_TIMESTAMP,
        bundle_id="fixed",
    )
    assert m1["overall_bundle_fingerprint"] == m2["overall_bundle_fingerprint"]


def test_invalid_overall_bundle_fingerprint_is_detected(tmp_path):
    _export(tmp_path)
    manifest_path = tmp_path / "bundle" / MANIFEST_NAME
    manifest = _load_manifest(tmp_path / "bundle")
    manifest["overall_bundle_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = verify_audit_bundle(tmp_path / "bundle")
    assert report["status"] == "invalid"
    assert any("overall bundle fingerprint mismatch" in issue for issue in report["issues"])


def test_bundle_manifest_has_no_secrets_or_absolute_paths(tmp_path):
    manifest = _export(tmp_path)
    blob = _canonical(manifest)
    for forbidden in ("JQUANTS_API_KEY", "EDINET_API_KEY", str(tmp_path), "/home/", "/tmp/"):
        assert forbidden not in blob


def test_fixed_timestamp_and_bundle_id_make_stable_manifest(tmp_path):
    m1 = export_audit_bundle(
        tmp_path / "one",
        baseline_path=BASELINE,
        ledger_path=LEDGER,
        synthetic=True,
        fixed_timestamp=GOLDEN_TIMESTAMP,
        bundle_id="fixed",
    )
    m2 = export_audit_bundle(
        tmp_path / "two",
        baseline_path=BASELINE,
        ledger_path=LEDGER,
        synthetic=True,
        fixed_timestamp=GOLDEN_TIMESTAMP,
        bundle_id="fixed",
    )
    assert m1 == m2

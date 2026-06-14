"""Tests for the append-only, hash-chained baseline history ledger."""

from __future__ import annotations

import json
from pathlib import Path

from jp_stock_analysis.modeling.baseline_history import (
    APPEND_APPENDED,
    APPEND_SKIPPED_NOT_PROMOTED,
    GENESIS_PARENT,
    STATUS_INVALID,
    STATUS_VALID,
    append_baseline_history_entry,
    load_ledger,
    summarize_baseline_history,
    verify_baseline_history,
)
from jp_stock_analysis.modeling.regression_baseline import (
    GOLDEN_TIMESTAMP,
    run_golden_synthetic_pipeline,
)
from jp_stock_analysis.modeling.run_compare import promote_pipeline_baseline

COMMITTED = Path("tests/fixtures/pipeline_baseline/baseline_history.jsonl")


def _promotion_record(tmp_path, note="note", status="promoted", deltas=None):
    return {
        "status": status,
        "source_run_id": "golden",
        "new_baseline_fingerprint": "a" * 64,
        "previous_baseline_fingerprint": "b" * 64,
        "reviewer_note": note,
        "approved": True,
        "approval_required": True,
        "synthetic": True,
        "baseline_path": "golden.json",
        "artifact_classification_counts": {"unchanged": 28},
        "headline_metric_deltas": deltas or [],
        "warnings": [],
    }


def _append(tmp_path, record, ts=GOLDEN_TIMESTAMP):
    ledger = tmp_path / "ledger.jsonl"
    return append_baseline_history_entry(ledger, record, created_at_utc=ts), ledger


# ------------------------------ ledger basics -------------------------------- #
def test_committed_genesis_fixture_is_valid():
    report = verify_baseline_history(COMMITTED)
    assert report["status"] == STATUS_VALID
    assert report["entry_count"] == 1
    entries = load_ledger(COMMITTED)
    assert entries[0]["parent_hash"] == GENESIS_PARENT
    assert entries[0]["entry_index"] == 0


def test_appended_entry_chains_to_prior(tmp_path):
    (e0, _s0), ledger = _append(tmp_path, _promotion_record(tmp_path, "first"))
    (e1, s1) = append_baseline_history_entry(
        ledger, _promotion_record(tmp_path, "second"), created_at_utc=GOLDEN_TIMESTAMP
    )
    assert s1 == APPEND_APPENDED
    assert e1["parent_hash"] == e0["entry_hash"]
    assert e1["entry_index"] == 1
    assert verify_baseline_history(ledger)["status"] == STATUS_VALID


def test_entry_hash_deterministic_for_identical_content(tmp_path):
    ea, _ = append_baseline_history_entry(
        tmp_path / "a.jsonl", _promotion_record(tmp_path, "x"), created_at_utc=GOLDEN_TIMESTAMP
    )
    eb, _ = append_baseline_history_entry(
        tmp_path / "b.jsonl", _promotion_record(tmp_path, "x"), created_at_utc=GOLDEN_TIMESTAMP
    )
    assert ea["entry_hash"] == eb["entry_hash"]


def test_changed_reviewer_note_changes_entry_hash(tmp_path):
    ea, _ = append_baseline_history_entry(
        tmp_path / "a.jsonl", _promotion_record(tmp_path, "note one"),
        created_at_utc=GOLDEN_TIMESTAMP,
    )
    eb, _ = append_baseline_history_entry(
        tmp_path / "b.jsonl", _promotion_record(tmp_path, "note two"),
        created_at_utc=GOLDEN_TIMESTAMP,
    )
    assert ea["entry_hash"] != eb["entry_hash"]


# ------------------------------ tamper detection ----------------------------- #
def test_tampered_entry_is_detected(tmp_path):
    (_e, _s), ledger = _append(tmp_path, _promotion_record(tmp_path, "orig"))
    entries = load_ledger(ledger)
    entries[0]["reviewer_note"] = "EDITED AFTER THE FACT"
    ledger.write_text("\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n")
    report = verify_baseline_history(ledger)
    assert report["status"] == STATUS_INVALID
    assert any("entry_hash mismatch" in i for i in report["issues"])


def test_broken_parent_link_is_detected(tmp_path):
    (_e0, _), ledger = _append(tmp_path, _promotion_record(tmp_path, "a"))
    append_baseline_history_entry(
        ledger, _promotion_record(tmp_path, "b"), created_at_utc=GOLDEN_TIMESTAMP
    )
    entries = load_ledger(ledger)
    entries[1]["parent_hash"] = "c" * 64  # break the link
    entries[1]["entry_hash"] = _recompute(entries[1])  # keep its own hash consistent
    ledger.write_text("\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n")
    report = verify_baseline_history(ledger)
    assert report["status"] == STATUS_INVALID
    assert any("parent_hash does not match" in i for i in report["issues"])


def test_out_of_order_entry_index_is_detected(tmp_path):
    (_e0, _), ledger = _append(tmp_path, _promotion_record(tmp_path, "a"))
    append_baseline_history_entry(
        ledger, _promotion_record(tmp_path, "b"), created_at_utc=GOLDEN_TIMESTAMP
    )
    entries = load_ledger(ledger)
    entries[1]["entry_index"] = 5
    entries[1]["entry_hash"] = _recompute(entries[1])
    ledger.write_text("\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n")
    report = verify_baseline_history(ledger)
    assert report["status"] == STATUS_INVALID
    assert any("out-of-order entry_index" in i for i in report["issues"])


def test_duplicate_entry_hash_is_detected(tmp_path):
    (e0, _), ledger = _append(tmp_path, _promotion_record(tmp_path, "a"))
    entries = load_ledger(ledger)
    entries.append(dict(entries[0]))  # exact duplicate line
    ledger.write_text("\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n")
    report = verify_baseline_history(ledger)
    assert report["status"] == STATUS_INVALID
    assert any("duplicate entry_hash" in i for i in report["issues"])


def test_invalid_jsonl_is_detected(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("{not valid json}\n", encoding="utf-8")
    report = verify_baseline_history(ledger)
    assert report["status"] == STATUS_INVALID
    assert any("invalid JSONL" in i for i in report["issues"])


def test_missing_required_field_is_detected(tmp_path):
    (_e, _), ledger = _append(tmp_path, _promotion_record(tmp_path, "a"))
    entries = load_ledger(ledger)
    del entries[0]["new_baseline_fingerprint"]
    ledger.write_text(json.dumps(entries[0], sort_keys=True) + "\n")
    report = verify_baseline_history(ledger)
    assert report["status"] == STATUS_INVALID
    assert any("missing required field" in i for i in report["issues"])


# ------------------------------ safety + summary ----------------------------- #
def test_entries_contain_no_secrets_or_abs_paths(tmp_path):
    (entry, _), _ = _append(tmp_path, _promotion_record(tmp_path, "a"))
    blob = json.dumps(entry)
    for forbidden in (str(tmp_path), "/home", "JQUANTS_API_KEY", "EDINET_API_KEY", "x-api-key"):
        assert forbidden not in blob


def test_summarize_baseline_history(tmp_path):
    (_e, _), ledger = _append(tmp_path, _promotion_record(tmp_path, "first"))
    summary = summarize_baseline_history(ledger)
    assert summary["entry_count"] == 1
    assert summary["chain_status"] == STATUS_VALID
    assert summary["research_only"] is True
    assert summary["entries"][0]["reviewer_note"] == "first"


# ------------------------------ promotion wiring ----------------------------- #
def test_blocked_promotion_does_not_append(tmp_path):
    a = run_golden_synthetic_pipeline(tmp_path / "a")
    ledger = tmp_path / "ledger.jsonl"
    record, updated = promote_pipeline_baseline(
        a, tmp_path / "b.json", reviewer_note="n", require_approval=True, approved=False,
        fixed_timestamp=GOLDEN_TIMESTAMP, ledger_path=ledger,
    )
    assert updated is False
    assert not ledger.exists()
    assert record["ledger_append_status"] is None


def test_approved_promotion_appends_exactly_one_entry(tmp_path):
    a = run_golden_synthetic_pipeline(tmp_path / "a")
    ledger = tmp_path / "ledger.jsonl"
    record, updated = promote_pipeline_baseline(
        a, tmp_path / "b.json", reviewer_note="approved", require_approval=True, approved=True,
        fixed_timestamp=GOLDEN_TIMESTAMP, ledger_path=ledger,
    )
    assert updated is True
    assert record["ledger_append_status"] == APPEND_APPENDED
    assert record["appended_entry_hash"]
    assert len(load_ledger(ledger)) == 1
    assert verify_baseline_history(ledger)["status"] == STATUS_VALID


def test_promotion_record_includes_appended_entry_hash(tmp_path):
    a = run_golden_synthetic_pipeline(tmp_path / "a")
    ledger = tmp_path / "ledger.jsonl"
    record, _ = promote_pipeline_baseline(
        a, tmp_path / "b.json", reviewer_note="approved", require_approval=True, approved=True,
        fixed_timestamp=GOLDEN_TIMESTAMP, ledger_path=ledger,
    )
    entry = load_ledger(ledger)[0]
    assert record["appended_entry_hash"] == entry["entry_hash"]


def test_broken_ledger_blocks_promotion_atomically(tmp_path):
    a = run_golden_synthetic_pipeline(tmp_path / "a")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("{broken json\n", encoding="utf-8")  # corrupt chain
    baseline_path = tmp_path / "b.json"
    record, updated = promote_pipeline_baseline(
        a, baseline_path, reviewer_note="approved", require_approval=True, approved=True,
        fixed_timestamp=GOLDEN_TIMESTAMP, ledger_path=ledger,
    )
    assert updated is False  # whole promotion blocked
    assert record["status"] == "blocked_broken_ledger"
    assert not baseline_path.exists()  # baseline NOT written (no partial state)


def test_non_promoted_record_is_not_appended(tmp_path):
    record = _promotion_record(tmp_path, "x", status="blocked_approval_required")
    (entry, status), _ = _append(tmp_path, record)
    assert entry is None
    assert status == APPEND_SKIPPED_NOT_PROMOTED


def _recompute(entry):
    from jp_stock_analysis.modeling.baseline_history import _content_hash

    return _content_hash(entry)

import json
import os

import pytest

from jp_stock_analysis.modeling import baseline_transaction as t
from jp_stock_analysis.modeling.baseline_history import load_ledger, verify_baseline_history


def _payload(v):
    return json.dumps({"version": v}).encode()


def _record(data):
    return {
        "status": "promoted",
        "approved": True,
        "approval_required": True,
        "new_baseline_fingerprint": t._fingerprint(json.loads(data)),
        "reviewer_note": "reviewed",
        "synthetic": True,
    }


def test_promotion_and_recovery_after_failure(tmp_path, monkeypatch):
    b, ledger = tmp_path / "baseline.json", tmp_path / "ledger.jsonl"
    d1, d2 = _payload(1), _payload(2)
    t.replace_pair(b, ledger, d1, _record(d1), None)
    before = b.read_bytes(), ledger.read_bytes()
    replace = os.replace
    count = 0

    def fault(src, dst):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("injected")
        return replace(src, dst)

    monkeypatch.setattr(t.os, "replace", fault)
    with pytest.raises(OSError):
        t.replace_pair(
            b,
            ledger,
            d2,
            {**_record(d2), "previous_baseline_fingerprint": t._fingerprint(json.loads(d1))},
            None,
        )
    assert not t.marker(b).exists()
    monkeypatch.setattr(t.os, "replace", replace)
    assert (b.read_bytes(), ledger.read_bytes()) == before
    assert verify_baseline_history(load_ledger(ledger))["status"] == "valid"


def test_crash_after_baseline_replace_and_explicit_recovery(tmp_path):
    import subprocess
    import sys

    b, ledger = tmp_path / "baseline.json", tmp_path / "ledger.jsonl"
    d1 = _payload(1)
    t.replace_pair(b, ledger, d1, _record(d1), None)
    before = b.read_bytes(), ledger.read_bytes()
    program = """
import os,sys,json
from pathlib import Path
from jp_stock_analysis.modeling import baseline_transaction as t
b,l=map(Path,sys.argv[1:]); d=json.dumps({'version':2}).encode()
r={'status':'promoted','approved':True,'approval_required':True,'synthetic':True,
   'reviewer_note':'reviewed','new_baseline_fingerprint':t._fingerprint(json.loads(d)),
   'previous_baseline_fingerprint':t._fingerprint(json.loads(b.read_bytes()))}
old=os.replace
count=0
def terminate(src,dst):
 global count
 count+=1
 old(src,dst)
 if count==1: os._exit(73)
t.os.replace=terminate
t.replace_pair(b,l,d,r,None)
"""
    proc = subprocess.run([sys.executable, "-c", program, str(b), str(ledger)], timeout=5)
    assert proc.returncode == 73
    with pytest.raises(ValueError, match="recovery required"):
        t.snapshot(b, ledger)
    t.recover(b, ledger)
    assert (b.read_bytes(), ledger.read_bytes()) == before


def test_concurrent_promotion_cannot_enter(tmp_path):
    import subprocess
    import sys

    b, ledger = tmp_path / "baseline.json", tmp_path / "ledger.jsonl"
    program = """
import sys,json
from pathlib import Path
from jp_stock_analysis.modeling import baseline_transaction as t
b,l=map(Path,sys.argv[1:]); d=json.dumps({'version':1}).encode()
r={'status':'promoted','approved':True,'approval_required':True,'synthetic':True,'reviewer_note':'reviewed','new_baseline_fingerprint':t._fingerprint(json.loads(d))}
try: t.replace_pair(b,l,d,r,None)
except BlockingIOError: sys.exit(42)
"""
    with t.locked(b, ledger):
        proc = subprocess.run([sys.executable, "-c", program, str(b), str(ledger)], timeout=5)
    assert proc.returncode == 42
    assert not b.exists() and not ledger.exists()


def test_official_direct_write_and_append_refused(tmp_path, monkeypatch):
    from jp_stock_analysis.modeling.baseline_history import append_baseline_history_entry
    from jp_stock_analysis.modeling.regression_baseline import write_baseline

    b, ledger = tmp_path / "baseline.json", tmp_path / "ledger.jsonl"
    monkeypatch.setattr(t, "OFFICIAL_BASELINE", b)
    monkeypatch.setattr(t, "OFFICIAL_LEDGER", ledger)
    with pytest.raises(ValueError, match="transactional"):
        write_baseline({"test": 1}, b)
    with pytest.raises(ValueError, match="transactional"):
        append_baseline_history_entry(ledger, _record(_payload(1)))
    assert not b.exists() and not ledger.exists()


def test_recovery_refuses_unknown_state(tmp_path, monkeypatch):
    b, ledger = tmp_path / "baseline.json", tmp_path / "ledger.jsonl"
    d1, d2 = _payload(1), _payload(2)
    t.replace_pair(b, ledger, d1, _record(d1), None)
    previous = t._fingerprint(json.loads(d1))
    original_replace = os.replace
    count = 0

    def fault(src, dst):
        nonlocal count
        count += 1
        if count >= 2:
            raise OSError("injected")
        return original_replace(src, dst)

    monkeypatch.setattr(t.os, "replace", fault)
    with pytest.raises(OSError):
        t.replace_pair(
            b, ledger, d2, {**_record(d2), "previous_baseline_fingerprint": previous}, None
        )
    monkeypatch.setattr(t.os, "replace", original_replace)
    b.write_bytes(b"unknown external modification")
    with pytest.raises(ValueError, match="unknown target state"):
        t.recover(b, ledger)
    assert b.read_bytes() == b"unknown external modification"
    assert t.marker(b).exists()


def test_snapshot_refuses_concurrent_promotion(tmp_path):
    import concurrent.futures
    import threading

    b, ledger = tmp_path / "baseline.json", tmp_path / "ledger.jsonl"
    d1 = _payload(1)
    t.replace_pair(b, ledger, d1, _record(d1), None)
    ready, release = threading.Event(), threading.Event()

    def hold_writer():
        with t.locked(b, ledger):
            ready.set()
            assert release.wait(timeout=3)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(hold_writer)
        assert ready.wait(timeout=3)
        try:
            with pytest.raises(BlockingIOError):
                t.snapshot(b, ledger)
        finally:
            release.set()
        future.result(timeout=3)
    assert t.snapshot(b, ledger)[0] == d1


def test_audit_during_writer_is_never_mixed(tmp_path, monkeypatch):
    import concurrent.futures
    import threading

    from jp_stock_analysis.modeling.audit_bundle import export_audit_bundle

    baseline, ledger = tmp_path / "baseline.json", tmp_path / "ledger.jsonl"
    first, second = _payload(1), _payload(2)
    t.replace_pair(baseline, ledger, first, _record(first), None)
    ready, release = threading.Event(), threading.Event()
    original_stage = t._stage

    def paused_stage(path, data):
        if path == baseline.resolve():
            ready.set()
            assert release.wait(timeout=3)
        return original_stage(path, data)

    monkeypatch.setattr(t, "_stage", paused_stage)
    record = {**_record(second), "previous_baseline_fingerprint": t._fingerprint(json.loads(first))}
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(t.replace_pair, baseline, ledger, second, record, None)
        assert ready.wait(timeout=3)
        try:
            with pytest.raises(BlockingIOError):
                export_audit_bundle(tmp_path / "bundle", baseline_path=baseline, ledger_path=ledger)
        finally:
            release.set()
        future.result(timeout=3)
    baseline_bytes, ledger_bytes = t.snapshot(baseline, ledger)
    assert baseline_bytes == second
    assert json.loads(ledger_bytes.splitlines()[-1])["new_baseline_fingerprint"] == t._fingerprint(
        json.loads(second)
    )


def test_direct_update_cli_refuses_official_path(tmp_path, monkeypatch):
    from jp_stock_analysis.cli import main

    baseline = tmp_path / "official.json"
    baseline.write_text('{"existing": true}')
    monkeypatch.setattr(t, "OFFICIAL_BASELINE", baseline)
    before = baseline.read_bytes()
    rc = main(
        [
            "check-pipeline-regression",
            "--synthetic",
            "--update-baseline",
            "--baseline-path",
            str(baseline),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc != 0
    assert baseline.read_bytes() == before


def test_failed_ledger_replace_returns_failure_and_restores(tmp_path, monkeypatch):
    from pathlib import Path

    from jp_stock_analysis.modeling.run_compare import promote_pipeline_baseline

    run = tmp_path / "run"
    run.mkdir()
    (run / "metric.json").write_text('{"value": 1}')
    baseline, ledger = tmp_path / "baseline.json", tmp_path / "ledger.jsonl"
    original = os.replace

    def fault(src, dst):
        if Path(dst) == ledger:
            raise OSError("injected ledger failure")
        return original(src, dst)

    monkeypatch.setattr(t.os, "replace", fault)
    record, updated = promote_pipeline_baseline(
        run, baseline, reviewer_note="reviewed", approved=True, ledger_path=ledger
    )
    assert not updated and record["status"] != "promoted"
    assert not baseline.exists() and not ledger.exists()
    assert not t.marker(baseline).exists() and not t.marker(ledger).exists()


def test_failed_baseline_stage_preserves_old_pair(tmp_path, monkeypatch):
    baseline, ledger = tmp_path / "baseline.json", tmp_path / "ledger.jsonl"
    first, second = _payload(1), _payload(2)
    t.replace_pair(baseline, ledger, first, _record(first), None)
    before = baseline.read_bytes(), ledger.read_bytes()
    original = t._stage

    def fail_first(path, data):
        if path == baseline:
            raise OSError("injected baseline stage failure")
        return original(path, data)

    monkeypatch.setattr(t, "_stage", fail_first)
    record = {**_record(second), "previous_baseline_fingerprint": t._fingerprint(json.loads(first))}
    with pytest.raises(OSError):
        t.replace_pair(baseline, ledger, second, record, None)
    assert (baseline.read_bytes(), ledger.read_bytes()) == before
    assert not t.marker(baseline).exists() and not t.marker(ledger).exists()


def test_promotion_appends_once_and_matches_head(tmp_path):
    from jp_stock_analysis.modeling.run_compare import promote_pipeline_baseline

    run = tmp_path / "run"
    run.mkdir()
    (run / "metric.json").write_text('{"value": 1}')
    baseline, ledger = tmp_path / "baseline.json", tmp_path / "ledger.jsonl"
    record, updated = promote_pipeline_baseline(
        run, baseline, reviewer_note="reviewed", approved=True, ledger_path=ledger
    )
    assert updated and record["status"] == "promoted"
    contents = load_ledger(ledger)
    assert len(contents) == 1
    assert contents[-1]["parent_hash"] == "GENESIS"
    assert contents[-1]["entry_hash"] == record["appended_entry_hash"]
    assert contents[-1]["new_baseline_fingerprint"] == t._fingerprint(
        json.loads(baseline.read_bytes())
    )
    assert verify_baseline_history(contents)["status"] == "valid"
    _, repeated = promote_pipeline_baseline(
        run, baseline, reviewer_note="reviewed", approved=True, ledger_path=ledger
    )
    assert not repeated and len(load_ledger(ledger)) == 1

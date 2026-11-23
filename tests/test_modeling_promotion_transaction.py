"""Failure and interruption coverage for baseline promotions."""

from jp_stock_analysis.cli import main


def test_cli_requires_approval_without_legacy_flag(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "metric.json").write_text('{"value": 1}')
    baseline = tmp_path / "baseline.json"
    rc = main(
        [
            "promote-pipeline-baseline",
            "--from-run",
            str(run),
            "--baseline-path",
            str(baseline),
            "--reviewer-note",
            "reviewed",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert not baseline.exists()


def test_fresh_check_uses_copied_baseline(tmp_path, monkeypatch):
    import json
    import shutil
    from pathlib import Path

    from jp_stock_analysis.modeling import regression_baseline
    from jp_stock_analysis.modeling.audit_bundle import export_audit_bundle

    fixture = Path("tests/fixtures/pipeline_baseline")
    baseline = tmp_path / "baseline.json"
    ledger = tmp_path / "ledger.jsonl"
    shutil.copyfile(fixture / "golden_pipeline_baseline.json", baseline)
    shutil.copyfile(fixture / "baseline_history.jsonl", ledger)
    original = json.loads(baseline.read_text())
    real = regression_baseline.run_golden_synthetic_pipeline

    def change_source(*args, **kwargs):
        payload = dict(original)
        payload["artifact_count"] = 123456
        baseline.write_text(json.dumps(payload))
        return real(*args, **kwargs)

    seen = []
    original_compare = regression_baseline.compare_to_baseline

    def capture(run, reference, **kwargs):
        seen.append(reference)
        return original_compare(run, reference, **kwargs)

    monkeypatch.setattr(regression_baseline, "run_golden_synthetic_pipeline", change_source)
    monkeypatch.setattr(regression_baseline, "compare_to_baseline", capture)
    export_audit_bundle(
        tmp_path / "bundle",
        baseline_path=baseline,
        ledger_path=ledger,
        synthetic=True,
        include_fresh_checks=True,
    )
    assert seen == [original]


def _git(args, cwd):
    import os
    import subprocess

    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        text=True,
        capture_output=True,
        env={"PATH": os.defpath, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
        timeout=10,
    ).stdout.strip()


def _paired_repos(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from jp_stock_analysis.modeling import baseline_transaction as t
    from jp_stock_analysis.modeling.baseline_history import _content_hash, _entry_content

    first, second = tmp_path / "repo", tmp_path / "other"
    first.mkdir()
    _git(["init", "-q"], first)
    _git(
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
        ],
        first,
    )
    _git(["worktree", "add", "-q", "-b", "other", str(second)], first)
    assert _git(["rev-parse", "--path-format=absolute", "--git-common-dir"], first) == _git(
        ["rev-parse", "--path-format=absolute", "--git-common-dir"], second
    )
    rel = Path("tests/fixtures/pipeline_baseline")
    for version, root in enumerate((first, second), 1):
        directory = root / rel
        directory.mkdir(parents=True)
        baseline = {"version": version}
        (directory / "golden_pipeline_baseline.json").write_text(json.dumps(baseline))
        record = {
            "status": "promoted",
            "approved": True,
            "approval_required": True,
            "new_baseline_fingerprint": t._fingerprint(baseline),
            "synthetic": True,
        }
        content = _entry_content(record, entry_index=0, parent_hash="GENESIS", created_at_utc=None)
        entry = {**content, "entry_hash": _content_hash(content)}
        (directory / "baseline_history.jsonl").write_text(json.dumps(entry) + "\n")
    monkeypatch.setattr(t, "REPO_ROOT", first)
    monkeypatch.setattr(t, "OFFICIAL_BASELINE", first / rel / "golden_pipeline_baseline.json")
    monkeypatch.setattr(t, "OFFICIAL_LEDGER", first / rel / "baseline_history.jsonl")
    return first / rel, second / rel


def test_sibling_worktrees_are_protected_and_paired(tmp_path, monkeypatch):
    import json
    import os

    import pytest

    from jp_stock_analysis.modeling import baseline_transaction as t
    from jp_stock_analysis.modeling.baseline_history import load_ledger
    from jp_stock_analysis.modeling.regression_baseline import load_baseline, write_baseline

    first, second = _paired_repos(tmp_path, monkeypatch)
    baseline = second / "golden_pipeline_baseline.json"
    ledger = second / "baseline_history.jsonl"
    assert t.official_baseline(first / baseline.name)
    assert t.official_baseline(baseline) and t.official_ledger(ledger)
    assert t.corresponding_ledger(baseline) == ledger
    assert load_baseline(baseline) == {"version": 2}
    assert load_ledger(ledger)[0]["new_baseline_fingerprint"] == t._fingerprint({"version": 2})
    old = baseline.read_bytes()
    with pytest.raises(ValueError, match="transactional"):
        write_baseline({"version": 9}, baseline)
    alias = tmp_path / "alias.json"
    alias.symlink_to(baseline)
    with pytest.raises(ValueError, match="transactional"):
        write_baseline({"version": 9}, alias)
    monkeypatch.chdir(second)
    with pytest.raises(ValueError, match="transactional"):
        write_baseline({"version": 9}, os.path.relpath(baseline, second))
    assert baseline.read_bytes() == old
    with pytest.raises(ValueError, match="different worktrees"):
        t.snapshot(baseline, first / ledger.name)
    with t.locked(alias):
        with pytest.raises(BlockingIOError):
            with t.locked(baseline):
                pass
    data, history = t.snapshot(baseline, ledger)
    assert json.loads(data) == {"version": 2} and json.loads(history)["entry_index"] == 0
    monkeypatch.setenv("GIT_DIR", "/nonexistent/forged")
    monkeypatch.setenv("GIT_WORK_TREE", "/nonexistent/forged")
    assert t.official_baseline(baseline)
    scratch = tmp_path / "research.json"
    write_baseline({"version": 3}, scratch)
    assert load_baseline(scratch) == {"version": 3}


def test_sibling_promotion_and_audit_snapshot(tmp_path, monkeypatch):
    import json

    import pytest

    from jp_stock_analysis.modeling import baseline_transaction as t
    from jp_stock_analysis.modeling.audit_bundle import export_audit_bundle, verify_audit_bundle
    from jp_stock_analysis.modeling.baseline_history import load_ledger, verify_baseline_history
    from jp_stock_analysis.modeling.regression_baseline import load_baseline
    from jp_stock_analysis.modeling.run_compare import promote_pipeline_baseline

    first, second = _paired_repos(tmp_path, monkeypatch)
    baseline = second / "golden_pipeline_baseline.json"
    ledger = second / "baseline_history.jsonl"
    first_before = (first / baseline.name).read_bytes(), (first / ledger.name).read_bytes()
    run = tmp_path / "run"
    run.mkdir()
    (run / "metric.json").write_text('{"value": 7}')
    record, updated = promote_pipeline_baseline(
        run,
        baseline,
        reviewer_note="synthetic test",
        approved=True,
        ledger_path=first / ledger.name,
    )
    assert not updated and record["status"] != "promoted"
    record, updated = promote_pipeline_baseline(
        run,
        baseline,
        reviewer_note="synthetic test",
        approved=True,
        ledger_path=ledger,
    )
    assert updated and record["status"] == "promoted"
    entries = load_ledger(ledger)
    assert len(entries) == 2 and entries[-1]["parent_hash"] == entries[0]["entry_hash"]
    assert entries[-1]["new_baseline_fingerprint"] == t._fingerprint(load_baseline(baseline))
    assert verify_baseline_history(entries)["status"] == "valid"
    assert (first / baseline.name).read_bytes() == first_before[0]
    assert (first / ledger.name).read_bytes() == first_before[1]
    bundle = tmp_path / "bundle"
    manifest = export_audit_bundle(
        bundle, baseline_path=baseline, ledger_path=ledger, include_fresh_checks=True
    )
    assert manifest["baseline_matches_ledger_head"]
    assert (bundle / "reports/pipeline_regression_report.json").is_file()
    assert verify_audit_bundle(bundle)["status"] == "valid"
    assert json.loads(
        (bundle / "baseline/golden_pipeline_baseline.json").read_text()
    ) == load_baseline(baseline)
    with pytest.raises(ValueError, match="different worktrees"):
        export_audit_bundle(
            tmp_path / "mixed", baseline_path=baseline, ledger_path=first / ledger.name
        )
    assert not t.marker(baseline).exists() and not t.marker(ledger).exists()


def test_sibling_journal_recovery(tmp_path, monkeypatch):
    import json
    import os

    import pytest

    from jp_stock_analysis.modeling import baseline_transaction as t

    _, second = _paired_repos(tmp_path, monkeypatch)
    baseline = second / "golden_pipeline_baseline.json"
    ledger = second / "baseline_history.jsonl"
    before = baseline.read_bytes(), ledger.read_bytes()
    new = json.dumps({"version": 3}).encode()
    record = {
        "status": "promoted",
        "approved": True,
        "approval_required": True,
        "reviewer_note": "synthetic",
        "synthetic": True,
        "previous_baseline_fingerprint": t._fingerprint({"version": 2}),
        "new_baseline_fingerprint": t._fingerprint({"version": 3}),
    }
    real = os.replace
    calls = 0

    def fault(src, dst):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError("injected failure")
        return real(src, dst)

    monkeypatch.setattr(t.os, "replace", fault)
    with pytest.raises(OSError):
        t.replace_pair(baseline, ledger, new, record, None)
    assert t.marker(baseline).is_file() and t.marker(ledger).is_file()
    with pytest.raises(ValueError, match="recovery required"):
        t.snapshot(baseline, ledger)
    monkeypatch.setattr(t.os, "replace", real)
    t.recover(baseline, ledger)
    assert (baseline.read_bytes(), ledger.read_bytes()) == before
    assert not t.marker(baseline).exists() and not t.marker(ledger).exists()


def test_current_worktree_fails_closed_when_git_identity_unavailable(tmp_path, monkeypatch):
    import pytest

    from jp_stock_analysis.modeling import baseline_transaction as t

    first, _ = _paired_repos(tmp_path, monkeypatch)

    def unavailable(_):
        raise ValueError("identity unavailable")

    monkeypatch.setattr(t, "_git_identity", unavailable)
    with pytest.raises(ValueError, match="identity unavailable"):
        t.official_baseline(first / "golden_pipeline_baseline.json")

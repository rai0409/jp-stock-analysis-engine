"""Cooperating baseline writers and readers: locking and crash evidence.

This protocol is local to a host and requires all application participants to
use it. It cannot prevent independent processes from editing files directly.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_DIR = REPO_ROOT / "tests/fixtures/pipeline_baseline"
OFFICIAL_BASELINE = OFFICIAL_DIR / "golden_pipeline_baseline.json"
OFFICIAL_LEDGER = OFFICIAL_DIR / "baseline_history.jsonl"


class TransactionFailure(OSError):
    """A failed promotion stage; persistent markers signal recovery required."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__(f"baseline transaction failed during {stage}")


def _same(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    try:
        return os.path.samefile(a, b)
    except FileNotFoundError:
        return False


def _git_identity(root: Path) -> tuple[Path, Path]:
    """Obtain root and common directory without inherited Git overrides."""
    env = {"PATH": os.defpath, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "rev-parse",
                "--path-format=absolute",
                "--show-toplevel",
                "--git-common-dir",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=3,
        )
        lines = proc.stdout.splitlines()
        if len(lines) != 2:
            raise ValueError("unexpected Git identity output")
        return Path(lines[0]).resolve(), Path(lines[1]).resolve()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("unable to verify worktree identity") from exc


def _managed_dir(path: str | Path, expected: Path) -> Path | None:
    candidate = Path(path).resolve()
    if candidate == expected.resolve():
        expected_dir = REPO_ROOT / "tests/fixtures/pipeline_baseline"
        if expected.parent == expected_dir and expected.name in {
            "golden_pipeline_baseline.json",
            "baseline_history.jsonl",
        }:
            trusted_root, _ = _git_identity(REPO_ROOT)
            if trusted_root != REPO_ROOT.resolve():
                raise ValueError("cannot verify application worktree")
        return expected.parent.resolve()
    relative = Path("tests/fixtures/pipeline_baseline") / expected.name
    parts = candidate.parts
    if len(parts) < len(relative.parts) or tuple(parts[-len(relative.parts) :]) != relative.parts:
        return expected.parent.resolve() if _same(candidate, expected) else None
    root = candidate.parents[len(relative.parts) - 1]
    if not (root / ".git").exists():
        raise ValueError("cannot verify potentially managed baseline worktree")
    actual_root, common = _git_identity(root)
    trusted_root, trusted_common = _git_identity(REPO_ROOT)
    if trusted_root != REPO_ROOT.resolve():
        raise ValueError("cannot verify application worktree")
    if actual_root != root:
        raise ValueError("candidate is not at worktree root")
    if common == trusted_common:
        if _same(candidate, expected):
            raise ValueError("hardlinked managed worktrees cannot be isolated")
        return candidate.parent
    return expected.parent.resolve() if _same(candidate, expected) else None


def official_baseline(path: str | Path) -> bool:
    return _managed_dir(path, OFFICIAL_BASELINE) is not None


def official_ledger(path: str | Path) -> bool:
    return _managed_dir(path, OFFICIAL_LEDGER) is not None


def corresponding_ledger(path: str | Path) -> Path:
    directory = _managed_dir(path, OFFICIAL_BASELINE)
    if directory is None:
        raise ValueError("not a managed baseline")
    return directory / OFFICIAL_LEDGER.name


def corresponding_baseline(path: str | Path) -> Path:
    directory = _managed_dir(path, OFFICIAL_LEDGER)
    if directory is None:
        raise ValueError("not a managed ledger")
    return directory / OFFICIAL_BASELINE.name


def validate_pair(baseline: Path, ledger: Path) -> bool:
    base_managed = official_baseline(baseline)
    ledger_managed = official_ledger(ledger)
    if base_managed != ledger_managed:
        raise ValueError("managed baseline and ledger must be used together")
    if base_managed and corresponding_ledger(baseline).resolve() != ledger.resolve():
        raise ValueError("baseline and ledger belong to different worktrees")
    return base_managed


def _canonical_resource(path: Path) -> Path:
    directory = _managed_dir(path, OFFICIAL_BASELINE)
    if directory is not None:
        return directory / OFFICIAL_BASELINE.name
    directory = _managed_dir(path, OFFICIAL_LEDGER)
    if directory is not None:
        return directory / OFFICIAL_LEDGER.name
    return path.resolve()


def marker(path: Path) -> Path:
    path = _canonical_resource(path)
    return path.with_name("." + path.name + ".promotion-journal")


@contextlib.contextmanager
def locked(*paths: Path) -> Iterator[None]:
    # All cooperating users and processes must resolve to the same lock inode.
    lockdir = Path("/tmp/jp-stock-baseline-locks")
    try:
        lockdir.mkdir(mode=0o1777)
        os.chmod(lockdir, 0o1777)
    except FileExistsError:
        pass
    mode = lockdir.stat().st_mode
    if lockdir.is_symlink() or mode & 0o1777 != 0o1777:
        raise OSError("unsafe lock directory")
    handles = []
    try:
        for name in sorted({str(_canonical_resource(p)) for p in paths}):
            name_hash = hashlib.sha256(name.encode()).hexdigest()
            lockfile = lockdir / name_hash
            try:
                fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o666)
                os.fchmod(fd, 0o666)
            except FileExistsError:
                fd = os.open(lockfile, os.O_RDWR | os.O_NOFOLLOW)
            handle = os.fdopen(fd, "r+b")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _stage(path: Path, data: bytes) -> Path:
    fd, filename = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        if path.exists():
            os.fchmod(fd, path.stat().st_mode & 0o777)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        Path(filename).unlink(missing_ok=True)
        raise
    return Path(filename)


def _digest(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def _raw(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def check_clear(*paths: Path) -> None:
    if any(marker(p).exists() or marker(p).is_symlink() for p in paths):
        raise ValueError("incomplete baseline promotion; recovery required")


def _parse_ledger(data: bytes) -> list[dict]:
    from jp_stock_analysis.modeling.baseline_history import verify_baseline_history

    entries = [json.loads(x) for x in data.decode().splitlines() if x.strip()]
    if verify_baseline_history(entries)["status"] != "valid":
        raise ValueError("broken baseline ledger")
    return entries


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def snapshot(baseline: Path, ledger: Path, *, strict: bool = True) -> tuple[bytes, bytes]:
    baseline, ledger = _canonical_resource(baseline), _canonical_resource(ledger)
    managed = validate_pair(baseline, ledger)
    with locked(baseline, ledger):
        check_clear(baseline, ledger)
        baseline_data, ledger_data = baseline.read_bytes(), ledger.read_bytes()
        if strict or managed:
            entries = _parse_ledger(ledger_data)
            if not entries or entries[-1]["new_baseline_fingerprint"] != _fingerprint(
                json.loads(baseline_data)
            ):
                raise ValueError("baseline and ledger head mismatch")
        return baseline_data, ledger_data


def replace_pair(
    baseline: Path, ledger: Path, new_baseline: bytes, record: dict, timestamp: str | None
) -> dict:
    """Replace both files under locks; leave journals on any uncertain outcome."""
    from jp_stock_analysis.modeling.baseline_history import (
        GENESIS_PARENT,
        _content_hash,
        _entry_content,
    )

    baseline, ledger = _canonical_resource(baseline), _canonical_resource(ledger)
    if baseline == ledger:
        raise ValueError("baseline and ledger must differ")
    validate_pair(baseline, ledger)
    baseline.parent.mkdir(parents=True, exist_ok=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with locked(baseline, ledger):
        check_clear(baseline, ledger)
        old_b, old_l = _raw(baseline), _raw(ledger)
        entries = _parse_ledger(old_l or b"")
        if official_baseline(baseline) and (not official_ledger(ledger) or not entries):
            raise ValueError("official promotion requires existing valid ledger")
        if entries and (
            old_b is None
            or entries[-1]["new_baseline_fingerprint"] != _fingerprint(json.loads(old_b))
        ):
            raise ValueError("baseline and ledger head mismatch")
        old_fp = _fingerprint(json.loads(old_b)) if old_b is not None else None
        if record.get("previous_baseline_fingerprint") != old_fp:
            raise ValueError("baseline changed during promotion or previous baseline mismatch")
        new_fp = _fingerprint(json.loads(new_baseline))
        if record.get("new_baseline_fingerprint") != new_fp:
            raise ValueError("record fingerprint does not match new baseline")
        if entries and entries[-1]["new_baseline_fingerprint"] == new_fp:
            raise ValueError("baseline already promoted")
        content = _entry_content(
            record,
            entry_index=len(entries),
            parent_hash=entries[-1]["entry_hash"] if entries else GENESIS_PARENT,
            created_at_utc=timestamp,
        )
        entry = {**content, "entry_hash": _content_hash(content)}
        new_l = (old_l or b"") + (
            json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode()
        _parse_ledger(new_l)
        before = [_digest(old_b), _digest(old_l)]
        after = [_digest(new_baseline), _digest(new_l)]
        payload = {
            "version": 1,
            "paths": [str(baseline), str(ledger)],
            "before": before,
            "after": after,
            "old": [
                base64.b64encode(x).decode() if x is not None else None for x in (old_b, old_l)
            ],
        }
        journal = (json.dumps(payload, sort_keys=True) + "\n").encode()
        staged = []
        stage = "baseline_write"
        try:
            staged.append(_stage(baseline, new_baseline))
            stage = "ledger_write"
            staged.append(_stage(ledger, new_l))
            stage = "journal_write"
            for p in (baseline, ledger):
                fd = os.open(marker(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "wb") as f:
                    f.write(journal)
                    f.flush()
                    os.fsync(f.fileno())
                _fsync_dir(p.parent)
            for index, (dst, src) in enumerate(zip((baseline, ledger), staged, strict=True)):
                stage = "baseline_write" if index == 0 else "ledger_write"
                os.replace(src, dst)
                _fsync_dir(dst.parent)
            stage = "verification"
            if [_digest(_raw(baseline)), _digest(_raw(ledger))] != after:
                raise OSError("post-write verification failed")
            stage = "journal_cleanup"
            for p in (baseline, ledger):
                marker(p).unlink()
                _fsync_dir(p.parent)
            return entry
        except Exception as exc:
            if marker(baseline).exists() or marker(ledger).exists():
                try:
                    _recover_locked(baseline, ledger)
                except (OSError, ValueError, KeyError, TypeError):
                    pass  # Retain journal and fail closed when recovery is uncertain.
            raise TransactionFailure(stage) from exc
        finally:
            for p in staged:
                p.unlink(missing_ok=True)


def recover(baseline: Path, ledger: Path) -> None:
    """Explicitly restore verified pre-transaction bytes; never guess states."""
    baseline, ledger = _canonical_resource(baseline), _canonical_resource(ledger)
    validate_pair(baseline, ledger)
    with locked(baseline, ledger):
        _recover_locked(baseline, ledger)


def _recover_locked(baseline: Path, ledger: Path) -> None:
    markers = [marker(p) for p in (baseline, ledger)]
    journals = [m.read_bytes() for m in markers if m.exists()]
    if not journals or any(j != journals[0] for j in journals):
        raise ValueError("journal missing or inconsistent")
    data = json.loads(journals[0])
    if data.get("version") != 1 or data.get("paths") != [str(baseline), str(ledger)]:
        raise ValueError("journal target mismatch")
    old = [base64.b64decode(v, validate=True) if v is not None else None for v in data["old"]]
    if [_digest(x) for x in old] != data["before"]:
        raise ValueError("journal backup corrupt")
    current = [_digest(_raw(p)) for p in (baseline, ledger)]
    if any(
        cur not in (before, after)
        for cur, before, after in zip(current, data["before"], data["after"], strict=True)
    ):
        raise ValueError("unknown target state; manual recovery required")
    for p, old_data in zip((baseline, ledger), old, strict=True):
        if old_data is None:
            p.unlink(missing_ok=True)
        else:
            os.replace(_stage(p, old_data), p)
        _fsync_dir(p.parent)
    for m in markers:
        m.unlink(missing_ok=True)
        _fsync_dir(m.parent)

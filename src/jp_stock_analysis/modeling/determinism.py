"""Canonicalization + artifact-tree comparison for the determinism gate.

Verifies reproducibility by comparing two pipeline output trees. It canonicalizes
**only declared volatile fields** (timestamps, run ids, absolute paths) — it never
ignores numeric differences, changed metrics, or missing/extra artifacts. A
difference in any real value or any file's presence is reported as a difference.

Research-only tooling; the determinism gate checks reproducibility, not model
validity.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SENTINEL = "<VOLATILE>"

# Field *names* whose values are reproducibly volatile and may be canonicalized.
DEFAULT_VOLATILE_KEYS = (
    "created_at_utc",
    "started_at_utc",
    "finished_at_utc",
    "elapsed_seconds",
    "absolute_path",
    "run_directory",
    "output_dir",
)

VERDICT_IDENTICAL = "identical"
VERDICT_DIFFERENT = "different"
VERDICT_ONLY_IN_A = "only_in_a"
VERDICT_ONLY_IN_B = "only_in_b"
VERDICT_UNREADABLE = "unreadable"

_TEXT_SUFFIXES = {".md", ".csv", ".txt"}


def canonicalize_json(obj: Any, *, volatile_keys: Sequence[str] = DEFAULT_VOLATILE_KEYS) -> Any:
    """Replace declared volatile field *values* with a sentinel; recurse."""
    keys = set(volatile_keys)
    if isinstance(obj, Mapping):
        return {
            k: (SENTINEL if k in keys else canonicalize_json(v, volatile_keys=volatile_keys))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [canonicalize_json(v, volatile_keys=volatile_keys) for v in obj]
    return obj


def canonicalize_text(text: str, *, volatile_values: Iterable[str] = ()) -> str:
    """Replace declared volatile literal substrings (run id, timestamp, paths)."""
    out = text
    for value in sorted((v for v in volatile_values if v), key=len, reverse=True):
        out = out.replace(value, SENTINEL)
    return out


def _list_files(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    if not base.is_dir():
        return {}
    return {
        str(p.relative_to(base)).replace("\\", "/"): p
        for p in sorted(base.rglob("*"))
        if p.is_file()
    }


def _compare_one(
    rel: str,
    a: Path | None,
    b: Path | None,
    *,
    volatile_keys: Sequence[str],
    volatile_values: Sequence[str],
) -> dict[str, Any]:
    if a is None:
        return {"path": rel, "verdict": VERDICT_ONLY_IN_B}
    if b is None:
        return {"path": rel, "verdict": VERDICT_ONLY_IN_A}

    suffix = Path(rel).suffix.lower()
    try:
        if suffix == ".json":
            ca = canonicalize_json(
                json.loads(a.read_text(encoding="utf-8")), volatile_keys=volatile_keys
            )
            cb = canonicalize_json(
                json.loads(b.read_text(encoding="utf-8")), volatile_keys=volatile_keys
            )
            same = json.dumps(ca, sort_keys=True) == json.dumps(cb, sort_keys=True)
        elif suffix in _TEXT_SUFFIXES:
            ta = canonicalize_text(a.read_text(encoding="utf-8"), volatile_values=volatile_values)
            tb = canonicalize_text(b.read_text(encoding="utf-8"), volatile_values=volatile_values)
            same = ta == tb
        else:
            same = a.read_bytes() == b.read_bytes()
    except (OSError, ValueError, json.JSONDecodeError):
        return {"path": rel, "verdict": VERDICT_UNREADABLE}
    return {"path": rel, "verdict": VERDICT_IDENTICAL if same else VERDICT_DIFFERENT}


def compare_artifact_trees(
    dir_a: str | Path,
    dir_b: str | Path,
    *,
    volatile_keys: Sequence[str] = DEFAULT_VOLATILE_KEYS,
    volatile_values: Sequence[str] = (),
) -> dict[str, Any]:
    """Compare two artifact trees; canonicalize only declared volatile fields."""
    files_a = _list_files(dir_a)
    files_b = _list_files(dir_b)
    rel_paths = sorted(set(files_a) | set(files_b))
    entries = [
        _compare_one(
            rel,
            files_a.get(rel),
            files_b.get(rel),
            volatile_keys=volatile_keys,
            volatile_values=volatile_values,
        )
        for rel in rel_paths
    ]
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
    identical = all(e["verdict"] == VERDICT_IDENTICAL for e in entries) and bool(entries)
    return {
        "overall": VERDICT_IDENTICAL if identical else VERDICT_DIFFERENT,
        "file_count": len(rel_paths),
        "counts": dict(sorted(counts.items())),
        "volatile_keys": list(volatile_keys),
        "ignored_volatile_values_count": len([v for v in volatile_values if v]),
        "entries": entries,
    }


def write_determinism_report(
    comparison: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Path]:
    """Write determinism_report.json / .md."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "determinism_report.json"
    payload = {
        "disclaimer": (
            "This output is for analytical and self-directed research purposes. It is "
            "not personalized financial advice. The determinism gate checks "
            "reproducibility, not model validity."
        ),
        "research_only": True,
        **comparison,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path = out_dir / "determinism_report.md"
    lines = [
        "# Pipeline Determinism Report",
        "",
        payload["disclaimer"],
        "",
        f"- Overall: **{comparison['overall'].upper()}**",
        f"- Files compared: {comparison['file_count']}",
        f"- Verdict counts: {comparison['counts']}",
        f"- Canonicalized volatile keys: {comparison['volatile_keys']}",
        "",
        "| artifact | verdict |",
        "| --- | --- |",
    ]
    for entry in comparison["entries"]:
        if entry["verdict"] != VERDICT_IDENTICAL:
            lines.append(f"| `{entry['path']}` | **{entry['verdict']}** |")
    if all(e["verdict"] == VERDICT_IDENTICAL for e in comparison["entries"]):
        lines.append("| _(all identical)_ | identical |")
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json_path": json_path, "markdown_path": md_path}


__all__ = [
    "DEFAULT_VOLATILE_KEYS",
    "SENTINEL",
    "VERDICT_DIFFERENT",
    "VERDICT_IDENTICAL",
    "VERDICT_ONLY_IN_A",
    "VERDICT_ONLY_IN_B",
    "canonicalize_json",
    "canonicalize_text",
    "compare_artifact_trees",
    "write_determinism_report",
]

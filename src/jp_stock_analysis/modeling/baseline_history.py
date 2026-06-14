"""Append-only, hash-chained baseline promotion ledger (research-only).

Every successful baseline promotion appends one immutable JSONL entry whose
``entry_hash`` is computed over the canonicalized entry content (excluding the
hash itself) and which references the previous entry's hash (``parent_hash``).
This makes the promotion history tamper-evident: a silent edit, a broken parent
link, an out-of-order index, or a duplicate hash is **detected, not hidden**.

It is an audit ledger. It does **not** prove model validity or market
performance; entries are synthetic-only unless generated from real-data runs. A
promotion means "approved reference", not "better model". Secrets are scrubbed and
committed entries carry no absolute paths or raw non-canonical timestamps.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jp_stock_analysis.modeling.audit import _canonical, _sha256, scrub_secrets

LEDGER_SCHEMA_VERSION = "baseline_history_v1"
GENESIS_PARENT = "GENESIS"  # stable parent_hash of the first (index 0) entry

RESEARCH_DISCLAIMER = (
    "This output is for analytical and self-directed research purposes. It is not "
    "personalized financial advice. The baseline history is an append-only, "
    "hash-chained audit ledger; it does not prove model validity or performance."
)

STATUS_VALID = "valid"
STATUS_INVALID = "invalid"

APPEND_APPENDED = "appended"
APPEND_SKIPPED_NOT_PROMOTED = "skipped_not_promoted"
APPEND_REFUSED_BROKEN_CHAIN = "refused_broken_chain"

# entry keys that carry the hash chain / non-content metadata; the entry_hash is
# computed over every key EXCEPT entry_hash.
_HASH_EXCLUDED_KEYS = ("entry_hash",)

_REQUIRED_FIELDS = (
    "ledger_schema_version",
    "entry_index",
    "parent_hash",
    "entry_hash",
    "promotion_record_fingerprint",
    "new_baseline_fingerprint",
    "approved",
    "synthetic",
)

_SECRET_KEY_MARKERS = ("api_key", "apikey", "token", "secret", "password", "credential")
_SECRET_VALUE_MARKERS = ("JQUANTS_API_KEY", "EDINET_API_KEY", "x-api-key")
_ABS_PATH_MARKERS = ("/tmp/", "/home/", "/Users/", "/var/")


def _content_hash(content: Mapping[str, Any]) -> str:
    """Deterministic hash over the entry content, excluding the hash field."""
    payload = {k: v for k, v in content.items() if k not in _HASH_EXCLUDED_KEYS}
    return _sha256(_canonical(payload).encode("utf-8"))


def _entry_content(
    promotion_record: Mapping[str, Any],
    *,
    entry_index: int,
    parent_hash: str,
    created_at_utc: str | None,
) -> dict[str, Any]:
    deltas = promotion_record.get("headline_metric_deltas") or []
    return {
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "entry_index": entry_index,
        "parent_hash": parent_hash,
        "promotion_record_fingerprint": _sha256(
            _canonical(dict(promotion_record)).encode("utf-8")
        ),
        "baseline_fingerprint": promotion_record.get("new_baseline_fingerprint"),
        "previous_baseline_fingerprint": promotion_record.get("previous_baseline_fingerprint"),
        "new_baseline_fingerprint": promotion_record.get("new_baseline_fingerprint"),
        "source_run_id": promotion_record.get("source_run_id"),
        "reviewer_note": promotion_record.get("reviewer_note", ""),
        "approved": bool(promotion_record.get("approved")),
        "approval_required": bool(promotion_record.get("approval_required")),
        "synthetic": bool(promotion_record.get("synthetic")),
        "research_only": True,
        "created_at_utc": created_at_utc,
        "headline_metric_delta_summary": {
            "count": len(deltas),
            "deltas": [
                {
                    "artifact": d.get("artifact"),
                    "metric": d.get("metric"),
                    "direction": d.get("direction"),
                }
                for d in deltas
            ],
        },
        "artifact_classification_counts": promotion_record.get(
            "artifact_classification_counts", {}
        ),
        "warnings": list(promotion_record.get("warnings", [])),
        "metadata": scrub_secrets(
            {"baseline_path": promotion_record.get("baseline_path")}
        ),
    }


def load_ledger(ledger_path: str | Path) -> list[dict[str, Any]]:
    """Parse a JSONL ledger into entries. Raises ValueError on malformed JSON."""
    path = Path(ledger_path)
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not raw.strip():
            continue
        try:
            entries.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_no + 1}: {exc}") from exc
    return entries


def append_baseline_history_entry(
    ledger_path: str | Path,
    promotion_record: Mapping[str, Any],
    *,
    created_at_utc: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Append a hash-chained entry from a successful promotion record.

    Returns ``(entry, status)``. Never appends for a non-promoted record, and
    refuses (without appending) if the existing chain is already broken.
    """
    if promotion_record.get("status") != "promoted":
        return None, APPEND_SKIPPED_NOT_PROMOTED

    path = Path(ledger_path)
    try:
        entries = load_ledger(path)
    except ValueError:
        return None, APPEND_REFUSED_BROKEN_CHAIN
    if entries:
        verification = verify_baseline_history(entries)
        if verification["status"] != STATUS_VALID:
            return None, APPEND_REFUSED_BROKEN_CHAIN

    parent_hash = entries[-1]["entry_hash"] if entries else GENESIS_PARENT
    content = _entry_content(
        promotion_record,
        entry_index=len(entries),
        parent_hash=parent_hash,
        created_at_utc=created_at_utc,
    )
    entry = {**content, "entry_hash": _content_hash(content)}

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return entry, APPEND_APPENDED


def _scan_unsafe(entry: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    for key in entry:
        if any(marker in str(key).lower() for marker in _SECRET_KEY_MARKERS):
            issues.append(f"secret-like field name: {key!r}")
    blob = _canonical(dict(entry))
    for marker in _SECRET_VALUE_MARKERS:
        if marker in blob:
            issues.append(f"secret-like value present: {marker}")
    for marker in _ABS_PATH_MARKERS:
        if marker in blob:
            issues.append(f"absolute path present: {marker}")
    return issues


def verify_baseline_history(
    ledger: str | Path | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify chain integrity. Returns a structured report; never raises on a bad
    chain — it reports the issues instead."""
    issues: list[str] = []
    if isinstance(ledger, (str, Path)):
        try:
            entries = load_ledger(ledger)
        except ValueError as exc:
            return {
                "disclaimer": RESEARCH_DISCLAIMER,
                "research_only": True,
                "status": STATUS_INVALID,
                "issues": [str(exc)],
                "entry_count": 0,
                "head_hash": None,
                "genesis_parent": GENESIS_PARENT,
                "first_entry": None,
                "last_entry": None,
            }
    else:
        entries = list(ledger)

    seen_hashes: set[str] = set()
    previous_hash = GENESIS_PARENT
    for position, entry in enumerate(entries):
        for field in _REQUIRED_FIELDS:
            if field not in entry:
                issues.append(f"entry {position}: missing required field {field!r}")
        if entry.get("entry_index") != position:
            issues.append(
                f"entry {position}: out-of-order entry_index {entry.get('entry_index')!r}"
            )
        if entry.get("parent_hash") != previous_hash:
            issues.append(
                f"entry {position}: parent_hash does not match previous entry hash"
            )
        recomputed = _content_hash(entry)
        if entry.get("entry_hash") != recomputed:
            issues.append(f"entry {position}: entry_hash mismatch (tampered content)")
        entry_hash = entry.get("entry_hash")
        if entry_hash in seen_hashes:
            issues.append(f"entry {position}: duplicate entry_hash")
        if isinstance(entry_hash, str):
            seen_hashes.add(entry_hash)
        issues.extend(f"entry {position}: {issue}" for issue in _scan_unsafe(entry))
        previous_hash = entry_hash if isinstance(entry_hash, str) else None

    return {
        "disclaimer": RESEARCH_DISCLAIMER,
        "research_only": True,
        "status": STATUS_VALID if not issues else STATUS_INVALID,
        "issues": issues,
        "entry_count": len(entries),
        "head_hash": entries[-1].get("entry_hash") if entries else None,
        "genesis_parent": GENESIS_PARENT,
        "first_entry": _entry_metadata(entries[0]) if entries else None,
        "last_entry": _entry_metadata(entries[-1]) if entries else None,
    }


def _entry_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_index": entry.get("entry_index"),
        "entry_hash": entry.get("entry_hash"),
        "parent_hash": entry.get("parent_hash"),
        "reviewer_note": entry.get("reviewer_note"),
        "new_baseline_fingerprint": entry.get("new_baseline_fingerprint"),
        "source_run_id": entry.get("source_run_id"),
        "synthetic": entry.get("synthetic"),
    }


def _short(value: Any) -> str:
    return str(value)[:12] + "…" if isinstance(value, str) and len(value) > 12 else str(value)


def summarize_baseline_history(
    ledger: str | Path | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compact, deterministic ledger summary (verification included)."""
    entries = (
        load_ledger(ledger) if isinstance(ledger, (str, Path)) else list(ledger)
    )
    verification = verify_baseline_history(entries)
    rows = [
        {
            "entry_index": entry.get("entry_index"),
            "entry_hash_short": _short(entry.get("entry_hash")),
            "parent_hash_short": _short(entry.get("parent_hash")),
            "reviewer_note": entry.get("reviewer_note"),
            "baseline_fingerprint_short": _short(entry.get("new_baseline_fingerprint")),
            "source_run_id": entry.get("source_run_id"),
            "synthetic": entry.get("synthetic"),
            "artifact_change_counts": entry.get("artifact_classification_counts", {}),
        }
        for entry in entries
    ]
    return {
        "disclaimer": RESEARCH_DISCLAIMER,
        "research_only": True,
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "entry_count": len(entries),
        "chain_status": verification["status"],
        "head_hash": verification["head_hash"],
        "genesis_parent": GENESIS_PARENT,
        "entries": rows,
    }


def write_history_outputs(
    summary: Mapping[str, Any], output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "baseline_history.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths = {"json_path": json_path}
    if write_markdown:
        md_path = out_dir / "baseline_history.md"
        lines = ["# Baseline Promotion History", "", str(summary["disclaimer"]), ""]
        lines += [
            f"- Entries: {summary['entry_count']}",
            f"- Chain status: **{summary['chain_status']}**",
            f"- Head hash: {_short(summary['head_hash'])}",
            "",
            "| # | entry | parent | note | baseline | synthetic |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in summary["entries"]:
            lines.append(
                f"| {row['entry_index']} | {row['entry_hash_short']} | "
                f"{row['parent_hash_short']} | {row['reviewer_note'] or '—'} | "
                f"{row['baseline_fingerprint_short']} | {row['synthetic']} |"
            )
        lines.append("")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


def write_verification_outputs(
    report: Mapping[str, Any], output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "baseline_lineage_verification.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths = {"json_path": json_path}
    if write_markdown:
        md_path = out_dir / "baseline_lineage_verification.md"
        lines = ["# Baseline Lineage Verification", "", str(report["disclaimer"]), ""]
        lines += [
            f"- Status: **{report['status']}**",
            f"- Entries: {report['entry_count']}",
            f"- Head hash: {_short(report['head_hash'])}",
            "",
        ]
        if report["issues"]:
            lines += ["## Issues", *[f"- {issue}" for issue in report["issues"]], ""]
        else:
            lines += ["_Chain intact: all entries hash-chained and untampered._", ""]
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


__all__ = [
    "APPEND_APPENDED",
    "APPEND_REFUSED_BROKEN_CHAIN",
    "APPEND_SKIPPED_NOT_PROMOTED",
    "GENESIS_PARENT",
    "LEDGER_SCHEMA_VERSION",
    "STATUS_INVALID",
    "STATUS_VALID",
    "append_baseline_history_entry",
    "load_ledger",
    "summarize_baseline_history",
    "verify_baseline_history",
    "write_history_outputs",
    "write_verification_outputs",
]

"""Pipeline run comparison + explicit baseline promotion (research-only).

``compare_runs`` diffs two arbitrary pipeline run directories (A vs B) into a
human-readable, neutral artifact/metric-delta report. ``promote_pipeline_baseline``
updates the approved golden baseline **only after explicit review/approval** and
writes an auditable provenance record.

This is diagnostic tooling. Metric deltas are **descriptive only** — they are
never labelled better/worse and imply no performance claim. A promoted baseline is
an *approved reference*, not an improvement. Synthetic outputs are not market
evidence; real-data interpretation still requires
``check-forward-readiness=ELIGIBLE`` and P0 strict no-look-ahead.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jp_stock_analysis.modeling.audit import scrub_secrets
from jp_stock_analysis.modeling.regression_baseline import (
    CLASS_CHANGED,
    CLASS_UNCHANGED,
    CLASS_UNREADABLE,
    CLASS_VOLATILE_ONLY,
    DEFAULT_EXCLUDE,
    GOLDEN_RUN_ID,
    _diff_details,
    _fingerprint,
    _list_tracked,
    _sha256,
    _volatile_values,
    capture_baseline,
    compare_to_baseline,
    load_baseline,
    write_baseline,
)

COMPARISON_SCHEMA_VERSION = "pipeline_run_comparison_v1"
PROMOTION_SCHEMA_VERSION = "baseline_promotion_v1"

RESEARCH_DISCLAIMER = (
    "This output is for analytical and self-directed research purposes. It is not "
    "personalized financial advice. Metric deltas are descriptive diagnostics only "
    "(never better/worse) and a promoted baseline is an approved reference, not a "
    "performance improvement."
)

CLASS_ONLY_IN_A = "only_in_a"
CLASS_ONLY_IN_B = "only_in_b"

DIR_INCREASED = "increased"
DIR_DECREASED = "decreased"
DIR_CHANGED = "changed"
DIR_UNCHANGED = "unchanged"

STATUS_IDENTICAL = "identical"
STATUS_CHANGED = "changed"
STATUS_MISSING_ARTIFACTS = "missing_artifacts"
STATUS_NEW_ARTIFACTS = "new_artifacts"
STATUS_UNREADABLE_ARTIFACTS = "unreadable_artifacts"

PROMOTION_BLOCKED_APPROVAL = "blocked_approval_required"
PROMOTION_BLOCKED_NOTE = "blocked_reviewer_note_required"
PROMOTION_PROMOTED = "promoted"


def _direction(a: Any, b: Any) -> str:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if b > a:
            return DIR_INCREASED
        if b < a:
            return DIR_DECREASED
        return DIR_UNCHANGED
    return DIR_CHANGED if a != b else DIR_UNCHANGED


def _csv_numeric_means(path: Path) -> dict[str, float]:
    """Best-effort per-column mean of numeric columns (neutral diagnostics)."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, ValueError):
        return {}
    if not rows:
        return {}
    means: dict[str, float] = {}
    for column in rows[0]:
        values: list[float] = []
        for row in rows:
            raw = (row.get(column) or "").strip()
            try:
                values.append(float(raw))
            except ValueError:
                values = []
                break
        if values:
            means[column] = sum(values) / len(values)
    return means


def _neutral_delta(
    rel: str, entry_a: Mapping[str, Any], entry_b: Mapping[str, Any],
    path_a: Path, path_b: Path,
) -> dict[str, Any]:
    """Neutral A-vs-B diff for a changed artifact (reuses the baseline diff logic)."""
    raw = _diff_details(rel, entry_a, entry_b)  # labels A=baseline, B=fresh
    details: dict[str, Any] = {"fingerprint_changed": True}
    if "json_top_level_keys" in raw:
        details["json_top_level_keys"] = raw["json_top_level_keys"]
    if "row_count" in raw:
        rc = raw["row_count"]
        details["row_count"] = {"a": rc["baseline"], "b": rc["fresh"]}
    if raw.get("columns_changed"):
        details["columns_changed"] = True
    if "line_count" in raw:
        lc = raw["line_count"]
        details["line_count"] = {"a": lc["baseline"], "b": lc["fresh"]}
    if "headline_metrics" in raw:
        details["headline_metrics"] = {
            metric: {
                "a": vals["baseline"],
                "b": vals["fresh"],
                "direction": _direction(vals["baseline"], vals["fresh"]),
            }
            for metric, vals in raw["headline_metrics"].items()
        }
    if path_a.suffix.lower() == ".csv" and path_b.suffix.lower() == ".csv":
        means_a = _csv_numeric_means(path_a)
        means_b = _csv_numeric_means(path_b)
        col_deltas = {
            col: {"a": means_a.get(col), "b": means_b.get(col),
                  "direction": _direction(means_a.get(col), means_b.get(col))}
            for col in sorted(set(means_a) | set(means_b))
            if means_a.get(col) != means_b.get(col)
        }
        if col_deltas:
            details["csv_numeric_column_deltas"] = col_deltas
    return details


def _detect_synthetic(run_dir: Path) -> bool | None:
    summary = run_dir / "pipeline_summary.json"
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            kind = payload.get("synthetic_vs_real")
            if kind in ("synthetic", "real"):
                return kind == "synthetic"
        except (ValueError, OSError):
            return None
    return None


def compare_runs(
    run_a: str | Path,
    run_b: str | Path,
    *,
    run_id_a: str | None = None,
    run_id_b: str | None = None,
    fixed_timestamp: str | None = None,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    strict_new_artifacts: bool = False,
) -> dict[str, Any]:
    """Diff two pipeline run directories. Neutral, deterministic, research-only."""
    dir_a, dir_b = Path(run_a), Path(run_b)
    vals_a = _volatile_values(run_id_a, fixed_timestamp)
    vals_b = _volatile_values(run_id_b, fixed_timestamp)
    paths_a = dict(_list_tracked(dir_a, exclude))
    paths_b = dict(_list_tracked(dir_b, exclude))
    fa = {rel: _fingerprint(p, rel, volatile_values=vals_a) for rel, p in paths_a.items()}
    fb = {rel: _fingerprint(p, rel, volatile_values=vals_b) for rel, p in paths_b.items()}

    entries: list[dict[str, Any]] = []
    headline_table: list[dict[str, Any]] = []
    for rel in sorted(set(fa) | set(fb)):
        ea, eb = fa.get(rel), fb.get(rel)
        if ea is not None and eb is None:
            entries.append({"relative_path": rel, "classification": CLASS_ONLY_IN_A})
            continue
        if ea is None and eb is not None:
            entries.append({"relative_path": rel, "classification": CLASS_ONLY_IN_B})
            continue
        assert ea is not None and eb is not None
        if ea.get("canonical_sha256") is None or eb.get("canonical_sha256") is None:
            entries.append({"relative_path": rel, "classification": CLASS_UNREADABLE})
            continue
        if ea["canonical_sha256"] != eb["canonical_sha256"]:
            delta = _neutral_delta(rel, ea, eb, paths_a[rel], paths_b[rel])
            entries.append(
                {"relative_path": rel, "classification": CLASS_CHANGED, "delta": delta}
            )
            for metric, vals in delta.get("headline_metrics", {}).items():
                headline_table.append({"artifact": rel, "metric": metric, **vals})
        elif ea["raw_sha256"] != eb["raw_sha256"]:
            entries.append({"relative_path": rel, "classification": CLASS_VOLATILE_ONLY})
        else:
            entries.append({"relative_path": rel, "classification": CLASS_UNCHANGED})

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["classification"]] = counts.get(entry["classification"], 0) + 1
    if counts.get(CLASS_UNREADABLE):
        status = STATUS_UNREADABLE_ARTIFACTS
    elif counts.get(CLASS_ONLY_IN_A):
        status = STATUS_MISSING_ARTIFACTS
    elif counts.get(CLASS_ONLY_IN_B):
        status = STATUS_NEW_ARTIFACTS
    elif counts.get(CLASS_CHANGED):
        status = STATUS_CHANGED
    else:
        status = STATUS_IDENTICAL
    synthetic = _detect_synthetic(dir_b)
    if synthetic is None:
        synthetic = _detect_synthetic(dir_a)
    return {
        "disclaimer": RESEARCH_DISCLAIMER,
        "research_only": True,
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "synthetic": synthetic,
        "synthetic_warning": (
            "SYNTHETIC FIXTURE RESULTS — not real market evidence." if synthetic else None
        ),
        "strict_new_artifacts": strict_new_artifacts,
        "comparison_status": status,
        "total_artifacts": len(entries),
        "counts": dict(sorted(counts.items())),
        "changed_artifacts": sorted(
            e["relative_path"] for e in entries if e["classification"] == CLASS_CHANGED
        ),
        "only_in_a": sorted(
            e["relative_path"] for e in entries if e["classification"] == CLASS_ONLY_IN_A
        ),
        "only_in_b": sorted(
            e["relative_path"] for e in entries if e["classification"] == CLASS_ONLY_IN_B
        ),
        "headline_metric_deltas": headline_table,
        "entries": entries,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_run_comparison_outputs(
    report: Mapping[str, Any], output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "run_comparison.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths = {"json_path": json_path}
    if write_markdown:
        md_path = out_dir / "run_comparison.md"
        lines = ["# Pipeline Run Comparison (A vs B)", "", str(report["disclaimer"]), ""]
        if report.get("synthetic_warning"):
            lines += [f"> **{report['synthetic_warning']}**", ""]
        lines += [
            f"- Status: **{report['comparison_status']}**",
            f"- Artifacts compared: {report['total_artifacts']}",
            f"- Counts: {report['counts']}",
            "",
        ]
        if report["headline_metric_deltas"]:
            lines += [
                "## Headline metric deltas (descriptive only — not better/worse)",
                "",
                "| artifact | metric | A | B | direction |",
                "| --- | --- | --- | --- | --- |",
            ]
            for row in report["headline_metric_deltas"]:
                lines.append(
                    f"| `{row['artifact']}` | {row['metric']} | {_fmt(row['a'])} | "
                    f"{_fmt(row['b'])} | {row['direction']} |"
                )
            lines.append("")
        if report["changed_artifacts"]:
            lines += ["## Changed artifacts", ""]
            lines += [f"- `{p}`" for p in report["changed_artifacts"]]
            lines.append("")
        if report["only_in_a"]:
            lines += ["## Only in A (missing in B)", *[f"- `{p}`" for p in report["only_in_a"]], ""]
        if report["only_in_b"]:
            lines += ["## Only in B (new in B)", *[f"- `{p}`" for p in report["only_in_b"]], ""]
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


# --------------------------------------------------------------------------- #
# Baseline promotion (explicit, auditable)
# --------------------------------------------------------------------------- #
def _baseline_fingerprint(baseline: Mapping[str, Any] | None) -> str | None:
    if baseline is None:
        return None
    return _sha256(json.dumps(baseline, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def promote_pipeline_baseline(
    from_run: str | Path,
    baseline_path: str | Path,
    *,
    reviewer_note: str = "",
    require_approval: bool = True,
    approved: bool = False,
    require_note: bool = True,
    previous_baseline_path: str | Path | None = None,
    run_id: str = GOLDEN_RUN_ID,
    fixed_timestamp: str | None = None,
    is_synthetic: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Promote a run to the approved golden baseline. Never promotes silently.

    Returns ``(promotion_record, updated)``. The baseline file is written only
    when approval (and a reviewer note, unless disabled) is supplied.
    """
    from_dir = Path(from_run)
    warnings: list[str] = []
    new_baseline = capture_baseline(
        from_dir, run_id=run_id, fixed_timestamp=fixed_timestamp, is_synthetic=is_synthetic
    )

    prior_path = Path(previous_baseline_path) if previous_baseline_path else Path(baseline_path)
    previous_baseline = load_baseline(prior_path) if prior_path.is_file() else None

    classification_counts: dict[str, int] = {}
    headline_metric_deltas: list[dict[str, Any]] = []
    if previous_baseline is not None:
        comparison = compare_to_baseline(
            from_dir, previous_baseline, run_id=run_id, fixed_timestamp=fixed_timestamp
        )
        classification_counts = comparison["counts"]
        for entry in comparison["entries"]:
            for metric, vals in entry.get("diff", {}).get("headline_metrics", {}).items():
                headline_metric_deltas.append(
                    {
                        "artifact": entry["relative_path"],
                        "metric": metric,
                        "previous": vals["baseline"],
                        "new": vals["fresh"],
                        "direction": _direction(vals["baseline"], vals["fresh"]),
                    }
                )

    blocked_reason: str | None = None
    if require_approval and not approved:
        blocked_reason = PROMOTION_BLOCKED_APPROVAL
        warnings.append("approval required but not given: baseline NOT updated")
    elif require_note and not reviewer_note.strip():
        blocked_reason = PROMOTION_BLOCKED_NOTE
        warnings.append("reviewer note required but empty: baseline NOT updated")

    updated = blocked_reason is None
    if updated:
        write_baseline(new_baseline, baseline_path)

    record = {
        "disclaimer": RESEARCH_DISCLAIMER,
        "research_only": True,
        "promotion_schema_version": PROMOTION_SCHEMA_VERSION,
        "status": PROMOTION_PROMOTED if updated else blocked_reason,
        "created_at_utc": fixed_timestamp,
        "source_run_id": run_id,
        "source_run_fingerprint": _baseline_fingerprint(new_baseline),
        "baseline_path": Path(baseline_path).name,  # basename only (no abs temp paths)
        "previous_baseline_fingerprint": _baseline_fingerprint(previous_baseline),
        "new_baseline_fingerprint": _baseline_fingerprint(new_baseline),
        "reviewer_note": reviewer_note,
        "approved": approved,
        "approval_required": require_approval,
        "artifact_classification_counts": dict(sorted(classification_counts.items())),
        "headline_metric_deltas": headline_metric_deltas,
        "warnings": warnings,
        "synthetic": is_synthetic,
        "synthetic_warning": (
            "SYNTHETIC FIXTURE RESULTS — not real market evidence." if is_synthetic else None
        ),
        "config": scrub_secrets(
            {"from_run": Path(from_run).name, "baseline_path": Path(baseline_path).name}
        ),
    }
    return record, updated


def write_promotion_record_outputs(
    record: Mapping[str, Any], output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "baseline_promotion_record.json"
    json_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths = {"json_path": json_path}
    if write_markdown:
        md_path = out_dir / "baseline_promotion_record.md"
        lines = ["# Baseline Promotion Record", "", str(record["disclaimer"]), ""]
        if record.get("synthetic_warning"):
            lines += [f"> **{record['synthetic_warning']}**", ""]
        lines += [
            f"- Status: **{record['status']}**",
            f"- Approved: {record['approved']} (approval required: {record['approval_required']})",
            f"- Reviewer note: {record['reviewer_note'] or '—'}",
            f"- Source run id: {record['source_run_id']}",
            f"- Previous baseline fingerprint: "
            f"{str(record['previous_baseline_fingerprint'])[:12]}…",
            f"- New baseline fingerprint: {str(record['new_baseline_fingerprint'])[:12]}…",
            f"- Classification vs previous: {record['artifact_classification_counts']}",
            "",
        ]
        if record["headline_metric_deltas"]:
            lines += [
                "## Headline metric deltas vs previous (descriptive only)",
                "",
                "| artifact | metric | previous | new | direction |",
                "| --- | --- | --- | --- | --- |",
            ]
            for row in record["headline_metric_deltas"]:
                lines.append(
                    f"| `{row['artifact']}` | {row['metric']} | {_fmt(row['previous'])} | "
                    f"{_fmt(row['new'])} | {row['direction']} |"
                )
            lines.append("")
        if record["warnings"]:
            lines += ["## Warnings", *[f"- {w}" for w in record["warnings"]], ""]
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


__all__ = [
    "CLASS_ONLY_IN_A",
    "CLASS_ONLY_IN_B",
    "COMPARISON_SCHEMA_VERSION",
    "PROMOTION_BLOCKED_APPROVAL",
    "PROMOTION_BLOCKED_NOTE",
    "PROMOTION_PROMOTED",
    "PROMOTION_SCHEMA_VERSION",
    "STATUS_CHANGED",
    "STATUS_IDENTICAL",
    "STATUS_MISSING_ARTIFACTS",
    "STATUS_NEW_ARTIFACTS",
    "compare_runs",
    "promote_pipeline_baseline",
    "write_promotion_record_outputs",
    "write_run_comparison_outputs",
]

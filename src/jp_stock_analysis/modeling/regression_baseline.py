"""Pipeline regression baseline & change detection (research-only).

Captures a small, canonicalized "golden" fingerprint set from a pipeline run and
compares fresh runs against it, classifying each artifact as
``unchanged`` / ``volatile_only`` / ``changed`` / ``missing`` / ``new`` /
``unreadable``. Unlike the determinism gate (which checks repeated-run
reproducibility), this detects *future* changes against a blessed reference.

It does NOT prove model validity or market performance. The golden baseline is
synthetic-only and is not market evidence. Baseline updates must be explicit
(``--update-baseline``) and reviewed. Secrets are never captured (the baseline
holds fingerprints + safe semantic metadata only); provenance manifests
(audit/artifact) are excluded by default because they are intentionally
run/commit-specific.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jp_stock_analysis.modeling.determinism import (
    DEFAULT_VOLATILE_KEYS,
    canonicalize_json,
    canonicalize_text,
)

SCHEMA_VERSION = "pipeline_regression_baseline_v1"

# Unlike the determinism gate (which keeps run_id fixed across the two runs), a
# golden baseline must be robust to the run id, so run_id is canonicalized here.
BASELINE_VOLATILE_KEYS = (*DEFAULT_VOLATILE_KEYS, "run_id")

RESEARCH_DISCLAIMER = (
    "This output is for analytical and self-directed research purposes. It is not "
    "personalized financial advice. The regression baseline detects unexpected "
    "pipeline output changes; it does not prove model validity or market performance."
)

# Provenance / self-referential artifacts are run- and commit-specific (they embed
# run-dependent fingerprints, git commit, etc.) or carry the optional regression
# hook, so they are not tracked by default. The deterministic metric/content
# artifacts (ranking, portfolio, ..., modeling_report) are what is tracked.
DEFAULT_EXCLUDE = (
    "audit_manifest.json",
    "audit_manifest.md",
    "artifact_manifest.json",
    "artifact_manifest.md",
    "pipeline_summary.json",
    "pipeline_summary.md",
)

# Safe, stable headline metrics surfaced in diffs (no secrets, small).
HEADLINE_METRICS: dict[str, tuple[str, ...]] = {
    "ranking/ranking_metrics.json": ("horizons.0.ic_mean", "horizons.0.icir",
                                     "horizons.0.coverage_count"),
    "portfolio/portfolio_metrics.json": ("spread_series.sharpe_like",
                                         "spread_series.mean_spread",
                                         "spread_series.hit_rate"),
    "neutralization/neutralized_metrics.json": ("neutralized_ic_mean", "raw_ic_mean"),
    "stability/model_stability.json": ("n_folds",),
    "readiness/forward_readiness.json": ("overall_status", "eligible_ticker_count"),
}

# Canonical synthetic golden settings — the committed fixture and the default
# `check-pipeline-regression --synthetic` run use exactly these, so a fresh run
# reproduces the committed baseline.
GOLDEN_RUN_ID = "golden"
GOLDEN_TIMESTAMP = "1970-01-01T00:00:00Z"

CLASS_UNCHANGED = "unchanged"
CLASS_VOLATILE_ONLY = "volatile_only"
CLASS_CHANGED = "changed"
CLASS_MISSING = "missing"
CLASS_NEW = "new"
CLASS_UNREADABLE = "unreadable"

_TEXT_SUFFIXES = {".md", ".csv", ".txt"}


def run_golden_synthetic_pipeline(
    output_dir: str | Path,
    *,
    run_id: str = GOLDEN_RUN_ID,
    fixed_timestamp: str = GOLDEN_TIMESTAMP,
) -> Path:
    """Run the canonical synthetic pipeline used for the golden baseline."""
    # imported here to avoid any import-time cost when only comparing baselines
    from jp_stock_analysis.modeling.dataset import build_modeling_dataset
    from jp_stock_analysis.modeling.fixtures import build_synthetic_bundle
    from jp_stock_analysis.modeling.pipeline import PipelineConfig, run_pipeline

    bundle = build_synthetic_bundle()
    dataset = build_modeling_dataset(
        bundle.fundamentals, bundle.prices, bundle.metadata, bundle.narratives,
        decision_dates=bundle.decision_dates, horizons=bundle.horizons,
        bundle_disclosure_date=bundle.bundle_disclosure_date, is_synthetic=True,
    )
    run_pipeline(
        dataset, bundle.prices, output_dir=output_dir, run_id=run_id,
        fixed_timestamp=fixed_timestamp, disclosure_date=bundle.bundle_disclosure_date,
        config=PipelineConfig(transaction_cost_bps=10.0, max_weight_per_name=0.34),
    )
    return Path(output_dir) / run_id


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get_path(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            cur = cur[idx] if -len(cur) <= idx < len(cur) else None
        elif isinstance(cur, Mapping):
            cur = cur.get(part)
        else:
            return None
        if cur is None:
            return None
    return cur


def _fingerprint(
    path: Path, rel: str, *, volatile_values: Sequence[str]
) -> dict[str, Any]:
    """Canonical + raw fingerprint and safe semantic metadata for one artifact."""
    data = path.read_bytes()
    entry: dict[str, Any] = {
        "relative_path": rel,
        "artifact_type": path.suffix.lstrip(".") or "file",
        "raw_sha256": _sha256(data),
    }
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            parsed = json.loads(data.decode("utf-8"))
            canonical = json.dumps(
                canonicalize_json(parsed, volatile_keys=BASELINE_VOLATILE_KEYS),
                sort_keys=True,
                ensure_ascii=False,
            )
            entry["canonical_sha256"] = _sha256(canonical.encode("utf-8"))
            if isinstance(parsed, Mapping):
                entry["json_top_level_keys"] = sorted(parsed.keys())
            headline = {
                p: _get_path(parsed, p) for p in HEADLINE_METRICS.get(rel, ())
            }
            if headline:
                entry["headline_metrics"] = headline
        elif suffix in _TEXT_SUFFIXES:
            text = data.decode("utf-8")
            canonical_text = canonicalize_text(text, volatile_values=volatile_values)
            entry["canonical_sha256"] = _sha256(canonical_text.encode("utf-8"))
            entry["line_count"] = len(text.splitlines())
            if suffix == ".csv":
                lines = data.decode("utf-8-sig", errors="replace").splitlines()
                if lines:
                    entry["columns"] = [c.strip() for c in lines[0].split(",")]
                    entry["row_count"] = max(0, len(lines) - 1)
        else:
            entry["canonical_sha256"] = _sha256(data)
    except (ValueError, UnicodeDecodeError):
        entry["canonical_sha256"] = None
        entry["unreadable"] = True
    return entry


def _list_tracked(run_dir: Path, exclude: Sequence[str]) -> list[tuple[str, Path]]:
    excluded = set(exclude)
    out = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(run_dir)).replace("\\", "/")
        if rel in excluded:
            continue
        out.append((rel, path))
    return sorted(out, key=lambda rp: rp[0])


def _volatile_values(run_id: str | None, fixed_timestamp: str | None) -> list[str]:
    return [v for v in (run_id, fixed_timestamp) if v]


def capture_baseline(
    run_dir: str | Path,
    *,
    run_id: str | None = None,
    fixed_timestamp: str | None = None,
    is_synthetic: bool = True,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
) -> dict[str, Any]:
    """Capture a canonicalized golden baseline from a pipeline run directory."""
    base = Path(run_dir)
    volatile_values = _volatile_values(run_id, fixed_timestamp)
    artifacts = [
        _fingerprint(path, rel, volatile_values=volatile_values)
        for rel, path in _list_tracked(base, exclude)
    ]
    artifacts.sort(key=lambda e: e["relative_path"])
    return {
        "disclaimer": RESEARCH_DISCLAIMER,
        "research_only": True,
        "schema_version": SCHEMA_VERSION,
        "synthetic": is_synthetic,
        "synthetic_warning": (
            "SYNTHETIC FIXTURE RESULTS — not real market evidence." if is_synthetic else None
        ),
        "ignored_volatile_keys": list(BASELINE_VOLATILE_KEYS),
        "ignored_volatile_value_kinds": ["run_id", "fixed_timestamp", "absolute_paths"],
        "excluded_artifacts": list(exclude),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def write_baseline(baseline: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def load_baseline(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _diff_details(
    rel: str, baseline_entry: Mapping[str, Any], fresh_entry: Mapping[str, Any]
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    base_keys = baseline_entry.get("json_top_level_keys")
    fresh_keys = fresh_entry.get("json_top_level_keys")
    if base_keys is not None and fresh_keys is not None and base_keys != fresh_keys:
        details["json_top_level_keys"] = {
            "added": sorted(set(fresh_keys) - set(base_keys)),
            "removed": sorted(set(base_keys) - set(fresh_keys)),
        }
    if baseline_entry.get("row_count") != fresh_entry.get("row_count"):
        details["row_count"] = {
            "baseline": baseline_entry.get("row_count"),
            "fresh": fresh_entry.get("row_count"),
        }
    if baseline_entry.get("columns") != fresh_entry.get("columns"):
        details["columns_changed"] = True
    if baseline_entry.get("line_count") != fresh_entry.get("line_count"):
        details["line_count"] = {
            "baseline": baseline_entry.get("line_count"),
            "fresh": fresh_entry.get("line_count"),
        }
    base_metrics = baseline_entry.get("headline_metrics") or {}
    fresh_metrics = fresh_entry.get("headline_metrics") or {}
    metric_diffs = {
        k: {"baseline": base_metrics.get(k), "fresh": fresh_metrics.get(k)}
        for k in sorted(set(base_metrics) | set(fresh_metrics))
        if base_metrics.get(k) != fresh_metrics.get(k)
    }
    if metric_diffs:
        details["headline_metrics"] = metric_diffs
    details["fingerprint_changed"] = True
    return details


def compare_to_baseline(
    run_dir: str | Path,
    baseline: Mapping[str, Any],
    *,
    run_id: str | None = None,
    fixed_timestamp: str | None = None,
    strict_new_artifacts: bool = False,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
) -> dict[str, Any]:
    """Compare a fresh run directory against a golden baseline."""
    base = Path(run_dir)
    volatile_values = _volatile_values(run_id, fixed_timestamp)
    baseline_by_path = {e["relative_path"]: e for e in baseline.get("artifacts", [])}
    fresh_by_path = {
        rel: _fingerprint(path, rel, volatile_values=volatile_values)
        for rel, path in _list_tracked(base, exclude)
    }

    entries: list[dict[str, Any]] = []
    for rel in sorted(set(baseline_by_path) | set(fresh_by_path)):
        base_entry = baseline_by_path.get(rel)
        fresh_entry = fresh_by_path.get(rel)
        if base_entry is not None and fresh_entry is None:
            entries.append({"relative_path": rel, "classification": CLASS_MISSING})
            continue
        if base_entry is None and fresh_entry is not None:
            entries.append({"relative_path": rel, "classification": CLASS_NEW})
            continue
        assert base_entry is not None and fresh_entry is not None
        if fresh_entry.get("unreadable") or base_entry.get("canonical_sha256") is None:
            entries.append({"relative_path": rel, "classification": CLASS_UNREADABLE})
            continue
        if fresh_entry["canonical_sha256"] != base_entry["canonical_sha256"]:
            entries.append(
                {
                    "relative_path": rel,
                    "classification": CLASS_CHANGED,
                    "diff": _diff_details(rel, base_entry, fresh_entry),
                }
            )
        elif fresh_entry["raw_sha256"] != base_entry["raw_sha256"]:
            entries.append({"relative_path": rel, "classification": CLASS_VOLATILE_ONLY})
        else:
            entries.append({"relative_path": rel, "classification": CLASS_UNCHANGED})

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["classification"]] = counts.get(entry["classification"], 0) + 1
    regression_artifacts = [
        e["relative_path"]
        for e in entries
        if e["classification"] in (CLASS_CHANGED, CLASS_MISSING)
        or (strict_new_artifacts and e["classification"] == CLASS_NEW)
    ]
    regression_detected = bool(regression_artifacts)
    return {
        "disclaimer": RESEARCH_DISCLAIMER,
        "research_only": True,
        "schema_version": baseline.get("schema_version", SCHEMA_VERSION),
        "synthetic": baseline.get("synthetic", True),
        "synthetic_warning": baseline.get("synthetic_warning"),
        "strict_new_artifacts": strict_new_artifacts,
        "total_artifacts": len(entries),
        "counts": dict(sorted(counts.items())),
        "regression_detected": regression_detected,
        "regression_artifacts": sorted(regression_artifacts),
        "entries": entries,
    }


def write_regression_report(
    report: Mapping[str, Any], output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pipeline_regression_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths = {"json_path": json_path}
    if write_markdown:
        md_path = out_dir / "pipeline_regression_report.md"
        lines = ["# Pipeline Regression Report", "", str(report["disclaimer"]), ""]
        if report.get("synthetic_warning"):
            lines += [f"> **{report['synthetic_warning']}**", ""]
        lines += [
            f"- Regression detected: **{report['regression_detected']}**",
            f"- Artifacts compared: {report['total_artifacts']}",
            f"- Counts: {report['counts']}",
            f"- Strict new artifacts: {report['strict_new_artifacts']}",
            "",
        ]
        flagged = [e for e in report["entries"] if e["classification"] != CLASS_UNCHANGED]
        if flagged:
            lines += ["| artifact | classification |", "| --- | --- |"]
            for e in flagged:
                lines.append(f"| `{e['relative_path']}` | **{e['classification']}** |")
        else:
            lines.append("_All tracked artifacts unchanged._")
        lines.append("")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


__all__ = [
    "CLASS_CHANGED",
    "CLASS_MISSING",
    "CLASS_NEW",
    "CLASS_UNCHANGED",
    "CLASS_UNREADABLE",
    "CLASS_VOLATILE_ONLY",
    "DEFAULT_EXCLUDE",
    "GOLDEN_RUN_ID",
    "GOLDEN_TIMESTAMP",
    "SCHEMA_VERSION",
    "capture_baseline",
    "compare_to_baseline",
    "load_baseline",
    "run_golden_synthetic_pipeline",
    "write_baseline",
    "write_regression_report",
]

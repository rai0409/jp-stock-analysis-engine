"""Offline audit bundle export and verification (research-only).

An audit bundle is a self-contained reproducibility artifact: baseline,
hash-chained ledger, optional promotion records, and optional determinism /
regression reports with canonical fingerprints. It is tamper-evident, synthetic
aware, and intentionally makes no market-performance or trading claim.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jp_stock_analysis.modeling.audit import _canonical, _sha256
from jp_stock_analysis.modeling.baseline_history import (
    STATUS_VALID,
    load_ledger,
    verify_baseline_history,
)

AUDIT_BUNDLE_SCHEMA_VERSION = "audit_bundle_v1"
MANIFEST_NAME = "audit_bundle_manifest.json"

RESEARCH_DISCLAIMER = (
    "This audit bundle is for analytical and self-directed research reproducibility. "
    "It is not personalized financial advice, does not prove model validity or "
    "market performance, and contains no buy/sell recommendation."
)
SYNTHETIC_WARNING = "SYNTHETIC FIXTURE RESULTS - not real market evidence."

STATUS_VALID_BUNDLE = "valid"
STATUS_INVALID_BUNDLE = "invalid"

_SECRET_VALUE_MARKERS = (
    "JQUANTS_API_KEY",
    "EDINET_API_KEY",
    "x-api-key",
    "api_key=",
    "apikey=",
    "bearer ",
)
_SECRET_KEY_MARKERS = ("api_key", "apikey", "token", "secret", "password", "credential")
_ABS_PATH_MARKERS = ("/home/", "/tmp/", "/var/", "/Users/")
_TEXT_SUFFIXES = {".md", ".txt", ".jsonl"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _baseline_fingerprint(baseline: Mapping[str, Any]) -> str:
    return _sha256(json.dumps(baseline, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _canonical_bytes(path: Path) -> bytes:
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix == ".json":
        try:
            return _canonical(_read_json(path)).encode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return data
    if suffix == ".jsonl":
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (ValueError, UnicodeDecodeError):
            return data
        return ("\n".join(_canonical(row) for row in rows) + "\n").encode("utf-8")
    return data


def _fingerprint_file(bundle_dir: Path, path: Path, artifact_type: str) -> dict[str, Any]:
    rel = str(path.relative_to(bundle_dir)).replace("\\", "/")
    data = path.read_bytes()
    canonical = _canonical_bytes(path)
    entry: dict[str, Any] = {
        "relative_path": rel,
        "artifact_type": artifact_type,
        "size_bytes": len(data),
        "canonical_sha256": _sha256(canonical),
        "raw_sha256": _sha256(data),
    }
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            parsed = _read_json(path)
            if isinstance(parsed, Mapping):
                entry["json_top_level_keys"] = sorted(parsed.keys())
        except ValueError:
            entry["warnings"] = ["unparseable JSON"]
    elif suffix == ".csv":
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
        except (OSError, UnicodeDecodeError, csv.Error):
            rows = []
        if rows:
            entry["columns"] = rows[0]
            entry["row_count"] = max(0, len(rows) - 1)
    if suffix in _TEXT_SUFFIXES:
        try:
            entry["line_count"] = len(path.read_text(encoding="utf-8").splitlines())
        except UnicodeDecodeError:
            entry["warnings"] = [*entry.get("warnings", []), "unreadable text"]
    return entry


def _scan_obj(obj: Any, *, prefix: str = "") -> list[str]:
    issues: list[str] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            if any(marker in str(key).lower() for marker in _SECRET_KEY_MARKERS):
                issues.append(f"secret-like field name: {full}")
            issues.extend(_scan_obj(value, prefix=full))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            issues.extend(_scan_obj(value, prefix=f"{prefix}[{idx}]"))
    elif isinstance(obj, str):
        lower = obj.lower()
        for marker in _SECRET_VALUE_MARKERS:
            if marker.lower() in lower:
                issues.append(f"secret-like value at {prefix or '<root>'}: {marker}")
        for marker in _ABS_PATH_MARKERS:
            if marker in obj:
                issues.append(f"absolute path at {prefix or '<root>'}: {marker}")
    return issues


def _scrub_bundle_json(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if any(marker in str(key).lower() for marker in _SECRET_KEY_MARKERS):
                out[str(key)] = "***REDACTED***"
            else:
                out[str(key)] = _scrub_bundle_json(value)
        return out
    if isinstance(obj, list):
        return [_scrub_bundle_json(value) for value in obj]
    return obj


def _stable_json_copy(src: Path, dst: Path) -> None:
    payload = _scrub_bundle_json(_read_json(src))
    _write_json(dst, payload)


def _stable_jsonl_copy(src: Path, dst: Path) -> None:
    rows = load_ledger(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _copy_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".json":
        _stable_json_copy(src, dst)
    elif src.suffix.lower() == ".jsonl":
        _stable_jsonl_copy(src, dst)
    else:
        shutil.copyfile(src, dst)


def _add_file(
    copied: list[tuple[str, str]],
    *,
    bundle_dir: Path,
    source: Path | None,
    dest_rel: str,
    artifact_type: str,
    warnings: list[str],
    required: bool = False,
) -> None:
    if source is None:
        return
    if not source.is_file():
        if required:
            warnings.append(f"required source missing: {dest_rel}")
        return
    dest = bundle_dir / dest_rel
    _copy_artifact(source, dest)
    copied.append((dest_rel, artifact_type))


def _discover_promotion_records(
    promotion_record: str | Path | None,
    promotion_record_dir: str | Path | None,
) -> list[Path]:
    paths: list[Path] = []
    if promotion_record:
        paths.append(Path(promotion_record))
    if promotion_record_dir:
        root = Path(promotion_record_dir)
        if root.is_dir():
            paths.extend(sorted(root.glob("*.json")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _pipeline_files(run_dir: str | Path | None) -> list[tuple[Path, str]]:
    if run_dir is None:
        return []
    base = Path(run_dir)
    names = (
        "pipeline_summary.json",
        "pipeline_summary.md",
        "artifact_manifest.json",
        "artifact_manifest.md",
        "audit_manifest.json",
        "audit_manifest.md",
    )
    return [(base / name, name) for name in names if (base / name).is_file()]


def _manifest_fingerprint(manifest_without_overall: Mapping[str, Any]) -> str:
    content_fingerprints = [
        {
            "relative_path": entry["relative_path"],
            "canonical_sha256": entry["canonical_sha256"],
        }
        for entry in manifest_without_overall.get("bundle_contents", [])
    ]
    payload = {
        "manifest": manifest_without_overall,
        "included_file_canonical_fingerprints": content_fingerprints,
    }
    return _sha256(_canonical(payload).encode("utf-8"))


def _ledger_head_baseline_fingerprint(entries: Sequence[Mapping[str, Any]]) -> str | None:
    if not entries:
        return None
    head = entries[-1]
    value = head.get("new_baseline_fingerprint") or head.get("baseline_fingerprint")
    return value if isinstance(value, str) else None


def _report_status(path: Path | None, status_key: str, fallback_key: str | None = None) -> str:
    if path is None or not path.is_file():
        return "unavailable"
    try:
        payload = _read_json(path)
    except ValueError:
        return "unreadable"
    value = payload.get(status_key)
    if value is None and fallback_key is not None:
        value = payload.get(fallback_key)
    return str(value) if value is not None else "included"


def export_audit_bundle(
    output_dir: str | Path,
    *,
    baseline_path: str | Path,
    ledger_path: str | Path,
    promotion_record_path: str | Path | None = None,
    promotion_record_dir: str | Path | None = None,
    pipeline_run_dir: str | Path | None = None,
    determinism_report_path: str | Path | None = None,
    regression_report_path: str | Path | None = None,
    synthetic: bool = True,
    fixed_timestamp: str | None = None,
    bundle_id: str | None = None,
    include_fresh_checks: bool = False,
) -> dict[str, Any]:
    """Export a deterministic, self-contained audit bundle directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    copied: list[tuple[str, str]] = []
    bundle_id = bundle_id or "audit_bundle"
    created_at = fixed_timestamp or "1970-01-01T00:00:00Z"

    baseline_src = Path(baseline_path)
    ledger_src = Path(ledger_path)
    _add_file(
        copied,
        bundle_dir=out,
        source=baseline_src,
        dest_rel="baseline/golden_pipeline_baseline.json",
        artifact_type="golden_baseline",
        warnings=warnings,
        required=True,
    )
    _add_file(
        copied,
        bundle_dir=out,
        source=ledger_src,
        dest_rel="ledger/baseline_history.jsonl",
        artifact_type="baseline_history_ledger",
        warnings=warnings,
        required=True,
    )

    for record in _discover_promotion_records(promotion_record_path, promotion_record_dir):
        _add_file(
            copied,
            bundle_dir=out,
            source=record,
            dest_rel=f"promotion_records/{record.name}",
            artifact_type="baseline_promotion_record",
            warnings=warnings,
        )

    det_src = Path(determinism_report_path) if determinism_report_path else None
    reg_src = Path(regression_report_path) if regression_report_path else None
    _add_file(
        copied,
        bundle_dir=out,
        source=det_src,
        dest_rel="reports/determinism_report.json",
        artifact_type="determinism_report",
        warnings=warnings,
    )
    if include_fresh_checks and synthetic and reg_src is None:
        from jp_stock_analysis.modeling.regression_baseline import (
            GOLDEN_RUN_ID,
            GOLDEN_TIMESTAMP,
            compare_to_baseline,
            run_golden_synthetic_pipeline,
            write_regression_report,
        )

        with tempfile.TemporaryDirectory(prefix="audit_bundle_regression_") as tmp:
            run_dir = run_golden_synthetic_pipeline(
                tmp, run_id=GOLDEN_RUN_ID, fixed_timestamp=GOLDEN_TIMESTAMP
            )
            report = compare_to_baseline(
                run_dir,
                _read_json(baseline_src),
                run_id=GOLDEN_RUN_ID,
                fixed_timestamp=GOLDEN_TIMESTAMP,
            )
            write_regression_report(report, out / "reports")
        copied.append(("reports/pipeline_regression_report.json", "pipeline_regression_report"))
        if (out / "reports/pipeline_regression_report.md").is_file():
            copied.append(("reports/pipeline_regression_report.md", "pipeline_regression_report"))
    else:
        _add_file(
            copied,
            bundle_dir=out,
            source=reg_src,
            dest_rel="reports/pipeline_regression_report.json",
            artifact_type="pipeline_regression_report",
            warnings=warnings,
        )

    for src, name in _pipeline_files(pipeline_run_dir):
        _add_file(
            copied,
            bundle_dir=out,
            source=src,
            dest_rel=f"pipeline_run/{name}",
            artifact_type="pipeline_run_manifest",
            warnings=warnings,
        )

    if include_fresh_checks:
        warnings.append("fresh checks requested: no network or market-data fetch was performed")

    baseline_payload = _read_json(out / "baseline/golden_pipeline_baseline.json")
    baseline_fp = _baseline_fingerprint(baseline_payload)
    bundled_ledger = out / "ledger/baseline_history.jsonl"
    ledger_verification = verify_baseline_history(bundled_ledger)
    ledger_entries = load_ledger(bundled_ledger)
    ledger_head_baseline = _ledger_head_baseline_fingerprint(ledger_entries)
    baseline_matches = (
        ledger_head_baseline is None or ledger_head_baseline == baseline_fp
    )
    if ledger_verification["status"] != STATUS_VALID:
        warnings.append("baseline ledger chain is invalid")
    if not baseline_matches:
        warnings.append("baseline fingerprint does not match ledger head")
    if det_src is None:
        warnings.append("determinism report unavailable: not supplied")
    if reg_src is None:
        warnings.append("regression report unavailable: not supplied")

    contents = [
        _fingerprint_file(out, out / rel, artifact_type)
        for rel, artifact_type in sorted(copied, key=lambda item: item[0])
    ]
    manifest: dict[str, Any] = {
        "disclaimer": RESEARCH_DISCLAIMER,
        "audit_bundle_schema_version": AUDIT_BUNDLE_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "created_at_utc": created_at,
        "synthetic": synthetic,
        "synthetic_warning": SYNTHETIC_WARNING if synthetic else None,
        "research_only": True,
        "not_financial_advice": True,
        "source_files_included": [entry["relative_path"] for entry in contents],
        "bundle_contents": contents,
        "ledger_verification_status": ledger_verification["status"],
        "ledger_head_hash": ledger_verification["head_hash"],
        "ledger_entry_count": ledger_verification["entry_count"],
        "baseline_fingerprint": baseline_fp,
        "ledger_head_baseline_fingerprint": ledger_head_baseline,
        "baseline_matches_ledger_head": baseline_matches,
        "determinism_status": _report_status(
            out / "reports/determinism_report.json", "overall"
        ),
        "regression_status": _report_status(
            out / "reports/pipeline_regression_report.json", "regression_detected"
        ),
        "include_fresh_checks": include_fresh_checks,
        "warnings": sorted(set(warnings)),
    }
    manifest["overall_bundle_fingerprint"] = _manifest_fingerprint(manifest)
    _write_json(out / MANIFEST_NAME, manifest)
    write_audit_bundle_manifest_markdown(manifest, out)
    return manifest


def _manifest_by_path(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(entry.get("relative_path")): entry
        for entry in manifest.get("bundle_contents", [])
        if isinstance(entry, Mapping)
    }


def _recompute_manifest_from_contents(
    bundle_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    recomputed = dict(manifest)
    entries: list[dict[str, Any]] = []
    for old in manifest.get("bundle_contents", []):
        if not isinstance(old, Mapping):
            continue
        rel = str(old.get("relative_path"))
        path = bundle_dir / rel
        if path.is_file():
            entries.append(
                _fingerprint_file(bundle_dir, path, str(old.get("artifact_type", "file")))
            )
        else:
            entries.append(dict(old))
    recomputed["bundle_contents"] = sorted(entries, key=lambda e: e["relative_path"])
    recomputed.pop("overall_bundle_fingerprint", None)
    return recomputed


def verify_audit_bundle(
    bundle_dir: str | Path,
    *,
    fail_on_invalid: bool = False,
) -> dict[str, Any]:
    """Verify bundle manifest, fingerprints, ledger lineage, and baseline link."""
    del fail_on_invalid  # CLI owns exit behavior; report is always returned.
    base = Path(bundle_dir)
    manifest_path = base / MANIFEST_NAME
    issues: list[str] = []
    warnings: list[str] = []
    if not manifest_path.is_file():
        return {
            "disclaimer": RESEARCH_DISCLAIMER,
            "research_only": True,
            "status": STATUS_INVALID_BUNDLE,
            "issues": [f"missing {MANIFEST_NAME}"],
            "warnings": [],
            "file_count": 0,
            "bundle_fingerprint": None,
            "ledger_verification": None,
            "baseline_ledger_match": None,
        }
    try:
        manifest = _read_json(manifest_path)
    except ValueError as exc:
        return {
            "disclaimer": RESEARCH_DISCLAIMER,
            "research_only": True,
            "status": STATUS_INVALID_BUNDLE,
            "issues": [f"invalid manifest JSON: {exc}"],
            "warnings": [],
            "file_count": 0,
            "bundle_fingerprint": None,
            "ledger_verification": None,
            "baseline_ledger_match": None,
        }

    issues.extend(f"manifest: {issue}" for issue in _scan_obj(manifest))
    by_path = _manifest_by_path(manifest)
    if "baseline/golden_pipeline_baseline.json" not in by_path:
        issues.append("missing required manifest entry: baseline/golden_pipeline_baseline.json")
    if "ledger/baseline_history.jsonl" not in by_path:
        issues.append("missing required manifest entry: ledger/baseline_history.jsonl")

    for rel, expected in by_path.items():
        if rel == MANIFEST_NAME or Path(rel).is_absolute() or ".." in Path(rel).parts:
            issues.append(f"unsafe manifest relative_path: {rel}")
            continue
        path = base / rel
        if not path.is_file():
            issues.append(f"missing bundled file: {rel}")
            continue
        actual = _fingerprint_file(base, path, str(expected.get("artifact_type", "file")))
        for key in ("size_bytes", "canonical_sha256", "raw_sha256"):
            if actual.get(key) != expected.get(key):
                issues.append(f"manifest/content mismatch for {rel}: {key}")
        if path.suffix.lower() == ".json":
            try:
                issues.extend(f"{rel}: {issue}" for issue in _scan_obj(_read_json(path)))
            except ValueError:
                issues.append(f"unreadable JSON: {rel}")

    actual_files = {
        str(path.relative_to(base)).replace("\\", "/")
        for path in base.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME and path.suffix != ".md"
    }
    expected_files = set(by_path)
    unexpected = sorted(actual_files - expected_files)
    if unexpected:
        warnings.append(f"unexpected unmanifested files: {unexpected}")

    baseline_path = base / "baseline/golden_pipeline_baseline.json"
    ledger_path = base / "ledger/baseline_history.jsonl"
    baseline_fp = None
    if baseline_path.is_file():
        try:
            baseline_fp = _baseline_fingerprint(_read_json(baseline_path))
        except ValueError:
            issues.append("baseline file is not readable JSON")
    else:
        issues.append("baseline file missing")
    if baseline_fp and baseline_fp != manifest.get("baseline_fingerprint"):
        issues.append("baseline fingerprint mismatch vs manifest")

    if ledger_path.is_file():
        ledger_verification = verify_baseline_history(ledger_path)
        if ledger_verification["status"] != STATUS_VALID:
            issues.extend(f"ledger: {issue}" for issue in ledger_verification["issues"])
        try:
            ledger_entries = load_ledger(ledger_path)
        except ValueError:
            ledger_entries = []
    else:
        ledger_verification = {"status": STATUS_INVALID_BUNDLE, "issues": ["missing ledger"]}
        ledger_entries = []
        issues.append("ledger file missing")

    ledger_head_baseline = _ledger_head_baseline_fingerprint(ledger_entries)
    baseline_matches = ledger_head_baseline is None or ledger_head_baseline == baseline_fp
    if manifest.get("ledger_head_hash") != ledger_verification.get("head_hash"):
        issues.append("ledger head hash mismatch vs manifest")
    if manifest.get("ledger_entry_count") != ledger_verification.get("entry_count"):
        issues.append("ledger entry count mismatch vs manifest")
    if manifest.get("ledger_head_baseline_fingerprint") != ledger_head_baseline:
        issues.append("ledger head baseline fingerprint mismatch vs manifest")
    if manifest.get("baseline_matches_ledger_head") != baseline_matches:
        issues.append("baseline/ledger match boolean mismatch vs manifest")
    if not baseline_matches:
        issues.append("baseline fingerprint does not match ledger head baseline fingerprint")

    for rel, expected in by_path.items():
        if expected.get("artifact_type") == "baseline_promotion_record":
            try:
                record = _read_json(base / rel)
            except ValueError:
                issues.append(f"promotion record unreadable: {rel}")
                continue
            entry_hash = record.get("appended_entry_hash")
            if entry_hash and entry_hash not in {e.get("entry_hash") for e in ledger_entries}:
                issues.append(f"promotion record {rel} appended_entry_hash not in ledger")

    if (base / "reports/determinism_report.json").is_file():
        try:
            _read_json(base / "reports/determinism_report.json")
        except ValueError:
            issues.append("determinism report unreadable")
    elif manifest.get("determinism_status") != "unavailable":
        issues.append("determinism report missing despite manifest status")

    if (base / "reports/pipeline_regression_report.json").is_file():
        try:
            _read_json(base / "reports/pipeline_regression_report.json")
        except ValueError:
            issues.append("regression report unreadable")
    elif manifest.get("regression_status") != "unavailable":
        issues.append("regression report missing despite manifest status")

    recomputed = _recompute_manifest_from_contents(base, manifest)
    recomputed_fp = _manifest_fingerprint(recomputed)
    if recomputed_fp != manifest.get("overall_bundle_fingerprint"):
        issues.append("overall bundle fingerprint mismatch")

    status = STATUS_VALID_BUNDLE if not issues else STATUS_INVALID_BUNDLE
    return {
        "disclaimer": RESEARCH_DISCLAIMER,
        "research_only": True,
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "file_count": len(by_path),
        "bundle_fingerprint": manifest.get("overall_bundle_fingerprint"),
        "recomputed_bundle_fingerprint": recomputed_fp,
        "ledger_verification": {
            "status": ledger_verification.get("status"),
            "entry_count": ledger_verification.get("entry_count"),
            "head_hash": ledger_verification.get("head_hash"),
        },
        "baseline_ledger_match": {
            "baseline_fingerprint": baseline_fp,
            "ledger_head_baseline_fingerprint": ledger_head_baseline,
            "matches": baseline_matches,
        },
    }


def write_audit_bundle_manifest_markdown(
    manifest: Mapping[str, Any], output_dir: str | Path
) -> Path:
    out = Path(output_dir)
    md = out / "audit_bundle_manifest.md"
    lines = ["# Audit Bundle Manifest", "", str(manifest["disclaimer"]), ""]
    if manifest.get("synthetic_warning"):
        lines += [f"> **{manifest['synthetic_warning']}**", ""]
    lines += [
        f"- Bundle id: `{manifest['bundle_id']}`",
        f"- Research only: {manifest['research_only']}",
        f"- Synthetic: {manifest['synthetic']}",
        f"- Ledger chain: **{manifest['ledger_verification_status']}**",
        f"- Baseline matches ledger head: {manifest['baseline_matches_ledger_head']}",
        f"- Bundle fingerprint: `{manifest['overall_bundle_fingerprint']}`",
        "",
        "| file | type | size | canonical sha256 |",
        "| --- | --- | ---: | --- |",
    ]
    for entry in manifest["bundle_contents"]:
        lines.append(
            f"| `{entry['relative_path']}` | {entry['artifact_type']} | "
            f"{entry['size_bytes']} | `{entry['canonical_sha256'][:12]}` |"
        )
    if manifest.get("warnings"):
        lines += ["", "## Warnings", *[f"- {warning}" for warning in manifest["warnings"]]]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def write_audit_bundle_verification_outputs(
    report: Mapping[str, Any], output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "audit_bundle_verification.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths = {"json_path": json_path}
    if write_markdown:
        md_path = out / "audit_bundle_verification.md"
        lines = ["# Audit Bundle Verification", "", str(report["disclaimer"]), ""]
        lines += [
            f"- Status: **{report['status']}**",
            f"- Files: {report['file_count']}",
            f"- Bundle fingerprint: `{report.get('bundle_fingerprint')}`",
            "",
        ]
        if report["issues"]:
            lines += ["## Issues", *[f"- {issue}" for issue in report["issues"]], ""]
        if report["warnings"]:
            lines += ["## Warnings", *[f"- {warning}" for warning in report["warnings"]], ""]
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


__all__ = [
    "AUDIT_BUNDLE_SCHEMA_VERSION",
    "MANIFEST_NAME",
    "STATUS_INVALID_BUNDLE",
    "STATUS_VALID_BUNDLE",
    "export_audit_bundle",
    "verify_audit_bundle",
    "write_audit_bundle_manifest_markdown",
    "write_audit_bundle_verification_outputs",
]

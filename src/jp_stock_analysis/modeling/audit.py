"""Consolidated audit manifest for reproducibility (research-only).

Produces a deterministic run manifest: input fingerprints, model versions,
feature/target columns, horizons, no-look-ahead status, a synthetic-vs-real
flag, output files, and warnings. Audit manifests improve reproducibility but do
**not** prove model validity, and contain no predictive or trading claim. Secrets
are scrubbed and never written.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RESEARCH_DISCLAIMER = (
    "This output is for analytical and self-directed research purposes. It is not "
    "personalized financial advice. An audit manifest improves reproducibility but "
    "does not prove model validity."
)

EPOCH_UTC = "1970-01-01T00:00:00Z"
_SECRET_MARKERS = ("key", "token", "secret", "password", "passwd", "credential", "api")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scrub_secrets(config: Mapping[str, Any]) -> dict[str, Any]:
    """Redact any config value whose key looks like a secret. Never emit keys."""
    out: dict[str, Any] = {}
    for key, value in config.items():
        if any(marker in str(key).lower() for marker in _SECRET_MARKERS):
            out[key] = "***REDACTED***"
        elif isinstance(value, Mapping):
            out[key] = scrub_secrets(value)
        else:
            out[key] = value
    return out


def fingerprint_file(path: str | Path, *, include_absolute_path: bool = False) -> dict[str, Any]:
    """Deterministic content fingerprint of a file (CSV row/column aware)."""
    p = Path(path)
    if not p.is_file():
        return {"name": p.name, "status": "missing"}
    data = p.read_bytes()
    info: dict[str, Any] = {
        "name": p.name,  # basename only (absolute temp paths are non-deterministic)
        "sha256": _sha256(data),
        "size_bytes": len(data),
        "status": "ok",
    }
    if include_absolute_path:
        info["absolute_path"] = str(p.resolve())
    if p.suffix.lower() == ".csv":
        lines = data.decode("utf-8-sig", errors="replace").splitlines()
        if lines:
            info["columns"] = [c.strip() for c in lines[0].split(",")]
            info["row_count"] = max(0, len(lines) - 1)
    return info


def fingerprint_records(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Deterministic fingerprint of in-memory records (e.g. a DataFrame's rows)."""
    columns = sorted({c for row in rows for c in row}) if rows else []
    payload = _canonical([{c: row.get(c) for c in columns} for row in rows])
    return {
        "name": name,
        "sha256": _sha256(payload.encode("utf-8")),
        "row_count": len(rows),
        "columns": columns,
        "status": "ok",
    }


def current_git_commit(repo_dir: str | Path = ".") -> str | None:
    """Best-effort git HEAD (offline; returns None on any failure)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit or None


def project_version() -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("jp-stock-analysis-engine")
        except PackageNotFoundError:
            return None
    except ImportError:  # pragma: no cover
        return None


def build_audit_manifest(
    *,
    command: str | Mapping[str, Any] | None = None,
    model_versions: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
    target_columns: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    no_look_ahead_status: str | None = None,
    is_synthetic: bool = False,
    input_fingerprints: Sequence[Mapping[str, Any]] | None = None,
    output_files: Sequence[str] | None = None,
    warnings: Sequence[str] | None = None,
    git_commit: str | None = None,
    version: str | None = None,
    run_id: str | None = None,
    created_at_utc: str | None = None,
    stable: bool = False,
) -> dict[str, Any]:
    """Build a deterministic run manifest.

    In ``stable`` mode (for tests) an unset ``run_id`` is derived from a hash of
    the inputs/config and ``created_at_utc`` defaults to the epoch, so identical
    inputs yield an identical manifest and changed inputs change the fingerprint.
    """
    scrubbed_command = (
        scrub_secrets(command) if isinstance(command, Mapping) else command
    )
    fingerprints = [dict(fp) for fp in (input_fingerprints or [])]
    body = {
        "command": scrubbed_command,
        "model_versions": list(model_versions or []),
        "feature_columns": list(feature_columns or []),
        "target_columns": list(target_columns or []),
        "horizons": list(horizons or []),
        "no_look_ahead_status": no_look_ahead_status,
        "synthetic_vs_real": "synthetic" if is_synthetic else "real",
        "input_fingerprints": fingerprints,
    }
    if run_id is None and stable:
        run_id = "run_" + _sha256(_canonical(body).encode("utf-8"))[:16]
    if created_at_utc is None and stable:
        created_at_utc = EPOCH_UTC

    manifest = {
        "disclaimer": RESEARCH_DISCLAIMER,
        "research_only": True,
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "project_version": version,
        "git_commit": git_commit,
        "synthetic_warning": (
            "SYNTHETIC FIXTURE RESULTS — not real market evidence." if is_synthetic else None
        ),
        **body,
        "output_files": list(output_files or []),
        "warnings": list(warnings or []),
    }
    return manifest


def write_audit_manifest_outputs(
    manifest: Mapping[str, Any], output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "audit_manifest.json"
    json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths = {"json_path": json_path}
    if write_markdown:
        md_path = out_dir / "audit_manifest.md"
        md_path.write_text(_markdown(manifest), encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


def _markdown(manifest: Mapping[str, Any]) -> str:
    lines = ["# Audit Manifest", "", str(manifest.get("disclaimer", "")), ""]
    if manifest.get("synthetic_warning"):
        lines += [f"> **{manifest['synthetic_warning']}**", ""]
    for key in (
        "run_id",
        "created_at_utc",
        "project_version",
        "git_commit",
        "synthetic_vs_real",
        "no_look_ahead_status",
        "model_versions",
        "feature_columns",
        "target_columns",
        "horizons",
    ):
        lines.append(f"- **{key}**: {manifest.get(key)}")
    lines.append("")
    lines.append("## Input fingerprints")
    for fp in manifest.get("input_fingerprints", []):
        lines.append(
            f"- `{fp.get('name')}` sha256={str(fp.get('sha256'))[:12]}… "
            f"rows={fp.get('row_count')} status={fp.get('status')}"
        )
    lines.append("")
    lines.append("## Output files")
    for out in manifest.get("output_files", []):
        lines.append(f"- {out}")
    if manifest.get("warnings"):
        lines += ["", "## Warnings", *[f"- {w}" for w in manifest["warnings"]]]
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "EPOCH_UTC",
    "build_audit_manifest",
    "current_git_commit",
    "fingerprint_file",
    "fingerprint_records",
    "project_version",
    "scrub_secrets",
    "write_audit_manifest_outputs",
]

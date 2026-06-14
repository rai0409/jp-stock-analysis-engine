"""Drift / stability monitoring across decision dates or folds (research-only).

Computes rolling mean/std, trailing-window z-scores, threshold flags, and a
stability band per metric (Rank IC, neutralized Rank IC, long-short spread,
Sharpe-like, turnover, exposure, feature coverage, prediction distribution).
Deterministic and offline; degenerate cases return a clear status. No predictive
or trading claim.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESEARCH_DISCLAIMER = (
    "This output is for analytical and self-directed research purposes. It is not "
    "personalized financial advice. Drift monitoring is a research diagnostic."
)

STATUS_OK = "ok"
STATUS_TOO_FEW_PERIODS = "too_few_periods"
STATUS_ALL_MISSING = "all_missing"

_MIN_PERIODS = 2


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


@dataclass(frozen=True)
class MetricMonitor:
    name: str
    status: str
    n_periods: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    stability_band: dict[str, float | None]
    worst_period: dict[str, Any] | None
    best_period: dict[str, Any] | None
    flagged_periods: list[dict[str, Any]]
    per_period: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "n_periods": self.n_periods,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "stability_band": self.stability_band,
            "worst_period": self.worst_period,
            "best_period": self.best_period,
            "flagged_periods": self.flagged_periods,
            "per_period": self.per_period,
        }


def monitor_metric(
    periods: Sequence[str],
    values: Sequence[float | None],
    name: str,
    *,
    window: int = 3,
    z_threshold: float = 2.0,
) -> MetricMonitor:
    """Rolling drift stats + threshold flags for one metric series."""
    if len(periods) != len(values):
        raise ValueError("periods and values must align")
    present = [(p, float(v)) for p, v in zip(periods, values, strict=True) if v is not None]
    if not present:
        return MetricMonitor(
            name, STATUS_ALL_MISSING, 0, None, None, None, None,
            {"lower": None, "upper": None}, None, None, [], [],
        )

    per_period: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    seen: list[float] = []
    for period, value in zip(periods, values, strict=True):
        if value is None:
            per_period.append({"period": period, "value": None, "status": "missing"})
            continue
        trailing = seen[-window:]
        roll_mean = _mean(trailing)
        roll_std = _std(trailing)
        if roll_mean is not None and roll_std not in (None, 0.0):
            zscore = (value - roll_mean) / roll_std  # type: ignore[operator]
        else:
            zscore = None  # zero / undefined trailing std -> no z (never div0)
        flag = zscore is not None and abs(zscore) > z_threshold
        row = {
            "period": period,
            "value": value,
            "rolling_mean": roll_mean,
            "rolling_std": roll_std,
            "zscore": zscore,
            "flagged": flag,
        }
        per_period.append(row)
        if flag:
            flagged.append(row)
        seen.append(value)

    series = [v for _p, v in present]
    mean = _mean(series)
    std = _std(series)
    band = {
        "lower": (mean - 2.0 * std) if (mean is not None and std is not None) else None,
        "upper": (mean + 2.0 * std) if (mean is not None and std is not None) else None,
    }
    worst = min(present, key=lambda pv: pv[1])
    best = max(present, key=lambda pv: pv[1])
    status = STATUS_OK if len(present) >= _MIN_PERIODS else STATUS_TOO_FEW_PERIODS
    return MetricMonitor(
        name=name,
        status=status,
        n_periods=len(present),
        mean=mean,
        std=std,
        min=worst[1],
        max=best[1],
        stability_band=band,
        worst_period={"period": worst[0], "value": worst[1]},
        best_period={"period": best[0], "value": best[1]},
        flagged_periods=flagged,
        per_period=per_period,
    )


@dataclass(frozen=True)
class MonitoringReport:
    is_synthetic: bool
    window: int
    z_threshold: float
    metrics: dict[str, MetricMonitor]
    disclaimer: str = RESEARCH_DISCLAIMER
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "research_only": True,
            "is_synthetic": self.is_synthetic,
            "synthetic_warning": (
                "SYNTHETIC FIXTURE RESULTS — not real market evidence."
                if self.is_synthetic
                else None
            ),
            "window": self.window,
            "z_threshold": self.z_threshold,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "warnings": self.warnings,
        }


def build_monitoring_report(
    periods: Sequence[str],
    metric_columns: Mapping[str, Sequence[float | None]],
    *,
    window: int = 3,
    z_threshold: float = 2.0,
    is_synthetic: bool = False,
) -> MonitoringReport:
    """Monitor several metric columns over the same ordered periods."""
    warnings: list[str] = []
    if len(periods) < _MIN_PERIODS:
        warnings.append(f"only {len(periods)} period(s): drift statistics are unreliable")
    metrics = {
        name: monitor_metric(periods, values, name, window=window, z_threshold=z_threshold)
        for name, values in metric_columns.items()
    }
    return MonitoringReport(
        is_synthetic=is_synthetic,
        window=window,
        z_threshold=z_threshold,
        metrics=metrics,
        warnings=warnings,
    )


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def write_monitoring_outputs(
    report: MonitoringReport, output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "monitoring.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = out_dir / "monitoring.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["metric", "status", "n_periods", "mean", "std", "min", "max", "n_flagged"]
        )
        for m in report.metrics.values():
            writer.writerow(
                [
                    m.name,
                    m.status,
                    m.n_periods,
                    _fmt(m.mean),
                    _fmt(m.std),
                    _fmt(m.min),
                    _fmt(m.max),
                    len(m.flagged_periods),
                ]
            )
    paths = {"json_path": json_path, "csv_path": csv_path}
    if write_markdown:
        md_path = out_dir / "monitoring.md"
        md_path.write_text(_markdown(report), encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


def _markdown(report: MonitoringReport) -> str:
    lines = ["# Model Monitoring & Drift (research diagnostics)", "", report.disclaimer, ""]
    if report.is_synthetic:
        lines += ["> **SYNTHETIC FIXTURE RESULTS — not real market evidence.**", ""]
    lines += [
        f"- Window {report.window}, z-threshold {report.z_threshold}",
        "",
        "| metric | status | n | mean | std | min | max | flagged |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in report.metrics.values():
        lines.append(
            f"| {m.name} | {m.status} | {m.n_periods} | {_fmt(m.mean)} | {_fmt(m.std)} | "
            f"{_fmt(m.min)} | {_fmt(m.max)} | {len(m.flagged_periods)} |"
        )
    if report.warnings:
        lines += ["", *[f"- _{w}_" for w in report.warnings]]
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "STATUS_ALL_MISSING",
    "STATUS_OK",
    "STATUS_TOO_FEW_PERIODS",
    "MetricMonitor",
    "MonitoringReport",
    "build_monitoring_report",
    "monitor_metric",
    "write_monitoring_outputs",
]

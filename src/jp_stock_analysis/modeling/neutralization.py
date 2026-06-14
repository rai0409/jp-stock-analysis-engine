"""Numerai-style neutralization and neutralized rank metrics (research-only).

Inspired by neutralized cross-sectional ranking concepts (regression-residual
feature/market neutralization, rank correlation, and a contribution delta). It is
**not** official Numerai scoring unless exactly matched, and it makes no
predictive or trading claim.

Core pieces:

- :func:`neutralize` — regression-residual neutralization of a prediction against
  arbitrary numeric exposure columns (deterministic least squares).
- :func:`neutralized_rank_ic` — neutralized Rank IC per decision date + ICIR.
- :func:`mmc_style_contribution` — an **MMC-STYLE** contribution delta of a
  candidate prediction after neutralizing against a base prediction (named
  "MMC-style", not official Numerai MMC).
- :func:`exposure_diagnostics` — pre/post-neutralization exposure correlations.

Degenerate cross-sections (constant prediction, no usable exposures, too few
points) return an explicit status, never an exception, and never fabricate.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from jp_stock_analysis.modeling.ranking_metrics import RESEARCH_DISCLAIMER, spearman

STATUS_OK = "ok"
STATUS_CONSTANT_PREDICTION = "constant_prediction"
STATUS_NO_EXPOSURES_USED = "no_exposures_used"
STATUS_INSUFFICIENT_POINTS = "insufficient_points"
STATUS_INSUFFICIENT_MODELS = "insufficient_models"


@dataclass(frozen=True)
class ExposureObservation:
    """One observation with a prediction, exposures, and a realised return."""

    decision_date: date
    ticker: str
    prediction: float | None
    forward_return: float | None
    exposures: dict[str, float | None] = field(default_factory=dict)
    sector: str | None = None


@dataclass(frozen=True)
class NeutralizationResult:
    neutralized: list[float | None]
    exposure_columns_used: list[str]
    skipped_exposures: dict[str, str]
    status: str
    proportion: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "proportion": self.proportion,
            "exposure_columns_used": self.exposure_columns_used,
            "skipped_exposures": self.skipped_exposures,
        }


def _column_present_mean(values: Sequence[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def neutralize(
    predictions: Sequence[float | None],
    exposures: Mapping[str, Sequence[float | None]],
    exposure_columns: Sequence[str],
    *,
    proportion: float = 1.0,
) -> NeutralizationResult:
    """Residual-neutralize ``predictions`` against the requested exposures.

    ``neutralized = prediction - proportion * X @ lstsq(X, prediction)`` over the
    rows where the prediction is present, with an intercept column. Requested
    columns that are absent, all-missing, or constant are reported in
    ``skipped_exposures`` (never silently ignored). Missing values inside a used
    column are mean-imputed for the design matrix only.
    """
    n = len(predictions)
    present_idx = [i for i, v in enumerate(predictions) if v is not None]
    skipped: dict[str, str] = {}
    used: list[str] = []
    columns: list[list[float]] = []

    for name in exposure_columns:
        if name not in exposures:
            skipped[name] = "missing_column"
            continue
        raw = exposures[name]
        if len(raw) != n:
            skipped[name] = "length_mismatch"
            continue
        mean = _column_present_mean(raw)
        if mean is None:
            skipped[name] = "all_missing"
            continue
        filled = [float(raw[i]) if raw[i] is not None else mean for i in present_idx]
        if max(filled) == min(filled):
            skipped[name] = "constant"
            continue
        columns.append(filled)
        used.append(name)

    neutralized: list[float | None] = list(predictions)

    def _result(status: str) -> NeutralizationResult:
        return NeutralizationResult(neutralized, used, skipped, status, proportion)

    if len(present_idx) < 2:
        return _result(STATUS_INSUFFICIENT_POINTS)

    y = np.asarray([float(predictions[i]) for i in present_idx], dtype=float)  # type: ignore[arg-type]
    if float(y.max()) == float(y.min()):
        return _result(STATUS_CONSTANT_PREDICTION)
    if not used:
        return _result(STATUS_NO_EXPOSURES_USED)

    design = np.column_stack([np.ones(len(present_idx))] + [np.asarray(c) for c in columns])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - proportion * (design @ beta)
    out = list(predictions)
    for position, i in enumerate(present_idx):
        out[i] = float(residual[position])
    return NeutralizationResult(out, used, skipped, STATUS_OK, proportion)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    if float(ax.std()) == 0.0 or float(ay.std()) == 0.0:
        return None
    return float(np.corrcoef(ax, ay)[0, 1])


@dataclass(frozen=True)
class ExposureDiagnostics:
    pre_neutralization_exposure_corr: dict[str, float | None]
    post_neutralization_exposure_corr: dict[str, float | None]
    max_abs_exposure_corr_before: float | None
    max_abs_exposure_corr_after: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_neutralization_exposure_corr": self.pre_neutralization_exposure_corr,
            "post_neutralization_exposure_corr": self.post_neutralization_exposure_corr,
            "max_abs_exposure_corr_before": self.max_abs_exposure_corr_before,
            "max_abs_exposure_corr_after": self.max_abs_exposure_corr_after,
        }


def exposure_diagnostics(
    predictions: Sequence[float | None],
    neutralized: Sequence[float | None],
    exposures: Mapping[str, Sequence[float | None]],
    exposure_columns: Sequence[str],
) -> ExposureDiagnostics:
    """Correlation of prediction vs each exposure, before and after neutralizing."""
    pre: dict[str, float | None] = {}
    post: dict[str, float | None] = {}
    for name in exposure_columns:
        if name not in exposures:
            continue
        raw = exposures[name]
        pre_pairs = [
            (float(predictions[i]), float(raw[i]))
            for i in range(len(predictions))
            if predictions[i] is not None and raw[i] is not None
        ]
        post_pairs = [
            (float(neutralized[i]), float(raw[i]))
            for i in range(len(neutralized))
            if neutralized[i] is not None and raw[i] is not None
        ]
        pre[name] = _pearson([p for p, _ in pre_pairs], [e for _, e in pre_pairs])
        post[name] = _pearson([p for p, _ in post_pairs], [e for _, e in post_pairs])
    pre_abs = [abs(v) for v in pre.values() if v is not None]
    post_abs = [abs(v) for v in post.values() if v is not None]
    return ExposureDiagnostics(
        pre_neutralization_exposure_corr=pre,
        post_neutralization_exposure_corr=post,
        max_abs_exposure_corr_before=max(pre_abs) if pre_abs else None,
        max_abs_exposure_corr_after=max(post_abs) if post_abs else None,
    )


@dataclass(frozen=True)
class NeutralizedICReport:
    horizon: int
    exposure_columns: list[str]
    proportion: float
    is_synthetic: bool
    rank_ic_by_date: dict[str, float | None]
    ic_mean: float | None
    ic_std: float | None
    icir: float | None
    raw_ic_mean: float | None
    diagnostics: ExposureDiagnostics
    skipped_exposures: dict[str, str]
    status: str
    model_label: str = "baseline_factor_ranker"
    disclaimer: str = RESEARCH_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "research_only": True,
            "model_label": self.model_label,
            "horizon": self.horizon,
            "is_synthetic": self.is_synthetic,
            "synthetic_warning": (
                "SYNTHETIC FIXTURE RESULTS — not real market evidence."
                if self.is_synthetic
                else None
            ),
            "status": self.status,
            "exposure_columns": self.exposure_columns,
            "proportion": self.proportion,
            "neutralized_ic_mean": self.ic_mean,
            "neutralized_ic_std": self.ic_std,
            "neutralized_icir": self.icir,
            "raw_ic_mean": self.raw_ic_mean,
            "rank_ic_by_date": self.rank_ic_by_date,
            "exposure_diagnostics": self.diagnostics.to_dict(),
            "skipped_exposures": self.skipped_exposures,
        }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def neutralized_rank_ic(
    observations: Sequence[ExposureObservation],
    *,
    horizon: int,
    exposure_columns: Sequence[str],
    proportion: float = 1.0,
    is_synthetic: bool = False,
    model_label: str = "baseline_factor_ranker",
) -> NeutralizedICReport:
    """Neutralize predictions per decision date, then Rank IC vs forward return."""
    by_date: dict[date, list[ExposureObservation]] = {}
    for obs in observations:
        by_date.setdefault(obs.decision_date, []).append(obs)

    ics: dict[date, float | None] = {}
    raw_ics: list[float] = []
    all_pred: list[float | None] = []
    all_neutral: list[float | None] = []
    all_exposures: dict[str, list[float | None]] = {name: [] for name in exposure_columns}
    skipped_union: dict[str, str] = {}

    for decision_date in sorted(by_date):
        group = by_date[decision_date]
        preds = [o.prediction for o in group]
        exposures = {
            name: [o.exposures.get(name) for o in group] for name in exposure_columns
        }
        result = neutralize(preds, exposures, exposure_columns, proportion=proportion)
        skipped_union.update(result.skipped_exposures)
        returns = [o.forward_return for o in group]
        pairs = [
            (result.neutralized[i], returns[i])
            for i in range(len(group))
            if result.neutralized[i] is not None and returns[i] is not None
        ]
        raw_pairs = [
            (preds[i], returns[i])
            for i in range(len(group))
            if preds[i] is not None and returns[i] is not None
        ]
        ics[decision_date] = (
            spearman([p for p, _ in pairs], [r for _, r in pairs]) if len(pairs) >= 2 else None
        )
        raw_ic = (
            spearman([p for p, _ in raw_pairs], [r for _, r in raw_pairs])
            if len(raw_pairs) >= 2
            else None
        )
        if raw_ic is not None:
            raw_ics.append(raw_ic)
        all_pred.extend(preds)
        all_neutral.extend(result.neutralized)
        for name in exposure_columns:
            all_exposures[name].extend(exposures[name])

    ic_values = [v for v in ics.values() if v is not None]
    ic_mean = _mean(ic_values)
    ic_std = _std(ic_values)
    icir = ic_mean / ic_std if (ic_mean is not None and ic_std not in (None, 0.0)) else None
    diagnostics = exposure_diagnostics(all_pred, all_neutral, all_exposures, exposure_columns)
    status = STATUS_OK if ic_values else STATUS_INSUFFICIENT_POINTS

    return NeutralizedICReport(
        horizon=horizon,
        exposure_columns=list(exposure_columns),
        proportion=proportion,
        is_synthetic=is_synthetic,
        rank_ic_by_date={d.isoformat(): ics[d] for d in sorted(ics)},
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        raw_ic_mean=_mean(raw_ics),
        diagnostics=diagnostics,
        skipped_exposures=dict(sorted(skipped_union.items())),
        status=status,
        model_label=model_label,
    )


@dataclass(frozen=True)
class MMCStyleObservation:
    decision_date: date
    ticker: str
    base_prediction: float | None
    candidate_prediction: float | None
    forward_return: float | None


@dataclass(frozen=True)
class MMCStyleReport:
    horizon: int
    is_synthetic: bool
    status: str
    base_ic_mean: float | None
    candidate_ic_mean: float | None
    neutralized_candidate_ic_mean: float | None
    contribution_delta: float | None
    delta_vs_base: float | None
    by_date: dict[str, float | None]
    disclaimer: str = RESEARCH_DISCLAIMER
    caveat: str = (
        "MMC-STYLE only: the candidate's mean neutralized-vs-base Rank IC. This is "
        "NOT official Numerai Meta Model Contribution unless exactly matched."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "caveat": self.caveat,
            "research_only": True,
            "horizon": self.horizon,
            "is_synthetic": self.is_synthetic,
            "synthetic_warning": (
                "SYNTHETIC FIXTURE RESULTS — not real market evidence."
                if self.is_synthetic
                else None
            ),
            "status": self.status,
            "base_ic_mean": self.base_ic_mean,
            "candidate_ic_mean": self.candidate_ic_mean,
            "neutralized_candidate_ic_mean": self.neutralized_candidate_ic_mean,
            "contribution_delta": self.contribution_delta,
            "delta_vs_base": self.delta_vs_base,
            "neutralized_ic_by_date": self.by_date,
        }


def _ic_vs_return(
    values: Sequence[float | None], returns: Sequence[float | None]
) -> float | None:
    """Spearman of aligned non-missing (value, return) pairs; None if <2 pairs."""
    pairs = [
        (values[i], returns[i])
        for i in range(len(values))
        if values[i] is not None and returns[i] is not None
    ]
    if len(pairs) < 2:
        return None
    return spearman([v for v, _ in pairs], [r for _, r in pairs])


def mmc_style_contribution(
    observations: Sequence[MMCStyleObservation],
    *,
    horizon: int,
    is_synthetic: bool = False,
) -> MMCStyleReport:
    """MMC-style: candidate Rank IC after neutralizing against the base model."""
    by_date: dict[date, list[MMCStyleObservation]] = {}
    for obs in observations:
        if obs.base_prediction is None and obs.candidate_prediction is None:
            continue
        by_date.setdefault(obs.decision_date, []).append(obs)

    base_ics: list[float] = []
    cand_ics: list[float] = []
    neutral_ics: list[float] = []
    by_date_out: dict[str, float | None] = {}

    for decision_date in sorted(by_date):
        group = by_date[decision_date]
        base = [o.base_prediction for o in group]
        candidate = [o.candidate_prediction for o in group]
        returns = [o.forward_return for o in group]
        result = neutralize(candidate, {"base": base}, ["base"], proportion=1.0)
        base_ic = _ic_vs_return(base, returns)
        cand_ic = _ic_vs_return(candidate, returns)
        neutral_ic = _ic_vs_return(result.neutralized, returns)
        by_date_out[decision_date.isoformat()] = neutral_ic
        if base_ic is not None:
            base_ics.append(base_ic)
        if cand_ic is not None:
            cand_ics.append(cand_ic)
        if neutral_ic is not None:
            neutral_ics.append(neutral_ic)

    if not neutral_ics:
        status = STATUS_INSUFFICIENT_POINTS
    else:
        status = STATUS_OK
    base_mean = _mean(base_ics)
    cand_mean = _mean(cand_ics)
    neutral_mean = _mean(neutral_ics)
    delta_vs_base = (
        cand_mean - base_mean if (cand_mean is not None and base_mean is not None) else None
    )
    return MMCStyleReport(
        horizon=horizon,
        is_synthetic=is_synthetic,
        status=status,
        base_ic_mean=base_mean,
        candidate_ic_mean=cand_mean,
        neutralized_candidate_ic_mean=neutral_mean,
        contribution_delta=neutral_mean,
        delta_vs_base=delta_vs_base,
        by_date=by_date_out,
    )


def sector_dummy_columns(sectors: Sequence[str | None]) -> dict[str, list[float]]:
    """One-hot sector exposure columns (deterministic, sorted sector names)."""
    names = sorted({s for s in sectors if s})
    return {
        f"sector::{name}": [1.0 if s == name else 0.0 for s in sectors] for name in names
    }


def write_neutralized_outputs(
    report: NeutralizedICReport,
    output_dir: str | Path,
    *,
    mmc: MMCStyleReport | None = None,
    write_markdown: bool = True,
) -> dict[str, Path]:
    """Write neutralized_metrics.json / .csv (+ optional .md)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    if mmc is not None:
        payload["mmc_style"] = mmc.to_dict()
    json_path = out_dir / "neutralized_metrics.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = out_dir / "neutralized_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["decision_date", "neutralized_rank_ic"])
        for d, v in report.rank_ic_by_date.items():
            writer.writerow([d, "" if v is None else f"{v:.6f}"])
    paths = {"json_path": json_path, "csv_path": csv_path}
    if write_markdown:
        md_path = out_dir / "neutralized_metrics.md"
        md_path.write_text(_markdown(report, mmc), encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _markdown(report: NeutralizedICReport, mmc: MMCStyleReport | None) -> str:
    lines = ["# Numerai-Style Neutralized Rank Metrics", "", report.disclaimer, ""]
    lines.append(
        "Inspired by neutralized cross-sectional ranking concepts; NOT official "
        "Numerai scoring unless exactly matched. Research-only; no trading claim."
    )
    lines.append("")
    if report.is_synthetic:
        lines += ["> **SYNTHETIC FIXTURE RESULTS — not real market evidence.**", ""]
    lines += [
        f"- Model: `{report.model_label}`  |  horizon: {report.horizon}  |  "
        f"status: **{report.status}**",
        f"- Exposures: {report.exposure_columns}  |  proportion: {report.proportion}",
        f"- Raw IC mean: {_fmt(report.raw_ic_mean)}  ->  "
        f"**Neutralized IC mean: {_fmt(report.ic_mean)}**  |  ICIR: {_fmt(report.icir)}",
        f"- Max |exposure corr| before/after: "
        f"{_fmt(report.diagnostics.max_abs_exposure_corr_before)} / "
        f"{_fmt(report.diagnostics.max_abs_exposure_corr_after)}",
    ]
    if report.skipped_exposures:
        lines.append(f"- Skipped exposures: {report.skipped_exposures}")
    if mmc is not None:
        lines += [
            "",
            "## MMC-style contribution delta",
            "",
            mmc.caveat,
            "",
            f"- status: **{mmc.status}**  |  base IC: {_fmt(mmc.base_ic_mean)}  |  "
            f"candidate IC: {_fmt(mmc.candidate_ic_mean)}",
            f"- neutralized-vs-base IC (contribution delta): "
            f"**{_fmt(mmc.contribution_delta)}**  |  delta vs base: {_fmt(mmc.delta_vs_base)}",
        ]
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "STATUS_CONSTANT_PREDICTION",
    "STATUS_INSUFFICIENT_MODELS",
    "STATUS_INSUFFICIENT_POINTS",
    "STATUS_NO_EXPOSURES_USED",
    "STATUS_OK",
    "ExposureDiagnostics",
    "ExposureObservation",
    "MMCStyleObservation",
    "MMCStyleReport",
    "NeutralizationResult",
    "NeutralizedICReport",
    "exposure_diagnostics",
    "mmc_style_contribution",
    "neutralize",
    "neutralized_rank_ic",
    "sector_dummy_columns",
    "write_neutralized_outputs",
]

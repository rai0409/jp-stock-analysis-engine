"""Model stability across folds and seeds (research diagnostics only).

Summarises how stable a metric (Rank IC, neutralized Rank IC, long-short spread,
Sharpe-like, hit rate) is across walk-forward folds, and a synthetic seed-noise
probe for robustness. Deterministic and offline; no predictive or trading claim.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jp_stock_analysis.modeling.portfolio_metrics import (
    evaluate_portfolio,
    observations_from_scored,
)
from jp_stock_analysis.modeling.ranking_metrics import (
    RESEARCH_DISCLAIMER,
    ScoredObservation,
    evaluate_horizon,
)
from jp_stock_analysis.modeling.walk_forward import WalkForwardFold

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_ALL_MISSING = "all_missing"
STATUS_SINGLE_FOLD = "single_fold"
STATUS_DETERMINISTIC = "deterministic_model"


@dataclass(frozen=True)
class MetricStability:
    name: str
    n: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    cv: float | None
    positive_period_rate: float | None
    worst_fold: dict[str, Any] | None
    best_fold: dict[str, Any] | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "cv": self.cv,
            "positive_period_rate": self.positive_period_rate,
            "worst_fold": self.worst_fold,
            "best_fold": self.best_fold,
            "status": self.status,
        }


def summarize_metric(values: Sequence[float | None], name: str) -> MetricStability:
    """Mean/std/min/max/CV/positive-rate of a metric across folds or seeds."""
    if not values:
        return MetricStability(
            name, 0, None, None, None, None, None, None, None, None, STATUS_EMPTY
        )
    indexed_present = [(i, float(v)) for i, v in enumerate(values) if v is not None]
    if not indexed_present:
        return MetricStability(
            name, 0, None, None, None, None, None, None, None, None, STATUS_ALL_MISSING
        )
    present = [v for _i, v in indexed_present]
    n = len(present)
    mean = sum(present) / n
    if n < 2:
        std: float | None = None
    elif max(present) == min(present):
        std = 0.0  # all identical: snap to exact zero (no float dust)
    else:
        std = math.sqrt(sum((v - mean) ** 2 for v in present) / (n - 1))
    cv = (std / abs(mean)) if (std is not None and mean != 0.0) else None
    positive_rate = sum(1 for v in present if v > 0) / n
    worst_i, worst_v = min(indexed_present, key=lambda iv: iv[1])
    best_i, best_v = max(indexed_present, key=lambda iv: iv[1])
    status = STATUS_OK if n >= 2 else STATUS_SINGLE_FOLD
    return MetricStability(
        name=name,
        n=n,
        mean=mean,
        std=std,
        min=worst_v,
        max=best_v,
        cv=cv,
        positive_period_rate=positive_rate,
        worst_fold={"index": worst_i, "value": worst_v},
        best_fold={"index": best_i, "value": best_v},
        status=status,
    )


def _fold_subset(
    scored: Sequence[ScoredObservation], fold: WalkForwardFold
) -> list[ScoredObservation]:
    test_dates = set(fold.test_periods)
    return [o for o in scored if o.decision_date in test_dates]


def compute_fold_metrics(
    scored: Sequence[ScoredObservation],
    folds: Sequence[WalkForwardFold],
    *,
    horizon: int,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
) -> dict[str, list[float | None]]:
    """Per-fold Rank IC, long-short spread, Sharpe-like, and hit rate."""
    metrics: dict[str, list[float | None]] = {
        "rank_ic": [],
        "long_short_spread": [],
        "sharpe_like": [],
        "hit_rate": [],
    }
    for fold in folds:
        subset = _fold_subset(scored, fold)
        ic = evaluate_horizon(subset, horizon).ic_mean if subset else None
        portfolio = evaluate_portfolio(
            observations_from_scored(subset, horizon),
            horizon=horizon,
            top_quantile=top_quantile,
            bottom_quantile=bottom_quantile,
        )
        metrics["rank_ic"].append(ic)
        metrics["long_short_spread"].append(portfolio.series.mean_spread)
        metrics["sharpe_like"].append(portfolio.series.sharpe_like)
        metrics["hit_rate"].append(portfolio.series.hit_rate)
    return metrics


def synthetic_seed_ic(
    scored: Sequence[ScoredObservation],
    *,
    horizon: int,
    seeds: Sequence[int],
    noise_scale: float = 0.05,
) -> list[float | None]:
    """Rank IC under deterministic seeded score noise (a robustness probe).

    SYNTHETIC ONLY: perturbs the prediction with seeded Gaussian noise scaled to
    the score spread and recomputes Rank IC. With a deterministic model the
    *model* does not change; this probes ranking sensitivity to small noise.
    """
    base = [float(o.score) for o in scored if o.score is not None]
    spread = (max(base) - min(base)) if base else 0.0
    out: list[float | None] = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        noise = rng.normal(0.0, noise_scale * spread, size=len(scored))
        perturbed = [
            ScoredObservation(
                decision_date=o.decision_date,
                ticker=o.ticker,
                score=(None if o.score is None else float(o.score) + float(noise[i])),
                sector=o.sector,
                labels=o.labels,
            )
            for i, o in enumerate(scored)
        ]
        out.append(evaluate_horizon(perturbed, horizon).ic_mean)
    return out


@dataclass(frozen=True)
class StabilityReport:
    horizon: int
    is_synthetic: bool
    fold_stability: dict[str, MetricStability]
    seed_stability: MetricStability | None
    n_folds: int
    disclaimer: str = RESEARCH_DISCLAIMER
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "research_only": True,
            "horizon": self.horizon,
            "is_synthetic": self.is_synthetic,
            "synthetic_warning": (
                "SYNTHETIC FIXTURE RESULTS — not real market evidence."
                if self.is_synthetic
                else None
            ),
            "n_folds": self.n_folds,
            "fold_stability": {k: v.to_dict() for k, v in self.fold_stability.items()},
            "seed_stability": self.seed_stability.to_dict() if self.seed_stability else None,
            "warnings": self.warnings,
        }


def build_stability_report(
    fold_metrics: Mapping[str, Sequence[float | None]],
    *,
    horizon: int,
    is_synthetic: bool = False,
    seed_ic: Sequence[float | None] | None = None,
    deterministic_model: bool = True,
) -> StabilityReport:
    """Summarise per-fold metrics (+ optional seed probe) into a report."""
    fold_stability = {
        name: summarize_metric(values, name) for name, values in fold_metrics.items()
    }
    n_folds = max((len(v) for v in fold_metrics.values()), default=0)
    seed_summary: MetricStability | None = None
    warnings: list[str] = []
    if seed_ic is not None:
        seed_summary = summarize_metric(seed_ic, "seed_rank_ic")
        if deterministic_model and seed_summary.std == 0.0:
            warnings.append(
                "deterministic model: seed variance is 0 (seed noise probe only)"
            )
    elif deterministic_model:
        warnings.append("deterministic model: per-seed retraining variance not applicable")
    return StabilityReport(
        horizon=horizon,
        is_synthetic=is_synthetic,
        fold_stability=fold_stability,
        seed_stability=seed_summary,
        n_folds=n_folds,
        warnings=warnings,
    )


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def write_stability_outputs(
    report: StabilityReport, output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "model_stability.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = out_dir / "model_stability.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["metric", "n", "mean", "std", "min", "max", "cv", "positive_period_rate", "status"]
        )
        for metric in report.fold_stability.values():
            writer.writerow(
                [
                    metric.name,
                    metric.n,
                    _fmt(metric.mean),
                    _fmt(metric.std),
                    _fmt(metric.min),
                    _fmt(metric.max),
                    _fmt(metric.cv),
                    _fmt(metric.positive_period_rate),
                    metric.status,
                ]
            )
    paths = {"json_path": json_path, "csv_path": csv_path}
    if write_markdown:
        md_path = out_dir / "model_stability.md"
        md_path.write_text(_markdown(report), encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


def _markdown(report: StabilityReport) -> str:
    lines = ["# Model Stability (research diagnostics)", "", report.disclaimer, ""]
    if report.is_synthetic:
        lines += ["> **SYNTHETIC FIXTURE RESULTS — not real market evidence.**", ""]
    lines += [
        f"- Horizon {report.horizon}, folds: {report.n_folds}",
        "",
        "| metric | n | mean | std | min | max | CV | positive rate | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in report.fold_stability.values():
        lines.append(
            f"| {m.name} | {m.n} | {_fmt(m.mean)} | {_fmt(m.std)} | {_fmt(m.min)} | "
            f"{_fmt(m.max)} | {_fmt(m.cv)} | {_fmt(m.positive_period_rate)} | {m.status} |"
        )
    if report.seed_stability is not None:
        s = report.seed_stability
        lines += [
            "",
            f"- Seed probe ({s.name}): mean {_fmt(s.mean)}, std {_fmt(s.std)}, status {s.status}",
        ]
    if report.warnings:
        lines += ["", *[f"- _{w}_" for w in report.warnings]]
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "STATUS_ALL_MISSING",
    "STATUS_DETERMINISTIC",
    "STATUS_EMPTY",
    "STATUS_OK",
    "STATUS_SINGLE_FOLD",
    "MetricStability",
    "StabilityReport",
    "build_stability_report",
    "compute_fold_metrics",
    "summarize_metric",
    "synthetic_seed_ic",
    "write_stability_outputs",
]

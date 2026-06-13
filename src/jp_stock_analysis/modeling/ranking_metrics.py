"""Cross-sectional ranking validation metrics (research-only).

Given a per-observation *score* (from the baseline factor ranker or an ML model)
and realised forward-return labels, this measures how well the score *ranks*
names cross-sectionally — Rank IC / Spearman, ICIR, quantile spreads, decile
tables, hit rates. It does **not** measure or claim trading profitability and
emits no buy/sell labels.

All statistics are computed without scipy: Spearman is Pearson on average ranks,
with explicit zero-variance handling (returns ``None``, never NaN). Output is
deterministic and carries a research-only / not-financial-advice disclaimer; on
synthetic data it is labelled synthetic and is not market evidence.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

RESEARCH_DISCLAIMER = (
    "This output is for analytical and self-directed research purposes. It is "
    "not personalized financial advice. It measures cross-sectional ranking "
    "association only and makes no predictive or trading-profitability claim."
)


@dataclass(frozen=True)
class ScoredObservation:
    """One scored observation with its realised labels."""

    decision_date: date
    ticker: str
    score: float | None
    sector: str | None
    labels: dict[str, float | None] = field(default_factory=dict)


def _ranks(values: Sequence[float]) -> list[float]:
    """Average (fractional) ranks, ties shared. 1-based."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None  # a constant series has no defined correlation
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    return sxy / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation; ``None`` for <2 points or zero variance."""
    if len(x) != len(y):
        raise ValueError("x and y must align")
    if len(x) < 2:
        return None
    return _pearson(_ranks(x), _ranks(y))


def _pairs(
    observations: Sequence[ScoredObservation], label_key: str, sector_neutral: bool
) -> dict[date, tuple[list[float], list[float]]]:
    """Aligned (score, label) lists per date, optionally sector-demeaned."""
    by_date: dict[date, list[ScoredObservation]] = {}
    for obs in observations:
        if obs.score is None or obs.labels.get(label_key) is None:
            continue
        by_date.setdefault(obs.decision_date, []).append(obs)

    out: dict[date, tuple[list[float], list[float]]] = {}
    for decision_date, group in by_date.items():
        scores = [float(o.score) for o in group]  # type: ignore[arg-type]
        labels = [float(o.labels[label_key]) for o in group]  # type: ignore[arg-type]
        if sector_neutral:
            scores = _sector_demean(scores, [o.sector for o in group])
            labels = _sector_demean(labels, [o.sector for o in group])
        out[decision_date] = (scores, labels)
    return out


def _sector_demean(values: list[float], sectors: Sequence[str | None]) -> list[float]:
    groups: dict[str | None, list[int]] = {}
    for index, sector in enumerate(sectors):
        groups.setdefault(sector, []).append(index)
    out = list(values)
    for indices in groups.values():
        mean = sum(values[i] for i in indices) / len(indices)
        for i in indices:
            out[i] = values[i] - mean
    return out


def _quantile_buckets(scores: Sequence[float], n: int) -> list[int]:
    """Assign each score to a quantile bucket 0..n-1 by rank (n-1 = top)."""
    ranks = _ranks(scores)
    size = len(scores)
    buckets = []
    for r in ranks:
        bucket = int((r - 1) / size * n)
        buckets.append(min(bucket, n - 1))
    return buckets


@dataclass(frozen=True)
class HorizonRankingMetrics:
    horizon: int
    label_key: str
    n_dates: int
    coverage_count: int
    missing_label_count: int
    ic_mean: float | None
    ic_std: float | None
    icir: float | None
    rank_ic_by_date: dict[str, float | None]
    sector_neutral_ic_mean: float | None
    quantile_spread_mean: float | None
    quantile_return_table: dict[str, float | None]
    hit_rate_top_positive: float | None
    hit_rate_top_above_median: float | None
    n_quantiles: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "label_key": self.label_key,
            "n_dates": self.n_dates,
            "coverage_count": self.coverage_count,
            "missing_label_count": self.missing_label_count,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "icir": self.icir,
            "rank_ic_by_date": self.rank_ic_by_date,
            "sector_neutral_ic_mean": self.sector_neutral_ic_mean,
            "quantile_spread_mean": self.quantile_spread_mean,
            "quantile_return_table": self.quantile_return_table,
            "hit_rate_top_positive": self.hit_rate_top_positive,
            "hit_rate_top_above_median": self.hit_rate_top_above_median,
            "n_quantiles": self.n_quantiles,
        }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def evaluate_horizon(
    observations: Sequence[ScoredObservation],
    horizon: int,
    *,
    label_prefix: str = "forward_return_h",
    n_quantiles: int = 5,
) -> HorizonRankingMetrics:
    """Compute all ranking metrics for one horizon."""
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")
    label_key = f"{label_prefix}{horizon}"
    scored = [o for o in observations if o.score is not None]
    coverage = sum(1 for o in scored if o.labels.get(label_key) is not None)
    missing = sum(1 for o in scored if o.labels.get(label_key) is None)

    pairs = _pairs(observations, label_key, sector_neutral=False)
    ics = {d: spearman(s, lbl) for d, (s, lbl) in pairs.items()}
    ic_values = [v for v in ics.values() if v is not None]
    ic_mean = _mean(ic_values)
    ic_std = _std(ic_values)
    icir = ic_mean / ic_std if (ic_mean is not None and ic_std not in (None, 0)) else None

    neutral_pairs = _pairs(observations, label_key, sector_neutral=True)
    neutral_ics = [
        v for v in (spearman(s, lbl) for s, lbl in neutral_pairs.values()) if v is not None
    ]
    sector_neutral_ic_mean = _mean(neutral_ics)

    spreads: list[float] = []
    quantile_returns: dict[int, list[float]] = {q: [] for q in range(n_quantiles)}
    top_positive = 0
    top_above_median = 0
    n_quantile_dates = 0
    for _date, (scores, labels) in pairs.items():
        if len(scores) < n_quantiles:
            continue
        n_quantile_dates += 1
        buckets = _quantile_buckets(scores, n_quantiles)
        per_bucket: dict[int, list[float]] = {q: [] for q in range(n_quantiles)}
        for bucket, label in zip(buckets, labels, strict=True):
            per_bucket[bucket].append(label)
        for q in range(n_quantiles):
            if per_bucket[q]:
                quantile_returns[q].append(sum(per_bucket[q]) / len(per_bucket[q]))
        top = per_bucket[n_quantiles - 1]
        bottom = per_bucket[0]
        if top and bottom:
            spreads.append(sum(top) / len(top) - sum(bottom) / len(bottom))
        if top:
            top_mean = sum(top) / len(top)
            if top_mean > 0:
                top_positive += 1
            universe_median = _median(labels)
            if universe_median is not None and top_mean > universe_median:
                top_above_median += 1

    quantile_table = {
        f"q{q + 1}": _mean(quantile_returns[q]) for q in range(n_quantiles)
    }
    return HorizonRankingMetrics(
        horizon=horizon,
        label_key=label_key,
        n_dates=len(pairs),
        coverage_count=coverage,
        missing_label_count=missing,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        rank_ic_by_date={d.isoformat(): ics[d] for d in sorted(ics)},
        sector_neutral_ic_mean=sector_neutral_ic_mean,
        quantile_spread_mean=_mean(spreads),
        quantile_return_table=quantile_table,
        hit_rate_top_positive=(top_positive / n_quantile_dates) if n_quantile_dates else None,
        hit_rate_top_above_median=(top_above_median / n_quantile_dates)
        if n_quantile_dates
        else None,
        n_quantiles=n_quantiles,
    )


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass(frozen=True)
class RankingReport:
    horizons: list[HorizonRankingMetrics]
    is_synthetic: bool
    model_label: str
    n_quantiles: int
    grade_dispersion: dict[str, int] = field(default_factory=dict)
    disclaimer: str = RESEARCH_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "model_label": self.model_label,
            "is_synthetic": self.is_synthetic,
            "synthetic_warning": (
                "SYNTHETIC FIXTURE RESULTS — not real market evidence."
                if self.is_synthetic
                else None
            ),
            "n_quantiles": self.n_quantiles,
            "grade_dispersion": self.grade_dispersion,
            "horizons": [h.to_dict() for h in self.horizons],
        }


def evaluate_ranking(
    observations: Sequence[ScoredObservation],
    horizons: Sequence[int],
    *,
    model_label: str = "baseline_factor_ranker",
    is_synthetic: bool = False,
    label_prefix: str = "forward_return_h",
    n_quantiles: int = 5,
    grade_dispersion: dict[str, int] | None = None,
) -> RankingReport:
    """Evaluate ranking metrics across every horizon."""
    metrics = [
        evaluate_horizon(
            observations, h, label_prefix=label_prefix, n_quantiles=n_quantiles
        )
        for h in sorted({int(h) for h in horizons})
    ]
    return RankingReport(
        horizons=metrics,
        is_synthetic=is_synthetic,
        model_label=model_label,
        n_quantiles=n_quantiles,
        grade_dispersion=dict(sorted((grade_dispersion or {}).items())),
    )


def write_ranking_outputs(
    report: RankingReport, output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    """Write ranking JSON / CSV (+ optional Markdown). Returns their paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ranking_metrics.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = out_dir / "ranking_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "horizon",
                "ic_mean",
                "ic_std",
                "icir",
                "sector_neutral_ic_mean",
                "quantile_spread_mean",
                "hit_rate_top_positive",
                "hit_rate_top_above_median",
                "coverage_count",
                "missing_label_count",
            ]
        )
        for h in report.horizons:
            writer.writerow(
                [
                    h.horizon,
                    _fmt(h.ic_mean),
                    _fmt(h.ic_std),
                    _fmt(h.icir),
                    _fmt(h.sector_neutral_ic_mean),
                    _fmt(h.quantile_spread_mean),
                    _fmt(h.hit_rate_top_positive),
                    _fmt(h.hit_rate_top_above_median),
                    h.coverage_count,
                    h.missing_label_count,
                ]
            )
    paths = {"json_path": json_path, "csv_path": csv_path}

    if write_markdown:
        md_path = out_dir / "ranking_metrics.md"
        md_path.write_text(_markdown(report), encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _markdown(report: RankingReport) -> str:
    lines = ["# Cross-Sectional Ranking Validation", "", report.disclaimer, ""]
    if report.is_synthetic:
        lines += ["> **SYNTHETIC FIXTURE RESULTS — not real market evidence.**", ""]
    lines.append(f"- Model: `{report.model_label}`")
    lines.append(f"- Quantiles: {report.n_quantiles}")
    if report.grade_dispersion:
        lines.append(f"- Reliability grade dispersion: {report.grade_dispersion}")
    lines += [
        "",
        "| horizon | IC mean | ICIR | sector-neutral IC | quantile spread | "
        "hit>0 | hit>median | coverage |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for h in report.horizons:
        lines.append(
            f"| {h.horizon} | {_fmt(h.ic_mean)} | {_fmt(h.icir)} | "
            f"{_fmt(h.sector_neutral_ic_mean)} | {_fmt(h.quantile_spread_mean)} | "
            f"{_fmt(h.hit_rate_top_positive)} | {_fmt(h.hit_rate_top_above_median)} | "
            f"{h.coverage_count} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "RESEARCH_DISCLAIMER",
    "HorizonRankingMetrics",
    "RankingReport",
    "ScoredObservation",
    "evaluate_horizon",
    "evaluate_ranking",
    "spearman",
    "write_ranking_outputs",
]

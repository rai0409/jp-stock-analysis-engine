"""Commercial-grade offline modeling report.

Assembles the full research report from a built dataset: coverage, eligible /
excluded observations, accounting-basis distribution, feature coverage, factor
score distribution, Rank IC and quantile spread by horizon, the walk-forward
fold table, a baseline-vs-(optional)-ML model comparison, the strict
no-look-ahead status, explicit limitations, and a synthetic-vs-real flag.

Every section is research-only. The report never emits a buy/sell label, never
claims predictive performance, and clearly marks synthetic-fixture results as
non-evidence.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from jp_stock_analysis.modeling.baseline_ranker import score_baseline, scored_observations
from jp_stock_analysis.modeling.dataset import ModelingDataset
from jp_stock_analysis.modeling.ml_models import (
    MODEL_BASELINE,
    MODEL_TYPES,
    available_backends,
    train_ranking_model,
)
from jp_stock_analysis.modeling.ranking_metrics import (
    RESEARCH_DISCLAIMER,
    RankingReport,
    evaluate_ranking,
)
from jp_stock_analysis.modeling.walk_forward import (
    MODE_EXPANDING,
    WalkForwardPlan,
    build_walk_forward_plan,
)
from jp_stock_analysis.schemas import PriceBar
from jp_stock_analysis.validation.no_lookahead import build_readiness_report

LIMITATIONS = (
    "This is research infrastructure, not a trading system: it produces no "
    "buy/sell signals and claims no predictive performance.",
    "Synthetic-fixture results are NOT market evidence; they only prove the "
    "pipeline runs deterministically.",
    "Real validation requires point-in-time disclosure dates and adjusted-close "
    "prices, and must pass the strict no-look-ahead readiness check first.",
    "Consolidated and non_consolidated fundamentals are never pooled; "
    "non_consolidated rows are excluded by default.",
    "LightGBM / CatBoost are optional; when absent the comparison reports them "
    "as skipped, not failed.",
)


@dataclass(frozen=True)
class ModelComparisonEntry:
    model_type: str
    status: str
    ic_by_horizon: dict[str, float | None]
    message: str = ""
    missing_dependency: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "status": self.status,
            "ic_by_horizon": self.ic_by_horizon,
            "message": self.message,
            "missing_dependency": self.missing_dependency,
        }


@dataclass(frozen=True)
class ModelingReport:
    dataset: ModelingDataset
    baseline_ranking: RankingReport
    walk_forward: WalkForwardPlan
    model_comparison: list[ModelComparisonEntry]
    readiness: dict[str, Any]
    factor_score_distribution: dict[str, float | None]
    backends: dict[str, bool]
    disclaimer: str = RESEARCH_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        ds = self.dataset
        return {
            "disclaimer": self.disclaimer,
            "synthetic": ds.is_synthetic,
            "synthetic_warning": (
                "SYNTHETIC FIXTURE RESULTS — not real market evidence."
                if ds.is_synthetic
                else None
            ),
            "data_coverage": {
                "decision_dates": [d.isoformat() for d in ds.decision_dates],
                "horizons": list(ds.horizons),
                "total_observations": len(ds.observations),
                "eligible_observations": len(ds.included()),
                "accounting_basis_distribution": ds.basis_counts(),
                "exclusions": ds.exclusion_counts(),
                "feature_coverage": ds.feature_coverage(),
                "label_coverage": ds.label_coverage(),
            },
            "factor_score_distribution": self.factor_score_distribution,
            "ranking_by_horizon": {
                str(h.horizon): {
                    "ic_mean": h.ic_mean,
                    "icir": h.icir,
                    "sector_neutral_ic_mean": h.sector_neutral_ic_mean,
                    "quantile_spread_mean": h.quantile_spread_mean,
                    "hit_rate_top_positive": h.hit_rate_top_positive,
                    "coverage_count": h.coverage_count,
                }
                for h in self.baseline_ranking.horizons
            },
            "walk_forward": self.walk_forward.to_dict(),
            "optional_backends": self.backends,
            "model_comparison": [m.to_dict() for m in self.model_comparison],
            "no_look_ahead_status": self.readiness,
            "limitations": list(LIMITATIONS),
        }


def _factor_score_distribution(dataset: ModelingDataset) -> dict[str, float | None]:
    scores = [
        s.factor_score for s in score_baseline(dataset) if s.factor_score is not None
    ]
    if not scores:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None, "std": None}
    return {
        "count": len(scores),
        "min": min(scores),
        "median": statistics.median(scores),
        "mean": statistics.fmean(scores),
        "max": max(scores),
        "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
    }


def build_modeling_report(
    dataset: ModelingDataset,
    prices: Mapping[str, Sequence[PriceBar]],
    *,
    bundle_disclosure_date: date | None = None,
    n_quantiles: int = 5,
    walk_forward_mode: str = MODE_EXPANDING,
    min_train_periods: int = 1,
    test_periods: int = 1,
    include_optional_models: bool = True,
) -> ModelingReport:
    """Build the full modeling report from a dataset and its price inputs."""
    horizons = list(dataset.horizons)

    baseline_scores = score_baseline(dataset)
    baseline_scored = [
        s for s in scored_observations(dataset, baseline_scores) if s.score is not None
    ]
    baseline_ranking = evaluate_ranking(
        baseline_scored,
        horizons,
        model_label=MODEL_BASELINE,
        is_synthetic=dataset.is_synthetic,
        n_quantiles=n_quantiles,
    )

    plan = build_walk_forward_plan(
        dataset.decision_dates,
        horizons=horizons,
        mode=walk_forward_mode,
        min_train_periods=min_train_periods,
        test_periods=test_periods,
    )

    comparison = _model_comparison(
        dataset, horizons, n_quantiles, include_optional_models
    )

    tickers = sorted({o.ticker for o in dataset.included()}) or sorted(
        {o.ticker for o in dataset.observations}
    )
    readiness = build_readiness_report(
        tickers, dict(prices), horizons, bundle_disclosure_date
    ).to_dict()

    return ModelingReport(
        dataset=dataset,
        baseline_ranking=baseline_ranking,
        walk_forward=plan,
        model_comparison=comparison,
        readiness=readiness,
        factor_score_distribution=_factor_score_distribution(dataset),
        backends=available_backends(),
    )


def _model_comparison(
    dataset: ModelingDataset,
    horizons: Sequence[int],
    n_quantiles: int,
    include_optional_models: bool,
) -> list[ModelComparisonEntry]:
    model_types = (
        list(MODEL_TYPES) if include_optional_models else [MODEL_BASELINE]
    )
    entries: list[ModelComparisonEntry] = []
    for model_type in model_types:
        # one primary horizon for training; evaluate the model's scores on all
        primary_horizon = sorted({int(h) for h in horizons})[0]
        result = train_ranking_model(
            dataset, model_type, horizon=primary_horizon, n_quantiles=n_quantiles
        )
        ic_by_horizon: dict[str, float | None] = {}
        if result.is_trained and result.scored:
            ranking = evaluate_ranking(
                result.scored,
                horizons,
                model_label=model_type,
                is_synthetic=dataset.is_synthetic,
                n_quantiles=n_quantiles,
            )
            ic_by_horizon = {str(h.horizon): h.ic_mean for h in ranking.horizons}
        entries.append(
            ModelComparisonEntry(
                model_type=model_type,
                status=result.status,
                ic_by_horizon=ic_by_horizon,
                message=result.message,
                missing_dependency=result.missing_dependency,
            )
        )
    return entries


def write_modeling_report_outputs(
    report: ModelingReport, output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "modeling_report.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths = {"json_path": json_path}
    if write_markdown:
        md_path = out_dir / "modeling_report.md"
        md_path.write_text(_markdown(report), encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _markdown(report: ModelingReport) -> str:
    data = report.to_dict()
    cov = data["data_coverage"]
    lines = ["# Offline Modeling Report", "", report.disclaimer, ""]
    if data["synthetic"]:
        lines += ["> **SYNTHETIC FIXTURE RESULTS — not real market evidence.**", ""]

    lines += ["## Data coverage", ""]
    lines.append(f"- Decision dates: {cov['decision_dates']}")
    lines.append(f"- Horizons: {cov['horizons']}")
    lines.append(
        f"- Eligible / total observations: "
        f"{cov['eligible_observations']} / {cov['total_observations']}"
    )
    lines.append(f"- Accounting basis distribution: {cov['accounting_basis_distribution']}")
    lines.append(f"- Exclusions: {cov['exclusions'] or 'none'}")
    lines.append("")

    lines += ["## Factor score distribution", ""]
    lines.append(f"- {report.factor_score_distribution}")
    lines.append("")

    lines += [
        "## Ranking by horizon (baseline factor ranker)",
        "",
        "| horizon | IC mean | ICIR | sector-neutral IC | quantile spread | hit>0 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for h in report.baseline_ranking.horizons:
        lines.append(
            f"| {h.horizon} | {_fmt(h.ic_mean)} | {_fmt(h.icir)} | "
            f"{_fmt(h.sector_neutral_ic_mean)} | {_fmt(h.quantile_spread_mean)} | "
            f"{_fmt(h.hit_rate_top_positive)} |"
        )
    lines.append("")

    lines += ["## Walk-forward folds", ""]
    wf = report.walk_forward
    lines.append(
        f"- Mode `{wf.mode}`, {len(wf.folds)} folds, "
        f"min train {wf.min_train_periods}, test {wf.test_periods}"
    )
    for f in report.walk_forward.folds:
        lines.append(
            f"  - fold {f.fold_index}: train {f.train_start}..{f.train_end} "
            f"({len(f.train_periods)}) -> test {f.test_start}..{f.test_end} "
            f"({len(f.test_periods)})"
        )
    lines.append("")

    lines += [
        "## Model comparison",
        "",
        f"- Optional backends: {report.backends}",
        "",
        "| model | status | IC by horizon |",
        "| --- | --- | --- |",
    ]
    for m in report.model_comparison:
        ic = ", ".join(f"h{k}={_fmt(v)}" for k, v in sorted(m.ic_by_horizon.items())) or "—"
        lines.append(f"| `{m.model_type}` | {m.status} | {ic} |")
    lines.append("")

    lines += ["## No-look-ahead status", ""]
    lines.append(f"- Overall: **{data['no_look_ahead_status']['overall_status'].upper()}**")
    lines.append(
        f"- Eligible tickers: {data['no_look_ahead_status']['eligible_ticker_count']} / "
        f"{data['no_look_ahead_status']['ticker_count']}"
    )
    lines.append(
        f"- Bundle disclosure date: {data['no_look_ahead_status']['bundle_disclosure_date']}"
    )
    lines.append("")

    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in data["limitations"]]
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "LIMITATIONS",
    "ModelComparisonEntry",
    "ModelingReport",
    "build_modeling_report",
    "write_modeling_report_outputs",
]

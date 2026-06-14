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

from jp_stock_analysis.modeling.audit import build_audit_manifest
from jp_stock_analysis.modeling.baseline_ranker import score_baseline, scored_observations
from jp_stock_analysis.modeling.constraints import (
    ConstraintConfig,
    PositionBook,
    apply_constraints,
)
from jp_stock_analysis.modeling.dataset import ModelingDataset, ModelingObservation
from jp_stock_analysis.modeling.ensemble import rank_average_ensemble, weighted_blend
from jp_stock_analysis.modeling.factors import ALL_FACTORS
from jp_stock_analysis.modeling.feature_importance import (
    coefficient_importance,
    permutation_importance,
)
from jp_stock_analysis.modeling.linear_models import ElasticNetRanker, RidgeRanker
from jp_stock_analysis.modeling.ml_models import (
    MODEL_BASELINE,
    MODEL_TYPES,
    available_backends,
    train_ranking_model,
)
from jp_stock_analysis.modeling.monitoring import build_monitoring_report
from jp_stock_analysis.modeling.neutralization import (
    ExposureObservation,
    MMCStyleObservation,
    MMCStyleReport,
    NeutralizedICReport,
    mmc_style_contribution,
    neutralized_rank_ic,
)
from jp_stock_analysis.modeling.portfolio_metrics import (
    STATUS_OK as PORTFOLIO_STATUS_OK,
)
from jp_stock_analysis.modeling.portfolio_metrics import (
    PortfolioReport,
    evaluate_portfolio,
    observations_from_scored,
)
from jp_stock_analysis.modeling.ranking_metrics import (
    RESEARCH_DISCLAIMER,
    RankingReport,
    ScoredObservation,
    evaluate_ranking,
    spearman,
)
from jp_stock_analysis.modeling.stability import (
    build_stability_report,
    compute_fold_metrics,
    synthetic_seed_ic,
)
from jp_stock_analysis.modeling.walk_forward import (
    MODE_EXPANDING,
    WalkForwardPlan,
    build_walk_forward_plan,
)
from jp_stock_analysis.schemas import PriceBar
from jp_stock_analysis.validation.no_lookahead import build_readiness_report

DEFAULT_NEUTRALIZE_FACTORS = ("momentum_60d", "leverage")

LIMITATIONS = (
    "This is research infrastructure, not a trading system: it produces no "
    "buy/sell signals and claims no predictive performance.",
    "The long-short spread evaluation is a research metric inspired by long-short "
    "competition scoring; it does not claim exchange/execution realism.",
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
    portfolio_by_horizon: dict[str, PortfolioReport]
    neutralized: NeutralizedICReport | None
    mmc_style: MMCStyleReport | None
    model_diversity: dict[str, Any]
    commercial_validation: dict[str, Any]
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
            "portfolio_long_short": {
                horizon: _portfolio_summary(report)
                for horizon, report in self.portfolio_by_horizon.items()
            },
            "neutralized_rank_metrics": (
                self.neutralized.to_dict() if self.neutralized is not None else None
            ),
            "mmc_style": (self.mmc_style.to_dict() if self.mmc_style is not None else None),
            "model_diversity": self.model_diversity,
            "commercial_validation": self.commercial_validation,
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


def _portfolio_summary(report: PortfolioReport) -> dict[str, Any]:
    """Compact per-horizon portfolio summary (full detail via evaluate-portfolio)."""
    series = report.series
    summary = {
        "status": report.status,
        "config": report.config,
        "sharpe_like": series.sharpe_like,
        "mean_spread": series.mean_spread,
        "hit_rate": series.hit_rate,
        "max_drawdown": series.max_drawdown,
        "observation_count": series.observation_count,
        "average_turnover": report.turnover.average_turnover,
        "max_turnover": report.turnover.max_turnover,
    }
    if report.transaction_cost is not None:
        summary["net_mean_spread"] = report.transaction_cost.net_mean_spread
        summary["transaction_cost_bps"] = report.transaction_cost.transaction_cost_bps
    return summary


def _exposure_observations(
    dataset: ModelingDataset,
    scored: Sequence[ScoredObservation],
    horizon: int,
    factor_columns: Sequence[str],
) -> tuple[list[ExposureObservation], list[str]]:
    """Build neutralization inputs: requested factor columns + sector dummies."""
    features_by_key = {(o.ticker, o.decision_date): o.features for o in dataset.included()}
    sectors = sorted({o.sector for o in scored if o.sector})
    sector_cols = [f"sector::{s}" for s in sectors]
    label_key = f"forward_return_h{horizon}"
    out: list[ExposureObservation] = []
    for obs in scored:
        features = features_by_key.get((obs.ticker, obs.decision_date), {})
        exposures: dict[str, float | None] = {
            col: features.get(col) for col in factor_columns
        }
        for sector in sectors:
            exposures[f"sector::{sector}"] = 1.0 if obs.sector == sector else 0.0
        out.append(
            ExposureObservation(
                decision_date=obs.decision_date,
                ticker=obs.ticker,
                prediction=obs.score,
                forward_return=obs.labels.get(label_key),
                exposures=exposures,
                sector=obs.sector,
            )
        )
    return out, [*factor_columns, *sector_cols]


def _train_linear(
    observations: Sequence[ModelingObservation], horizon: int, model
) -> tuple[Any, list[ScoredObservation]]:
    """Fit a linear ranker on the labelled observations; return (model, scored)."""
    label_key = f"forward_return_h{horizon}"
    labelled = [o for o in observations if o.labels.get(label_key) is not None]
    labelled.sort(key=lambda o: (o.decision_date, o.ticker))
    matrix = [[o.features.get(f) for f in ALL_FACTORS] for o in labelled]
    target = [float(o.labels[label_key]) for o in labelled]  # type: ignore[arg-type]
    model.fit(matrix, target, list(ALL_FACTORS))
    predictions = model.predict(matrix) if labelled else []
    scored = [
        ScoredObservation(
            decision_date=o.decision_date,
            ticker=o.ticker,
            score=float(predictions[i]),
            sector=o.sector,
            labels=o.labels,
        )
        for i, o in enumerate(labelled)
    ]
    return model, scored


def build_model_diversity(
    dataset: ModelingDataset,
    baseline_scored: Sequence[ScoredObservation],
    *,
    horizon: int,
    n_quantiles: int,
) -> dict[str, Any]:
    """Linear baselines, ensemble/blend, stability, and feature importance.

    Synthetic-only integration path: trains a Ridge and a real Elastic Net on the
    in-memory dataset, ensembles them with the baseline, summarises walk-forward
    stability, and reports feature importance. Research diagnostics only.
    """
    included = dataset.included()
    ridge, ridge_scored = _train_linear(included, horizon, RidgeRanker(alpha=1.0))
    elastic, elastic_scored = _train_linear(
        included, horizon, ElasticNetRanker(alpha=0.05, l1_ratio=0.5, max_iter=2000)
    )

    predictions = {
        "baseline": list(baseline_scored),
        "ridge": ridge_scored,
        "elastic_net": elastic_scored,
    }
    ensemble = rank_average_ensemble(predictions, is_synthetic=dataset.is_synthetic)
    blend = weighted_blend(
        predictions,
        {"baseline": 1.0, "ridge": 1.0, "elastic_net": 1.0},
        is_synthetic=dataset.is_synthetic,
    )
    ensemble_ic = (
        evaluate_ranking(
            ensemble.scored, [horizon], model_label="rank_average_ensemble",
            is_synthetic=dataset.is_synthetic, n_quantiles=n_quantiles,
        ).horizons[0].ic_mean
        if ensemble.scored
        else None
    )
    blend_ic = (
        evaluate_ranking(
            blend.scored, [horizon], model_label="weighted_blend",
            is_synthetic=dataset.is_synthetic, n_quantiles=n_quantiles,
        ).horizons[0].ic_mean
        if blend.scored
        else None
    )

    plan = build_walk_forward_plan(dataset.decision_dates, horizons=[horizon])
    fold_metrics = compute_fold_metrics(baseline_scored, plan.folds, horizon=horizon)
    seed_ic = synthetic_seed_ic(baseline_scored, horizon=horizon, seeds=[0, 1, 2, 3])
    stability = build_stability_report(
        fold_metrics, horizon=horizon, is_synthetic=dataset.is_synthetic, seed_ic=seed_ic
    )

    coef_importance = coefficient_importance(
        elastic.model_metadata["scaled_coefficients"], is_synthetic=dataset.is_synthetic
    )
    label_key = f"forward_return_h{horizon}"
    labelled = sorted(
        (o for o in included if o.labels.get(label_key) is not None),
        key=lambda o: (o.decision_date, o.ticker),
    )
    perm = permutation_importance(
        elastic,
        [[o.features.get(f) for f in ALL_FACTORS] for o in labelled],
        list(ALL_FACTORS),
        [o.decision_date for o in labelled],
        [o.labels[label_key] for o in labelled],
        seed=0,
        is_synthetic=dataset.is_synthetic,
    )

    return {
        "horizon": horizon,
        "linear_models": {
            "ridge": _linear_summary(ridge),
            "elastic_net": _linear_summary(elastic),
        },
        "ensemble": {
            "rank_average": {**ensemble.to_dict(), "ic_mean": ensemble_ic},
            "weighted_blend": {**blend.to_dict(), "ic_mean": blend_ic},
        },
        "stability": stability.to_dict(),
        "feature_importance": {
            "coefficient": coef_importance.to_dict(),
            "permutation": perm.to_dict(),
        },
        "limitations": [
            "Linear/ensemble/stability/importance are research diagnostics only.",
            "Elastic Net is a real coordinate-descent L1/L2 model; sparsity is not "
            "proof of alpha.",
            "Feature importance is explanatory, not causal.",
        ],
    }


def _linear_summary(model) -> dict[str, Any]:
    meta = model.model_metadata
    return {
        "model_version": meta["model_version"],
        "status": meta["status"],
        "intercept": meta["intercept"],
        "coefficients": meta["coefficients"],
        "scaled_coefficients": meta["scaled_coefficients"],
        "sparsity": meta["sparsity"],
        "selected_features": meta["selected_features"],
        "n_iter": meta.get("n_iter"),
        "converged": meta.get("converged"),
        "final_objective": meta.get("final_objective"),
        "warnings": meta["warnings"],
    }


def build_commercial_validation(
    dataset: ModelingDataset,
    baseline_scored: Sequence[ScoredObservation],
    portfolio: PortfolioReport,
    *,
    horizon: int,
    model_versions: Sequence[str],
) -> dict[str, Any]:
    """Constraints, cost/exposure decomposition, drift monitoring, audit stub.

    Synthetic-only integration path. The constrained book is NOT a recommended
    portfolio — it is a feasibility approximation. Research diagnostics only.
    """
    # constraints on the latest valid date's long/short book
    ok_dates = [s for s in portfolio.per_date if s.status == PORTFOLIO_STATUS_OK]
    constraints_summary: dict[str, Any] = {"status": "no_valid_dates"}
    if ok_dates:
        latest = ok_dates[-1]
        sector_of = {
            o.ticker: o.sector
            for o in dataset.included()
            if o.decision_date == latest.decision_date
        }
        book = PositionBook(
            long_weights=dict(latest.long_weights),
            short_weights=dict(latest.short_weights),
            sector_of=sector_of,
            adv_of=None,  # synthetic fixtures carry no ADV; never fabricated
        )
        result = apply_constraints(
            book, ConstraintConfig(max_weight_per_name=0.34, max_sector_weight=0.6)
        )
        constraints_summary = {
            "status": result.status,
            "decision_date": latest.decision_date.isoformat(),
            "liquidity_adv_available": False,
            "long_gross_before": result.long.gross_before,
            "long_gross_after": result.long.gross_after,
            "short_gross_before": result.short.gross_before,
            "short_gross_after": result.short.gross_after,
            "applied_constraints": result.applied_constraints,
            "warnings": result.warnings,
        }

    # drift monitoring over decision dates: spread, turnover, per-date Rank IC
    periods: list[str] = []
    spreads: list[float | None] = []
    turnover_by_date = {
        row["decision_date"]: row["total_turnover"] for row in portfolio.turnover.per_date
    }
    label_key = f"forward_return_h{horizon}"
    scored_by_date: dict[date, list[ScoredObservation]] = {}
    for obs in baseline_scored:
        scored_by_date.setdefault(obs.decision_date, []).append(obs)
    ic_series: list[float | None] = []
    turnover_series: list[float | None] = []
    for spread in sorted(portfolio.per_date, key=lambda s: s.decision_date):
        iso = spread.decision_date.isoformat()
        periods.append(iso)
        spreads.append(spread.spread_return)
        turnover_series.append(turnover_by_date.get(iso))
        group = scored_by_date.get(spread.decision_date, [])
        pairs = [
            (float(o.score), float(o.labels[label_key]))
            for o in group
            if o.score is not None and o.labels.get(label_key) is not None
        ]
        ic_series.append(
            spearman([s for s, _ in pairs], [r for _, r in pairs]) if len(pairs) >= 2 else None
        )
    monitoring = build_monitoring_report(
        periods,
        {"long_short_spread": spreads, "rank_ic": ic_series, "turnover": turnover_series},
        is_synthetic=dataset.is_synthetic,
    )

    audit = build_audit_manifest(
        command={"report": "modeling-report", "horizon": horizon},
        model_versions=list(model_versions),
        feature_columns=list(ALL_FACTORS),
        target_columns=[label_key],
        horizons=list(dataset.horizons),
        no_look_ahead_status=None,
        is_synthetic=dataset.is_synthetic,
        warnings=["embedded audit stub; use build-audit-manifest for input fingerprints"],
        stable=True,
    )

    return {
        "constraints": constraints_summary,
        "portfolio_commercial": portfolio.commercial,
        "monitoring": monitoring.to_dict(),
        "audit_manifest_stub": {
            "run_id": audit["run_id"],
            "synthetic_vs_real": audit["synthetic_vs_real"],
            "model_versions": audit["model_versions"],
        },
        "limitations": [
            "Commercial-validation infrastructure, NOT commercial-ready proof.",
            "The constrained book is a feasibility approximation, not a recommended "
            "portfolio.",
            "Liquidity/ADV constraints require real liquidity data to be meaningful "
            "(none in synthetic fixtures; ADV is never fabricated).",
            "Audit manifests improve reproducibility but do not prove model validity.",
        ],
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
    portfolio_top_quantile: float = 0.2,
    portfolio_bottom_quantile: float = 0.2,
    portfolio_rank_weighted: bool = False,
    transaction_cost_bps: float = 0.0,
    neutralize_factors: Sequence[str] = DEFAULT_NEUTRALIZE_FACTORS,
    neutralize_proportion: float = 1.0,
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

    # JPX-style long-short evaluation per horizon
    portfolio_by_horizon: dict[str, PortfolioReport] = {}
    for horizon in horizons:
        portfolio_by_horizon[str(horizon)] = evaluate_portfolio(
            observations_from_scored(baseline_scored, horizon),
            horizon=horizon,
            model_label=MODEL_BASELINE,
            is_synthetic=dataset.is_synthetic,
            top_quantile=portfolio_top_quantile,
            bottom_quantile=portfolio_bottom_quantile,
            rank_weighted=portfolio_rank_weighted,
            transaction_cost_bps=transaction_cost_bps,
        )

    # Numerai-style neutralized rank metrics (primary horizon)
    primary_horizon = sorted({int(h) for h in horizons})[0]
    exposure_obs, exposure_columns = _exposure_observations(
        dataset, baseline_scored, primary_horizon, neutralize_factors
    )
    neutralized = neutralized_rank_ic(
        exposure_obs,
        horizon=primary_horizon,
        exposure_columns=exposure_columns,
        proportion=neutralize_proportion,
        is_synthetic=dataset.is_synthetic,
        model_label=MODEL_BASELINE,
    )

    mmc_style = _mmc_style_report(
        dataset, baseline_scored, primary_horizon, n_quantiles
    )

    model_diversity = build_model_diversity(
        dataset, baseline_scored, horizon=primary_horizon, n_quantiles=n_quantiles
    )

    model_versions = ["baseline_factor_ranker_v1"] + [
        v["model_version"]
        for v in model_diversity["linear_models"].values()
        if v.get("model_version")
    ]
    commercial_validation = build_commercial_validation(
        dataset,
        baseline_scored,
        portfolio_by_horizon[str(primary_horizon)],
        horizon=primary_horizon,
        model_versions=model_versions,
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
        portfolio_by_horizon=portfolio_by_horizon,
        neutralized=neutralized,
        mmc_style=mmc_style,
        model_diversity=model_diversity,
        commercial_validation=commercial_validation,
    )


def _mmc_style_report(
    dataset: ModelingDataset,
    baseline_scored: Sequence[ScoredObservation],
    horizon: int,
    n_quantiles: int,
) -> MMCStyleReport | None:
    """MMC-style delta vs the first trained optional model, if one is available.

    Requires >=2 model predictions; when no optional backend is installed there
    is only the baseline, so this returns ``None`` (reported as unavailable).
    """
    candidate = None
    for model_type in MODEL_TYPES:
        if model_type == MODEL_BASELINE:
            continue
        result = train_ranking_model(
            dataset, model_type, horizon=horizon, n_quantiles=n_quantiles
        )
        if result.is_trained and result.scored:
            candidate = result.scored
            break
    if candidate is None:
        return None

    label_key = f"forward_return_h{horizon}"
    base_by_key = {(o.ticker, o.decision_date): o.score for o in baseline_scored}
    obs: list[MMCStyleObservation] = []
    for cand in candidate:
        obs.append(
            MMCStyleObservation(
                decision_date=cand.decision_date,
                ticker=cand.ticker,
                base_prediction=base_by_key.get((cand.ticker, cand.decision_date)),
                candidate_prediction=cand.score,
                forward_return=cand.labels.get(label_key),
            )
        )
    return mmc_style_contribution(obs, horizon=horizon, is_synthetic=dataset.is_synthetic)


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


def _top_coefs(linear_summary: dict[str, Any], n: int = 3) -> list[tuple[str, float]]:
    coefs = linear_summary.get("scaled_coefficients", {})
    ranked = sorted(coefs.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    return [(name, round(float(value), 4)) for name, value in ranked[:n]]


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

    lines += [
        "## JPX-style long-short spread evaluation",
        "",
        "Research metric inspired by long-short competition scoring; no execution "
        "realism, no trading signal.",
        "",
        "| horizon | status | Sharpe-like | mean spread | hit>0 | max DD | "
        "avg turnover | net mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for horizon, summary in data["portfolio_long_short"].items():
        lines.append(
            f"| {horizon} | {summary['status']} | {_fmt(summary['sharpe_like'])} | "
            f"{_fmt(summary['mean_spread'])} | {_fmt(summary['hit_rate'])} | "
            f"{_fmt(summary['max_drawdown'])} | {_fmt(summary['average_turnover'])} | "
            f"{_fmt(summary.get('net_mean_spread'))} |"
        )
    lines.append("")

    neutral = data["neutralized_rank_metrics"]
    lines += ["## Numerai-style neutralized rank metrics", ""]
    if neutral is None:
        lines.append("- unavailable")
    else:
        diag = neutral["exposure_diagnostics"]
        lines.append(
            "Inspired by neutralized ranking concepts; NOT official Numerai scoring."
        )
        lines.append("")
        lines.append(
            f"- horizon {neutral['horizon']}, exposures {neutral['exposure_columns']} "
            f"(status {neutral['status']})"
        )
        lines.append(
            f"- Raw IC mean {_fmt(neutral['raw_ic_mean'])} -> neutralized IC mean "
            f"**{_fmt(neutral['neutralized_ic_mean'])}** (ICIR {_fmt(neutral['neutralized_icir'])})"
        )
        lines.append(
            f"- Max |exposure corr| before/after: "
            f"{_fmt(diag['max_abs_exposure_corr_before'])} / "
            f"{_fmt(diag['max_abs_exposure_corr_after'])}"
        )
    lines.append("")

    mmc = data["mmc_style"]
    lines += ["## MMC-style contribution delta", ""]
    if mmc is None:
        lines.append("- unavailable (requires >=2 trained model predictions)")
    else:
        lines.append(mmc["caveat"])
        lines.append("")
        lines.append(
            f"- status {mmc['status']}, contribution delta "
            f"**{_fmt(mmc['contribution_delta'])}**, delta vs base {_fmt(mmc['delta_vs_base'])}"
        )
    lines.append("")

    md = data["model_diversity"]
    lines += ["## Model diversity, stability & explainability", ""]
    lines.append(
        "Research diagnostics only. Elastic Net is a real coordinate-descent L1/L2 "
        "model; feature importance is explanatory, not causal."
    )
    lines.append("")
    rid = md["linear_models"]["ridge"]
    en = md["linear_models"]["elastic_net"]
    lines.append(f"- Ridge: status {rid['status']}, top {_top_coefs(rid)}")
    lines.append(
        f"- Elastic Net: status {en['status']}, converged {en['converged']} "
        f"(n_iter {en['n_iter']}), sparsity {_fmt(en['sparsity'])}, "
        f"selected {en['selected_features']}"
    )
    ra = md["ensemble"]["rank_average"]
    wb = md["ensemble"]["weighted_blend"]
    lines.append(
        f"- Ensemble (rank-average) models {ra['model_names']}: IC {_fmt(ra['ic_mean'])}, "
        f"diversity {_fmt(ra['diversity_score'])}"
    )
    lines.append(f"- Weighted blend: IC {_fmt(wb['ic_mean'])} (status {wb['status']})")
    lines.append(f"- Pairwise prediction corr: {ra['pairwise_correlations']}")
    rank_ic_stab = md["stability"]["fold_stability"].get("rank_ic", {})
    lines.append(
        f"- Stability (Rank IC across {md['stability']['n_folds']} folds): "
        f"mean {_fmt(rank_ic_stab.get('mean'))}, std {_fmt(rank_ic_stab.get('std'))}, "
        f"positive rate {_fmt(rank_ic_stab.get('positive_period_rate'))}"
    )
    perm_rows = md["feature_importance"]["permutation"]["rows"][:3]
    lines.append(
        "- Top permutation importance: "
        + ", ".join(f"{r['feature']}={_fmt(r['importance'])}" for r in perm_rows)
    )
    lines.append("")

    cv = data["commercial_validation"]
    lines += ["## Commercial validation (research diagnostics)", ""]
    lines.append(
        "Constraints, cost/exposure decomposition, and drift monitoring. The "
        "constrained book is a feasibility approximation, NOT a recommended portfolio."
    )
    lines.append("")
    con = cv["constraints"]
    lines.append(
        f"- Constraints (latest date): status {con['status']}, ADV available "
        f"{con.get('liquidity_adv_available')}, applied {con.get('applied_constraints')}"
    )
    pc = cv["portfolio_commercial"]
    if pc and pc.get("status") == "ok":
        lines.append(
            f"- Gross vs net mean spread: {_fmt(pc['cost_decomposition']['gross_mean_spread'])} "
            f"-> {_fmt(pc['cost_decomposition']['net_mean_spread'])}"
        )
        lines.append(
            f"- Benchmark-relative Sharpe-like (long excess vs universe): "
            f"{_fmt(pc['benchmark_relative']['benchmark_relative_sharpe_like'])}"
        )
        lines.append(
            f"- Net sector exposure: {pc['exposure']['net_sector_exposure']}"
        )
        lines.append(
            f"- Concentration: effective N long "
            f"{_fmt(pc['concentration']['effective_n_long'])}, top weight long "
            f"{_fmt(pc['concentration']['top_weight_long_mean'])}"
        )
    flagged = sum(len(m["flagged_periods"]) for m in cv["monitoring"]["metrics"].values())
    lines.append(f"- Drift flags across monitored metrics: {flagged}")
    lines.append(f"- Audit manifest run_id (stub): `{cv['audit_manifest_stub']['run_id']}`")
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

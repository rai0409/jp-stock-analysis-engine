"""Deterministic end-to-end offline modeling pipeline runner (research-only).

Runs the P1–P4 modeling steps in a fixed order into a stamped run directory,
recording per-step status/inputs/outputs/warnings, then builds a real audit
manifest (fingerprinting the actual produced files), an artifact-manifest index,
and a consolidated pipeline summary.

This is an offline research pipeline runner, NOT a trading system. It emits no
buy/sell signal, claims no predictive performance, and clearly labels synthetic
output as non-evidence. No-look-ahead BLOCKED is surfaced, never bypassed. Real
liquidity constraints require real ADV data (never fabricated).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from jp_stock_analysis.modeling.audit import (
    build_artifact_manifest,
    build_audit_manifest,
    fingerprint_file,
    write_artifact_manifest_outputs,
    write_audit_manifest_outputs,
)
from jp_stock_analysis.modeling.baseline_ranker import score_baseline, scored_observations
from jp_stock_analysis.modeling.constraints import (
    ConstraintConfig,
    PositionBook,
    apply_constraints,
)
from jp_stock_analysis.modeling.dataset import ModelingDataset, write_dataset_outputs
from jp_stock_analysis.modeling.ensemble import rank_average_ensemble, weighted_blend
from jp_stock_analysis.modeling.factors import ALL_FACTORS
from jp_stock_analysis.modeling.feature_importance import (
    coefficient_importance,
    permutation_importance,
)
from jp_stock_analysis.modeling.linear_models import ElasticNetRanker, RidgeRanker
from jp_stock_analysis.modeling.monitoring import build_monitoring_report, write_monitoring_outputs
from jp_stock_analysis.modeling.neutralization import (
    ExposureObservation,
    neutralized_rank_ic,
    write_neutralized_outputs,
)
from jp_stock_analysis.modeling.portfolio_metrics import (
    evaluate_portfolio,
    observations_from_scored,
    write_portfolio_outputs,
)
from jp_stock_analysis.modeling.ranking_metrics import (
    ScoredObservation,
    evaluate_ranking,
    spearman,
    write_ranking_outputs,
)
from jp_stock_analysis.modeling.report import build_modeling_report, write_modeling_report_outputs
from jp_stock_analysis.modeling.stability import (
    build_stability_report,
    compute_fold_metrics,
    synthetic_seed_ic,
    write_stability_outputs,
)
from jp_stock_analysis.modeling.walk_forward import build_walk_forward_plan
from jp_stock_analysis.schemas import PriceBar
from jp_stock_analysis.validation.no_lookahead import (
    build_readiness_report,
    write_readiness_outputs,
)

RESEARCH_DISCLAIMER = (
    "This output is for analytical and self-directed research purposes. It is not "
    "personalized financial advice. Offline research pipeline; synthetic output is "
    "not market evidence."
)

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_BLOCKED = "blocked"
STATUS_ERROR = "error"


@dataclass(frozen=True)
class PipelineConfig:
    primary_horizon: int | None = None
    linear_models: tuple[str, ...] = ("ridge", "elastic_net")
    alpha: float = 0.05
    l1_ratio: float = 0.5
    portfolio_top_quantile: float = 0.2
    portfolio_bottom_quantile: float = 0.2
    portfolio_rank_weighted: bool = False
    transaction_cost_bps: float = 0.0
    max_weight_per_name: float | None = None
    max_sector_weight: float | None = None
    max_participation_rate: float | None = None
    min_adv: float | None = None
    monitoring_window: int = 3
    monitoring_threshold: float = 2.0
    neutralize_factors: tuple[str, ...] = ("momentum_60d", "leverage")


@dataclass
class StepRecord:
    name: str
    status: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_reason: str | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "warnings": self.warnings,
            "skipped_reason": self.skipped_reason,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
        }


def _exposure_observations(dataset, scored, horizon, factor_columns):
    features_by_key = {(o.ticker, o.decision_date): o.features for o in dataset.included()}
    sectors = sorted({o.sector for o in scored if o.sector})
    label_key = f"forward_return_h{horizon}"
    out = []
    for s in scored:
        features = features_by_key.get((s.ticker, s.decision_date), {})
        exposures = {col: features.get(col) for col in factor_columns}
        for sector in sectors:
            exposures[f"sector::{sector}"] = 1.0 if s.sector == sector else 0.0
        out.append(
            ExposureObservation(
                decision_date=s.decision_date,
                ticker=s.ticker,
                prediction=s.score,
                forward_return=s.labels.get(label_key),
                exposures=exposures,
                sector=s.sector,
            )
        )
    return out, [*factor_columns, *(f"sector::{s}" for s in sectors)]


def _train_linear(dataset, horizon, model):
    label_key = f"forward_return_h{horizon}"
    labelled = sorted(
        (o for o in dataset.included() if o.labels.get(label_key) is not None),
        key=lambda o: (o.decision_date, o.ticker),
    )
    matrix = [[o.features.get(f) for f in ALL_FACTORS] for o in labelled]
    target = [float(o.labels[label_key]) for o in labelled]
    model.fit(matrix, target, list(ALL_FACTORS))
    predictions = model.predict(matrix) if labelled else []
    scored = [
        ScoredObservation(o.decision_date, o.ticker, float(predictions[i]), o.sector, o.labels)
        for i, o in enumerate(labelled)
    ]
    return model, scored, labelled, matrix


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_pipeline(
    dataset: ModelingDataset,
    prices: Mapping[str, Sequence[PriceBar]],
    *,
    output_dir: str | Path,
    run_id: str = "run",
    fixed_timestamp: str | None = None,
    disclosure_date: date | None = None,
    config: PipelineConfig | None = None,
    input_files: Sequence[str] = (),
    adv: Mapping[str, float] | None = None,
    git_commit: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Run the full pipeline into ``output_dir/run_id`` and return the summary."""
    config = config or PipelineConfig()
    horizons = list(dataset.horizons)
    primary = config.primary_horizon or sorted({int(h) for h in horizons})[0]
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = fixed_timestamp
    steps: list[StepRecord] = []

    def rel(paths: Mapping[str, Path]) -> list[str]:
        return sorted(str(p.relative_to(run_dir)).replace("\\", "/") for p in paths.values())

    def record(name: str, status: str, outputs=None, warnings=None, skipped=None, inputs=None):
        step = StepRecord(
            name=name,
            status=status,
            inputs=list(inputs or []),
            outputs=list(outputs or []),
            warnings=list(warnings or []),
            skipped_reason=skipped,
            started_at_utc=stamp,
            finished_at_utc=stamp,
        )
        steps.append(step)
        return step

    # 1. dataset
    ds_paths = write_dataset_outputs(dataset, run_dir / "dataset")
    record("build_modeling_dataset", STATUS_OK, outputs=rel(ds_paths))

    baseline_scored = [
        s for s in scored_observations(dataset, score_baseline(dataset)) if s.score is not None
    ]

    # 2. readiness (surface BLOCKED, never bypass)
    tickers = sorted({o.ticker for o in dataset.included()}) or sorted(
        {o.ticker for o in dataset.observations}
    )
    readiness = build_readiness_report(tickers, dict(prices), horizons, disclosure_date)
    rd_paths = write_readiness_outputs(readiness, run_dir / "readiness")
    record(
        "check_forward_readiness",
        STATUS_BLOCKED if readiness.overall_status != "eligible" else STATUS_OK,
        outputs=rel(rd_paths),
        warnings=(
            ["no-look-ahead readiness BLOCKED — surfaced, not bypassed"]
            if readiness.overall_status != "eligible"
            else []
        ),
    )

    # 3. ranking
    ranking = evaluate_ranking(
        baseline_scored, horizons, model_label="baseline_factor_ranker",
        is_synthetic=dataset.is_synthetic,
    )
    rk_paths = write_ranking_outputs(ranking, run_dir / "ranking")
    record("evaluate_ranking", STATUS_OK, outputs=rel(rk_paths))

    # 4. portfolio
    portfolio = evaluate_portfolio(
        observations_from_scored(baseline_scored, primary),
        horizon=primary,
        is_synthetic=dataset.is_synthetic,
        top_quantile=config.portfolio_top_quantile,
        bottom_quantile=config.portfolio_bottom_quantile,
        rank_weighted=config.portfolio_rank_weighted,
        transaction_cost_bps=config.transaction_cost_bps,
    )
    pf_paths = write_portfolio_outputs(portfolio, run_dir / "portfolio")
    record("evaluate_portfolio", STATUS_OK, outputs=rel(pf_paths))

    # 5. neutralization
    exposure_obs, exposure_cols = _exposure_observations(
        dataset, baseline_scored, primary, config.neutralize_factors
    )
    neutralized = neutralized_rank_ic(
        exposure_obs, horizon=primary, exposure_columns=exposure_cols,
        is_synthetic=dataset.is_synthetic,
    )
    nt_paths = write_neutralized_outputs(neutralized, run_dir / "neutralization")
    record("evaluate_neutralized_ranking", STATUS_OK, outputs=rel(nt_paths))

    # 6. linear models + 7. ensemble
    predictions: dict[str, list[ScoredObservation]] = {"baseline": baseline_scored}
    fitted: dict[str, Any] = {}
    linear_outputs: list[str] = []
    for model_type in config.linear_models:
        if model_type == "ridge":
            model = RidgeRanker(alpha=1.0)
        elif model_type == "elastic_net":
            model = ElasticNetRanker(
                alpha=config.alpha, l1_ratio=config.l1_ratio, max_iter=2000
            )
        else:
            record("train_linear_model", STATUS_SKIPPED, skipped=f"unknown model {model_type}")
            continue
        model, scored, labelled, matrix = _train_linear(dataset, primary, model)
        predictions[model_type] = scored
        fitted[model_type] = (model, labelled, matrix)
        sub = run_dir / "linear" / model_type
        sub.mkdir(parents=True, exist_ok=True)
        _write_json(sub / "model_metadata.json", model.model_metadata)
        with (sub / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["ticker", "decision_date", "prediction"])
            for s in scored:
                writer.writerow([s.ticker, s.decision_date.isoformat(), f"{s.score:.8f}"])
        linear_outputs += [
            f"linear/{model_type}/model_metadata.json",
            f"linear/{model_type}/predictions.csv",
        ]
    record("train_linear_models", STATUS_OK, outputs=sorted(linear_outputs))

    ensemble = rank_average_ensemble(predictions, is_synthetic=dataset.is_synthetic)
    blend = weighted_blend(
        predictions, {k: 1.0 for k in predictions}, is_synthetic=dataset.is_synthetic
    )
    _write_json(
        run_dir / "ensemble" / "ensemble.json",
        {"rank_average": ensemble.to_dict(), "weighted_blend": blend.to_dict()},
    )
    record("build_ensemble", STATUS_OK, outputs=["ensemble/ensemble.json"])

    # 8. stability
    plan = build_walk_forward_plan(dataset.decision_dates, horizons=[primary])
    fold_metrics = compute_fold_metrics(baseline_scored, plan.folds, horizon=primary)
    seed_ic = synthetic_seed_ic(baseline_scored, horizon=primary, seeds=[0, 1, 2, 3])
    stability = build_stability_report(
        fold_metrics, horizon=primary, is_synthetic=dataset.is_synthetic, seed_ic=seed_ic
    )
    st_paths = write_stability_outputs(stability, run_dir / "stability")
    record("evaluate_stability", STATUS_OK, outputs=rel(st_paths))

    # 9. feature importance
    fi_status = STATUS_OK
    fi_skip = None
    if "elastic_net" in fitted:
        model, labelled, matrix = fitted["elastic_net"]
        label_key = f"forward_return_h{primary}"
        coef = coefficient_importance(
            model.model_metadata["scaled_coefficients"], is_synthetic=dataset.is_synthetic
        )
        perm = permutation_importance(
            model, matrix, list(ALL_FACTORS),
            [o.decision_date for o in labelled],
            [o.labels[label_key] for o in labelled],
            seed=0, is_synthetic=dataset.is_synthetic,
        )
        _write_json(
            run_dir / "feature_importance" / "feature_importance.json",
            {"coefficient": coef.to_dict(), "permutation": perm.to_dict()},
        )
        fi_outputs = ["feature_importance/feature_importance.json"]
    else:
        fi_status = STATUS_SKIPPED
        fi_skip = "elastic_net not trained; coefficient importance unavailable"
        fi_outputs = []
    record("compute_feature_importance", fi_status, outputs=fi_outputs, skipped=fi_skip)

    # 10. constraints
    constraints_status = STATUS_OK
    con_warnings: list[str] = []
    ok_dates = [s for s in portfolio.per_date if s.status == "ok"]
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
            adv_of=dict(adv) if adv else None,  # ADV never fabricated
        )
        result = apply_constraints(
            book,
            ConstraintConfig(
                max_weight_per_name=config.max_weight_per_name,
                max_sector_weight=config.max_sector_weight,
                max_participation_rate=config.max_participation_rate,
                min_adv=config.min_adv,
            ),
        )
        _write_json(
            run_dir / "constraints" / "constrained_portfolio.json", result.to_dict()
        )
        constraints_status = result.status
        con_warnings = result.warnings
        con_outputs = ["constraints/constrained_portfolio.json"]
    else:
        constraints_status = STATUS_SKIPPED
        con_outputs = []
        con_warnings = ["no valid long-short date to constrain"]
    record("apply_constraints", constraints_status, outputs=con_outputs, warnings=con_warnings)

    # 11. monitoring
    periods, spreads, ic_series, turnover_series = [], [], [], []
    turnover_by_date = {
        row["decision_date"]: row["total_turnover"] for row in portfolio.turnover.per_date
    }
    label_key = f"forward_return_h{primary}"
    scored_by_date: dict[date, list[ScoredObservation]] = {}
    for obs in baseline_scored:
        scored_by_date.setdefault(obs.decision_date, []).append(obs)
    for spread in sorted(portfolio.per_date, key=lambda s: s.decision_date):
        iso = spread.decision_date.isoformat()
        periods.append(iso)
        spreads.append(spread.spread_return)
        turnover_series.append(turnover_by_date.get(iso))
        grp = scored_by_date.get(spread.decision_date, [])
        pairs = [
            (float(o.score), float(o.labels[label_key]))
            for o in grp
            if o.score is not None and o.labels.get(label_key) is not None
        ]
        ic_series.append(
            spearman([s for s, _ in pairs], [r for _, r in pairs]) if len(pairs) >= 2 else None
        )
    monitoring = build_monitoring_report(
        periods,
        {"long_short_spread": spreads, "rank_ic": ic_series, "turnover": turnover_series},
        window=config.monitoring_window,
        z_threshold=config.monitoring_threshold,
        is_synthetic=dataset.is_synthetic,
    )
    mon_paths = write_monitoring_outputs(monitoring, run_dir / "monitoring")
    record("evaluate_monitoring", STATUS_OK, outputs=rel(mon_paths))

    # 12. consolidated modeling report
    report = build_modeling_report(
        dataset, prices, bundle_disclosure_date=disclosure_date,
        transaction_cost_bps=config.transaction_cost_bps,
    )
    rp_paths = write_modeling_report_outputs(report, run_dir)
    record("build_modeling_report", STATUS_OK, outputs=rel(rp_paths))

    # 13. audit manifest (fingerprint REAL produced output files + real inputs)
    output_files = sorted(
        str(p.relative_to(run_dir)).replace("\\", "/")
        for p in run_dir.rglob("*")
        if p.is_file()
    )
    input_fingerprints = [fingerprint_file(p) for p in input_files]
    model_versions = ["baseline_factor_ranker_v1"] + [
        fitted[m][0].model_metadata["model_version"] for m in fitted
    ]
    audit = build_audit_manifest(
        command={"command": "run-modeling-pipeline", "run_id": run_id},
        model_versions=model_versions,
        feature_columns=list(ALL_FACTORS),
        target_columns=[f"forward_return_h{primary}"],
        horizons=horizons,
        no_look_ahead_status=readiness.overall_status,
        is_synthetic=dataset.is_synthetic,
        input_fingerprints=input_fingerprints,
        output_files=output_files,
        warnings=[w for step in steps for w in step.warnings],
        git_commit=git_commit,
        version=version,
        run_id=run_id,
        created_at_utc=fixed_timestamp,
        stable=True,
    )
    au_paths = write_audit_manifest_outputs(audit, run_dir)
    record("build_audit_manifest", STATUS_OK, outputs=rel(au_paths))

    # 14. artifact manifest index (exclude self + summary, written after)
    step_by_path = {out: step.name for step in steps for out in step.outputs}
    artifact_manifest = build_artifact_manifest(
        run_dir,
        step_by_path=step_by_path,
        is_synthetic=dataset.is_synthetic,
        exclude=("artifact_manifest.json", "artifact_manifest.md",
                 "pipeline_summary.json", "pipeline_summary.md"),
    )
    am_paths = write_artifact_manifest_outputs(artifact_manifest, run_dir)
    record("build_artifact_manifest", STATUS_OK, outputs=rel(am_paths))

    # 15. pipeline summary
    summary = {
        "disclaimer": RESEARCH_DISCLAIMER,
        "research_only": True,
        "run_id": run_id,
        "created_at_utc": fixed_timestamp,
        "synthetic_vs_real": "synthetic" if dataset.is_synthetic else "real",
        "synthetic_warning": (
            "SYNTHETIC FIXTURE RESULTS — not real market evidence."
            if dataset.is_synthetic
            else None
        ),
        "primary_horizon": primary,
        "horizons": horizons,
        "no_look_ahead_status": readiness.overall_status,
        "step_count": len(steps),
        "steps": [s.to_dict() for s in steps],
        "artifact_manifest": "artifact_manifest.json",
        "audit_manifest": "audit_manifest.json",
        "modeling_report": "modeling_report.json",
        "limitations": [
            "Offline research pipeline runner, not a trading system.",
            "Synthetic pipeline outputs are not market evidence.",
            "Real-data results require check-forward-readiness=ELIGIBLE; no-look-ahead "
            "BLOCKED is surfaced, not bypassed.",
            "The determinism gate checks reproducibility, not model validity.",
            "Liquidity constraints require real ADV/liquidity data to be meaningful.",
        ],
    }
    _write_json(run_dir / "pipeline_summary.json", summary)
    (run_dir / "pipeline_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    record("build_pipeline_summary", STATUS_OK, outputs=["pipeline_summary.json",
                                                         "pipeline_summary.md"])
    # rewrite summary to include the final step (summary lists itself)
    summary["steps"] = [s.to_dict() for s in steps]
    summary["step_count"] = len(steps)
    _write_json(run_dir / "pipeline_summary.json", summary)
    (run_dir / "pipeline_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")

    summary["run_directory"] = str(run_dir)
    summary["paths"] = {
        "pipeline_summary": str(run_dir / "pipeline_summary.json"),
        "artifact_manifest": str(am_paths["json_path"]),
        "audit_manifest": str(au_paths["json_path"]),
        "modeling_report": str(rp_paths["json_path"]),
    }
    return summary


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Pipeline Summary", "", str(summary["disclaimer"]), ""]
    if summary.get("synthetic_warning"):
        lines += [f"> **{summary['synthetic_warning']}**", ""]
    lines += [
        f"- run_id: `{summary['run_id']}`",
        f"- created_at_utc: {summary['created_at_utc']}",
        f"- synthetic_vs_real: {summary['synthetic_vs_real']}",
        f"- no-look-ahead status: **{str(summary['no_look_ahead_status']).upper()}**",
        f"- primary horizon: {summary['primary_horizon']}",
        "",
        "| step | status | outputs | skipped reason |",
        "| --- | --- | --- | --- |",
    ]
    for step in summary["steps"]:
        lines.append(
            f"| {step['name']} | {step['status']} | {len(step['outputs'])} | "
            f"{step.get('skipped_reason') or '—'} |"
        )
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in summary["limitations"]]
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "RESEARCH_DISCLAIMER",
    "STATUS_BLOCKED",
    "STATUS_OK",
    "STATUS_SKIPPED",
    "PipelineConfig",
    "StepRecord",
    "run_pipeline",
]

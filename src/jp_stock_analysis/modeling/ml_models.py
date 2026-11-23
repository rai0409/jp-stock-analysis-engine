"""Optional ML ranking adapters (LightGBM / CatBoost) + the baseline.

LightGBM and CatBoost are **optional**. They are not mandatory dependencies of
this project; if they are not installed, the adapters return a clear
``optional_dependency_missing`` result instead of raising at import time, and the
test-suite still passes. Install them via the ``lightgbm`` / ``catboost`` /
``all-modeling`` extras only if you want those backends.

Validation: every trained model is evaluated **out of sample** via purged
walk-forward folds (see ``modeling/out_of_sample.py``). A score is only ever
produced for a decision date the model was not trained on, and training rows
whose forward-return window overlaps the test window are purged. Features are
within-date centred ranks; regressors fit a within-date rank target; rankers fit
within-date quantile relevance grades.

Every model output is research-only: a per-name *score* used purely for
cross-sectional ranking validation. No buy/sell signal, no predictive claim.
"""

from __future__ import annotations

import importlib.util
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from jp_stock_analysis.errors import JPStockAnalysisError
from jp_stock_analysis.modeling.baseline_ranker import score_baseline, scored_observations
from jp_stock_analysis.modeling.dataset import ModelingDataset, ModelingObservation
from jp_stock_analysis.modeling.factors import ALL_FACTORS
from jp_stock_analysis.modeling.out_of_sample import (
    VALIDATION_SCHEME,
    cross_sectional_rank_matrix,
    cross_sectional_target,
    labelled_observations,
    ranked_matrix,
    walk_forward_oos,
)
from jp_stock_analysis.modeling.ranking_metrics import ScoredObservation
from jp_stock_analysis.schemas import PriceBar

MODEL_BASELINE = "baseline_factor_ranker"
MODEL_LGBM_RANKER = "lightgbm_ranker"
MODEL_LGBM_REGRESSOR = "lightgbm_regressor"
MODEL_CATBOOST_RANKER = "catboost_ranker"
MODEL_CATBOOST_REGRESSOR = "catboost_regressor"

MODEL_TREE_ENSEMBLE = "tree_ensemble"

# The adopted production-research model: purged walk-forward OOS GBDTs, rank-
# averaged with an OOS Ridge anchor. Regressors (not LambdaRank/YetiRank) are
# used because a long-short book needs both tails ordered, not just the top.
PRIMARY_TREE_MEMBERS = ("lightgbm_regressor", "catboost_regressor")

MODEL_TYPES = (
    MODEL_BASELINE,
    MODEL_LGBM_RANKER,
    MODEL_LGBM_REGRESSOR,
    MODEL_CATBOOST_RANKER,
    MODEL_CATBOOST_REGRESSOR,
)

_BACKEND_BY_MODEL = {
    MODEL_LGBM_RANKER: "lightgbm",
    MODEL_LGBM_REGRESSOR: "lightgbm",
    MODEL_CATBOOST_RANKER: "catboost",
    MODEL_CATBOOST_REGRESSOR: "catboost",
}

# LightGBM's sklearn wrappers (LGBMRanker / LGBMRegressor) need scikit-learn.
_BACKEND_REQUIREMENTS = {
    "lightgbm": ("lightgbm", "sklearn"),
    "catboost": ("catboost",),
}

STATUS_TRAINED = "trained"
STATUS_MISSING_DEPENDENCY = "optional_dependency_missing"
STATUS_INSUFFICIENT_DATA = "insufficient_data"

RESEARCH_DISCLAIMER = (
    "This output is for analytical and self-directed research purposes. It is "
    "not personalized financial advice. Model scores are for cross-sectional "
    "ranking validation only and make no predictive or trading claim."
)


class OptionalDependencyMissing(JPStockAnalysisError):
    """Raised (or reported via status) when an optional ML backend is absent."""


def backend_available(name: str) -> bool:
    """Whether an optional backend (and its runtime requirements) is importable."""
    modules = _BACKEND_REQUIREMENTS.get(name, (name,))
    return all(importlib.util.find_spec(module) is not None for module in modules)


def available_backends() -> dict[str, bool]:
    return {"lightgbm": backend_available("lightgbm"), "catboost": backend_available("catboost")}


@dataclass(frozen=True)
class ModelResult:
    """Outcome of a (possibly skipped) model training run."""

    model_type: str
    status: str
    horizon: int
    scored: list[ScoredObservation] = field(default_factory=list)
    missing_dependency: str | None = None
    message: str = ""
    model_version: str = ""
    validation: str = VALIDATION_SCHEME
    oos_folds: list[dict[str, Any]] = field(default_factory=list)
    disclaimer: str = RESEARCH_DISCLAIMER

    @property
    def is_trained(self) -> bool:
        return self.status == STATUS_TRAINED

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "status": self.status,
            "horizon": self.horizon,
            "missing_dependency": self.missing_dependency,
            "message": self.message,
            "model_version": self.model_version,
            "validation": self.validation,
            "oos_folds": self.oos_folds,
            "scored_count": len(self.scored),
            "disclaimer": self.disclaimer,
        }


def resolve_threads(n_jobs: int) -> int:
    """Safe GBDT thread count: ``<= 0`` means "all physical cores".

    Never exceeds half the logical CPUs (≈ physical cores). Oversubscribing
    LightGBM's OpenMP pool with hyper-threads (e.g. ``n_jobs=-1`` -> all
    logical CPUs) was measured to stall training indefinitely under WSL2, while
    the physical-core count trains normally.
    """
    physical = max(1, (os.cpu_count() or 2) // 2)
    if n_jobs <= 0:
        return physical
    return min(n_jobs, physical)


def _ranks_for(
    dataset: ModelingDataset | None,
    rows: Sequence[ModelingObservation],
    names: Sequence[str],
) -> list[list[float | None]]:
    """Cached within-date ranks (dataset-wide) or per-call ranks as a fallback."""
    if dataset is not None:
        return ranked_matrix(dataset, rows, names)
    return cross_sectional_rank_matrix(rows, names)


def _dense(matrix: Sequence[Sequence[float | None]]) -> list[list[float]]:
    """``None`` -> NaN. GBDT backends route missing values natively."""
    return [[math.nan if v is None else float(v) for v in row] for row in matrix]


def train_ranking_model(
    dataset: ModelingDataset,
    model_type: str,
    *,
    horizon: int,
    n_quantiles: int = 5,
    params: dict[str, Any] | None = None,
    prices: Mapping[str, Sequence[PriceBar]] | None = None,
    min_train_periods: int = 1,
    embargo_days: int = 0,
    refit_every: int = 1,
    n_jobs: int = 1,
) -> ModelResult:
    """Train (or gracefully skip) a ranking model with purged walk-forward OOS.

    ``refit_every`` refits every N decision dates (e.g. 4 = monthly on weekly
    decisions). ``n_jobs`` sets GBDT threads (``-1`` = physical cores, capped —
    see :func:`resolve_threads`); results stay reproducible (LightGBM
    ``deterministic=True``; CatBoost fixed seed).

    Never raises on missing optional dependencies. ``prices`` (when given) makes
    the purge use exact label end dates; otherwise a conservative calendar
    estimate is used.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")

    if model_type == MODEL_BASELINE:
        # Fixed-weight, unfitted ranker: it sees no labels, so every score is
        # already out of sample by construction.
        scores = score_baseline(dataset)
        scored = [s for s in scored_observations(dataset, scores) if s.score is not None]
        return ModelResult(
            model_type=model_type,
            status=STATUS_TRAINED,
            horizon=horizon,
            scored=scored,
            model_version="baseline_factor_ranker_v1",
            validation="unfitted_fixed_weights",
        )

    backend = _BACKEND_BY_MODEL[model_type]
    if not backend_available(backend):
        missing = [
            m for m in _BACKEND_REQUIREMENTS[backend] if importlib.util.find_spec(m) is None
        ]
        return ModelResult(
            model_type=model_type,
            status=STATUS_MISSING_DEPENDENCY,
            horizon=horizon,
            missing_dependency=backend,
            message=(
                f"optional dependency '{backend}' is not installed (missing: "
                f"{', '.join(missing)}); install the '{backend}' extra to enable "
                f"{model_type}. Skipped, not failed."
            ),
        )

    observations, _label_key = labelled_observations(dataset, horizon)
    if len(observations) < n_quantiles + 1:
        return ModelResult(
            model_type=model_type,
            status=STATUS_INSUFFICIENT_DATA,
            horizon=horizon,
            message=f"too few labelled observations ({len(observations)}) to train",
        )

    names = list(dataset.feature_names)
    n_jobs = resolve_threads(n_jobs)
    if backend == "lightgbm":
        fit_predict, version = _lightgbm_fit_predict(
            model_type, n_quantiles, {"n_jobs": n_jobs, **(params or {})}, names, dataset
        )
    else:
        fit_predict, version = _catboost_fit_predict(
            model_type, n_quantiles, {"thread_count": n_jobs, **(params or {})}, names, dataset
        )

    oos = walk_forward_oos(
        dataset,
        horizon,
        fit_predict,
        prices=prices,
        min_train_periods=min_train_periods,
        embargo_days=embargo_days,
        min_train_rows=n_quantiles + 1,
        refit_every=refit_every,
    )
    folds = [f.to_dict() for f in oos.folds]
    if not oos.ok:
        return ModelResult(
            model_type=model_type,
            status=STATUS_INSUFFICIENT_DATA,
            horizon=horizon,
            message=f"no out-of-sample predictions: {oos.status} — {oos.message}",
            model_version=version,
            oos_folds=folds,
        )
    return ModelResult(
        model_type=model_type,
        status=STATUS_TRAINED,
        horizon=horizon,
        scored=oos.scored,
        model_version=version,
        oos_folds=folds,
    )


def _relevance_labels(
    observations: Sequence[ModelingObservation], label_key: str, n_quantiles: int
) -> tuple[list[ModelingObservation], list[int]]:
    """Per-date quantile relevance grades (0..n-1) for ranking objectives.

    Dates with fewer names than ``n_quantiles`` cannot be graded and are
    **dropped** (grading them all 0 would teach the ranker that every name on
    that date is equally irrelevant — pure label noise).
    """
    from jp_stock_analysis.modeling.ranking_metrics import _quantile_buckets

    by_date: dict[date, list[ModelingObservation]] = {}
    for obs in observations:
        by_date.setdefault(obs.decision_date, []).append(obs)
    kept: list[ModelingObservation] = []
    relevance: list[int] = []
    for decision_date in sorted(by_date):
        group = sorted(by_date[decision_date], key=lambda o: o.ticker)
        if len(group) < n_quantiles:
            continue
        values = [float(o.labels[label_key]) for o in group]  # type: ignore[arg-type]
        kept.extend(group)
        relevance.extend(_quantile_buckets(values, n_quantiles))
    return kept, relevance


def _group_sizes(observations: Sequence[ModelingObservation]) -> list[int]:
    sizes: list[int] = []
    current: date | None = None
    count = 0
    for obs in observations:
        if obs.decision_date != current:
            if current is not None:
                sizes.append(count)
            current = obs.decision_date
            count = 0
        count += 1
    if current is not None:
        sizes.append(count)
    return sizes


def _lightgbm_params(n_train: int, params: dict[str, Any] | None) -> dict[str, Any]:
    """Conservative, deterministic GBDT defaults for low-SNR return data.

    Cross-sectional return prediction has a very low signal-to-noise ratio, so
    the defaults favour heavy regularisation: shallow trees, a minimum leaf size
    that scales with the sample, L2 shrinkage, row/feature subsampling and a
    small learning rate. Single-threaded + fixed seed => bit-reproducible.
    """
    base = {
        "verbose": -1,
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": max(5, n_train // 50),
        "reg_lambda": 5.0,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 0,
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": 1,
    }
    base.update(params or {})
    return base


def _lightgbm_fit_predict(
    model_type: str,
    n_quantiles: int,
    params: dict[str, Any] | None,
    names: Sequence[str] = ALL_FACTORS,
    dataset: ModelingDataset | None = None,
):  # pragma: no cover - exercised only when lightgbm is installed
    import lightgbm as lgb
    import numpy as np

    names = list(names)

    def fit_predict(train, test, label_key):
        if model_type == MODEL_LGBM_RANKER:
            rows, relevance = _relevance_labels(train, label_key, n_quantiles)
            if not rows:
                return [0.0] * len(test)
            features = np.asarray(_dense(_ranks_for(dataset, rows, names)))
            model = lgb.LGBMRanker(**_lightgbm_params(len(rows), params))
            model.fit(features, np.asarray(relevance), group=_group_sizes(rows))
        else:
            features = np.asarray(_dense(_ranks_for(dataset, train, names)))
            target = np.asarray(cross_sectional_target(train, label_key), dtype=float)
            model = lgb.LGBMRegressor(**_lightgbm_params(len(train), params))
            model.fit(features, target)
        x_test = np.asarray(_dense(_ranks_for(dataset, test, names)))
        return [float(p) for p in model.predict(x_test)]

    return fit_predict, f"lightgbm_{lgb.__version__}"


def _catboost_fit_predict(
    model_type: str,
    n_quantiles: int,
    params: dict[str, Any] | None,
    names: Sequence[str] = ALL_FACTORS,
    dataset: ModelingDataset | None = None,
):  # pragma: no cover - exercised only when catboost is installed
    import catboost as cb
    import numpy as np

    names = list(names)
    base_params = {
        "verbose": False,
        "iterations": 300,
        "learning_rate": 0.03,
        "depth": 3,
        "l2_leaf_reg": 10.0,
        "random_seed": 0,
        "thread_count": 1,
        "allow_writing_files": False,  # never write catboost_info/ outside outputs
        # NaN stays NaN: CatBoost treats it as "missing", which 0.0 would not.
        "nan_mode": "Min",
        **(params or {}),
    }

    def fit_predict(train, test, label_key):
        x_test = np.asarray(_dense(_ranks_for(dataset, test, names)))
        if model_type == MODEL_CATBOOST_RANKER:
            rows, relevance = _relevance_labels(train, label_key, n_quantiles)
            if not rows:
                return [0.0] * len(test)
            features = np.asarray(_dense(_ranks_for(dataset, rows, names)))
            group_id = [o.decision_date.toordinal() for o in rows]
            pool = cb.Pool(features, label=relevance, group_id=group_id)
            model = cb.CatBoost({"loss_function": "YetiRank", **base_params})
            model.fit(pool)
            return [float(p) for p in model.predict(cb.Pool(x_test))]
        features = np.asarray(_dense(_ranks_for(dataset, train, names)))
        target = np.asarray(cross_sectional_target(train, label_key), dtype=float)
        model = cb.CatBoostRegressor(**base_params)
        model.fit(features, target)
        return [float(p) for p in model.predict(x_test)]

    return fit_predict, f"catboost_{cb.__version__}"


@dataclass(frozen=True)
class PrimaryModelResult:
    """The model whose OOS scores drive ranking/portfolio/monitoring steps."""

    model_label: str
    horizon: int
    scored: list[ScoredObservation]
    members: dict[str, dict[str, Any]]
    fallback_used: bool
    warnings: list[str] = field(default_factory=list)
    ensemble_diversity: float | None = None
    # trained member results (not serialised) so reports can reuse them
    member_results: dict[str, ModelResult] = field(default_factory=dict, repr=False)
    validation: str = VALIDATION_SCHEME
    disclaimer: str = RESEARCH_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_label": self.model_label,
            "horizon": self.horizon,
            "validation": self.validation,
            "fallback_used": self.fallback_used,
            "members": self.members,
            "ensemble_diversity": self.ensemble_diversity,
            "scored_count": len(self.scored),
            "scored_dates": sorted({s.decision_date.isoformat() for s in self.scored}),
            "warnings": list(self.warnings),
            "disclaimer": self.disclaimer,
        }


def train_primary_model(
    dataset: ModelingDataset,
    *,
    horizon: int,
    model: str = MODEL_TREE_ENSEMBLE,
    prices: Mapping[str, Sequence[PriceBar]] | None = None,
    tree_members: Sequence[str] = PRIMARY_TREE_MEMBERS,
    include_ridge: bool = True,
    n_quantiles: int = 5,
    min_train_periods: int = 1,
    embargo_days: int = 0,
    refit_every: int = 1,
    n_jobs: int = 1,
) -> PrimaryModelResult:
    """Train the adopted primary model (default: OOS tree ensemble).

    ``model`` is ``tree_ensemble`` (rank-average of the OOS GBDT regressors and,
    optionally, an OOS Ridge anchor) or any single ``MODEL_TYPES`` entry. When no
    tree backend is installed or no OOS fold is trainable, it falls back to the
    unfitted baseline ranker and says so explicitly — it never silently swaps in
    in-sample scores.
    """
    from jp_stock_analysis.modeling.ensemble import rank_average_ensemble
    from jp_stock_analysis.modeling.linear_models import RidgeRanker
    from jp_stock_analysis.modeling.out_of_sample import linear_walk_forward_oos

    if model != MODEL_TREE_ENSEMBLE and model not in MODEL_TYPES:
        raise ValueError(
            f"unknown primary model {model!r}; expected {MODEL_TREE_ENSEMBLE!r} or one of "
            f"{MODEL_TYPES}"
        )
    warnings: list[str] = []
    members: dict[str, dict[str, Any]] = {}
    member_results: dict[str, ModelResult] = {}
    member_scores: dict[str, list[ScoredObservation]] = {}

    wanted = list(tree_members) if model == MODEL_TREE_ENSEMBLE else [model]
    for member in wanted:
        result = train_ranking_model(
            dataset,
            member,
            horizon=horizon,
            n_quantiles=n_quantiles,
            prices=prices,
            min_train_periods=min_train_periods,
            embargo_days=embargo_days,
            refit_every=refit_every,
            n_jobs=n_jobs,
        )
        members[member] = result.to_dict()
        member_results[member] = result
        if result.is_trained and result.scored:
            member_scores[member] = result.scored
        else:
            warnings.append(f"{member}: {result.status} — {result.message}")

    has_tree = any(m in member_scores for m in wanted if m != MODEL_BASELINE)
    if model == MODEL_TREE_ENSEMBLE and include_ridge and has_tree:
        run = linear_walk_forward_oos(
            dataset,
            horizon,
            lambda: RidgeRanker(alpha=1.0),
            prices=prices,
            min_train_periods=min_train_periods,
            embargo_days=embargo_days,
            refit_every=refit_every,
        )
        members["ridge"] = run.result.to_dict()
        if run.result.ok:
            member_scores["ridge"] = run.result.scored

    if model == MODEL_BASELINE or (model != MODEL_BASELINE and not has_tree):
        if model != MODEL_BASELINE:
            warnings.append(
                "no tree model produced out-of-sample scores; falling back to the "
                "unfitted baseline factor ranker"
            )
        baseline = train_ranking_model(dataset, MODEL_BASELINE, horizon=horizon)
        return PrimaryModelResult(
            model_label=MODEL_BASELINE,
            horizon=horizon,
            scored=baseline.scored,
            members=members,
            fallback_used=model != MODEL_BASELINE,
            warnings=warnings,
            validation=baseline.validation,
            member_results=member_results,
        )

    if len(member_scores) == 1:
        (label, scored), = member_scores.items()
        return PrimaryModelResult(
            model_label=label, horizon=horizon, scored=scored, members=members,
            fallback_used=False, warnings=warnings, member_results=member_results,
        )

    ensemble = rank_average_ensemble(member_scores, is_synthetic=dataset.is_synthetic)
    warnings.extend(ensemble.warnings)
    return PrimaryModelResult(
        model_label=MODEL_TREE_ENSEMBLE,
        horizon=horizon,
        scored=ensemble.scored,
        members=members,
        fallback_used=False,
        warnings=warnings,
        ensemble_diversity=ensemble.diversity_score,
        member_results=member_results,
    )


__all__ = [
    "MODEL_BASELINE",
    "MODEL_TREE_ENSEMBLE",
    "PRIMARY_TREE_MEMBERS",
    "PrimaryModelResult",
    "train_primary_model",
    "MODEL_CATBOOST_RANKER",
    "MODEL_CATBOOST_REGRESSOR",
    "MODEL_LGBM_RANKER",
    "MODEL_LGBM_REGRESSOR",
    "MODEL_TYPES",
    "STATUS_INSUFFICIENT_DATA",
    "STATUS_MISSING_DEPENDENCY",
    "STATUS_TRAINED",
    "ModelResult",
    "OptionalDependencyMissing",
    "available_backends",
    "backend_available",
    "train_ranking_model",
]

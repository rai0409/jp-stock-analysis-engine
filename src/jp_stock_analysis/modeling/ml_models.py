"""Optional ML ranking adapters (LightGBM / CatBoost) + the baseline.

LightGBM and CatBoost are **optional**. They are not mandatory dependencies of
this project; if they are not installed, the adapters return a clear
``optional_dependency_missing`` result instead of raising at import time, and the
test-suite still passes. Install them via the ``lightgbm`` / ``catboost`` /
``all-modeling`` extras only if you want those backends.

Every model output is research-only: a per-name *score* used purely for
cross-sectional ranking validation. No buy/sell signal, no predictive claim.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from jp_stock_analysis.errors import JPStockAnalysisError
from jp_stock_analysis.modeling.baseline_ranker import score_baseline, scored_observations
from jp_stock_analysis.modeling.dataset import ModelingDataset, ModelingObservation
from jp_stock_analysis.modeling.factors import ALL_FACTORS
from jp_stock_analysis.modeling.ranking_metrics import ScoredObservation

MODEL_BASELINE = "baseline_factor_ranker"
MODEL_LGBM_RANKER = "lightgbm_ranker"
MODEL_LGBM_REGRESSOR = "lightgbm_regressor"
MODEL_CATBOOST_RANKER = "catboost_ranker"
MODEL_CATBOOST_REGRESSOR = "catboost_regressor"

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
    """Whether an optional backend importable without importing it eagerly."""
    return importlib.util.find_spec(name) is not None


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
            "scored_count": len(self.scored),
            "disclaimer": self.disclaimer,
        }


def _feature_matrix(
    observations: Sequence[ModelingObservation],
) -> tuple[list[list[float]], list[str]]:
    """Dense feature matrix (missing -> NaN; GBDT backends handle NaN natively)."""
    matrix: list[list[float]] = []
    for obs in observations:
        row = [
            float(obs.features[name]) if obs.features.get(name) is not None else math.nan
            for name in ALL_FACTORS
        ]
        matrix.append(row)
    return matrix, list(ALL_FACTORS)


def _labelled(
    dataset: ModelingDataset, horizon: int
) -> tuple[list[ModelingObservation], str]:
    label_key = f"forward_return_h{horizon}"
    obs = [o for o in dataset.included() if o.labels.get(label_key) is not None]
    obs.sort(key=lambda o: (o.decision_date, o.ticker))
    return obs, label_key


def _scored_from_predictions(
    observations: Sequence[ModelingObservation],
    predictions: Sequence[float],
) -> list[ScoredObservation]:
    out: list[ScoredObservation] = []
    for obs, prediction in zip(observations, predictions, strict=False):
        out.append(
            ScoredObservation(
                decision_date=obs.decision_date,
                ticker=obs.ticker,
                score=float(prediction),
                sector=obs.sector,
                labels=obs.labels,
            )
        )
    return out


def train_ranking_model(
    dataset: ModelingDataset,
    model_type: str,
    *,
    horizon: int,
    n_quantiles: int = 5,
    params: dict[str, Any] | None = None,
) -> ModelResult:
    """Train (or gracefully skip) a ranking model; never raises on missing deps."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")

    if model_type == MODEL_BASELINE:
        scores = score_baseline(dataset)
        scored = [s for s in scored_observations(dataset, scores) if s.score is not None]
        return ModelResult(
            model_type=model_type,
            status=STATUS_TRAINED,
            horizon=horizon,
            scored=scored,
            model_version="baseline_factor_ranker_v1",
        )

    backend = _BACKEND_BY_MODEL[model_type]
    if not backend_available(backend):
        return ModelResult(
            model_type=model_type,
            status=STATUS_MISSING_DEPENDENCY,
            horizon=horizon,
            missing_dependency=backend,
            message=(
                f"optional dependency '{backend}' is not installed; install the "
                f"'{backend}' extra to enable {model_type}. Skipped, not failed."
            ),
        )

    observations, label_key = _labelled(dataset, horizon)
    if len(observations) < n_quantiles + 1:
        return ModelResult(
            model_type=model_type,
            status=STATUS_INSUFFICIENT_DATA,
            horizon=horizon,
            message=f"too few labelled observations ({len(observations)}) to train",
        )

    if backend == "lightgbm":
        return _train_lightgbm(
            model_type, dataset, observations, label_key, horizon, n_quantiles, params
        )
    return _train_catboost(
        model_type, dataset, observations, label_key, horizon, n_quantiles, params
    )


def _relevance_labels(
    observations: Sequence[ModelingObservation], label_key: str, n_quantiles: int
) -> list[int]:
    """Per-date quantile relevance labels (0..n-1) for ranking objectives."""
    from jp_stock_analysis.modeling.ranking_metrics import _quantile_buckets

    by_date: dict[date, list[int]] = {}
    for index, obs in enumerate(observations):
        by_date.setdefault(obs.decision_date, []).append(index)
    relevance = [0] * len(observations)
    for indices in by_date.values():
        values = [float(observations[i].labels[label_key]) for i in indices]  # type: ignore[arg-type]
        if len(values) < n_quantiles:
            continue
        buckets = _quantile_buckets(values, n_quantiles)
        for position, index in enumerate(indices):
            relevance[index] = buckets[position]
    return relevance


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


def _train_lightgbm(
    model_type: str,
    dataset: ModelingDataset,
    observations: Sequence[ModelingObservation],
    label_key: str,
    horizon: int,
    n_quantiles: int,
    params: dict[str, Any] | None,
) -> ModelResult:  # pragma: no cover - exercised only when lightgbm is installed
    import lightgbm as lgb
    import numpy as np

    matrix, _names = _feature_matrix(observations)
    features = np.asarray(matrix, dtype=float)
    base_params = {"verbose": -1, "min_data_in_leaf": 1, "num_leaves": 7, **(params or {})}

    if model_type == MODEL_LGBM_RANKER:
        relevance = np.asarray(_relevance_labels(observations, label_key, n_quantiles))
        model = lgb.LGBMRanker(**base_params)
        model.fit(features, relevance, group=_group_sizes(observations))
    else:
        target = np.asarray(
            [float(o.labels[label_key]) for o in observations], dtype=float
        )
        model = lgb.LGBMRegressor(**base_params)
        model.fit(features, target)

    predictions = model.predict(features)
    return ModelResult(
        model_type=model_type,
        status=STATUS_TRAINED,
        horizon=horizon,
        scored=_scored_from_predictions(observations, list(predictions)),
        model_version=f"lightgbm_{lgb.__version__}",
    )


def _train_catboost(
    model_type: str,
    dataset: ModelingDataset,
    observations: Sequence[ModelingObservation],
    label_key: str,
    horizon: int,
    n_quantiles: int,
    params: dict[str, Any] | None,
) -> ModelResult:  # pragma: no cover - exercised only when catboost is installed
    import catboost as cb
    import numpy as np

    matrix, _names = _feature_matrix(observations)
    features = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
    base_params = {"verbose": False, "iterations": 50, "depth": 3, **(params or {})}

    if model_type == MODEL_CATBOOST_RANKER:
        relevance = np.asarray(_relevance_labels(observations, label_key, n_quantiles))
        group_id = []
        for size, obs in zip(
            _group_sizes(observations), _unique_dates(observations), strict=False
        ):
            group_id.extend([obs.toordinal()] * size)
        pool = cb.Pool(features, label=relevance, group_id=group_id)
        model = cb.CatBoost({"loss_function": "YetiRank", **base_params})
        model.fit(pool)
        predictions = model.predict(pool)
    else:
        target = np.asarray([float(o.labels[label_key]) for o in observations], dtype=float)
        model = cb.CatBoostRegressor(**base_params)
        model.fit(features, target)
        predictions = model.predict(features)

    return ModelResult(
        model_type=model_type,
        status=STATUS_TRAINED,
        horizon=horizon,
        scored=_scored_from_predictions(observations, list(predictions)),
        model_version=f"catboost_{cb.__version__}",
    )


def _unique_dates(observations: Sequence[ModelingObservation]) -> list[date]:
    seen: list[date] = []
    for obs in observations:
        if not seen or seen[-1] != obs.decision_date:
            seen.append(obs.decision_date)
    return seen


__all__ = [
    "MODEL_BASELINE",
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

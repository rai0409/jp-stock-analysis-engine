"""Transparent baseline factor ranker (analysis_only).

A fully explainable, deterministic cross-sectional ranker: within each decision
date it direction-adjusts every factor (lower-is-better factors inverted),
winsorises and z-scores them, averages available factors within each group
(equal weight), then combines groups with configurable weights (conservative
equal weights by default). It outputs a ``factor_score``, ``factor_rank``, a
``sector_neutral_factor_score``, the ``missing_feature_count``, and a
``model_version``.

It emits **no** buy/sell signal and makes no predictive claim — it is a ranking
baseline for research validation only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from jp_stock_analysis.modeling.dataset import ModelingDataset, ModelingObservation
from jp_stock_analysis.modeling.factors import (
    FACTOR_DIRECTION,
    FACTOR_GROUPS,
    sector_zscore,
    winsorize,
    zscore,
)
from jp_stock_analysis.modeling.ranking_metrics import ScoredObservation

MODEL_VERSION = "baseline_factor_ranker_v1"


@dataclass(frozen=True)
class BaselineScore:
    ticker: str
    decision_date: date
    factor_score: float | None
    factor_rank: int | None
    sector_neutral_factor_score: float | None
    missing_feature_count: int
    available_group_count: int
    model_version: str = MODEL_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "decision_date": self.decision_date.isoformat(),
            "factor_score": self.factor_score,
            "factor_rank": self.factor_rank,
            "sector_neutral_factor_score": self.sector_neutral_factor_score,
            "missing_feature_count": self.missing_feature_count,
            "available_group_count": self.available_group_count,
            "model_version": self.model_version,
        }


def _directional(values: Sequence[float | None], factor: str) -> list[float | None]:
    """Negate lower-is-better factors so higher z-score is always more favourable."""
    if FACTOR_DIRECTION.get(factor, True):
        return list(values)
    return [None if v is None else -v for v in values]


def _group_scores(
    observations: Sequence[ModelingObservation], *, sector_neutral: bool
) -> tuple[list[dict[str, float | None]], list[int]]:
    """Per-observation group z-score means and available-group counts."""
    sectors = [o.sector for o in observations]
    # factor -> normalised value per observation
    normalised: dict[str, list[float | None]] = {}
    for group in FACTOR_GROUPS.values():
        for factor in group:
            raw = _directional([o.features.get(factor) for o in observations], factor)
            clipped = winsorize(raw)
            normalised[factor] = (
                sector_zscore(clipped, sectors) if sector_neutral else zscore(clipped)
            )

    group_means: list[dict[str, float | None]] = []
    available_counts: list[int] = []
    for index in range(len(observations)):
        per_group: dict[str, float | None] = {}
        available = 0
        for group_name, factors in FACTOR_GROUPS.items():
            present = [
                normalised[factor][index]
                for factor in factors
                if normalised[factor][index] is not None
            ]
            if present:
                per_group[group_name] = sum(present) / len(present)
                available += 1
            else:
                per_group[group_name] = None
        group_means.append(per_group)
        available_counts.append(available)
    return group_means, available_counts


def _combine(
    per_group: Mapping[str, float | None], group_weights: Mapping[str, float]
) -> float | None:
    total_weight = 0.0
    accumulated = 0.0
    for group_name, value in per_group.items():
        if value is None:
            continue
        weight = group_weights.get(group_name, 0.0)
        if weight <= 0:
            continue
        accumulated += weight * value
        total_weight += weight
    if total_weight == 0:
        return None
    return accumulated / total_weight


def score_baseline(
    dataset: ModelingDataset,
    *,
    group_weights: Mapping[str, float] | None = None,
) -> list[BaselineScore]:
    """Score every included observation, ranked within each decision date."""
    weights = dict.fromkeys(FACTOR_GROUPS, 1.0)
    if group_weights:
        for key, value in group_weights.items():
            if value < 0:
                raise ValueError("group weights must be non-negative")
            weights[key] = value

    by_date: dict[date, list[ModelingObservation]] = {}
    for obs in dataset.included():
        by_date.setdefault(obs.decision_date, []).append(obs)

    scores: list[BaselineScore] = []
    for decision_date in sorted(by_date):
        group = sorted(by_date[decision_date], key=lambda o: o.ticker)
        plain_groups, available_counts = _group_scores(group, sector_neutral=False)
        neutral_groups, _ = _group_scores(group, sector_neutral=True)

        raw_scores = [_combine(per_group, weights) for per_group in plain_groups]
        neutral_scores = [_combine(per_group, weights) for per_group in neutral_groups]
        ranks = _rank_desc(raw_scores)

        for index, obs in enumerate(group):
            scores.append(
                BaselineScore(
                    ticker=obs.ticker,
                    decision_date=decision_date,
                    factor_score=raw_scores[index],
                    factor_rank=ranks[index],
                    sector_neutral_factor_score=neutral_scores[index],
                    missing_feature_count=obs.missing_feature_count,
                    available_group_count=available_counts[index],
                )
            )
    return scores


def _rank_desc(scores: Sequence[float | None]) -> list[int | None]:
    """1 = highest score. ``None`` scores are unranked (rank ``None``)."""
    ranked = sorted(
        (i for i, s in enumerate(scores) if s is not None),
        key=lambda i: (-scores[i], i),  # type: ignore[operator]
    )
    out: list[int | None] = [None] * len(scores)
    for position, index in enumerate(ranked):
        out[index] = position + 1
    return out


def scored_observations(
    dataset: ModelingDataset, scores: Sequence[BaselineScore]
) -> list[ScoredObservation]:
    """Join baseline scores to dataset labels for ranking-metric evaluation."""
    labels_by_key = {
        (o.ticker, o.decision_date): o.labels for o in dataset.included()
    }
    sector_by_key = {(o.ticker, o.decision_date): o.sector for o in dataset.included()}
    out: list[ScoredObservation] = []
    for score in scores:
        key = (score.ticker, score.decision_date)
        out.append(
            ScoredObservation(
                decision_date=score.decision_date,
                ticker=score.ticker,
                score=score.factor_score,
                sector=sector_by_key.get(key),
                labels=labels_by_key.get(key, {}),
            )
        )
    return out


__all__ = [
    "MODEL_VERSION",
    "BaselineScore",
    "score_baseline",
    "scored_observations",
]

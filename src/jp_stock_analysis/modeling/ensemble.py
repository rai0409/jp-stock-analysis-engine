"""Ensemble / blending of multiple model predictions (research-only).

Combines several models' per-observation predictions into a single ranking
signal — by **rank averaging** or a **weighted blend** of per-decision-date
standardized predictions — plus simple diversity diagnostics (pairwise
correlation, a diversity score). Outputs are ``ScoredObservation`` lists, so they
flow straight into Rank IC, portfolio metrics, and neutralization.

Deterministic, offline, scale-robust (it ranks / standardizes within each
decision date). No predictive or trading claim; synthetic results are not
evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from jp_stock_analysis.modeling.ranking_metrics import (
    RESEARCH_DISCLAIMER,
    ScoredObservation,
    _ranks,
    spearman,
)

METHOD_RANK_AVERAGE = "rank_average"
METHOD_WEIGHTED_BLEND = "weighted_blend"

STATUS_OK = "ok"
STATUS_NO_COMMON_OBSERVATIONS = "no_common_observations"
STATUS_MISSING_COLUMN = "missing_column"
STATUS_NEAR_IDENTICAL = "near_identical_models"

_NEAR_IDENTICAL_CORR = 0.999


@dataclass(frozen=True)
class EnsembleResult:
    method: str
    model_names: list[str]
    scored: list[ScoredObservation]
    pairwise_correlations: dict[str, float | None]
    diversity_score: float | None
    status: str
    warnings: list[str] = field(default_factory=list)
    is_synthetic: bool = False
    disclaimer: str = RESEARCH_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "research_only": True,
            "method": self.method,
            "model_names": self.model_names,
            "status": self.status,
            "warnings": self.warnings,
            "pairwise_correlations": self.pairwise_correlations,
            "diversity_score": self.diversity_score,
            "scored_count": len(self.scored),
            "is_synthetic": self.is_synthetic,
            "synthetic_warning": (
                "SYNTHETIC FIXTURE RESULTS — not real market evidence."
                if self.is_synthetic
                else None
            ),
        }


def _index(predictions: Mapping[str, Sequence[ScoredObservation]]):
    """Per-model {(ticker, decision_date): score} for non-missing scores."""
    return {
        name: {
            (o.ticker, o.decision_date): o.score
            for o in obs
            if o.score is not None
        }
        for name, obs in predictions.items()
    }


def _common_keys(indexed: Mapping[str, Mapping[tuple[str, date], float]]) -> list[tuple[str, date]]:
    if not indexed:
        return []
    common: set[tuple[str, date]] | None = None
    for mapping in indexed.values():
        keys = set(mapping)
        common = keys if common is None else (common & keys)
    return sorted(common or [], key=lambda k: (k[1], k[0]))


def _sectors_labels(predictions: Mapping[str, Sequence[ScoredObservation]]):
    sectors: dict[tuple[str, date], str | None] = {}
    labels: dict[tuple[str, date], dict[str, float | None]] = {}
    for obs in predictions.values():
        for o in obs:
            key = (o.ticker, o.decision_date)
            sectors.setdefault(key, o.sector)
            labels.setdefault(key, o.labels)
    return sectors, labels


def _pairwise(indexed, keys) -> tuple[dict[str, float | None], float | None]:
    names = sorted(indexed)
    corrs: dict[str, float | None] = {}
    values: list[float] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            xs = [indexed[a][k] for k in keys]
            ys = [indexed[b][k] for k in keys]
            corr = spearman(xs, ys) if len(keys) >= 2 else None
            corrs[f"{a}|{b}"] = corr
            if corr is not None:
                values.append(corr)
    diversity = (1.0 - (sum(values) / len(values))) if values else None
    return corrs, diversity


def _build_scored(
    keys, scores, predictions
) -> list[ScoredObservation]:
    sectors, labels = _sectors_labels(predictions)
    return [
        ScoredObservation(
            decision_date=key[1],
            ticker=key[0],
            score=scores[key],
            sector=sectors.get(key),
            labels=labels.get(key, {}),
        )
        for key in keys
    ]


def rank_average_ensemble(
    predictions: Mapping[str, Sequence[ScoredObservation]],
    *,
    is_synthetic: bool = False,
) -> EnsembleResult:
    """Average per-decision-date ranks across models (deterministic ties)."""
    names = sorted(predictions)
    indexed = _index(predictions)
    keys = _common_keys(indexed)
    warnings: list[str] = []
    if not keys:
        return EnsembleResult(
            METHOD_RANK_AVERAGE, names, [], {}, None, STATUS_NO_COMMON_OBSERVATIONS,
            ["no observations shared by all models"], is_synthetic,
        )

    by_date: dict[date, list[tuple[str, date]]] = {}
    for key in keys:
        by_date.setdefault(key[1], []).append(key)

    rank_sums: dict[tuple[str, date], float] = {}
    for _d, group in by_date.items():
        for name in names:
            ranks = _ranks([indexed[name][k] for k in group])
            for key, rank in zip(group, ranks, strict=True):
                rank_sums[key] = rank_sums.get(key, 0.0) + rank
    scores = {key: rank_sums[key] / len(names) for key in keys}

    corrs, diversity = _pairwise(indexed, keys)
    status = STATUS_OK
    if diversity is not None and diversity < (1.0 - _NEAR_IDENTICAL_CORR):
        status = STATUS_NEAR_IDENTICAL
        warnings.append("models are near-identical: ensemble adds little diversity")
    return EnsembleResult(
        METHOD_RANK_AVERAGE, names, _build_scored(keys, scores, predictions),
        corrs, diversity, status, warnings, is_synthetic,
    )


def _zscore_by_date(indexed_model, keys) -> dict[tuple[str, date], float]:
    by_date: dict[date, list[tuple[str, date]]] = {}
    for key in keys:
        by_date.setdefault(key[1], []).append(key)
    out: dict[tuple[str, date], float] = {}
    for _d, group in by_date.items():
        vals = [indexed_model[k] for k in group]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var**0.5
        for key in group:
            out[key] = 0.0 if std == 0.0 else (indexed_model[key] - mean) / std
    return out


def weighted_blend(
    predictions: Mapping[str, Sequence[ScoredObservation]],
    weights: Mapping[str, float],
    *,
    is_synthetic: bool = False,
) -> EnsembleResult:
    """Weighted blend of per-decision-date standardized predictions."""
    names = sorted(predictions)
    missing = [name for name in weights if name not in predictions]
    if missing:
        return EnsembleResult(
            METHOD_WEIGHTED_BLEND, names, [], {}, None, STATUS_MISSING_COLUMN,
            [f"weight references unknown model(s): {sorted(missing)}"], is_synthetic,
        )
    usable = {name: float(w) for name, w in weights.items() if w > 0 and name in predictions}
    total = sum(usable.values())
    if not usable or total <= 0:
        return EnsembleResult(
            METHOD_WEIGHTED_BLEND, names, [], {}, None, STATUS_MISSING_COLUMN,
            ["no usable positive weights"], is_synthetic,
        )
    normalized = {name: w / total for name, w in usable.items()}

    indexed = _index({name: predictions[name] for name in normalized})
    keys = _common_keys(indexed)
    if not keys:
        return EnsembleResult(
            METHOD_WEIGHTED_BLEND, sorted(normalized), [], {}, None,
            STATUS_NO_COMMON_OBSERVATIONS, ["no observations shared by weighted models"],
            is_synthetic,
        )

    standardized = {name: _zscore_by_date(indexed[name], keys) for name in normalized}
    scores = {
        key: sum(normalized[name] * standardized[name][key] for name in normalized)
        for key in keys
    }
    corrs, diversity = _pairwise(indexed, keys)
    return EnsembleResult(
        METHOD_WEIGHTED_BLEND, sorted(normalized),
        _build_scored(keys, scores, predictions), corrs, diversity, STATUS_OK,
        [f"normalized weights: {normalized}"], is_synthetic,
    )


__all__ = [
    "METHOD_RANK_AVERAGE",
    "METHOD_WEIGHTED_BLEND",
    "STATUS_MISSING_COLUMN",
    "STATUS_NEAR_IDENTICAL",
    "STATUS_NO_COMMON_OBSERVATIONS",
    "STATUS_OK",
    "EnsembleResult",
    "rank_average_ensemble",
    "weighted_blend",
]

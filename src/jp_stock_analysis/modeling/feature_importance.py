"""Feature importance: coefficient-based and permutation-based (research-only).

Explanatory diagnostics only — **not causal proof** and not a predictive claim.
Coefficient importance uses absolute standardized coefficients; permutation
importance (synthetic/offline, deterministic seed) measures the Rank IC
degradation when one feature is shuffled. Degenerate metrics return a clear
status rather than raising.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from jp_stock_analysis.modeling.ranking_metrics import RESEARCH_DISCLAIMER, spearman

STATUS_OK = "ok"
STATUS_ALL_ZERO = "all_zero_coefficients"
STATUS_METRIC_UNAVAILABLE = "metric_unavailable"

EXPLANATORY_CAVEAT = (
    "Feature importance is explanatory research output only, not causal proof "
    "and not a predictive claim."
)


@dataclass(frozen=True)
class ImportanceRow:
    feature: str
    importance: float
    metric_delta: float | None
    direction: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "importance": self.importance,
            "metric_delta": self.metric_delta,
            "direction": self.direction,
            "status": self.status,
        }


@dataclass(frozen=True)
class ImportanceReport:
    method: str
    base_metric: float | None
    rows: list[ImportanceRow]
    status: str
    is_synthetic: bool = False
    caveat: str = EXPLANATORY_CAVEAT
    disclaimer: str = RESEARCH_DISCLAIMER
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "caveat": self.caveat,
            "research_only": True,
            "method": self.method,
            "status": self.status,
            "base_metric": self.base_metric,
            "is_synthetic": self.is_synthetic,
            "synthetic_warning": (
                "SYNTHETIC FIXTURE RESULTS — not real market evidence."
                if self.is_synthetic
                else None
            ),
            "rows": [r.to_dict() for r in self.rows],
            "warnings": self.warnings,
        }


def coefficient_importance(
    scaled_coefficients: dict[str, float], *, is_synthetic: bool = False
) -> ImportanceReport:
    """Normalized |standardized coefficient| importance (sums to 1 when possible)."""
    names = list(scaled_coefficients)
    abs_coefs = {name: abs(float(c)) for name, c in scaled_coefficients.items()}
    total = sum(abs_coefs.values())
    if total == 0.0:
        rows = [
            ImportanceRow(name, 0.0, None, "zero", STATUS_ALL_ZERO) for name in names
        ]
        return ImportanceReport(
            "coefficient", None, rows, STATUS_ALL_ZERO, is_synthetic,
            warnings=["all coefficients are zero"],
        )
    rows = [
        ImportanceRow(
            feature=name,
            importance=abs_coefs[name] / total,
            metric_delta=None,
            direction="nonzero" if abs_coefs[name] > 0 else "zero",
            status=STATUS_OK,
        )
        for name in names
    ]
    rows.sort(key=lambda r: (-r.importance, r.feature))
    return ImportanceReport("coefficient", None, rows, STATUS_OK, is_synthetic)


def _per_date_rank_ic(
    predictions: Sequence[float],
    decision_dates: Sequence[date],
    forward_returns: Sequence[float | None],
) -> float | None:
    by_date: dict[date, list[int]] = {}
    for i, d in enumerate(decision_dates):
        if forward_returns[i] is not None:
            by_date.setdefault(d, []).append(i)
    ics: list[float] = []
    for indices in by_date.values():
        if len(indices) < 2:
            continue
        ic = spearman(
            [predictions[i] for i in indices],
            [float(forward_returns[i]) for i in indices],  # type: ignore[arg-type]
        )
        if ic is not None:
            ics.append(ic)
    return sum(ics) / len(ics) if ics else None


def permutation_importance(
    model: Any,
    X: Sequence[Sequence[float | None]],
    feature_names: Sequence[str],
    decision_dates: Sequence[date],
    forward_returns: Sequence[float | None],
    *,
    seed: int = 0,
    is_synthetic: bool = False,
) -> ImportanceReport:
    """Rank-IC degradation when each feature is permuted (deterministic seed)."""
    if not list(X):
        return ImportanceReport(
            "permutation", None, [], STATUS_METRIC_UNAVAILABLE, is_synthetic,
            warnings=["no observations to permute"],
        )
    base_predictions = model.predict(X)
    base_ic = _per_date_rank_ic(base_predictions, decision_dates, forward_returns)
    if base_ic is None:
        return ImportanceReport(
            "permutation", None, [], STATUS_METRIC_UNAVAILABLE, is_synthetic,
            warnings=["base Rank IC unavailable (need >=2 names per date with labels)"],
        )

    matrix = [list(row) for row in X]
    n = len(matrix)
    rng = np.random.RandomState(seed)
    rows: list[ImportanceRow] = []
    for j, name in enumerate(feature_names):
        permuted = [list(r) for r in matrix]
        order = rng.permutation(n)
        column = [matrix[i][j] for i in range(n)]
        for dst, src in enumerate(order):
            permuted[dst][j] = column[src]
        permuted_ic = _per_date_rank_ic(
            model.predict(permuted), decision_dates, forward_returns
        )
        if permuted_ic is None:
            rows.append(ImportanceRow(name, 0.0, None, "unknown", STATUS_METRIC_UNAVAILABLE))
            continue
        degradation = base_ic - permuted_ic  # higher => more informative
        rows.append(
            ImportanceRow(
                feature=name,
                importance=degradation,
                metric_delta=permuted_ic - base_ic,
                direction="informative" if degradation > 0 else "uninformative_or_noise",
                status=STATUS_OK,
            )
        )
    rows.sort(key=lambda r: (-r.importance, r.feature))
    return ImportanceReport("permutation", base_ic, rows, STATUS_OK, is_synthetic)


__all__ = [
    "EXPLANATORY_CAVEAT",
    "STATUS_ALL_ZERO",
    "STATUS_METRIC_UNAVAILABLE",
    "STATUS_OK",
    "ImportanceReport",
    "ImportanceRow",
    "coefficient_importance",
    "permutation_importance",
]

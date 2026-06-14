"""Tests for feature importance. Deterministic, offline."""

from __future__ import annotations

from datetime import date

import numpy as np

from jp_stock_analysis.modeling.feature_importance import (
    STATUS_ALL_ZERO,
    STATUS_METRIC_UNAVAILABLE,
    STATUS_OK,
    coefficient_importance,
    permutation_importance,
)
from jp_stock_analysis.modeling.linear_models import ElasticNetRanker

FEATURES = ["f0", "f1", "f2", "f3"]


def test_coefficient_importance_sums_to_one():
    report = coefficient_importance({"f0": 2.0, "f1": -1.0, "f2": 0.0})
    assert report.status == STATUS_OK
    assert round(sum(r.importance for r in report.rows), 6) == 1.0
    assert report.rows[0].feature == "f0"  # largest |coef| first


def test_all_zero_coefficients_status():
    report = coefficient_importance({"f0": 0.0, "f1": 0.0})
    assert report.status == STATUS_ALL_ZERO
    assert all(r.importance == 0.0 for r in report.rows)


def test_sparse_elastic_net_produces_sparse_importance():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    y = X @ np.array([3.0, -2.0, 0.0, 0.0]) + 0.01 * rng.randn(200)
    model = ElasticNetRanker(alpha=0.1, l1_ratio=1.0, max_iter=5000).fit(
        X.tolist(), y.tolist(), FEATURES
    )
    report = coefficient_importance(model.model_metadata["scaled_coefficients"])
    importance = {r.feature: r.importance for r in report.rows}
    assert importance["f2"] == 0.0 and importance["f3"] == 0.0  # sparse -> zero importance
    assert importance["f0"] > 0.0 and importance["f1"] > 0.0


def test_permutation_importance_identifies_informative_feature():
    rng = np.random.RandomState(0)
    n = 240
    X = rng.randn(n, 4)
    y = X @ np.array([3.0, -2.0, 0.0, 0.0]) + 0.01 * rng.randn(n)
    model = ElasticNetRanker(alpha=0.02, l1_ratio=0.5, max_iter=5000).fit(
        X.tolist(), y.tolist(), FEATURES
    )
    # spread across 3 decision dates so per-date Rank IC is defined
    dates = [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1)]
    decision_dates = [dates[i % 3] for i in range(n)]
    report = permutation_importance(
        model, X.tolist(), FEATURES, decision_dates, y.tolist(), seed=0
    )
    assert report.status == STATUS_OK
    importance = {r.feature: r.importance for r in report.rows}
    # permuting an informative feature degrades Rank IC more than a noise feature
    assert importance["f0"] > importance["f2"]
    assert importance["f0"] > importance["f3"]
    assert report.rows[0].feature in ("f0", "f1")


def test_permutation_importance_metric_unavailable():
    rng = np.random.RandomState(1)
    X = rng.randn(4, 4)
    y = [1.0, 2.0, 3.0, 4.0]
    model = ElasticNetRanker(alpha=0.1, l1_ratio=0.5).fit(X.tolist(), y, FEATURES)
    # each row a distinct date -> no date has >=2 names -> Rank IC unavailable
    dates = [date(2025, 1, i + 1) for i in range(4)]
    report = permutation_importance(model, X.tolist(), FEATURES, dates, y, seed=0)
    assert report.status == STATUS_METRIC_UNAVAILABLE


def test_importance_is_explanatory_not_causal_caveat():
    report = coefficient_importance({"f0": 1.0})
    assert "not causal proof" in report.caveat

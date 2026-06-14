"""Tests for Ridge and the real coordinate-descent Elastic Net. Deterministic."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from jp_stock_analysis.modeling.linear_models import (
    STATUS_CONSTANT_TARGET,
    STATUS_FITTED,
    STATUS_INSUFFICIENT_ROWS,
    STATUS_NOT_CONVERGED,
    ElasticNetRanker,
    RidgeRanker,
)
from jp_stock_analysis.modeling.portfolio_metrics import (
    evaluate_portfolio,
    observations_from_scored,
)
from jp_stock_analysis.modeling.ranking_metrics import ScoredObservation, evaluate_horizon

FEATURES = ["f0", "f1", "f2", "f3"]


def _regression(n=200, seed=0, coefs=(3.0, -2.0, 0.0, 0.0), noise=0.01):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, len(coefs))
    y = X @ np.asarray(coefs) + noise * rng.randn(n)
    return X.tolist(), y.tolist()


# ----------------------------- Ridge ----------------------------------------- #
def test_ridge_fit_predict_deterministic():
    X, y = _regression()
    a = RidgeRanker(alpha=1.0).fit_predict(X, y, FEATURES)
    b = RidgeRanker(alpha=1.0).fit_predict(X, y, FEATURES)
    assert a == b


def test_ridge_recovers_signal_direction():
    X, y = _regression(coefs=(3.0, -2.0, 0.0, 0.0))
    model = RidgeRanker(alpha=1.0).fit(X, y, FEATURES)
    assert model.status == STATUS_FITTED
    coefs = model.coefficients
    assert coefs["f0"] > 0.5 and coefs["f1"] < -0.5
    assert abs(coefs["f2"]) < 0.5 and abs(coefs["f3"]) < 0.5


def test_ridge_handles_singular_matrix():
    X, y = _regression(coefs=(2.0, 0.0, 0.0, 0.0))
    # make f1 a perfect copy of f0 -> collinear / singular Gram
    for row in X:
        row[1] = row[0]
    model = RidgeRanker(alpha=1.0).fit(X, y, FEATURES)
    assert model.status == STATUS_FITTED  # ridge regularization absorbs it
    assert all(np.isfinite(v) for v in model.coefficients.values())


def test_ridge_imputes_missing_deterministically():
    X, y = _regression()
    X[0][2] = None
    X[5][2] = None
    model = RidgeRanker(alpha=1.0).fit(X, y, FEATURES)
    impute = model.model_metadata["imputation_values"]
    assert "f2" in impute and np.isfinite(impute["f2"])
    # predicting a row with a missing value reuses the stored median (no crash)
    out = model.predict([[None, 0.0, None, 0.0]])
    assert np.isfinite(out[0])


def test_ridge_constant_target_status():
    X, _y = _regression()
    model = RidgeRanker(alpha=1.0).fit(X, [5.0] * len(X), FEATURES)
    assert model.status == STATUS_CONSTANT_TARGET
    assert model.predict(X[:3]) == [5.0, 5.0, 5.0]  # stable constant prediction


def test_insufficient_rows_status():
    model = RidgeRanker().fit([[1.0, 2.0, 3.0, 4.0]], [1.0], FEATURES)
    assert model.status == STATUS_INSUFFICIENT_ROWS


# ----------------------------- Elastic Net ----------------------------------- #
def test_elastic_net_deterministic():
    X, y = _regression()
    a = ElasticNetRanker(alpha=0.05, l1_ratio=0.5).fit_predict(X, y, FEATURES)
    b = ElasticNetRanker(alpha=0.05, l1_ratio=0.5).fit_predict(X, y, FEATURES)
    assert a == b


def test_l1_ratio_zero_is_ridge_like_no_sparsity():
    X, y = _regression()
    model = ElasticNetRanker(alpha=0.1, l1_ratio=0.0, max_iter=5000).fit(X, y, FEATURES)
    assert model.sparsity == 0.0  # pure L2 -> no exact zeros
    assert model.coefficients["f0"] > 0.5 and model.coefficients["f1"] < -0.5


def test_l1_ratio_one_is_sparse_on_irrelevant_features():
    X, y = _regression(coefs=(3.0, -2.0, 0.0, 0.0))
    model = ElasticNetRanker(alpha=0.1, l1_ratio=1.0, max_iter=5000).fit(X, y, FEATURES)
    assert model.coefficients["f2"] == 0.0  # soft-threshold removes irrelevant
    assert model.coefficients["f3"] == 0.0
    assert "f0" in model.selected_features and "f1" in model.selected_features
    assert model.sparsity > 0.0


def test_intermediate_l1_ratio_combines_sparsity_and_shrinkage():
    X, y = _regression(coefs=(3.0, -2.0, 0.0, 0.0))
    en = ElasticNetRanker(alpha=0.1, l1_ratio=0.5, max_iter=5000).fit(X, y, FEATURES)
    assert en.coefficients["f2"] == 0.0  # irrelevant zeroed
    assert abs(en.coefficients["f0"]) > 0.0  # informative shrunk but kept


def test_correlated_features_handled_stably():
    X, y = _regression(coefs=(2.0, 2.0, 0.0, 0.0))
    for row in X:
        row[1] = row[0] + 0.001 * row[1]  # nearly collinear with f0
    model = ElasticNetRanker(alpha=0.05, l1_ratio=0.5, max_iter=5000).fit(X, y, FEATURES)
    assert model.status == STATUS_FITTED
    assert all(np.isfinite(v) for v in model.coefficients.values())


def test_missing_imputed_with_training_median_reused_at_predict():
    X, y = _regression()
    X[1][2] = None
    model = ElasticNetRanker(alpha=0.05, l1_ratio=0.5).fit(X, y, FEATURES)
    median = model.model_metadata["imputation_values"]["f2"]
    # predict a row whose f2 is missing -> uses the same training median
    explicit = model.predict([[0.0, 0.0, median, 0.0]])
    imputed = model.predict([[0.0, 0.0, None, 0.0]])
    assert explicit == imputed


def test_zero_variance_feature_does_not_crash():
    X, y = _regression()
    for row in X:
        row[2] = 7.0  # constant feature
    model = ElasticNetRanker(alpha=0.05, l1_ratio=0.5).fit(X, y, FEATURES)
    assert model.status in (STATUS_FITTED, STATUS_NOT_CONVERGED)
    assert model.coefficients["f2"] == 0.0
    assert any("zero-variance" in w for w in model.warnings)


def test_objective_history_is_non_increasing():
    X, y = _regression()
    model = ElasticNetRanker(alpha=0.1, l1_ratio=0.5, max_iter=2000).fit(X, y, FEATURES)
    history = model.model_metadata["objective_history"]
    assert len(history) >= 2
    assert all(history[i + 1] <= history[i] + 1e-9 for i in range(len(history) - 1))


def test_convergence_metadata_populated():
    X, y = _regression()
    meta = ElasticNetRanker(alpha=0.1, l1_ratio=0.5).fit(X, y, FEATURES).model_metadata
    for key in ("n_iter", "converged", "max_coefficient_change", "final_objective"):
        assert key in meta
    assert meta["converged"] is True


def test_non_convergence_returns_status_not_exception():
    X, y = _regression()
    model = ElasticNetRanker(alpha=0.05, l1_ratio=0.5, max_iter=1).fit(X, y, FEATURES)
    assert model.status == STATUS_NOT_CONVERGED
    assert model.model_metadata["converged"] is False
    assert any("did not converge" in w for w in model.warnings)


def test_elastic_net_recovers_signal_direction():
    X, y = _regression(coefs=(4.0, -3.0, 0.0, 0.0))
    model = ElasticNetRanker(alpha=0.02, l1_ratio=0.5, max_iter=5000).fit(X, y, FEATURES)
    assert model.coefficients["f0"] > 0 and model.coefficients["f1"] < 0


def test_predictions_integrate_with_rank_ic_and_portfolio():
    X, y = _regression(coefs=(3.0, -2.0, 0.0, 0.0))
    model = ElasticNetRanker(alpha=0.02, l1_ratio=0.5, max_iter=5000).fit(X, y, FEATURES)
    preds = model.predict(X)
    d = date(2025, 1, 1)
    scored = [
        ScoredObservation(d, f"t{i}", float(preds[i]), "a", {"forward_return_h5": y[i]})
        for i in range(len(y))
    ]
    ic = evaluate_horizon(scored, 5)
    assert ic.ic_mean is not None  # integrates cleanly
    port = evaluate_portfolio(observations_from_scored(scored, 5), horizon=5)
    # one decision date -> a Sharpe series is undefined; integration still runs
    assert port.status in ("ok", "no_valid_dates", "degenerate_series")
    assert port.per_date[0].status == "ok"  # the single date's spread computed


def test_no_sklearn_import():
    from pathlib import Path

    source = Path("src/jp_stock_analysis/modeling/linear_models.py").read_text(encoding="utf-8")
    import_lines = [
        ln for ln in source.splitlines() if ln.strip().startswith(("import ", "from "))
    ]
    assert not any("sklearn" in ln for ln in import_lines)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        RidgeRanker(alpha=-1.0)
    with pytest.raises(ValueError):
        ElasticNetRanker(alpha=-0.1)
    with pytest.raises(ValueError):
        ElasticNetRanker(l1_ratio=1.5)
    with pytest.raises(ValueError):
        ElasticNetRanker(max_iter=0)

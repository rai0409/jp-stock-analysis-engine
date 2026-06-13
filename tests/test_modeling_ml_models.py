"""Tests for the optional ML ranking adapters.

LightGBM / CatBoost are optional; these tests assert graceful skip behaviour
when they are absent and never require them to be installed.
"""

from __future__ import annotations

import pytest

from jp_stock_analysis.modeling.dataset import build_modeling_dataset
from jp_stock_analysis.modeling.fixtures import build_synthetic_bundle
from jp_stock_analysis.modeling.ml_models import (
    MODEL_BASELINE,
    MODEL_CATBOOST_RANKER,
    MODEL_LGBM_RANKER,
    MODEL_LGBM_REGRESSOR,
    STATUS_MISSING_DEPENDENCY,
    STATUS_TRAINED,
    available_backends,
    backend_available,
    train_ranking_model,
)


def _dataset():
    b = build_synthetic_bundle()
    return build_modeling_dataset(
        b.fundamentals,
        b.prices,
        b.metadata,
        b.narratives,
        decision_dates=b.decision_dates,
        horizons=b.horizons,
        bundle_disclosure_date=b.bundle_disclosure_date,
        is_synthetic=True,
    )


def test_baseline_model_always_trains():
    result = train_ranking_model(_dataset(), MODEL_BASELINE, horizon=20)
    assert result.status == STATUS_TRAINED
    assert result.scored
    assert "not personalized financial advice" in result.disclaimer


@pytest.mark.parametrize(
    "model_type", [MODEL_LGBM_RANKER, MODEL_LGBM_REGRESSOR, MODEL_CATBOOST_RANKER]
)
def test_optional_backend_skips_cleanly_when_absent(model_type):
    backend = "lightgbm" if "lightgbm" in model_type else "catboost"
    if backend_available(backend):
        pytest.skip(f"{backend} installed; missing-dependency path not applicable")
    result = train_ranking_model(_dataset(), model_type, horizon=20)
    assert result.status == STATUS_MISSING_DEPENDENCY
    assert result.missing_dependency == backend
    assert "not installed" in result.message
    assert result.scored == []  # nothing fabricated


def test_optional_backend_trains_when_installed():
    if not backend_available("lightgbm"):
        pytest.skip("lightgbm not installed")
    result = train_ranking_model(_dataset(), MODEL_LGBM_REGRESSOR, horizon=20)
    assert result.status == STATUS_TRAINED
    assert result.scored


def test_available_backends_reports_booleans():
    backends = available_backends()
    assert set(backends) == {"lightgbm", "catboost"}
    assert all(isinstance(v, bool) for v in backends.values())


def test_unknown_model_type_raises():
    with pytest.raises(ValueError):
        train_ranking_model(_dataset(), "not_a_model", horizon=5)

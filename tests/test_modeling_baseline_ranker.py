"""Tests for the transparent baseline factor ranker."""

from __future__ import annotations

from jp_stock_analysis.modeling.baseline_ranker import (
    MODEL_VERSION,
    score_baseline,
    scored_observations,
)
from jp_stock_analysis.modeling.dataset import build_modeling_dataset
from jp_stock_analysis.modeling.fixtures import build_synthetic_bundle


def _synthetic_dataset(**overrides):
    bundle = build_synthetic_bundle()
    params = dict(
        decision_dates=bundle.decision_dates,
        horizons=bundle.horizons,
        bundle_disclosure_date=bundle.bundle_disclosure_date,
        is_synthetic=True,
    )
    params.update(overrides)
    return build_modeling_dataset(
        bundle.fundamentals, bundle.prices, bundle.metadata, bundle.narratives, **params
    )


def test_scores_are_deterministic():
    ds = _synthetic_dataset()
    first = [s.to_dict() for s in score_baseline(ds)]
    second = [s.to_dict() for s in score_baseline(ds)]
    assert first == second


def test_ranks_are_dense_and_one_based_per_date():
    ds = _synthetic_dataset()
    scores = score_baseline(ds)
    by_date: dict = {}
    for s in scores:
        by_date.setdefault(s.decision_date, []).append(s)
    for group in by_date.values():
        ranks = sorted(s.factor_rank for s in group if s.factor_rank is not None)
        assert ranks == list(range(1, len(ranks) + 1))  # 1..n, no gaps


def test_score_carries_model_version_and_no_signal_fields():
    ds = _synthetic_dataset()
    score = score_baseline(ds)[0]
    payload = score.to_dict()
    assert payload["model_version"] == MODEL_VERSION
    # analysis_only: no buy/sell field anywhere
    assert "signal" not in payload
    assert "label" not in payload


def test_higher_factor_score_outranks_lower():
    ds = _synthetic_dataset()
    scores = [s for s in score_baseline(ds) if s.factor_score is not None]
    for a in scores:
        for b in scores:
            if a.decision_date == b.decision_date and a.factor_score > b.factor_score:
                assert a.factor_rank < b.factor_rank


def test_missing_feature_count_propagates():
    ds = _synthetic_dataset()
    scores = score_baseline(ds)
    # the deliberately-missing-fields ticker has a positive missing count somewhere
    assert any(s.missing_feature_count > 0 for s in scores)


def test_scored_observations_join_labels_for_evaluation():
    ds = _synthetic_dataset()
    scores = score_baseline(ds)
    scored = scored_observations(ds, scores)
    assert len(scored) == len(scores)
    assert all("forward_return_h5" in s.labels for s in scored)


def test_custom_group_weights_change_scores():
    ds = _synthetic_dataset()
    default = {s.ticker + s.decision_date.isoformat(): s.factor_score for s in score_baseline(ds)}
    quality_only = score_baseline(
        ds,
        group_weights={
            "value": 0.0,
            "quality": 1.0,
            "growth": 0.0,
            "momentum": 0.0,
            "risk": 0.0,
            "disclosure": 0.0,
        },
    )
    weighted = {s.ticker + s.decision_date.isoformat(): s.factor_score for s in quality_only}
    assert default != weighted

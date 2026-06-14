"""Tests for ensemble / blending. Deterministic, offline."""

from __future__ import annotations

from datetime import date

from jp_stock_analysis.modeling.ensemble import (
    METHOD_RANK_AVERAGE,
    STATUS_MISSING_COLUMN,
    STATUS_OK,
    rank_average_ensemble,
    weighted_blend,
)
from jp_stock_analysis.modeling.ranking_metrics import ScoredObservation, evaluate_horizon

DATES = [date(2025, 1, 1), date(2025, 2, 1)]


def _model(scores_by_date, sector="a"):
    obs = []
    for d in DATES:
        for i, s in enumerate(scores_by_date):
            obs.append(
                ScoredObservation(d, f"t{i}", float(s), sector, {"forward_return_h5": float(i)})
            )
    return obs


def test_rank_average_of_identical_predictions_preserves_order():
    a = _model([1.0, 2.0, 3.0, 4.0])
    result = rank_average_ensemble({"m1": a, "m2": list(a)})
    # identical models are (correctly) flagged near-identical; order is preserved
    assert result.status in (STATUS_OK, "near_identical_models")
    # identical inputs -> ensemble ranks identical to either input's order
    by_ticker = {(o.ticker, o.decision_date): o.score for o in result.scored}
    for d in DATES:
        order = sorted(
            [t for (t, dd) in by_ticker if dd == d],
            key=lambda t: by_ticker[(t, d)],
        )
        assert order == ["t0", "t1", "t2", "t3"]


def test_weighted_blend_deterministic():
    a = _model([1.0, 2.0, 3.0, 4.0])
    b = _model([4.0, 3.0, 2.0, 1.0])
    r1 = weighted_blend({"a": a, "b": b}, {"a": 0.7, "b": 0.3})
    r2 = weighted_blend({"a": a, "b": b}, {"a": 0.7, "b": 0.3})
    assert [o.score for o in r1.scored] == [o.score for o in r2.scored]
    assert r1.status == STATUS_OK


def test_missing_prediction_column_returns_status():
    a = _model([1.0, 2.0, 3.0, 4.0])
    result = weighted_blend({"a": a}, {"a": 0.5, "missing": 0.5})
    assert result.status == STATUS_MISSING_COLUMN


def test_pairwise_correlation_detects_identical_vs_diverse():
    a = _model([1.0, 2.0, 3.0, 4.0])
    identical = rank_average_ensemble({"a": a, "b": list(a)})
    assert identical.pairwise_correlations["a|b"] == 1.0
    diverse = rank_average_ensemble({"a": a, "b": _model([4.0, 3.0, 2.0, 1.0])})
    assert diverse.pairwise_correlations["a|b"] < 0.0
    assert diverse.diversity_score > identical.diversity_score


def test_near_identical_models_flagged():
    a = _model([1.0, 2.0, 3.0, 4.0])
    result = rank_average_ensemble({"a": a, "b": list(a)})
    assert result.status == "near_identical_models"


def test_ensemble_output_integrates_with_rank_ic():
    a = _model([1.0, 2.0, 3.0, 4.0])
    b = _model([1.5, 2.5, 2.4, 4.5])
    result = rank_average_ensemble({"a": a, "b": b})
    assert result.method == METHOD_RANK_AVERAGE
    ic = evaluate_horizon(result.scored, 5)
    assert ic.coverage_count == len(result.scored)


def test_to_dict_is_research_only_and_labels_synthetic():
    a = _model([1.0, 2.0, 3.0, 4.0])
    result = rank_average_ensemble({"a": a, "b": _model([2, 1, 4, 3])}, is_synthetic=True)
    payload = result.to_dict()
    assert payload["research_only"] is True
    assert "not real market evidence" in payload["synthetic_warning"]

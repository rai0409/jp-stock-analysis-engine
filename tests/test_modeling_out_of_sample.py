"""Regression tests for the finance-correctness fixes in the modeling layer.

- point-in-time fundamentals (per-statement disclosure dates)
- unadjusted price x reported shares for valuation
- purged walk-forward out-of-sample scoring (no in-sample evaluation)
- cross-sectional rank transforms
- IC significance (Newey-West) and overlap-aware equity compounding

Synthetic, deterministic, offline. Tickers are fixed placeholders.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from jp_stock_analysis.modeling.dataset import (
    EXCLUDE_DISCLOSURE_AFTER_DECISION,
    build_modeling_dataset,
)
from jp_stock_analysis.modeling.factors import compute_factors
from jp_stock_analysis.modeling.fixtures import build_synthetic_bundle
from jp_stock_analysis.modeling.linear_models import RidgeRanker
from jp_stock_analysis.modeling.ml_models import (
    _BACKEND_BY_MODEL,
    MODEL_BASELINE,
    MODEL_CATBOOST_RANKER,
    MODEL_CATBOOST_REGRESSOR,
    MODEL_LGBM_RANKER,
    MODEL_LGBM_REGRESSOR,
    STATUS_INSUFFICIENT_DATA,
    STATUS_TRAINED,
    _relevance_labels,
    backend_available,
    train_ranking_model,
)
from jp_stock_analysis.modeling.out_of_sample import (
    STATUS_NO_FOLDS,
    conservative_calendar_label_end,
    cross_sectional_rank_matrix,
    cross_sectional_target,
    linear_walk_forward_oos,
    walk_forward_oos,
)
from jp_stock_analysis.modeling.portfolio_metrics import (
    DateSpread,
    non_overlapping_spreads,
)
from jp_stock_analysis.modeling.ranking_metrics import (
    MIN_DATES_FOR_TSTAT,
    evaluate_horizon,
    newey_west_tstat,
    overlap_lag,
)
from jp_stock_analysis.modeling.report import _model_comparison
from jp_stock_analysis.schemas import FinancialStatement, PriceBar


def _bars(ticker: str, start: date, n: int, price: float = 100.0) -> list[PriceBar]:
    return [
        PriceBar(ticker=ticker, date=start + timedelta(days=i), close=price + i)
        for i in range(n)
    ]


def _synthetic_dataset():
    b = build_synthetic_bundle()
    return b, build_modeling_dataset(
        b.fundamentals,
        b.prices,
        b.metadata,
        b.narratives,
        decision_dates=b.decision_dates,
        horizons=b.horizons,
        bundle_disclosure_date=b.bundle_disclosure_date,
        is_synthetic=True,
    )


# --------------------------------------------------------------------------- #
# point-in-time fundamentals
# --------------------------------------------------------------------------- #
def test_point_in_time_uses_statement_public_at_each_decision_date():
    fy2023 = FinancialStatement(
        ticker="0001", fiscal_year=2023, accounting_basis="consolidated",
        revenue=100.0, net_income=10.0, equity=50.0, total_assets=100.0,
        disclosure_date=date(2024, 5, 10),
    )
    fy2024 = FinancialStatement(
        ticker="0001", fiscal_year=2024, accounting_basis="consolidated",
        revenue=200.0, net_income=30.0, equity=60.0, total_assets=120.0,
        disclosure_date=date(2025, 5, 10),
    )
    prices = {"0001": _bars("0001", date(2024, 1, 1), 600)}
    ds = build_modeling_dataset(
        {"0001": [fy2023, fy2024]},
        prices,
        decision_dates=[date(2024, 4, 1), date(2024, 12, 1), date(2025, 6, 1)],
        horizons=[5],
    )
    by_date = {o.decision_date: o for o in ds.observations}
    # before any disclosure -> excluded, not silently using a future statement
    assert by_date[date(2024, 4, 1)].exclusion_reason == EXCLUDE_DISCLOSURE_AFTER_DECISION
    # FY2024 not yet public -> FY2023 figures (ROE 10/50 = 20%)
    assert by_date[date(2024, 12, 1)].features["roe"] == pytest.approx(20.0)
    assert by_date[date(2024, 12, 1)].features["revenue_growth_yoy"] is None
    # after FY2024 disclosure -> FY2024 figures and YoY vs FY2023
    assert by_date[date(2025, 6, 1)].features["roe"] == pytest.approx(50.0)
    assert by_date[date(2025, 6, 1)].features["revenue_growth_yoy"] == pytest.approx(100.0)


def test_valuation_uses_unadjusted_close_with_reported_shares():
    statement = FinancialStatement(
        ticker="0002", fiscal_year=2024, net_income=100.0, shares_outstanding=10.0
    )
    # 2-for-1 split history: adjusted close is half the traded close
    bars = [
        PriceBar(ticker="0002", date=date(2025, 1, 1), close=200.0, adjusted_close=100.0),
        PriceBar(ticker="0002", date=date(2025, 1, 2), close=200.0, adjusted_close=100.0),
    ]
    result = compute_factors(statement, None, bars)
    # market cap = 200 * 10 = 2000 -> earnings yield 5%
    assert result.features["earnings_yield"] == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# out-of-sample scoring
# --------------------------------------------------------------------------- #
def test_oos_never_scores_a_date_it_trained_on():
    _b, ds = _synthetic_dataset()
    seen: list[tuple[set[date], set[date]]] = []

    def fit_predict(train, test, _label_key):
        seen.append(({o.decision_date for o in train}, {o.decision_date for o in test}))
        return [0.0] * len(test)

    result = walk_forward_oos(ds, 5, fit_predict, min_train_rows=1)
    assert result.ok
    first_date = min(ds.decision_dates)
    assert all(s.decision_date != first_date for s in result.scored)
    for train_dates, test_dates in seen:
        assert max(train_dates) < min(test_dates)


def test_purge_drops_training_rows_whose_label_overlaps_test():
    b, ds = _synthetic_dataset()
    # decision dates are 30 rows apart; a 60-row label always overlaps the next
    result = walk_forward_oos(
        ds, 60, lambda tr, te, k: [0.0] * len(te), prices=b.prices, min_train_rows=1
    )
    assert not result.scored
    assert all(f.train_rows == 0 for f in result.folds)
    assert sum(f.purged_rows for f in result.folds) > 0


def test_single_decision_date_has_no_oos_folds():
    b = build_synthetic_bundle()
    ds = build_modeling_dataset(
        b.fundamentals, b.prices, b.metadata, b.narratives,
        decision_dates=b.decision_dates[:1], horizons=(5,),
        bundle_disclosure_date=b.bundle_disclosure_date, is_synthetic=True,
    )
    result = walk_forward_oos(ds, 5, lambda tr, te, k: [0.0] * len(te))
    assert result.status == STATUS_NO_FOLDS
    assert result.scored == []


def test_linear_oos_scores_only_held_out_dates():
    b, ds = _synthetic_dataset()
    run = linear_walk_forward_oos(ds, 5, lambda: RidgeRanker(alpha=1.0), prices=b.prices)
    assert run.result.ok
    assert {s.decision_date for s in run.result.scored} == set(ds.decision_dates[1:])
    assert run.last_model is not None


def test_calendar_label_end_is_conservative():
    assert conservative_calendar_label_end(date(2025, 1, 6), 20) > date(2025, 2, 3)


# --------------------------------------------------------------------------- #
# cross-sectional transforms
# --------------------------------------------------------------------------- #
def test_rank_transforms_are_within_date_and_centered():
    _b, ds = _synthetic_dataset()
    rows = sorted(ds.included(), key=lambda o: (o.decision_date, o.ticker))
    labelled = [o for o in rows if o.labels.get("forward_return_h5") is not None]
    target = cross_sectional_target(labelled, "forward_return_h5")
    for d in ds.decision_dates:
        values = [t for t, o in zip(target, labelled, strict=True) if o.decision_date == d]
        assert min(values) == pytest.approx(-0.5)
        assert max(values) == pytest.approx(0.5)
        assert sum(values) == pytest.approx(0.0, abs=1e-9)
    matrix = cross_sectional_rank_matrix(rows)
    flat = [v for row in matrix for v in row if v is not None]
    assert all(-0.5 <= v <= 0.5 for v in flat)


# --------------------------------------------------------------------------- #
# significance and compounding
# --------------------------------------------------------------------------- #
def test_newey_west_tstat_shrinks_with_positive_autocorrelation():
    values = [0.05, 0.06, 0.04, 0.05, -0.01, -0.02, 0.0, 0.07, 0.08, 0.06]
    plain = newey_west_tstat(values, 0)
    hac = newey_west_tstat(values, 3)
    assert plain is not None and hac is not None
    assert abs(hac) < abs(plain)


def test_overlap_lag_counts_overlapping_decision_dates():
    dates = [date(2025, 1, 1) + timedelta(days=7 * i) for i in range(6)]
    assert overlap_lag(dates, 20) == 3  # 28-day span: 7, 14, 21 days later overlap
    assert overlap_lag(dates, 5) == 0


def test_equity_curve_compounds_only_non_overlapping_periods():
    spreads = [
        DateSpread(
            decision_date=date(2025, 1, 1) + timedelta(days=7 * i),
            status="ok",
            spread_return=1.0,
            long_leg_return=None,
            short_leg_return=None,
            count_long=2,
            count_short=2,
            universe_count=10,
        )
        for i in range(8)
    ]
    kept = non_overlapping_spreads(spreads, 20)
    gaps = [(b.decision_date - a.decision_date).days for a, b in zip(kept, kept[1:], strict=False)]
    assert all(g >= 28 for g in gaps)
    assert len(kept) < len(spreads)


# --------------------------------------------------------------------------- #
# tree models (LightGBM / CatBoost) — skipped only if the backend is absent
# --------------------------------------------------------------------------- #
TREE_MODELS = [
    pytest.param(m, marks=pytest.mark.skipif(
        not backend_available(_BACKEND_BY_MODEL[m]), reason=f"{m} backend not installed"
    ))
    for m in (
        MODEL_LGBM_RANKER, MODEL_LGBM_REGRESSOR,
        MODEL_CATBOOST_RANKER, MODEL_CATBOOST_REGRESSOR,
    )
]


@pytest.mark.parametrize("model_type", TREE_MODELS)
def test_tree_model_scores_only_held_out_dates(model_type):
    b, ds = _synthetic_dataset()
    result = train_ranking_model(ds, model_type, horizon=5, prices=b.prices)
    assert result.status == STATUS_TRAINED
    assert result.validation == "walk_forward_purged_oos"
    assert {s.decision_date for s in result.scored} == set(ds.decision_dates[1:])
    assert [f["status"] for f in result.oos_folds] == ["ok", "ok"]


@pytest.mark.parametrize("model_type", TREE_MODELS)
def test_tree_model_is_deterministic(model_type):
    b, ds = _synthetic_dataset()
    first = train_ranking_model(ds, model_type, horizon=5, prices=b.prices)
    second = train_ranking_model(ds, model_type, horizon=5, prices=b.prices)
    assert [s.score for s in first.scored] == [s.score for s in second.scored]


@pytest.mark.parametrize("model_type", TREE_MODELS)
def test_tree_model_refuses_when_purge_leaves_no_training_rows(model_type):
    b, ds = _synthetic_dataset()
    result = train_ranking_model(ds, model_type, horizon=60, prices=b.prices)
    assert result.status == STATUS_INSUFFICIENT_DATA
    assert result.scored == []  # nothing in-sample is substituted
    assert all(f["train_rows"] == 0 and f["purged_rows"] > 0 for f in result.oos_folds)


def test_ranker_relevance_drops_dates_too_small_to_grade():
    _b, ds = _synthetic_dataset()
    rows = [o for o in ds.included() if o.labels.get("forward_return_h5") is not None]
    first_date = min(o.decision_date for o in rows)
    small = [o for o in rows if o.decision_date == first_date][:3]
    big = [o for o in rows if o.decision_date != first_date]
    kept, relevance = _relevance_labels(small + big, "forward_return_h5", 5)
    assert first_date not in {o.decision_date for o in kept}
    assert len(kept) == len(relevance) == len(big)


def test_lightgbm_backend_requires_scikit_learn(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "sklearn" else real(name, *a, **k),
    )
    assert backend_available("lightgbm") is False


def test_tstat_withheld_for_small_samples():
    _b, ds = _synthetic_dataset()
    scored = train_ranking_model(ds, MODEL_BASELINE, horizon=5).scored
    horizon = evaluate_horizon(scored, 5)
    assert horizon.n_dates < MIN_DATES_FOR_TSTAT
    assert horizon.ic_tstat is None
    assert horizon.ic_tstat_newey_west is None


def test_model_comparison_uses_common_dates():
    b, ds = _synthetic_dataset()
    entries = _model_comparison(ds, [5], 5, True, b.prices)
    by_model = {e.model_type: e for e in entries}
    baseline = by_model[MODEL_BASELINE]
    trained_oos = [
        e for e in entries if e.model_type != MODEL_BASELINE and e.status == STATUS_TRAINED
    ]
    if not trained_oos:
        assert baseline.common_date_count == len(ds.decision_dates)
        return
    # baseline scores every date; OOS models skip the first -> common = the rest
    assert baseline.common_date_count == len(ds.decision_dates) - 1
    assert all(e.common_date_count == baseline.common_date_count for e in trained_oos)

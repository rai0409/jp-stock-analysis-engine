"""Tests for the extended trading feature set and tradable dataset options.

Hand-checkable synthetic bars; point-in-time behaviour of guidance, margin and
earnings-calendar inputs; next-open labels; liquidity filter; the rich fixture
end-to-end with the tree models; and the CLI switches. Offline, deterministic.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from jp_stock_analysis.cli import main
from jp_stock_analysis.modeling import ml_models
from jp_stock_analysis.modeling.dataset import (
    ENTRY_NEXT_OPEN,
    EXCLUDE_ILLIQUID,
    FEATURE_SET_EXTENDED,
    build_modeling_dataset,
    forward_return,
)
from jp_stock_analysis.modeling.factors import ALL_FACTORS
from jp_stock_analysis.modeling.fixtures import build_rich_synthetic_bundle
from jp_stock_analysis.modeling.ml_models import MODEL_TREE_ENSEMBLE, train_primary_model
from jp_stock_analysis.modeling.trading_features import (
    TRADING_FEATURES,
    adjusted_ohlc,
    compute_trading_features,
)
from jp_stock_analysis.schemas import FinancialStatement, MarginBalance, PriceBar

START = date(2025, 1, 6)


def _bars(n: int, *, ticker: str = "0001", split_at: int | None = None) -> list[PriceBar]:
    """Linear uptrend with fixed candles; optional 2:1 split (raw halves)."""
    bars = []
    for i in range(n):
        adj_close = 100.0 + i
        raw_factor = 0.5 if (split_at is not None and i >= split_at) else 1.0
        # adjusted series is continuous; raw prices before the split are 2x adjusted
        scale = 1.0 if (split_at is None or i >= split_at) else 2.0
        raw_close = adj_close * scale
        bars.append(
            PriceBar(
                ticker=ticker,
                date=START + timedelta(days=i),
                open=(adj_close - 0.5) * scale,
                high=(adj_close + 1.0) * scale,
                low=(adj_close - 1.0) * scale,
                close=raw_close,
                adjusted_close=adj_close,
                volume=1000.0 / scale * (1.0 / raw_factor if raw_factor != 1.0 else 1.0),
            )
        )
    return bars


def test_feature_names_are_core_plus_trading():
    assert len(set(TRADING_FEATURES)) == len(TRADING_FEATURES)
    assert not set(TRADING_FEATURES) & set(ALL_FACTORS)


def test_candle_features_hand_checked():
    bars = _bars(30)
    f = compute_trading_features(
        statement=None, statement_public_date=None, earlier_disclosures=[],
        bars_as_of=bars, decision_date=bars[-1].date,
    )
    # last bar: O=128.5 H=130 L=128 C=129 -> range 2, body 0.5
    assert f["body_ratio"] == pytest.approx(0.25)
    assert f["upper_shadow_ratio"] == pytest.approx(0.5)
    assert f["lower_shadow_ratio"] == pytest.approx(0.25)
    assert f["ret_1d"] == pytest.approx((129 / 128 - 1) * 100)
    assert f["gap_1d"] == pytest.approx((128.5 / 128 - 1) * 100)
    assert f["intraday_ret_1d"] == pytest.approx((129 / 128.5 - 1) * 100)
    assert f["rsi_14"] == pytest.approx(100.0)  # monotonic uptrend: no losses
    assert f["ma_dev_5"] == pytest.approx((129 / 127 - 1) * 100)


def test_adjusted_ohlc_is_continuous_across_split():
    bars = _bars(40, split_at=20)
    rows = adjusted_ohlc(bars)
    jumps = [rows[i].close / rows[i - 1].close - 1 for i in range(1, len(rows))]
    assert max(abs(j) for j in jumps) < 0.02  # no artificial -50% split gap
    assert rows[10].open == pytest.approx(109.5)  # raw 219 scaled by 0.5


def test_valuation_uses_raw_price_and_guidance():
    bars = _bars(30)
    statement = FinancialStatement(
        ticker="0001", fiscal_year=2024, shares_outstanding=1000.0, total_assets=50_000.0,
        net_income=1_000.0, operating_cash_flow=1_500.0, capital_expenditure=500.0,
        gross_profit=10_000.0, forecast_net_income=2_580.0, forecast_dividends_per_share=2.58,
    )
    f = compute_trading_features(
        statement=statement, statement_public_date=bars[0].date, earlier_disclosures=[],
        bars_as_of=bars, decision_date=bars[-1].date,
    )
    mcap = 129.0 * 1000.0
    assert f["forecast_earnings_yield"] == pytest.approx(2_580 / mcap * 100)
    assert f["forecast_dividend_yield"] == pytest.approx(2.0)
    assert f["fcf_yield"] == pytest.approx(1_000 / mcap * 100)
    assert f["log_market_cap"] == pytest.approx(math.log(mcap))
    assert f["gross_profitability"] == pytest.approx(20.0)
    assert f["accruals"] == pytest.approx(-1.0)


def test_guidance_revision_and_surprise():
    bars = _bars(30)
    old = FinancialStatement(ticker="0001", fiscal_year=2024, forecast_fiscal_year=2025,
                             forecast_eps=100.0)
    new = FinancialStatement(ticker="0001", fiscal_year=2025, fiscal_period="1Q",
                             forecast_fiscal_year=2025, forecast_eps=110.0)
    f = compute_trading_features(
        statement=new, statement_public_date=bars[20].date, earlier_disclosures=[old],
        bars_as_of=bars, decision_date=bars[-1].date,
    )
    assert f["forecast_eps_revision"] == pytest.approx(10.0)
    assert f["days_since_disclosure"] == 9.0
    actual = FinancialStatement(ticker="0001", fiscal_year=2025, eps=88.0)
    f2 = compute_trading_features(
        statement=actual, statement_public_date=bars[25].date,
        earlier_disclosures=[old, new], bars_as_of=bars, decision_date=bars[-1].date,
    )
    assert f2["earnings_surprise"] == pytest.approx(-20.0)  # vs the latest guidance 110


def test_margin_and_calendar_are_point_in_time():
    bars = _bars(30)
    decision = bars[-1].date
    unpublished = MarginBalance(ticker="0001", date=decision - timedelta(days=1),
                                long_balance=9e9, short_balance=1.0)
    published = MarginBalance(ticker="0001", date=decision - timedelta(days=10),
                              long_balance=3000.0, short_balance=999.0,
                              published_date=decision - timedelta(days=7))
    f = compute_trading_features(
        statement=None, statement_public_date=None, earlier_disclosures=[], bars_as_of=bars,
        decision_date=decision, margin=[published, unpublished],
        earnings_dates=[decision + timedelta(days=10), decision + timedelta(days=100)],
    )
    assert f["margin_ratio_log"] == pytest.approx(math.log(3001 / 1000))
    assert f["days_to_next_earnings"] == 10.0
    far = compute_trading_features(
        statement=None, statement_public_date=None, earlier_disclosures=[], bars_as_of=bars,
        decision_date=decision, earnings_dates=[decision + timedelta(days=100)],
    )
    assert far["days_to_next_earnings"] is None  # beyond the known-schedule lead


def test_next_open_label():
    bars = _bars(30)
    decision = bars[9].date
    value = forward_return(bars, decision, 5, entry=ENTRY_NEXT_OPEN)
    # buy open of row 10 (109.5), sell open of row 15 (114.5)
    assert value == pytest.approx((114.5 / 109.5 - 1) * 100)
    with pytest.raises(ValueError):
        forward_return(bars, decision, 5, entry="whenever")


def _rich_dataset(**kwargs):
    b = build_rich_synthetic_bundle()
    ds = build_modeling_dataset(
        b.fundamentals, b.prices, b.metadata, decision_dates=b.decision_dates,
        horizons=b.horizons, is_synthetic=True, feature_set=FEATURE_SET_EXTENDED,
        entry=ENTRY_NEXT_OPEN, earnings_dates=b.earnings_dates, margin=b.margin, **kwargs,
    )
    return b, ds


def test_rich_extended_dataset_coverage():
    _b, ds = _rich_dataset()
    assert ds.feature_names == ALL_FACTORS + TRADING_FEATURES
    assert ds.entry == ENTRY_NEXT_OPEN
    coverage = ds.feature_coverage()
    covered = [name for name in TRADING_FEATURES if coverage[name] > 0]
    assert len(covered) >= len(TRADING_FEATURES) - 2  # all families populated


def test_liquidity_filter_excludes_untradable_names():
    _b, ds = _rich_dataset(min_traded_value=1e15)
    assert ds.exclusion_counts().get(EXCLUDE_ILLIQUID) == len(ds.observations)


@pytest.mark.skipif(
    not (ml_models.backend_available("lightgbm") and ml_models.backend_available("catboost")),
    reason="tree backends not installed",
)
def test_tree_ensemble_uses_extended_features():
    b, ds = _rich_dataset()
    primary = train_primary_model(ds, horizon=5, prices=b.prices)
    assert primary.model_label == MODEL_TREE_ENSEMBLE
    # next_open purge = h(5) + entry row + MAX_EXIT_DELAY_ROWS(5) = 11 rows > the
    # fixture's 10-row spacing, so the 2nd date has no purged training rows
    # either: the first two dates are training-only.
    assert {s.decision_date for s in primary.scored} == set(ds.decision_dates[2:])


def test_cli_extended_dataset(tmp_path):
    out = tmp_path / "ds"
    rc = main([
        "build-modeling-dataset", "--synthetic-rich", "--feature-set", "extended",
        "--entry", "next_open", "--horizons", "5,20", "--output-dir", str(out),
    ])
    assert rc == 0
    summary = json.loads((out / "modeling_dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["entry"] == "next_open"
    assert summary["feature_count"] == len(ALL_FACTORS) + len(TRADING_FEATURES)
    header = (out / "modeling_dataset.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "body_ratio" in header and "forecast_eps_revision" in header


# --------------------------------------------------------------------------- #
# round-5 fixes: unfillable sessions, label windows, refit cadence, caching
# --------------------------------------------------------------------------- #
def _locked(bar: PriceBar) -> PriceBar:
    return bar.model_copy(update={"high": bar.open, "low": bar.open, "close": bar.open})


def test_locked_entry_session_is_unfillable():
    bars = _bars(30)
    bars[10] = _locked(bars[10])  # session after the decision is limit-locked
    assert forward_return(bars, bars[9].date, 5, entry=ENTRY_NEXT_OPEN) is None


def test_locked_exit_rolls_to_next_tradable_open():
    bars = _bars(30)
    bars[15] = _locked(bars[15])  # scheduled exit session is locked
    value = forward_return(bars, bars[9].date, 5, entry=ENTRY_NEXT_OPEN)
    # exit rolls to row 16's open (115.5); entry is row 10's open (109.5)
    assert value == pytest.approx((115.5 / 109.5 - 1) * 100)


def test_label_end_index_adds_exit_delay_for_next_open():
    from jp_stock_analysis.modeling.dataset import MAX_EXIT_DELAY_ROWS
    from jp_stock_analysis.modeling.out_of_sample import label_end_index_for

    b, ds = _rich_dataset()
    obs = ds.included()[0]
    ends = label_end_index_for(ds, b.prices)
    dates = sorted(bar.date for bar in b.prices[obs.ticker])
    first_after = next(i for i, d in enumerate(dates) if d > obs.decision_date)
    assert ends.end(obs, 5) == dates[first_after + 5 + MAX_EXIT_DELAY_ROWS]


@pytest.mark.skipif(
    not ml_models.backend_available("lightgbm"), reason="lightgbm not installed"
)
def test_refit_every_groups_test_dates_per_fold():
    b, ds = _rich_dataset()
    monthly = ml_models.train_ranking_model(
        ds, "lightgbm_regressor", horizon=5, prices=b.prices, refit_every=3
    )
    assert monthly.status == ml_models.STATUS_TRAINED
    # 10 decision dates, first is training-only -> folds cover 3 dates each
    assert len(monthly.oos_folds) == 3
    assert all(f["test_start"] <= f["test_end"] for f in monthly.oos_folds)


def test_ranked_feature_cache_matches_direct_ranks():
    from jp_stock_analysis.modeling.out_of_sample import (
        cross_sectional_rank_matrix,
        ranked_matrix,
    )

    _b, ds = _rich_dataset()
    rows = sorted(ds.included(), key=lambda o: (o.decision_date, o.ticker))
    assert ranked_matrix(ds, rows) == cross_sectional_rank_matrix(rows, ds.feature_names)


def test_feature_window_matches_full_history():
    """Windowed (400-row) features equal features computed on the full history."""
    b, ds = _rich_dataset()
    obs = ds.included()[-1]
    full = compute_trading_features(
        statement=None, statement_public_date=None, earlier_disclosures=[],
        bars_as_of=[x for x in b.prices[obs.ticker] if x.date <= obs.decision_date],
        decision_date=obs.decision_date,
    )
    for name in ("ret_5d", "mom_12_1", "rsi_14", "atr_14_pct", "high_52w_ratio"):
        assert obs.features[name] == pytest.approx(full[name]), name


def test_gbdt_threads_never_exceed_physical_cores():
    import os

    physical = max(1, (os.cpu_count() or 2) // 2)
    assert ml_models.resolve_threads(-1) == physical
    assert ml_models.resolve_threads(10_000) == physical
    assert ml_models.resolve_threads(1) == 1

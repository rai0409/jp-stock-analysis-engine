"""Tests for JPX-style long-short portfolio metrics. Deterministic, offline."""

from __future__ import annotations

from datetime import date

import pytest

from jp_stock_analysis.modeling.portfolio_metrics import (
    STATUS_CONSTANT_PREDICTIONS,
    STATUS_INSUFFICIENT_NAMES,
    STATUS_NO_VALID_DATES,
    STATUS_OK,
    DateSpread,
    PortfolioObservation,
    compute_turnover,
    evaluate_portfolio,
    summarize_spread_series,
    write_portfolio_outputs,
)

DATES = [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1)]
SLOPES = {DATES[0]: 1.0, DATES[1]: 1.5, DATES[2]: 0.8}


def _obs(d, ticker, score, ret, **kw):
    return PortfolioObservation(d, ticker, score, ret, kw.get("sector"), kw.get("weight"))


def _monotone(reverse: bool = False):
    obs = []
    for d in DATES:
        for i in range(6):
            ret = (5 - i if reverse else i) * SLOPES[d]
            obs.append(_obs(d, f"t{i}", float(i), float(ret)))
    return obs


def test_monotone_positive_signal_positive_spread_and_sharpe():
    report = evaluate_portfolio(
        _monotone(), horizon=5, top_quantile=0.34, bottom_quantile=0.34
    )
    assert report.status == STATUS_OK
    assert report.series.mean_spread > 0
    assert report.series.sharpe_like > 0
    assert all(s.spread_return > 0 for s in report.per_date)


def test_reversed_signal_negative_spread():
    report = evaluate_portfolio(
        _monotone(reverse=True), horizon=5, top_quantile=0.34, bottom_quantile=0.34
    )
    assert report.series.mean_spread < 0
    assert all(s.spread_return < 0 for s in report.per_date)


def test_constant_prediction_is_degenerate():
    obs = [_obs(DATES[0], f"t{i}", 1.0, float(i)) for i in range(6)]
    report = evaluate_portfolio(obs, horizon=5, top_quantile=0.34, bottom_quantile=0.34)
    assert report.per_date[0].status == STATUS_CONSTANT_PREDICTIONS
    assert report.series.sharpe_like is None
    assert report.status == STATUS_NO_VALID_DATES


def test_too_few_names_returns_status():
    obs = [_obs(DATES[0], "t0", 1.0, 2.0)]
    report = evaluate_portfolio(obs, horizon=5, top_quantile=0.34, bottom_quantile=0.34)
    assert report.per_date[0].status == STATUS_INSUFFICIENT_NAMES
    assert report.per_date[0].spread_return is None


def test_rank_weighted_and_equal_weight_are_deterministic_and_differ():
    equal_a = evaluate_portfolio(_monotone(), horizon=5, top_quantile=0.5, bottom_quantile=0.5)
    equal_b = evaluate_portfolio(_monotone(), horizon=5, top_quantile=0.5, bottom_quantile=0.5)
    ranked = evaluate_portfolio(
        _monotone(), horizon=5, top_quantile=0.5, bottom_quantile=0.5, rank_weighted=True
    )
    assert equal_a.to_dict() == equal_b.to_dict()  # deterministic
    assert ranked.series.mean_spread != equal_a.series.mean_spread  # weighting matters


def test_turnover_within_bounds():
    report = evaluate_portfolio(_monotone(), horizon=5, top_quantile=0.34, bottom_quantile=0.34)
    turnover = compute_turnover(report.per_date)
    for row in turnover.per_date:
        assert 0.0 <= row["total_turnover"] <= 2.0
    if turnover.average_turnover is not None:
        assert 0.0 <= turnover.average_turnover <= 2.0


def test_transaction_cost_reduces_net_when_turnover_positive():
    # membership flips between dates -> turnover > 0
    obs = []
    for j, d in enumerate(DATES):
        for i in range(6):
            score = float(i if j % 2 == 0 else 5 - i)
            obs.append(_obs(d, f"t{i}", score, float(i) + j))
    gross = evaluate_portfolio(obs, horizon=5, top_quantile=0.34, bottom_quantile=0.34)
    net = evaluate_portfolio(
        obs, horizon=5, top_quantile=0.34, bottom_quantile=0.34, transaction_cost_bps=50.0
    )
    assert net.transaction_cost is not None
    assert net.turnover.average_turnover and net.turnover.average_turnover > 0
    assert net.transaction_cost.net_mean_spread < gross.series.mean_spread


def test_equity_curve_and_max_drawdown_are_correct():
    spreads = [
        DateSpread(DATES[0], STATUS_OK, 10.0, 10.0, 0.0, 1, 1, 4),
        DateSpread(DATES[1], STATUS_OK, -50.0, -50.0, 0.0, 1, 1, 4),
    ]
    summary = summarize_spread_series(spreads)
    # equity: 1*1.10 = 1.10 ; 1.10*0.50 = 0.55 ; drawdown = 0.55/1.10 - 1 = -50%
    assert round(summary.equity_curve[0]["equity"], 6) == 1.10
    assert round(summary.equity_curve[1]["equity"], 6) == 0.55
    assert round(summary.max_drawdown, 6) == -50.0
    assert summary.best_period["spread_return"] == 10.0
    assert summary.worst_period["spread_return"] == -50.0


def test_missing_returns_handled_deterministically():
    obs = [
        _obs(DATES[0], "t0", 5.0, None),  # missing return -> excluded
        _obs(DATES[0], "t1", 4.0, 4.0),
        _obs(DATES[0], "t2", 3.0, 3.0),
        _obs(DATES[0], "t3", 2.0, None),  # missing
        _obs(DATES[0], "t4", 1.0, 1.0),
    ]
    report = evaluate_portfolio(obs, horizon=5, top_quantile=0.34, bottom_quantile=0.34)
    assert report.per_date[0].universe_count == 3  # only the 3 with returns


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        evaluate_portfolio(_monotone(), horizon=5, mode="bogus")
    with pytest.raises(ValueError):
        evaluate_portfolio(_monotone(), horizon=5, transaction_cost_bps=-1.0)


def test_outputs_written_and_synthetic_labelled(tmp_path):
    report = evaluate_portfolio(
        _monotone(), horizon=5, top_quantile=0.34, bottom_quantile=0.34, is_synthetic=True
    )
    paths = write_portfolio_outputs(report, tmp_path / "out")
    assert paths["json_path"].exists()
    assert paths["csv_path"].exists()
    assert "SYNTHETIC" in paths["markdown_path"].read_text(encoding="utf-8")

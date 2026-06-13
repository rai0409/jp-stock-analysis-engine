"""Tests for the modeling-dataset builder and its no-look-ahead guardrails."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from jp_stock_analysis.modeling.dataset import (
    EXCLUDE_DISCLOSURE_AFTER_DECISION,
    EXCLUDE_NON_CONSOLIDATED,
    build_modeling_dataset,
    forward_return,
    write_dataset_outputs,
)
from jp_stock_analysis.schemas import CompanyMetadata, FinancialStatement, PriceBar


def _bars(ticker: str, start: date, n: int, step: float = 1.0) -> list[PriceBar]:
    bars = []
    day = start
    price = 100.0
    for _ in range(n):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        bars.append(PriceBar(ticker=ticker, date=day, close=price, adjusted_close=price))
        price += step
        day += timedelta(days=1)
    return bars


def _statement(ticker: str, basis: str = "consolidated") -> FinancialStatement:
    return FinancialStatement(
        ticker=ticker,
        fiscal_year=2025,
        accounting_basis=basis,
        revenue=1000.0,
        operating_income=150.0,
        net_income=100.0,
        equity=1000.0,
        total_assets=2000.0,
        shares_outstanding=10.0,
    )


def test_forward_return_uses_only_rows_after_decision_date():
    start = date(2025, 1, 1)
    bars = _bars("1301", start, 10, step=0.0)  # flat -> 0% return
    # set an upward step after decision
    bars = _bars("1301", start, 10, step=10.0)
    decision = bars[2].date  # base = bars[3], target = bars[3+2]
    ret = forward_return(bars, decision, 2)
    assert ret is not None
    base = bars[3].close
    target = bars[5].close
    assert round(ret, 6) == round((target / base - 1) * 100, 6)


def test_forward_return_none_when_too_few_rows():
    bars = _bars("1301", date(2025, 1, 1), 4)
    assert forward_return(bars, bars[1].date, 5) is None


def test_disclosure_after_decision_is_excluded():
    decision = date(2025, 3, 1)
    bars = _bars("1301", date(2025, 1, 1), 80)
    ds = build_modeling_dataset(
        {"1301": [_statement("1301")]},
        {"1301": bars},
        {},
        decision_dates=[decision],
        horizons=[5],
        bundle_disclosure_date=date(2025, 6, 1),  # AFTER the decision date
    )
    obs = ds.observations[0]
    assert obs.included is False
    assert obs.exclusion_reason == EXCLUDE_DISCLOSURE_AFTER_DECISION


def test_disclosure_on_or_before_decision_is_included():
    decision = date(2025, 3, 1)
    bars = _bars("1301", date(2025, 1, 1), 80)
    ds = build_modeling_dataset(
        {"1301": [_statement("1301")]},
        {"1301": bars},
        {},
        decision_dates=[decision],
        horizons=[5],
        bundle_disclosure_date=date(2025, 2, 28),
    )
    assert ds.observations[0].included is True


def test_non_consolidated_excluded_by_default_and_kept_with_flag():
    decision = date(2025, 3, 1)
    bars = _bars("1301", date(2025, 1, 1), 80)
    common = dict(
        prices={"1301": bars},
        metadata={},
        decision_dates=[decision],
        horizons=[5],
        bundle_disclosure_date=date(2025, 2, 1),
    )
    funds = {"1301": [_statement("1301", basis="non_consolidated")]}

    default = build_modeling_dataset(funds, **common)
    assert default.observations[0].included is False
    assert default.observations[0].exclusion_reason == EXCLUDE_NON_CONSOLIDATED
    assert default.basis_counts() == {"non_consolidated": 1}

    kept = build_modeling_dataset(funds, include_non_consolidated=True, **common)
    assert kept.observations[0].included is True


def test_mixed_basis_is_labelled_mixed():
    decision = date(2025, 3, 1)
    bars = _bars("1301", date(2025, 1, 1), 80)
    funds = {
        "1301": [
            _statement("1301", basis="consolidated"),
            FinancialStatement(
                ticker="1301", fiscal_year=2024, accounting_basis="non_consolidated"
            ),
        ]
    }
    ds = build_modeling_dataset(
        funds,
        {"1301": bars},
        {},
        decision_dates=[decision],
        horizons=[5],
        bundle_disclosure_date=date(2025, 2, 1),
    )
    assert ds.observations[0].accounting_basis == "mixed"


def test_excess_return_is_sector_demeaned():
    decision = date(2025, 3, 1)
    funds = {t: [_statement(t)] for t in ("1301", "1302")}
    prices = {
        "1301": _bars("1301", date(2025, 1, 1), 80, step=10.0),  # rising
        "1302": _bars("1302", date(2025, 1, 1), 80, step=1.0),  # rising slower
    }
    metadata = {
        "1301": CompanyMetadata(ticker="1301", sector="tech"),
        "1302": CompanyMetadata(ticker="1302", sector="tech"),
    }
    ds = build_modeling_dataset(
        funds,
        prices,
        metadata,
        decision_dates=[decision],
        horizons=[5],
        bundle_disclosure_date=date(2025, 2, 1),
    )
    included = {o.ticker: o for o in ds.included()}
    e1 = included["1301"].labels["excess_return_h5"]
    e2 = included["1302"].labels["excess_return_h5"]
    assert e1 is not None and e2 is not None
    # demeaned within the 2-member sector -> opposite signs summing to ~0
    assert round(e1 + e2, 6) == 0.0


def test_dataset_is_deterministic_and_writes_outputs(tmp_path):
    decision = date(2025, 3, 1)
    bars = _bars("1301", date(2025, 1, 1), 80)
    args = (
        {"1301": [_statement("1301")]},
        {"1301": bars},
        {},
    )
    kwargs = dict(
        decision_dates=[decision], horizons=[5, 20], bundle_disclosure_date=date(2025, 2, 1)
    )
    first = build_modeling_dataset(*args, **kwargs)
    second = build_modeling_dataset(*args, **kwargs)
    assert first.to_rows() == second.to_rows()

    paths = write_dataset_outputs(first, tmp_path / "out")
    assert paths["csv_path"].exists()
    assert paths["summary_path"].exists()


def test_requires_decision_dates():
    with pytest.raises(ValueError):
        build_modeling_dataset({}, {}, {}, decision_dates=[], horizons=[5])

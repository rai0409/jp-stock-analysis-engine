"""Survivorship-bias control: listing/delisting files (CSV / Excel / directory).

Covers the loader (merge order, aliases, Excel codes, validation), dataset
eligibility (not yet listed / delisted), delisting-adjusted labels (cash-out
price, source delisting return, explicit penalty, last trade), and the CLI.
Synthetic, deterministic, offline; tickers are fixed placeholders.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from jp_stock_analysis.cli import main
from jp_stock_analysis.errors import DataValidationError
from jp_stock_analysis.modeling.dataset import (
    ENTRY_NEXT_OPEN,
    EXCLUDE_DELISTED,
    EXCLUDE_NOT_YET_LISTED,
    build_modeling_dataset,
    write_dataset_outputs,
)
from jp_stock_analysis.providers.listing_status import (
    listing_status_summary,
    load_listing_status,
)
from jp_stock_analysis.schemas import FinancialStatement, ListingStatus, PriceBar

START = date(2025, 1, 6)


def _bars(ticker: str, n: int, *, factor: float = 1.0) -> list[PriceBar]:
    """Close 100,101,...; open = close - 0.5; ``factor`` = adjusted/raw ratio."""
    return [
        PriceBar(
            ticker=ticker, date=START + timedelta(days=i), open=(100.0 + i - 0.5),
            high=101.0 + i, low=99.0 + i, close=100.0 + i,
            adjusted_close=(100.0 + i) * factor, volume=1000.0,
        )
        for i in range(n)
    ]


def _statement(ticker: str) -> list[FinancialStatement]:
    return [FinancialStatement(
        ticker=ticker, fiscal_year=2024, accounting_basis="consolidated", revenue=1e9,
        net_income=1e8, equity=5e8, total_assets=1e9, shares_outstanding=1e6,
        disclosure_date=START,
    )]


def _dataset(listing: dict[str, ListingStatus], *, n: int = 30, decision_row: int = 10,
             horizons=(5,), entry: str = "next_close", penalty: float | None = None,
             factor: float = 1.0):
    ticker = "9001"
    bars = _bars(ticker, n, factor=factor)
    return bars, build_modeling_dataset(
        {ticker: _statement(ticker)}, {ticker: bars},
        decision_dates=[bars[decision_row].date], horizons=horizons, entry=entry,
        listing_status=listing, delisting_penalty_pct=penalty,
    )


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
def test_csv_loader_with_japanese_headers(tmp_path):
    path = tmp_path / "delist.csv"
    path.write_text(
        "銘柄コード,上場日,上場廃止日,廃止理由,TOB価格\n"
        "9001,2010/04/01,2025/02/03,TOB,1500\n"
        "9002,2015-01-05,,,\n",
        encoding="utf-8",
    )
    statuses = load_listing_status(path)
    assert statuses["9001"].delisting_date == date(2025, 2, 3)
    assert statuses["9001"].final_price == 1500.0
    assert statuses["9001"].delisting_reason == "TOB"
    assert statuses["9002"].delisting_date is None
    summary = listing_status_summary(statuses)
    assert summary == {"tickers": 2, "delisted": 1, "with_final_price": 1,
                       "with_delisting_return": 0, "exit_at_last_trade": 0}


def test_directory_merge_later_file_overrides(tmp_path):
    folder = tmp_path / "listing"
    folder.mkdir()
    (folder / "2025_01.csv").write_text(
        "ticker,delisting_date,delisting_reason\n9001,2025-03-01,announced\n", encoding="utf-8"
    )
    (folder / "2025_02.csv").write_text(
        "ticker,delisting_date,final_price\n9001,2025-02-20,1200\n", encoding="utf-8"
    )
    (folder / "notes.txt").write_text("ignored", encoding="utf-8")
    status = load_listing_status(folder)["9001"]
    assert status.delisting_date == date(2025, 2, 20)  # later file wins
    assert status.final_price == 1200.0
    assert status.delisting_reason == "announced"  # kept: later file left it empty
    assert status.source == "2025_01.csv|2025_02.csv"


def test_excel_loader_normalises_numeric_codes(tmp_path):
    pytest.importorskip("openpyxl")
    import pandas as pd

    path = tmp_path / "delist.xlsx"
    pd.DataFrame(
        {"コード": [9001, 9003], "上場廃止日": ["2025-02-03", None],
         "delisting_return_pct": [-35.0, None]}
    ).to_excel(path, index=False)
    statuses = load_listing_status(path)
    assert set(statuses) == {"9001", "9003"}
    assert statuses["9001"].delisting_return_pct == -35.0


def test_loader_validation(tmp_path):
    bad_date = tmp_path / "a.csv"
    bad_date.write_text("ticker,delisting_date\n9001,next week\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_listing_status(bad_date)
    reversed_dates = tmp_path / "b.csv"
    reversed_dates.write_text(
        "ticker,listing_date,delisting_date\n9001,2025-05-01,2025-01-01\n", encoding="utf-8"
    )
    with pytest.raises(DataValidationError):
        load_listing_status(reversed_dates)
    with pytest.raises(DataValidationError):
        load_listing_status(tmp_path / "missing.csv")


# --------------------------------------------------------------------------- #
# dataset eligibility and delisting-adjusted labels
# --------------------------------------------------------------------------- #
def test_decisions_before_listing_and_after_delisting_are_excluded():
    bars = _bars("9001", 30)
    listing = {"9001": ListingStatus(ticker="9001", listing_date=bars[12].date,
                                     delisting_date=bars[20].date)}
    ds = build_modeling_dataset(
        {"9001": _statement("9001")}, {"9001": bars},
        decision_dates=[bars[10].date, bars[15].date, bars[20].date], horizons=(5,),
        listing_status=listing,
    )
    reasons = [o.exclusion_reason for o in ds.observations]
    assert reasons == [EXCLUDE_NOT_YET_LISTED, None, EXCLUDE_DELISTED]


def test_delisting_beyond_window_keeps_normal_label():
    bars, ds = _dataset({"9001": ListingStatus(ticker="9001")})  # no delisting
    _bars2, ds2 = _dataset({"9001": ListingStatus(ticker="9001", delisting_date=date(2030, 1, 1))})
    assert ds.observations[0].labels == ds2.observations[0].labels
    assert ds2.observations[0].label_flags == {}


def test_last_trade_exit_when_source_has_no_value():
    bars = _bars("9001", 30)
    delist = bars[13].date  # entry row 11, last trade row 12
    _b, ds = _dataset({"9001": ListingStatus(ticker="9001", delisting_date=delist)})
    obs = ds.observations[0]
    assert obs.labels["forward_return_h5"] == pytest.approx((112 / 111 - 1) * 100)
    assert obs.label_flags["forward_return_h5"] == "delisting_adjusted:last_trade"


def test_final_price_and_source_return_and_penalty():
    bars = _bars("9001", 30)
    delist = bars[13].date
    _b, tob = _dataset({"9001": ListingStatus(ticker="9001", delisting_date=delist,
                                              final_price=150.0)})
    assert tob.observations[0].labels["forward_return_h5"] == pytest.approx((150 / 111 - 1) * 100)
    _b, dlret = _dataset({"9001": ListingStatus(ticker="9001", delisting_date=delist,
                                                delisting_return_pct=-50.0)})
    assert dlret.observations[0].labels["forward_return_h5"] == pytest.approx(
        (112 * 0.5 / 111 - 1) * 100
    )
    _b, pen = _dataset({"9001": ListingStatus(ticker="9001", delisting_date=delist)},
                       penalty=-30.0)
    assert pen.observations[0].labels["forward_return_h5"] == pytest.approx(
        (112 * 0.7 / 111 - 1) * 100
    )
    assert pen.observations[0].label_flags["forward_return_h5"] == (
        "delisting_adjusted:configured_penalty"
    )
    assert pen.survivorship["delisting_penalty_pct"] == -30.0


def test_final_price_is_converted_to_adjusted_units():
    bars = _bars("9001", 30)
    _b, ds = _dataset(
        {"9001": ListingStatus(ticker="9001", delisting_date=bars[13].date, final_price=150.0)},
        factor=0.5,  # e.g. a later 2:1 split: adjusted = raw / 2
    )
    # base = adjusted close of row 11 = 55.5; exit = 150 raw * 0.5 = 75
    assert ds.observations[0].labels["forward_return_h5"] == pytest.approx((75 / 55.5 - 1) * 100)


def test_next_open_delisting_label():
    bars = _bars("9001", 30)
    _b, ds = _dataset(
        {"9001": ListingStatus(ticker="9001", delisting_date=bars[13].date)},
        entry=ENTRY_NEXT_OPEN,
    )
    # buy open of row 11 (110.5); exit at last trade close row 12 (112)
    assert ds.observations[0].labels["forward_return_h5"] == pytest.approx((112 / 110.5 - 1) * 100)


def test_delisted_before_entry_has_no_label():
    bars = _bars("9001", 30)
    _b, ds = _dataset({"9001": ListingStatus(ticker="9001", delisting_date=bars[11].date)})
    obs = ds.observations[0]
    assert obs.labels["forward_return_h5"] is None
    assert obs.label_flags["forward_return_h5"] == "delisted_before_entry"


def test_outputs_record_flags_and_summary(tmp_path):
    bars = _bars("9001", 30)
    _b, ds = _dataset({"9001": ListingStatus(ticker="9001", delisting_date=bars[13].date)})
    paths = write_dataset_outputs(ds, tmp_path)
    header = paths["csv_path"].read_text(encoding="utf-8").splitlines()[0]
    assert header.endswith("label_flags")
    summary = json.loads(paths["summary_path"].read_text(encoding="utf-8"))
    assert summary["survivorship"]["delisting_adjusted_labels"] == {
        "delisting_adjusted:last_trade": 1
    }


def test_no_listing_status_leaves_outputs_unchanged(tmp_path):
    _bars_, ds = _dataset({})
    paths = write_dataset_outputs(ds, tmp_path)
    assert "label_flags" not in paths["csv_path"].read_text(encoding="utf-8").splitlines()[0]
    assert "survivorship" not in json.loads(paths["summary_path"].read_text(encoding="utf-8"))


def test_cli_listing_status_directory(tmp_path):
    folder = tmp_path / "listing"
    folder.mkdir()
    (folder / "delist.csv").write_text(
        "ticker,delisting_date\n2001,2025-04-01\n", encoding="utf-8"
    )
    out = tmp_path / "ds"
    rc = main([
        "build-modeling-dataset", "--synthetic-rich", "--horizons", "5,20",
        "--listing-status", str(folder), "--output-dir", str(out),
    ])
    assert rc == 0
    summary = json.loads((out / "modeling_dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["survivorship"]["delisted_tickers"] == 1
    assert summary["exclusions"].get(EXCLUDE_DELISTED, 0) >= 1

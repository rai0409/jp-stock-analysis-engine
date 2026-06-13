"""Deterministic tests for the offline forward-return validation harness.

No network, no randomness: price series and screening payloads are built
inline so look-ahead, missing-horizon, and grouping behavior are all exact.
"""

from __future__ import annotations

import csv
import json
from datetime import date

import pytest

from jp_stock_analysis.cli import main
from jp_stock_analysis.providers.local_csv import load_prices_csv
from jp_stock_analysis.validation.forward_returns import (
    FORWARD_RETURN_DISCLAIMER,
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_NO_PRICE_DATA,
    STATUS_OK,
    build_forward_return_report,
    load_forward_return_report,
    write_forward_return_outputs,
)


def _screening_payload(entries, results=None):
    return {
        "disclaimer": "x",
        "signal_mode": "screening",
        "result_count": len(entries),
        "screening": entries,
        "results": results if results is not None else [],
    }


def _entry(ticker, **overrides):
    base = {
        "ticker": ticker,
        "rank": 1,
        "final_score": 70.0,
        "confidence_score": 80.0,
        "data_coverage_score": 100.0,
        "screening_score": 56.0,
        "screening_eligible": True,
        "reliability_grade": "high",
    }
    base.update(overrides)
    return base


def _write_prices(path, rows):
    """rows: list of (ticker, 'YYYY-MM-DD', close)."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticker", "date", "close"])
        for ticker, d, close in rows:
            writer.writerow([ticker, d, close])


def _linear_rows(ticker, dates, start, step):
    return [(ticker, d, round(start + step * i, 4)) for i, d in enumerate(dates)]


# Ten consecutive weekdays starting 2024-03-04 (analysis_date 2024-03-01 is a Fri).
WEEKDAYS = [
    "2024-03-04",
    "2024-03-05",
    "2024-03-06",
    "2024-03-07",
    "2024-03-08",
    "2024-03-11",
    "2024-03-12",
    "2024-03-13",
    "2024-03-14",
    "2024-03-15",
]


def test_single_ticker_forward_return(tmp_path):
    prices_path = tmp_path / "prices.csv"
    # base=100 at index0 (2024-03-04), +1/row; horizon 5 -> index5 = 105 -> +5%.
    _write_prices(prices_path, _linear_rows("1001", WEEKDAYS, 100.0, 1.0))
    payload = _screening_payload(
        [_entry("1001")], [{"ticker": "1001", "analysis_date": "2024-03-01"}]
    )
    report = build_forward_return_report(
        payload, load_prices_csv(prices_path), [5]
    )
    ticker = report.per_ticker_forward_returns[0]
    cell = ticker.returns[0]
    assert cell.status == STATUS_OK
    assert cell.base_date == date(2024, 3, 4)
    assert cell.base_price == 100.0
    assert cell.target_date == date(2024, 3, 11)
    assert cell.forward_return == pytest.approx(5.0)


def test_multiple_horizons(tmp_path):
    prices_path = tmp_path / "prices.csv"
    _write_prices(prices_path, _linear_rows("1001", WEEKDAYS, 100.0, 1.0))
    payload = _screening_payload(
        [_entry("1001")], [{"ticker": "1001", "analysis_date": "2024-03-01"}]
    )
    report = build_forward_return_report(
        payload, load_prices_csv(prices_path), [1, 5, 9]
    )
    returns = {r.horizon: r for r in report.per_ticker_forward_returns[0].returns}
    assert returns[1].forward_return == pytest.approx(1.0)
    assert returns[5].forward_return == pytest.approx(5.0)
    assert returns[9].forward_return == pytest.approx(9.0)


def test_missing_horizon_marked_not_interpolated(tmp_path):
    prices_path = tmp_path / "prices.csv"
    # Only 6 rows after analysis date: horizon 5 ok (index5), horizon 20/60 missing.
    _write_prices(prices_path, _linear_rows("1001", WEEKDAYS[:6], 100.0, 1.0))
    payload = _screening_payload(
        [_entry("1001")], [{"ticker": "1001", "analysis_date": "2024-03-01"}]
    )
    report = build_forward_return_report(
        payload, load_prices_csv(prices_path), [5, 20, 60]
    )
    returns = {r.horizon: r for r in report.per_ticker_forward_returns[0].returns}
    assert returns[5].status == STATUS_OK
    assert returns[20].status == STATUS_INSUFFICIENT_HISTORY
    assert returns[20].forward_return is None
    assert returns[60].status == STATUS_INSUFFICIENT_HISTORY


def test_no_lookahead_on_or_before_analysis_date(tmp_path):
    prices_path = tmp_path / "prices.csv"
    # Include a row ON the analysis date and one before; neither may become base.
    rows = [
        ("1001", "2024-02-28", 9999.0),  # before
        ("1001", "2024-03-01", 8888.0),  # on the analysis date
    ] + _linear_rows("1001", WEEKDAYS, 100.0, 1.0)
    _write_prices(prices_path, rows)
    payload = _screening_payload(
        [_entry("1001")], [{"ticker": "1001", "analysis_date": "2024-03-01"}]
    )
    report = build_forward_return_report(
        payload, load_prices_csv(prices_path), [5]
    )
    cell = report.per_ticker_forward_returns[0].returns[0]
    # base must be the first row strictly AFTER 2024-03-01, i.e. 2024-03-04 @ 100.
    assert cell.base_date == date(2024, 3, 4)
    assert cell.base_price == 100.0
    assert cell.target_price == 105.0


def test_no_price_data_for_ticker(tmp_path):
    prices_path = tmp_path / "prices.csv"
    _write_prices(prices_path, _linear_rows("9999", WEEKDAYS, 100.0, 1.0))
    payload = _screening_payload(
        [_entry("1001")], [{"ticker": "1001", "analysis_date": "2024-03-01"}]
    )
    report = build_forward_return_report(
        payload, load_prices_csv(prices_path), [5]
    )
    ticker = report.per_ticker_forward_returns[0]
    assert ticker.returns[0].status == STATUS_NO_PRICE_DATA
    assert any("no price rows" in w for w in ticker.warnings)


def _multi_ticker_report(tmp_path, horizons=(5, 20)):
    prices_path = tmp_path / "prices.csv"
    rows = []
    # 1001: full history (eligible/high), positive
    rows += _linear_rows("1001", WEEKDAYS, 100.0, 1.0)
    # 1002: full history (eligible/medium), negative
    rows += _linear_rows("1002", WEEKDAYS, 200.0, -1.0)
    # 1003: short history (ineligible/low) -> all missing
    rows += _linear_rows("1003", WEEKDAYS[:3], 50.0, 1.0)
    _write_prices(prices_path, rows)
    results = [
        {"ticker": "1001", "analysis_date": "2024-03-01"},
        {"ticker": "1002", "analysis_date": "2024-03-01"},
        {"ticker": "1003", "analysis_date": "2024-03-01"},
    ]
    payload = _screening_payload(
        [
            _entry(
                "1001",
                final_score=82.0,
                screening_score=78.0,
                reliability_grade="high",
                screening_eligible=True,
            ),
            _entry(
                "1002",
                final_score=64.0,
                screening_score=42.0,
                reliability_grade="medium",
                screening_eligible=True,
            ),
            _entry(
                "1003",
                final_score=75.0,
                screening_score=8.0,
                reliability_grade="low",
                screening_eligible=False,
            ),
        ],
        results,
    )
    return build_forward_return_report(
        payload, load_prices_csv(prices_path), list(horizons)
    )


def _group(report, dimension, group, horizon):
    return next(
        g
        for g in report.grouped_summary
        if g.dimension == dimension and g.group == group and g.horizon == horizon
    )


def test_grouping_by_screening_eligible(tmp_path):
    report = _multi_ticker_report(tmp_path)
    eligible = _group(report, "screening_eligible", "true", 5)
    ineligible = _group(report, "screening_eligible", "false", 5)
    assert eligible.count == 2
    assert eligible.available_horizon_count == 2
    assert ineligible.count == 1
    assert ineligible.available_horizon_count == 0
    # 1001 +5%, 1002 -2.5% -> one positive of two
    assert eligible.hit_rate_positive == pytest.approx(0.5)


def test_grouping_by_reliability_grade(tmp_path):
    report = _multi_ticker_report(tmp_path)
    high = _group(report, "reliability_grade", "high", 5)
    medium = _group(report, "reliability_grade", "medium", 5)
    low = _group(report, "reliability_grade", "low", 5)
    assert high.mean_forward_return == pytest.approx(5.0)
    assert medium.mean_forward_return == pytest.approx(-2.5)
    assert low.available_horizon_count == 0
    assert low.mean_forward_return is None


def test_grouping_by_screening_score_bucket(tmp_path):
    report = _multi_ticker_report(tmp_path)
    # 1001 screening_score 78 -> "70-80"; 1002 42 -> "40-50"; 1003 8 -> "0-10"
    top = _group(report, "screening_score_bucket", "70-80", 5)
    mid = _group(report, "screening_score_bucket", "40-50", 5)
    bottom = _group(report, "screening_score_bucket", "0-10", 5)
    assert top.mean_forward_return == pytest.approx(5.0)
    assert mid.mean_forward_return == pytest.approx(-2.5)
    assert bottom.available_horizon_count == 0


def test_grouping_by_final_score_bucket(tmp_path):
    report = _multi_ticker_report(tmp_path)
    # 1001 final 82 -> "80-90"; 1002 64 -> "60-70"; 1003 75 -> "70-80"
    assert _group(report, "final_score_bucket", "80-90", 5).mean_forward_return == (
        pytest.approx(5.0)
    )
    assert _group(report, "final_score_bucket", "60-70", 5).mean_forward_return == (
        pytest.approx(-2.5)
    )
    assert _group(report, "final_score_bucket", "70-80", 5).available_horizon_count == 0


def test_analysis_date_override_used_when_missing(tmp_path):
    prices_path = tmp_path / "prices.csv"
    _write_prices(prices_path, _linear_rows("1001", WEEKDAYS, 100.0, 1.0))
    # results section omits analysis_date entirely
    payload = _screening_payload([_entry("1001")], results=[])
    report = build_forward_return_report(
        payload,
        load_prices_csv(prices_path),
        [5],
        analysis_date_override=date(2024, 3, 1),
    )
    ticker = report.per_ticker_forward_returns[0]
    assert ticker.analysis_date_source == "override"
    assert ticker.returns[0].status == STATUS_OK


def test_missing_analysis_date_without_override_warns(tmp_path):
    prices_path = tmp_path / "prices.csv"
    _write_prices(prices_path, _linear_rows("1001", WEEKDAYS, 100.0, 1.0))
    payload = _screening_payload([_entry("1001")], results=[])
    report = build_forward_return_report(payload, load_prices_csv(prices_path), [5])
    ticker = report.per_ticker_forward_returns[0]
    assert ticker.analysis_date_source == "missing"
    assert ticker.returns[0].forward_return is None
    assert report.warnings


def test_deterministic_output_ordering(tmp_path):
    prices_path = tmp_path / "prices.csv"
    # Provide tickers out of sorted order in the screening list.
    rows = (
        _linear_rows("3003", WEEKDAYS, 100.0, 1.0)
        + _linear_rows("1001", WEEKDAYS, 100.0, 1.0)
        + _linear_rows("2002", WEEKDAYS, 100.0, 1.0)
    )
    _write_prices(prices_path, rows)
    payload = _screening_payload(
        [_entry("3003"), _entry("1001"), _entry("2002")],
        [
            {"ticker": "3003", "analysis_date": "2024-03-01"},
            {"ticker": "1001", "analysis_date": "2024-03-01"},
            {"ticker": "2002", "analysis_date": "2024-03-01"},
        ],
    )
    prices = load_prices_csv(prices_path)
    report_a = build_forward_return_report(payload, prices, [5])
    report_b = build_forward_return_report(payload, prices, [5])
    tickers = [t.ticker for t in report_a.per_ticker_forward_returns]
    assert tickers == ["1001", "2002", "3003"]
    assert report_a.to_dict() == report_b.to_dict()


def test_outputs_written_and_disclaimer_present(tmp_path):
    report = _multi_ticker_report(tmp_path)
    out_dir = tmp_path / "out"
    paths = write_forward_return_outputs(report, out_dir)
    assert paths["json_path"].exists()
    assert paths["csv_path"].exists()
    assert paths["markdown_path"].exists()
    payload = json.loads(paths["json_path"].read_text(encoding="utf-8"))
    assert payload["disclaimer"] == FORWARD_RETURN_DISCLAIMER


def test_markdown_has_no_trading_language(tmp_path):
    report = _multi_ticker_report(tmp_path)
    out_dir = tmp_path / "out"
    paths = write_forward_return_outputs(report, out_dir)
    md = paths["markdown_path"].read_text(encoding="utf-8").lower()
    assert "not financial advice" in md
    for forbidden in (
        "buy_signal",
        "sell_signal",
        "hold_signal",
        "position sizing",
        "portfolio construction",
    ):
        # the only allowed occurrences are inside the explicit negative disclaimer
        if forbidden in ("position sizing", "portfolio construction"):
            assert f"no {forbidden}" in md
        else:
            assert forbidden not in md


def test_cli_smoke(tmp_path):
    prices_path = tmp_path / "prices.csv"
    _write_prices(
        prices_path,
        _linear_rows("1001", WEEKDAYS, 100.0, 1.0)
        + _linear_rows("1002", WEEKDAYS[:6], 200.0, -1.0),
    )
    screening_path = tmp_path / "screening.json"
    payload = _screening_payload(
        [
            _entry("1001", reliability_grade="high"),
            _entry("1002", reliability_grade="medium", screening_score=42.0),
        ],
        [
            {"ticker": "1001", "analysis_date": "2024-03-01"},
            {"ticker": "1002", "analysis_date": "2024-03-01"},
        ],
    )
    screening_path.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "out"
    rc = main(
        [
            "validate-forward-returns",
            "--screening-json",
            str(screening_path),
            "--prices",
            str(prices_path),
            "--output-dir",
            str(out_dir),
            "--horizons",
            "5,20,60",
        ]
    )
    assert rc == 0
    assert (out_dir / "forward_returns.json").exists()
    assert (out_dir / "forward_returns.csv").exists()
    assert (out_dir / "forward_returns.md").exists()

    report = load_forward_return_report(screening_path, prices_path, [5, 20, 60])
    # 1002 short series -> 20/60 missing, 5 available
    r1002 = next(t for t in report.per_ticker_forward_returns if t.ticker == "1002")
    statuses = {r.horizon: r.status for r in r1002.returns}
    assert statuses[5] == STATUS_OK
    assert statuses[20] == STATUS_INSUFFICIENT_HISTORY


def test_cli_invalid_horizons_returns_error(tmp_path):
    prices_path = tmp_path / "prices.csv"
    _write_prices(prices_path, _linear_rows("1001", WEEKDAYS, 100.0, 1.0))
    screening_path = tmp_path / "screening.json"
    screening_path.write_text(
        json.dumps(
            _screening_payload(
                [_entry("1001")], [{"ticker": "1001", "analysis_date": "2024-03-01"}]
            )
        ),
        encoding="utf-8",
    )
    rc = main(
        [
            "validate-forward-returns",
            "--screening-json",
            str(screening_path),
            "--prices",
            str(prices_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--horizons",
            "0,-3",
        ]
    )
    assert rc == 1

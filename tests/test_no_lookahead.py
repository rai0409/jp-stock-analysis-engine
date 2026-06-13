"""Deterministic tests for the strict no-look-ahead readiness check.

No network: disclosure dates and price series are built inline, so eligibility
and every blocked-reason are exact.
"""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta

from jp_stock_analysis.cli import main
from jp_stock_analysis.providers.local_csv import load_prices_csv
from jp_stock_analysis.schemas import PriceBar
from jp_stock_analysis.validation.no_lookahead import (
    REASON_INSUFFICIENT_FORWARD_ROWS,
    REASON_MISSING_DISCLOSURE_DATE,
    REASON_MISSING_PRICE_DATA,
    REASON_PRICE_COVERAGE_ENDS_BEFORE,
    STATUS_BLOCKED,
    STATUS_ELIGIBLE,
    build_readiness_report,
    write_readiness_outputs,
)

DISCLOSURE = date(2026, 3, 27)


def _bars(ticker, start, n):
    """n consecutive weekday bars from `start`."""
    bars = []
    d = start
    c = 0
    while c < n:
        if d.weekday() < 5:
            bars.append(PriceBar(ticker=ticker, date=d, close=100.0 + c))
            c += 1
        d += timedelta(days=1)
    return bars


def _h(report, ticker, horizon):
    t = next(t for t in report.per_ticker if t.ticker == ticker)
    return next(h for h in t.horizons if h.horizon == horizon)


def test_eligible_when_enough_forward_rows():
    # 70 weekday bars strictly after disclosure -> horizon 60 needs 61, eligible.
    prices = {"1001": _bars("1001", date(2026, 3, 30), 70)}
    report = build_readiness_report(["1001"], prices, [5, 20, 60], DISCLOSURE)
    for horizon in (5, 20, 60):
        cell = _h(report, "1001", horizon)
        assert cell.status == STATUS_ELIGIBLE
        assert cell.reason is None
    assert report.overall_status == STATUS_ELIGIBLE
    assert report.to_dict()["eligible_tickers"] == ["1001"]


def test_blocked_when_price_coverage_ends_before_disclosure():
    # All bars are on/before the disclosure date -> zero rows strictly after.
    prices = {"1001": _bars("1001", date(2026, 3, 2), 19)}  # ends 2026-03-26
    assert max(b.date for b in prices["1001"]) < DISCLOSURE
    report = build_readiness_report(["1001"], prices, [5, 20, 60], DISCLOSURE)
    for horizon in (5, 20, 60):
        cell = _h(report, "1001", horizon)
        assert cell.status == STATUS_BLOCKED
        assert cell.reason == REASON_PRICE_COVERAGE_ENDS_BEFORE
        assert cell.forward_rows_after == 0
    assert report.overall_status == STATUS_BLOCKED


def test_insufficient_forward_rows_after_disclosure():
    # 10 bars strictly after disclosure: enough for h5 (needs 6), not h20/h60.
    prices = {"1001": _bars("1001", date(2026, 3, 30), 10)}
    report = build_readiness_report(["1001"], prices, [5, 20, 60], DISCLOSURE)
    assert _h(report, "1001", 5).status == STATUS_ELIGIBLE
    assert _h(report, "1001", 20).status == STATUS_BLOCKED
    assert _h(report, "1001", 20).reason == REASON_INSUFFICIENT_FORWARD_ROWS
    assert _h(report, "1001", 60).reason == REASON_INSUFFICIENT_FORWARD_ROWS
    # overall is eligible because at least one ticker/horizon is eligible
    assert report.overall_status == STATUS_ELIGIBLE


def test_missing_price_data():
    report = build_readiness_report(["9999"], {}, [5], DISCLOSURE)
    cell = _h(report, "9999", 5)
    assert cell.status == STATUS_BLOCKED
    assert cell.reason == REASON_MISSING_PRICE_DATA
    assert report.overall_status == STATUS_BLOCKED


def test_missing_disclosure_date():
    prices = {"1001": _bars("1001", date(2026, 3, 30), 70)}
    report = build_readiness_report(["1001"], prices, [5], bundle_disclosure_date=None)
    cell = _h(report, "1001", 5)
    assert cell.status == STATUS_BLOCKED
    assert cell.reason == REASON_MISSING_DISCLOSURE_DATE


def test_boundary_exactly_enough_rows():
    # horizon 5 needs base + 5 = 6 rows strictly after; exactly 6 -> eligible.
    prices = {"1001": _bars("1001", date(2026, 3, 30), 6)}
    assert _h(build_readiness_report(["1001"], prices, [5], DISCLOSURE), "1001", 5).status == (
        STATUS_ELIGIBLE
    )
    # exactly 5 -> one short -> blocked
    prices5 = {"1001": _bars("1001", date(2026, 3, 30), 5)}
    cell = _h(build_readiness_report(["1001"], prices5, [5], DISCLOSURE), "1001", 5)
    assert cell.status == STATUS_BLOCKED
    assert cell.reason == REASON_INSUFFICIENT_FORWARD_ROWS


def test_rows_on_or_before_disclosure_never_counted():
    # bars span before, on, and after the disclosure date; only strictly-after count.
    before = _bars("1001", date(2026, 3, 2), 18)  # up to ~2026-03-25
    on = [PriceBar(ticker="1001", date=DISCLOSURE, close=999.0)]  # on the date
    after = _bars("1001", date(2026, 3, 30), 3)
    prices = {"1001": before + on + after}
    report = build_readiness_report(["1001"], prices, [5], DISCLOSURE)
    cell = _h(report, "1001", 5)
    assert cell.forward_rows_after == 3  # the on-date row is excluded
    assert cell.status == STATUS_BLOCKED
    assert cell.reason == REASON_INSUFFICIENT_FORWARD_ROWS


def test_deterministic_ordering_and_outputs(tmp_path):
    prices = {
        "3003": _bars("3003", date(2026, 3, 30), 2),
        "1001": _bars("1001", date(2026, 3, 30), 70),
    }
    report = build_readiness_report(["3003", "1001"], prices, [5, 20, 60], DISCLOSURE)
    assert [t.ticker for t in report.per_ticker] == ["1001", "3003"]
    paths = write_readiness_outputs(report, tmp_path / "out")
    assert paths["json_path"].exists()
    assert paths["markdown_path"].exists()
    payload = json.loads(paths["json_path"].read_text(encoding="utf-8"))
    assert payload["bundle_disclosure_date"] == "2026-03-27"
    assert payload["disclaimer"]
    assert "not personalized financial advice" in payload["disclaimer"]
    # rebuild -> identical JSON (deterministic)
    report2 = build_readiness_report(["1001", "3003"], prices, [60, 20, 5], DISCLOSURE)
    assert report2.to_dict() == report.to_dict()


def _write_fundamentals(path, tickers):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticker", "fiscal_year", "revenue"])
        for t in tickers:
            writer.writerow([t, 2025, 1000])


def _write_prices(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticker", "date", "close"])
        for r in rows:
            writer.writerow(r)


def test_cli_smoke_blocked(tmp_path, capsys):
    fundamentals = tmp_path / "fundamentals.csv"
    _write_fundamentals(fundamentals, ["3928", "4107", "4264"])
    # price series ending before the disclosure date 2026-03-27
    prices = tmp_path / "prices.csv"
    rows = []
    for t in ("3928", "4107", "4264"):
        for b in _bars(t, date(2026, 3, 2), 19):  # ends ~2026-03-26
            rows.append([t, b.date.isoformat(), b.close])
    _write_prices(prices, rows)
    index_json = tmp_path / "index.json"
    index_json.write_text(json.dumps({"target_date": "2026-03-27", "documents": []}))

    out = tmp_path / "out"
    rc = main(
        [
            "check-forward-readiness",
            "--fundamentals", str(fundamentals),
            "--prices", str(prices),
            "--disclosure-index", str(index_json),
            "--output-dir", str(out),
            "--horizons", "5,20,60",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "BLOCKED" in captured
    payload = json.loads((out / "forward_readiness.json").read_text(encoding="utf-8"))
    assert payload["overall_status"] == "blocked"
    assert payload["bundle_disclosure_date"] == "2026-03-27"
    assert payload["blocked_reason_counts"].get(REASON_PRICE_COVERAGE_ENDS_BEFORE) == 9
    # markdown has no trading language
    md = (out / "forward_readiness.md").read_text(encoding="utf-8").lower()
    assert "not financial advice" in md
    for forbidden in ("buy_signal", "sell_signal", "hold_signal"):
        assert forbidden not in md


def test_cli_requires_disclosure_source(tmp_path):
    fundamentals = tmp_path / "fundamentals.csv"
    _write_fundamentals(fundamentals, ["3928"])
    rc = main(
        [
            "check-forward-readiness",
            "--fundamentals", str(fundamentals),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 1  # no --disclosure-index or --disclosure-date


def test_load_prices_csv_roundtrip_for_readiness(tmp_path):
    """Readiness consumes the same prices CSV shape as the rest of the pipeline."""
    prices_path = tmp_path / "prices.csv"
    _write_prices(prices_path, [["1001", "2026-03-30", 100], ["1001", "2026-03-31", 101]])
    prices = load_prices_csv(prices_path)
    report = build_readiness_report(["1001"], prices, [5], DISCLOSURE)
    cell = _h(report, "1001", 5)
    assert cell.forward_rows_after == 2
    assert cell.status == STATUS_BLOCKED
    assert cell.reason == REASON_INSUFFICIENT_FORWARD_ROWS

"""Strict no-look-ahead readiness check for forward-return validation.

Why this exists
---------------
``forward_returns.py`` already enforces *price-axis* no-look-ahead: a forward
return never uses a price on or before the decision date. That is necessary but
not sufficient for an honest study. The other axis is *disclosure-axis*
no-look-ahead: the screening scores fed into a decision must only use
information that was public **on or before** the decision date.

For the topix1000 annual bundle the fundamentals/disclosures become public on
the bundle's disclosure/target date (e.g. 2026-03-27). A strict study therefore
may only pick a decision date **on or after** that disclosure date, and then
needs enough *future* price rows to measure each horizon. This module checks,
per ticker and horizon, whether such a window exists — and, when it does not,
reports the exact blocking reason. It fabricates nothing and makes no predictive
or trading claim.

Eligibility rule (matches what ``validate-forward-returns`` can actually compute)
---------------------------------------------------------------------------------
The forward-return harness, for horizon ``N``, takes the first price row
strictly after the decision date as the *base* and the row ``N`` positions later
as the *target*. So computing horizon ``N`` needs at least ``N + 1`` price rows
strictly after the decision date (base + ``N``). A horizon is:

- ``eligible``  — ``>= N + 1`` price rows strictly after the disclosure date;
- ``blocked``   — otherwise, with one of these reasons:

  - ``missing_disclosure_date`` — no disclosure date for the ticker;
  - ``missing_price_data`` — no price rows at all for the ticker;
  - ``price_coverage_ends_before_disclosure_date`` — zero price rows strictly
    after the disclosure date (the series ends on/before it);
  - ``insufficient_forward_rows`` — some, but fewer than ``N + 1``, rows after.

Determinism
-----------
Tickers are processed in sorted order and horizons ascending, so JSON / Markdown
output is byte-stable. No network, no fabrication.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from jp_stock_analysis.providers.local_csv import load_prices_csv
from jp_stock_analysis.validation.forward_returns import (
    FORWARD_RETURN_DISCLAIMER,
    _parse_date,
)

STATUS_ELIGIBLE = "eligible"
STATUS_BLOCKED = "blocked"

REASON_MISSING_DISCLOSURE_DATE = "missing_disclosure_date"
REASON_MISSING_PRICE_DATA = "missing_price_data"
REASON_PRICE_COVERAGE_ENDS_BEFORE = "price_coverage_ends_before_disclosure_date"
REASON_INSUFFICIENT_FORWARD_ROWS = "insufficient_forward_rows"

DEFAULT_HORIZONS = (5, 20, 60)


@dataclass(frozen=True)
class HorizonReadiness:
    """Readiness verdict for one ticker at one horizon."""

    horizon: int
    status: str
    reason: str | None
    forward_rows_after: int
    rows_required: int  # N + 1 (base + N)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "status": self.status,
            "reason": self.reason,
            "forward_rows_after": self.forward_rows_after,
            "rows_required": self.rows_required,
        }


@dataclass(frozen=True)
class TickerReadiness:
    """Readiness for one ticker across all horizons."""

    ticker: str
    disclosure_date: date | None
    price_min_date: date | None
    price_max_date: date | None
    price_row_count: int
    forward_rows_after: int
    horizons: list[HorizonReadiness]

    @property
    def any_eligible(self) -> bool:
        return any(h.status == STATUS_ELIGIBLE for h in self.horizons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "disclosure_date": self.disclosure_date.isoformat()
            if self.disclosure_date
            else None,
            "price_min_date": self.price_min_date.isoformat()
            if self.price_min_date
            else None,
            "price_max_date": self.price_max_date.isoformat()
            if self.price_max_date
            else None,
            "price_row_count": self.price_row_count,
            "forward_rows_after": self.forward_rows_after,
            "any_eligible": self.any_eligible,
            "horizons": [h.to_dict() for h in self.horizons],
        }


@dataclass(frozen=True)
class ReadinessReport:
    """Bundle-level strict no-look-ahead readiness verdict."""

    bundle_disclosure_date: date | None
    horizons: list[int]
    disclaimer: str
    per_ticker: list[TickerReadiness]

    @property
    def overall_status(self) -> str:
        """``eligible`` if any ticker/horizon is eligible, else ``blocked``."""
        return (
            STATUS_ELIGIBLE
            if any(t.any_eligible for t in self.per_ticker)
            else STATUS_BLOCKED
        )

    @property
    def blocked_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ticker in self.per_ticker:
            for h in ticker.horizons:
                if h.status == STATUS_BLOCKED and h.reason:
                    counts[h.reason] = counts.get(h.reason, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        eligible_tickers = sorted(t.ticker for t in self.per_ticker if t.any_eligible)
        return {
            "disclaimer": self.disclaimer,
            "bundle_disclosure_date": self.bundle_disclosure_date.isoformat()
            if self.bundle_disclosure_date
            else None,
            "horizons": list(self.horizons),
            "ticker_count": len(self.per_ticker),
            "overall_status": self.overall_status,
            "eligible_ticker_count": len(eligible_tickers),
            "eligible_tickers": eligible_tickers,
            "blocked_reason_counts": self.blocked_reason_counts,
            "per_ticker": [t.to_dict() for t in self.per_ticker],
        }


def _assess_horizon(
    horizon: int,
    disclosure_date: date | None,
    bars: list,
    forward_rows_after: int,
    price_max_date: date | None,
) -> HorizonReadiness:
    required = horizon + 1  # base row + N forward rows
    if disclosure_date is None:
        reason: str | None = REASON_MISSING_DISCLOSURE_DATE
        status = STATUS_BLOCKED
    elif not bars:
        reason = REASON_MISSING_PRICE_DATA
        status = STATUS_BLOCKED
    elif forward_rows_after == 0:
        # No rows strictly after the disclosure date: the series ends on/before it.
        reason = REASON_PRICE_COVERAGE_ENDS_BEFORE
        status = STATUS_BLOCKED
    elif forward_rows_after < required:
        reason = REASON_INSUFFICIENT_FORWARD_ROWS
        status = STATUS_BLOCKED
    else:
        reason = None
        status = STATUS_ELIGIBLE
    return HorizonReadiness(
        horizon=horizon,
        status=status,
        reason=reason,
        forward_rows_after=forward_rows_after,
        rows_required=required,
    )


def build_readiness_report(
    tickers: list[str],
    prices: dict[str, list],
    horizons: list[int],
    bundle_disclosure_date: date | None,
    disclosure_dates: dict[str, date] | None = None,
) -> ReadinessReport:
    """Assess strict no-look-ahead readiness for each ticker/horizon.

    ``prices`` maps ticker -> sorted ``PriceBar`` list (as
    :func:`load_prices_csv` returns). ``disclosure_dates`` may override the
    bundle date per ticker; otherwise ``bundle_disclosure_date`` is used.
    """
    if not horizons:
        raise ValueError("at least one horizon is required")
    horizons = sorted({int(h) for h in horizons})
    for h in horizons:
        if h < 1:
            raise ValueError("horizons must be positive trading-row offsets")

    disclosure_dates = disclosure_dates or {}
    universe = sorted({str(t).strip() for t in tickers if str(t).strip()})

    per_ticker: list[TickerReadiness] = []
    for ticker in universe:
        disclosure_date = disclosure_dates.get(ticker, bundle_disclosure_date)
        bars = prices.get(ticker, [])
        dates = sorted(bar.date for bar in bars)
        price_min = dates[0] if dates else None
        price_max = dates[-1] if dates else None
        if disclosure_date is not None:
            forward_after = sum(1 for d in dates if d > disclosure_date)
        else:
            forward_after = 0
        horizon_readiness = [
            _assess_horizon(h, disclosure_date, bars, forward_after, price_max)
            for h in horizons
        ]
        per_ticker.append(
            TickerReadiness(
                ticker=ticker,
                disclosure_date=disclosure_date,
                price_min_date=price_min,
                price_max_date=price_max,
                price_row_count=len(bars),
                forward_rows_after=forward_after,
                horizons=horizon_readiness,
            )
        )

    return ReadinessReport(
        bundle_disclosure_date=bundle_disclosure_date,
        horizons=horizons,
        disclaimer=FORWARD_RETURN_DISCLAIMER,
        per_ticker=per_ticker,
    )


def load_bundle_disclosure_date(index_json_path: str | Path) -> date | None:
    """Read the bundle ``target_date`` from a topix1000 export ``index.json``."""
    payload = json.loads(Path(index_json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return _parse_date(payload.get("target_date"))


def load_tickers_from_fundamentals(fundamentals_csv_path: str | Path) -> list[str]:
    """Read the ticker universe (first column ``ticker``/``code``) from a CSV."""
    path = Path(fundamentals_csv_path)
    tickers: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [f.strip().lower() for f in (reader.fieldnames or [])]
        col = None
        for candidate in ("ticker", "code", "symbol"):
            if candidate in fieldnames:
                # map back to the real header preserving original case
                col = (reader.fieldnames or [])[fieldnames.index(candidate)]
                break
        if col is None:
            raise ValueError(
                f"fundamentals CSV {path} has no ticker/code column "
                f"(header: {', '.join(reader.fieldnames or []) or 'empty'})"
            )
        for row in reader:
            value = (row.get(col) or "").strip()
            if value:
                tickers.append(value)
    return tickers


def load_readiness_report(
    fundamentals_csv_path: str | Path,
    prices_path: str | Path | None,
    horizons: list[int],
    index_json_path: str | Path | None = None,
    disclosure_date_override: date | None = None,
) -> ReadinessReport:
    """Load inputs from disk and build the readiness report. No network."""
    tickers = load_tickers_from_fundamentals(fundamentals_csv_path)
    bundle_date = disclosure_date_override
    if bundle_date is None and index_json_path is not None:
        bundle_date = load_bundle_disclosure_date(index_json_path)
    prices = load_prices_csv(prices_path) if prices_path else {}
    return build_readiness_report(tickers, prices, horizons, bundle_date)


def _format_date(value: date | None) -> str:
    return value.isoformat() if value else "—"


def write_readiness_outputs(
    report: ReadinessReport,
    output_dir: str | Path,
    write_markdown: bool = True,
) -> dict[str, Path]:
    """Write readiness.json (+ optional readiness.md) and return their paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "forward_readiness.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths = {"json_path": json_path}

    if write_markdown:
        md_path = out_dir / "forward_readiness.md"
        lines: list[str] = []
        lines.append("# Strict No-Look-Ahead Readiness")
        lines.append("")
        lines.append(report.disclaimer)
        lines.append("")
        lines.append(
            "This report checks whether a strict no-look-ahead forward-return "
            "study is possible (disclosure-axis): the decision date must be on or "
            "after the bundle disclosure date, with enough later price rows to "
            "measure each horizon. It contains no trading signals and is not "
            "financial advice."
        )
        lines.append("")
        lines.append(
            f"- Bundle disclosure date: **{_format_date(report.bundle_disclosure_date)}**"
        )
        lines.append(f"- Horizons (trading rows): {report.horizons}")
        lines.append(f"- Overall status: **{report.overall_status.upper()}**")
        lines.append(
            f"- Eligible tickers: {report.to_dict()['eligible_ticker_count']} / "
            f"{len(report.per_ticker)}"
        )
        if report.blocked_reason_counts:
            lines.append("- Blocked (ticker×horizon) reason counts:")
            for reason, count in report.blocked_reason_counts.items():
                lines.append(f"  - `{reason}`: {count}")
        lines.append("")
        lines.append(
            "| ticker | disclosure | price_min | price_max | rows | fwd_after | "
            + " | ".join(f"h{h}" for h in report.horizons)
            + " |"
        )
        lines.append(
            "| --- | --- | --- | --- | --- | --- | "
            + " | ".join("---" for _ in report.horizons)
            + " |"
        )
        for t in report.per_ticker:
            by_h = {h.horizon: h for h in t.horizons}
            cells = []
            for h in report.horizons:
                hr = by_h[h]
                cells.append(
                    hr.status if hr.status == STATUS_ELIGIBLE else (hr.reason or "blocked")
                )
            lines.append(
                f"| {t.ticker} | {_format_date(t.disclosure_date)} | "
                f"{_format_date(t.price_min_date)} | {_format_date(t.price_max_date)} | "
                f"{t.price_row_count} | {t.forward_rows_after} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths["markdown_path"] = md_path

    return paths


__all__ = [
    "DEFAULT_HORIZONS",
    "HorizonReadiness",
    "ReadinessReport",
    "REASON_INSUFFICIENT_FORWARD_ROWS",
    "REASON_MISSING_DISCLOSURE_DATE",
    "REASON_MISSING_PRICE_DATA",
    "REASON_PRICE_COVERAGE_ENDS_BEFORE",
    "STATUS_BLOCKED",
    "STATUS_ELIGIBLE",
    "TickerReadiness",
    "build_readiness_report",
    "load_bundle_disclosure_date",
    "load_readiness_report",
    "load_tickers_from_fundamentals",
    "write_readiness_outputs",
]

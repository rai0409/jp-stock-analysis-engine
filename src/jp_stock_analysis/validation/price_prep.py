"""Local real-price CSV preparation for forward-return validation.

The forward-return harness (:mod:`jp_stock_analysis.validation.forward_returns`)
needs a local ``ticker,date,close`` prices CSV covering dates *after* the
analysis date. This module accepts a user-supplied local raw price CSV in one
of several common schemas, validates and normalizes it, and writes the exact
``ticker,date,close`` shape the harness consumes.

What this is NOT
----------------
This is strictly an offline file-preparation step. It does **not** fetch data
(no network, no J-Quants, no EDINET), does not fabricate or interpolate
prices, and emits no trading signals. It only reshapes and validates a CSV the
user already has locally.

Accepted source header schemas (case-insensitive, extra columns ignored):

- ``ticker,date,close``
- ``ticker,date,open,high,low,close,volume`` (close is taken; OHLCV ignored)
- ``code,date,close``
- ``Code,Date,Close``
- ``LocalCode,Date,Close``

Normalization:

- ticker -> string, trimmed, upper-cased, trailing ``.T`` suffix removed
  (e.g. ``7203.T`` -> ``7203``); alphanumeric listing codes such as ``286A``
  are preserved.
- date -> ``YYYY-MM-DD`` (accepts ``YYYY-MM-DD``, ``YYYY/MM/DD``, ``YYYYMMDD``).
- rows whose ticker is not requested are dropped.
- output is sorted by ``(ticker, date)`` for deterministic, byte-stable output.

Validation (any failure raises :class:`DataValidationError`, CLI exits non-zero):

- header must expose a ticker, a date, and a close column;
- ``close`` must be numeric (commas allowed, e.g. ``1,234``);
- every requested ticker must appear at least once;
- when ``min_rows_after`` is given, every requested ticker must have at least
  that many rows dated on or after ``from_date``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from jp_stock_analysis.errors import DataValidationError

_TICKER_ALIASES = ("ticker", "code", "localcode", "local_code")
_DATE_ALIASES = ("date", "trade_date")
_CLOSE_ALIASES = ("close", "close_price")

_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d")


@dataclass(frozen=True)
class PreparePriceCsvResult:
    """Deterministic summary of a price-CSV preparation run."""

    input_path: str
    output_path: str
    tickers: list[str]
    from_date: str
    min_rows_after: int | None
    total_rows_written: int
    rows_per_ticker: dict[str, int]
    rows_after_from_date: dict[str, int]
    warnings: list[str] = field(default_factory=list)


def normalize_ticker(raw: str | None) -> str:
    """Trim, upper-case, and drop a trailing ``.T`` suffix. ``""`` when empty."""
    if raw is None:
        return ""
    text = raw.strip().upper()
    if text.endswith(".T"):
        text = text[:-2]
    return text


def _normalize_date(raw: str | None) -> date | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _resolve_column(fieldnames: list[str], aliases: tuple[str, ...]) -> str | None:
    """Return the actual header matching the first alias present (case-insensitive)."""
    lowered = {name.strip().lower(): name for name in fieldnames if name is not None}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _parse_tickers(raw: str) -> list[str]:
    seen: list[str] = []
    for part in raw.split(","):
        ticker = normalize_ticker(part)
        if ticker and ticker not in seen:
            seen.append(ticker)
    if not seen:
        raise DataValidationError("--tickers must contain at least one ticker")
    return seen


def prepare_price_csv(
    input_path: str | Path,
    output_path: str | Path,
    tickers: list[str],
    from_date: date,
    *,
    min_rows_after: int | None = None,
) -> PreparePriceCsvResult:
    """Validate and normalize a local raw price CSV to ``ticker,date,close``.

    Raises :class:`DataValidationError` on any structural or coverage failure;
    never fabricates or interpolates prices.
    """
    requested = [normalize_ticker(t) for t in tickers]
    requested = list(dict.fromkeys(t for t in requested if t))
    if not requested:
        raise DataValidationError("no valid tickers requested")
    requested_set = set(requested)

    in_path = Path(input_path)
    if not in_path.exists():
        raise DataValidationError(f"input CSV not found: {in_path}")

    with in_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        ticker_col = _resolve_column(fieldnames, _TICKER_ALIASES)
        date_col = _resolve_column(fieldnames, _DATE_ALIASES)
        close_col = _resolve_column(fieldnames, _CLOSE_ALIASES)
        missing = [
            label
            for label, col in (
                ("ticker", ticker_col),
                ("date", date_col),
                ("close", close_col),
            )
            if col is None
        ]
        if missing:
            raise DataValidationError(
                "input CSV is missing required column(s): "
                + ", ".join(missing)
                + f" (header was: {', '.join(fieldnames) or 'empty'})"
            )

        rows: list[tuple[str, date, float]] = []
        for index, raw in enumerate(reader, start=2):  # row 1 is the header
            ticker = normalize_ticker(raw.get(ticker_col))
            if ticker not in requested_set:
                continue
            parsed_date = _normalize_date(raw.get(date_col))
            if parsed_date is None:
                raise DataValidationError(
                    f"row {index}: unparseable date {raw.get(date_col)!r} for ticker {ticker}"
                )
            close = _to_float(raw.get(close_col))
            if close is None:
                raise DataValidationError(
                    f"row {index}: non-numeric close {raw.get(close_col)!r} for ticker {ticker}"
                )
            rows.append((ticker, parsed_date, close))

    rows.sort(key=lambda item: (item[0], item[1]))

    rows_per_ticker = {ticker: 0 for ticker in requested}
    rows_after = {ticker: 0 for ticker in requested}
    for ticker, parsed_date, _close in rows:
        rows_per_ticker[ticker] += 1
        if parsed_date >= from_date:
            rows_after[ticker] += 1

    absent = sorted(ticker for ticker, count in rows_per_ticker.items() if count == 0)
    if absent:
        raise DataValidationError(
            "requested ticker(s) absent from input CSV: " + ", ".join(absent)
        )

    if min_rows_after is not None:
        short = sorted(
            f"{ticker} ({rows_after[ticker]})"
            for ticker in requested
            if rows_after[ticker] < min_rows_after
        )
        if short:
            raise DataValidationError(
                f"insufficient rows on or after {from_date.isoformat()}: "
                f"need >= {min_rows_after} per ticker, got "
                + ", ".join(short)
            )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["ticker", "date", "close"])
        for ticker, parsed_date, close in rows:
            writer.writerow([ticker, parsed_date.isoformat(), _format_close(close)])

    warnings = [
        "prepared CSV passes the source 'close' column through unchanged; "
        "whether it holds raw or adjusted close depends on the input. Raw close "
        "does not account for corporate actions (splits, dividends)",
    ]
    return PreparePriceCsvResult(
        input_path=str(in_path),
        output_path=str(out_path),
        tickers=requested,
        from_date=from_date.isoformat(),
        min_rows_after=min_rows_after,
        total_rows_written=len(rows),
        rows_per_ticker=rows_per_ticker,
        rows_after_from_date=rows_after,
        warnings=warnings,
    )


def _format_close(value: float) -> str:
    """Render close without trailing ``.0`` for integer-valued prices."""
    if value == int(value):
        return str(int(value))
    return repr(value)


__all__ = ["PreparePriceCsvResult", "prepare_price_csv", "normalize_ticker", "_parse_tickers"]

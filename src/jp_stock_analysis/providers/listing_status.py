"""Load listing / delisting records from updatable CSV or Excel files.

Survivorship-bias control: a backtest that only contains today's listed
companies silently drops every firm that went bankrupt, was taken private or
merged away. These loaders read one or more *updatable* files — CSV
(``.csv``) or Excel (``.xlsx`` / ``.xlsm`` / ``.xls``) — or every such file in
a directory, and merge them into one record per ticker.

Merging rule (deterministic): files are read in sorted path order and rows in
file order; for each ticker a later non-empty value overrides an earlier one,
field by field. Appending a new monthly file therefore updates the master
without editing old files.

Accepted column names (case-insensitive; Japanese aliases supported):

- ticker: ``ticker`` / ``code`` / ``コード`` / ``銘柄コード``
- listing date: ``listing_date`` / ``上場日``
- delisting date: ``delisting_date`` / ``上場廃止日`` / ``廃止日``
- reason: ``delisting_reason`` / ``reason`` / ``廃止理由``
- final price: ``final_price`` / ``tob_price`` / ``最終価格`` / ``TOB価格``
- delisting return (percent): ``delisting_return_pct`` / ``dlret_pct``

Excel support needs ``openpyxl`` (``pip install -e ".[excel]"``); without it a
clear ``DataValidationError`` is raised. Nothing is fetched from the network.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

from jp_stock_analysis.errors import DataValidationError
from jp_stock_analysis.schemas import ListingStatus

EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")
SUPPORTED_SUFFIXES = (".csv", *EXCEL_SUFFIXES)

_ALIASES = {
    "ticker": "ticker", "code": "ticker", "symbol": "ticker", "コード": "ticker",
    "銘柄コード": "ticker",
    "listing_date": "listing_date", "listed_date": "listing_date", "上場日": "listing_date",
    "delisting_date": "delisting_date", "delisted_date": "delisting_date",
    "上場廃止日": "delisting_date", "廃止日": "delisting_date",
    "delisting_reason": "delisting_reason", "reason": "delisting_reason",
    "廃止理由": "delisting_reason", "上場廃止理由": "delisting_reason",
    "final_price": "final_price", "tob_price": "final_price", "最終価格": "final_price",
    "tob価格": "final_price",
    "delisting_return_pct": "delisting_return_pct", "dlret_pct": "delisting_return_pct",
}
_FIELDS = (
    "listing_date", "delisting_date", "delisting_reason", "final_price",
    "delisting_return_pct",
)


def _canonical(key: Any) -> str | None:
    if key is None:
        return None
    text = str(key).strip()
    return _ALIASES.get(text.lower()) or _ALIASES.get(text)


def _to_date(value: Any, where: str) -> date | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise DataValidationError(f"{where}: unrecognised date {text!r}")


def _to_float(value: Any, where: str) -> float | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise DataValidationError(f"{where}: not a number {text!r}") from exc


def _normalize_ticker(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():  # Excel numeric codes
        text = text[:-2]
    return text


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    try:
        import pandas as pd

        frame = pd.read_excel(path, dtype=object)
    except ImportError as exc:  # openpyxl / xlrd missing
        raise DataValidationError(
            f"{path.name}: reading Excel needs openpyxl (pip install -e \".[excel]\")"
        ) from exc
    frame = frame.where(frame.notna(), None)
    return frame.to_dict(orient="records")


def _expand(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(
                p for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                and not p.name.startswith(("~$", "."))
            )
        elif path.is_file():
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise DataValidationError(f"unsupported listing-status file type: {path.name}")
            files.append(path)
        else:
            raise DataValidationError(f"listing-status path not found: {path}")
    return sorted(set(files))


def load_listing_status(paths: str | Path | Iterable[str | Path]) -> dict[str, ListingStatus]:
    """Merge listing/delisting records from CSV/Excel files or directories."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    merged: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[str]] = {}
    for file in _expand(paths):
        for index, raw in enumerate(_read_rows(file)):
            row = {_canonical(k): v for k, v in raw.items() if _canonical(k)}
            where = f"{file.name} row {index + 1}"
            if not row.get("ticker") or str(row["ticker"]).strip() == "":
                raise DataValidationError(f"{where}: ticker is required")
            ticker = _normalize_ticker(row["ticker"])
            values = {
                "listing_date": _to_date(row.get("listing_date"), where),
                "delisting_date": _to_date(row.get("delisting_date"), where),
                "delisting_reason": (str(row["delisting_reason"]).strip() or None)
                if row.get("delisting_reason") is not None else None,
                "final_price": _to_float(row.get("final_price"), where),
                "delisting_return_pct": _to_float(row.get("delisting_return_pct"), where),
            }
            record = merged.setdefault(ticker, dict.fromkeys(_FIELDS))
            for field, value in values.items():
                if value is not None:
                    record[field] = value
            sources.setdefault(ticker, []).append(file.name)
    out: dict[str, ListingStatus] = {}
    for ticker, record in merged.items():
        status = ListingStatus(
            ticker=ticker, source="|".join(dict.fromkeys(sources[ticker])), **record
        )
        if (
            status.listing_date and status.delisting_date
            and status.delisting_date < status.listing_date
        ):
            raise DataValidationError(f"{ticker}: delisting_date precedes listing_date")
        out[ticker] = status
    return dict(sorted(out.items()))


def listing_status_summary(statuses: Mapping[str, ListingStatus]) -> dict[str, Any]:
    delisted = [s for s in statuses.values() if s.delisting_date]
    return {
        "tickers": len(statuses),
        "delisted": len(delisted),
        "with_final_price": sum(1 for s in delisted if s.final_price is not None),
        "with_delisting_return": sum(1 for s in delisted if s.delisting_return_pct is not None),
        "exit_at_last_trade": sum(
            1 for s in delisted if s.final_price is None and s.delisting_return_pct is None
        ),
    }


__all__ = ["SUPPORTED_SUFFIXES", "listing_status_summary", "load_listing_status"]

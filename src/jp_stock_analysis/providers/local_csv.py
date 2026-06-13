"""Local CSV / text-file providers. No network access.

Column names are normalized: lower-cased, stripped, and mapped through alias
tables so common variations (``code``/``symbol`` for ticker, ``adj_close`` for
adjusted close, Japanese headers, etc.) load predictably. Missing optional
columns yield ``None`` values; rows missing required fields are rejected with
``DataValidationError``.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from jp_stock_analysis.errors import DataValidationError
from jp_stock_analysis.schemas import (
    CompanyMetadata,
    DisclosureDocument,
    FinancialStatement,
    PriceBar,
)

_PRICE_ALIASES = {
    "ticker": "ticker",
    "code": "ticker",
    "symbol": "ticker",
    "銘柄コード": "ticker",
    "date": "date",
    "trade_date": "date",
    "日付": "date",
    "open": "open",
    "始値": "open",
    "high": "high",
    "高値": "high",
    "low": "low",
    "安値": "low",
    "close": "close",
    "close_price": "close",
    "終値": "close",
    "adjusted_close": "adjusted_close",
    "adj_close": "adjusted_close",
    "adjclose": "adjusted_close",
    "調整後終値": "adjusted_close",
    "volume": "volume",
    "vol": "volume",
    "出来高": "volume",
}

_FUNDAMENTALS_ALIASES = {
    "ticker": "ticker",
    "code": "ticker",
    "symbol": "ticker",
    "fiscal_year": "fiscal_year",
    "year": "fiscal_year",
    "fy": "fiscal_year",
    "fiscal_period": "fiscal_period",
    "period": "fiscal_period",
    "accounting_basis": "accounting_basis",
    "basis": "accounting_basis",
    "revenue": "revenue",
    "sales": "revenue",
    "net_sales": "revenue",
    "売上高": "revenue",
    "operating_income": "operating_income",
    "operating_profit": "operating_income",
    "営業利益": "operating_income",
    "net_income": "net_income",
    "net_profit": "net_income",
    "profit": "net_income",
    "当期純利益": "net_income",
    "eps": "eps",
    "bps": "bps",
    "dividends_per_share": "dividends_per_share",
    "dps": "dividends_per_share",
    "dividend": "dividends_per_share",
    "shares_outstanding": "shares_outstanding",
    "shares": "shares_outstanding",
    "total_assets": "total_assets",
    "総資産": "total_assets",
    "equity": "equity",
    "net_assets": "equity",
    "shareholders_equity": "equity",
    "自己資本": "equity",
    "operating_cash_flow": "operating_cash_flow",
    "ocf": "operating_cash_flow",
    "capital_expenditure": "capital_expenditure",
    "capex": "capital_expenditure",
}

_METADATA_ALIASES = {
    "ticker": "ticker",
    "code": "ticker",
    "symbol": "ticker",
    "company_name": "company_name",
    "name": "company_name",
    "company": "company_name",
    "社名": "company_name",
    "sector": "sector",
    "industry": "sector",
    "業種": "sector",
    "market": "market",
    "市場": "market",
}

_FUNDAMENTALS_FLOAT_FIELDS = (
    "revenue",
    "operating_income",
    "net_income",
    "eps",
    "bps",
    "dividends_per_share",
    "shares_outstanding",
    "total_assets",
    "equity",
    "operating_cash_flow",
    "capital_expenditure",
)


def _normalize_row(row: dict[str, str], aliases: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        canonical = aliases.get(key.strip().lower()) or aliases.get(key.strip())
        if canonical and canonical not in out:
            out[canonical] = (value or "").strip()
    return out


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    as_float = _to_float(value)
    return None if as_float is None else int(as_float)


def _to_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise DataValidationError(f"CSV file not found: {csv_path}")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_prices_csv(path: str | Path) -> dict[str, list[PriceBar]]:
    """Load daily price bars grouped by ticker, sorted by date ascending."""
    grouped: dict[str, list[PriceBar]] = {}
    for index, raw in enumerate(_read_rows(path)):
        row = _normalize_row(raw, _PRICE_ALIASES)
        ticker = row.get("ticker")
        bar_date = _to_date(row.get("date"))
        close = _to_float(row.get("close"))
        if not ticker or bar_date is None or close is None:
            raise DataValidationError(
                f"prices CSV row {index + 1}: ticker, date, and close are required"
            )
        grouped.setdefault(ticker, []).append(
            PriceBar(
                ticker=ticker,
                date=bar_date,
                open=_to_float(row.get("open")),
                high=_to_float(row.get("high")),
                low=_to_float(row.get("low")),
                close=close,
                adjusted_close=_to_float(row.get("adjusted_close")),
                volume=_to_float(row.get("volume")),
            )
        )
    for bars in grouped.values():
        bars.sort(key=lambda bar: bar.date)
    return grouped


def load_fundamentals_csv(path: str | Path) -> dict[str, list[FinancialStatement]]:
    """Load financial statements grouped by ticker, sorted by fiscal year."""
    grouped: dict[str, list[FinancialStatement]] = {}
    for index, raw in enumerate(_read_rows(path)):
        row = _normalize_row(raw, _FUNDAMENTALS_ALIASES)
        ticker = row.get("ticker")
        if not ticker:
            raise DataValidationError(f"fundamentals CSV row {index + 1}: ticker is required")
        figures = {field: _to_float(row.get(field)) for field in _FUNDAMENTALS_FLOAT_FIELDS}
        grouped.setdefault(ticker, []).append(
            FinancialStatement(
                ticker=ticker,
                fiscal_year=_to_int(row.get("fiscal_year")),
                fiscal_period=row.get("fiscal_period") or None,
                accounting_basis=row.get("accounting_basis") or None,
                source_metadata={"source": str(path)},
                **figures,
            )
        )
    for statements in grouped.values():
        statements.sort(key=lambda s: (s.fiscal_year is None, s.fiscal_year or 0))
    return grouped


def load_company_metadata_csv(path: str | Path) -> dict[str, CompanyMetadata]:
    """Load company metadata keyed by ticker."""
    metadata: dict[str, CompanyMetadata] = {}
    for index, raw in enumerate(_read_rows(path)):
        row = _normalize_row(raw, _METADATA_ALIASES)
        ticker = row.get("ticker")
        if not ticker:
            raise DataValidationError(f"metadata CSV row {index + 1}: ticker is required")
        metadata[ticker] = CompanyMetadata(
            ticker=ticker,
            company_name=row.get("company_name") or None,
            sector=row.get("sector") or None,
            market=row.get("market") or None,
            source_metadata={"source": str(path)},
        )
    return metadata


def load_disclosure_texts(directory: str | Path) -> dict[str, DisclosureDocument]:
    """Load ``<ticker>.txt`` disclosure files from a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise DataValidationError(f"disclosure directory not found: {dir_path}")
    documents: dict[str, DisclosureDocument] = {}
    for text_file in sorted(dir_path.glob("*.txt")):
        ticker = text_file.stem
        documents[ticker] = DisclosureDocument(
            ticker=ticker,
            text=text_file.read_text(encoding="utf-8"),
            document_type="local_text",
            source=str(text_file),
        )
    return documents

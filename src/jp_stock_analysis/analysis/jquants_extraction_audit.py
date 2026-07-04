"""Audit local J-Quants extraction quality without modifying raw inputs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jp_stock_analysis.analysis.jquants_commercial_common import jst_now_iso, to_jsonable


DEFAULT_FINANCIAL_SUMMARY = Path("/mnt/d/jquants_price_store/financial_summary_raw.csv")
DEFAULT_PRICES = Path("/mnt/d/jquants_price_store/prices_daily_bars.csv")
DEFAULT_ANALYSIS_DIR = Path("analysis/jquants_fin_summary")
LOGIC_VERSION = "jquants_extraction_audit_v1.0"

OUTPUT_FILES = {
    "report_json": "extraction_audit_report.json",
    "report_md": "extraction_audit_report.md",
    "missing_columns": "extraction_missing_columns.csv",
    "missing_values": "extraction_missing_values.csv",
    "duplicate_keys": "extraction_duplicate_keys.csv",
    "coverage_by_date": "extraction_coverage_by_date.csv",
    "coverage_by_ticker": "extraction_coverage_by_ticker.csv",
    "unusable_rows": "extraction_unusable_rows.csv",
    "required_fields_matrix": "extraction_required_fields_matrix.csv",
}

RANKING_FILES = {
    "positive_reacted": "commercial_quality_positive_reacted_ranking_enriched.csv",
    "positive_unreacted": "commercial_quality_positive_unreacted_ranking_enriched.csv",
    "negative_reacted": "commercial_quality_negative_reacted_ranking_enriched.csv",
    "negative_reversal": "commercial_quality_negative_reversal_ranking_enriched.csv",
    "excluded": "commercial_quality_rankings_excluded_audit_enriched.csv",
    "unclassified": "commercial_quality_rankings_unclassified_audit_enriched.csv",
}


@dataclass(frozen=True)
class FieldSpec:
    dataset: str
    field_purpose: str
    expected_candidates: tuple[str, ...]
    missing_policy: str
    used_by: str
    impact_if_missing: str


FIELD_SPECS = [
    FieldSpec(
        "financial_summary_raw",
        "ticker",
        ("ticker", "Code", "LocalCode", "code"),
        "required",
        "event classification, price join",
        "cannot join prices",
    ),
    FieldSpec(
        "financial_summary_raw",
        "company_name",
        ("CompanyName", "company_name", "name", "Name"),
        "optional",
        "display enrichment",
        "display name unavailable",
    ),
    FieldSpec(
        "financial_summary_raw",
        "disclosure_date",
        ("DiscDate", "disclosure_date", "date"),
        "required",
        "event date, price reaction",
        "cannot time align event",
    ),
    FieldSpec(
        "financial_summary_raw",
        "disclosure_time",
        ("DisclosedTime", "DiscTime", "first_time", "disclosure_time"),
        "warn",
        "event time bucket",
        "reaction base date becomes conservative or unknown",
    ),
    FieldSpec(
        "financial_summary_raw",
        "document_type",
        ("TypeOfDocument", "DocType", "doctype", "document_type"),
        "required",
        "event classification",
        "cannot classify disclosure type reliably",
    ),
    FieldSpec(
        "financial_summary_raw",
        "sales",
        ("Sales", "NCSales"),
        "warn",
        "financial revision features",
        "sales change features unavailable for affected events",
    ),
    FieldSpec(
        "financial_summary_raw",
        "operating_profit",
        ("OP", "NCOP"),
        "warn",
        "financial revision features",
        "profit change features unavailable for affected events",
    ),
    FieldSpec(
        "financial_summary_raw",
        "net_profit",
        ("NP", "NCNP"),
        "warn",
        "financial revision features",
        "net profit change features unavailable for affected events",
    ),
    FieldSpec(
        "financial_summary_raw",
        "forecast_sales",
        ("FSales", "NxFSales", "FNCSales", "NxFNCSales"),
        "warn",
        "revision detection",
        "forecast revision features may be incomplete",
    ),
    FieldSpec(
        "financial_summary_raw",
        "forecast_dividend",
        ("FDivAnn", "NxFDivAnn", "DivAnn"),
        "warn",
        "dividend revision detection",
        "dividend revision features may be incomplete",
    ),
    FieldSpec(
        "prices_daily_bars",
        "ticker",
        ("ticker", "Code", "LocalCode", "code"),
        "required",
        "price join",
        "cannot join events to prices",
    ),
    FieldSpec(
        "prices_daily_bars",
        "price_date",
        ("date", "Date"),
        "required",
        "price timeline",
        "cannot compute returns",
    ),
    FieldSpec(
        "prices_daily_bars",
        "adjusted_close",
        ("adj_close", "AdjClose", "AdjC"),
        "required",
        "price reaction, abnormal return",
        "cannot compute adjusted returns",
    ),
    FieldSpec(
        "prices_daily_bars",
        "open",
        ("open", "Open", "O"),
        "warn",
        "event-day diagnostics",
        "intraday diagnostics incomplete",
    ),
    FieldSpec(
        "prices_daily_bars",
        "high",
        ("high", "High", "H"),
        "warn",
        "price sanity checks",
        "OHLC sanity checks incomplete",
    ),
    FieldSpec(
        "prices_daily_bars",
        "low",
        ("low", "Low", "L"),
        "warn",
        "price sanity checks",
        "OHLC sanity checks incomplete",
    ),
    FieldSpec(
        "prices_daily_bars",
        "close",
        ("close", "Close", "C"),
        "warn",
        "price sanity checks",
        "OHLC sanity checks incomplete",
    ),
    FieldSpec(
        "prices_daily_bars",
        "volume",
        ("volume", "Volume", "Vo"),
        "warn",
        "liquidity features",
        "volume based liquidity incomplete",
    ),
    FieldSpec(
        "prices_daily_bars",
        "turnover_value",
        ("turnover_value", "TurnoverValue", "Va"),
        "warn",
        "liquidity features",
        "turnover based liquidity incomplete",
    ),
    FieldSpec(
        "prices_daily_bars",
        "adjustment_factor",
        ("adjustment_factor", "AdjustmentFactor", "AdjFactor"),
        "optional",
        "corporate action diagnostics",
        "adjustment diagnostics unavailable",
    ),
]

REQUIRED_MATRIX_COLUMNS = [
    "dataset",
    "field_purpose",
    "expected_candidates",
    "resolved_column",
    "status",
    "missing_policy",
    "used_by",
    "impact_if_missing",
]
MISSING_COLUMNS_COLUMNS = [
    "dataset",
    "field_purpose",
    "expected_candidates",
    "status",
    "missing_policy",
    "impact_if_missing",
]
MISSING_VALUES_COLUMNS = ["dataset", "column", "missing_count", "missing_ratio", "severity", "impact"]
DUPLICATE_COLUMNS = ["dataset", "key_columns", "key_value", "duplicate_count", "severity"]
COVERAGE_DATE_COLUMNS = ["dataset", "date", "row_count", "ticker_count", "missing_core_fields_count"]
COVERAGE_TICKER_COLUMNS = [
    "dataset",
    "ticker",
    "row_count",
    "date_min",
    "date_max",
    "missing_core_fields_count",
    "coverage_score",
]
UNUSABLE_COLUMNS = ["dataset", "row_index", "ticker", "date_or_disc_date", "issue_type", "issue_detail", "severity"]


def normalize_ticker_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    text = re.sub(r"\.0$", "", text)
    text = re.sub(r"\D", "", text)
    if len(text) == 5 and text.endswith("0"):
        text = text[:4]
    if not text:
        return None
    return text.zfill(4)[-4:]


def normalize_ticker_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_ticker_value)


def load_csv_readonly(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        return pd.DataFrame(), [f"missing input file: {path}"]
    try:
        return pd.read_csv(path, dtype={"ticker": str, "Code": str, "LocalCode": str}), []
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), [f"empty csv: {path}"]
    except Exception as exc:
        return pd.DataFrame(), [f"failed to read csv {path}: {exc}"]


def load_json_readonly(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"missing json file: {path}"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except Exception as exc:
        return {}, [f"failed to read json {path}: {exc}"]


def resolve_field(columns: list[str] | pd.Index, candidates: tuple[str, ...], missing_policy: str = "warn") -> dict[str, str]:
    cols = list(columns)
    lower = {str(col).lower(): str(col) for col in cols}
    for idx, candidate in enumerate(candidates):
        if candidate in cols:
            return {
                "resolved_column": candidate,
                "status": "present" if idx == 0 else "alias_found",
            }
        if candidate.lower() in lower:
            return {
                "resolved_column": lower[candidate.lower()],
                "status": "present" if idx == 0 else "alias_found",
            }
    if missing_policy == "required":
        status = "missing_should_exclude"
    elif missing_policy == "optional":
        status = "missing_but_optional"
    else:
        status = "missing_should_warn"
    return {"resolved_column": "", "status": status}


def build_required_fields_matrix(columns_by_dataset: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in FIELD_SPECS:
        resolved = resolve_field(columns_by_dataset.get(spec.dataset, []), spec.expected_candidates, spec.missing_policy)
        rows.append(
            {
                "dataset": spec.dataset,
                "field_purpose": spec.field_purpose,
                "expected_candidates": "|".join(spec.expected_candidates),
                "resolved_column": resolved["resolved_column"],
                "status": resolved["status"],
                "missing_policy": spec.missing_policy,
                "used_by": spec.used_by,
                "impact_if_missing": spec.impact_if_missing,
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_MATRIX_COLUMNS)


def _severity_for_missing(column: str, ratio: float, core_columns: set[str]) -> str:
    if column in core_columns and ratio > 0:
        return "high"
    if ratio >= 0.95:
        return "medium"
    if ratio >= 0.50:
        return "low"
    return "info"


def missing_values_frame(dataset: str, df: pd.DataFrame, core_columns: set[str]) -> pd.DataFrame:
    if df.empty and not len(df.columns):
        return pd.DataFrame(columns=MISSING_VALUES_COLUMNS)
    rows = []
    total = max(len(df), 1)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        ratio = float(missing / total)
        rows.append(
            {
                "dataset": dataset,
                "column": col,
                "missing_count": missing,
                "missing_ratio": ratio,
                "severity": _severity_for_missing(col, ratio, core_columns),
                "impact": "core field affected" if col in core_columns else "monitor if used downstream",
            }
        )
    return pd.DataFrame(rows, columns=MISSING_VALUES_COLUMNS)


def duplicate_keys_frame(dataset: str, df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if df.empty or any(col not in df.columns for col in key_columns):
        return pd.DataFrame(columns=DUPLICATE_COLUMNS)
    grouped = df.dropna(subset=key_columns).groupby(key_columns, dropna=False).size().reset_index(name="duplicate_count")
    grouped = grouped[grouped["duplicate_count"] > 1]
    rows = []
    for _, row in grouped.iterrows():
        key_value = "|".join(str(row[col]) for col in key_columns)
        rows.append(
            {
                "dataset": dataset,
                "key_columns": "|".join(key_columns),
                "key_value": key_value,
                "duplicate_count": int(row["duplicate_count"]),
                "severity": "high",
            }
        )
    return pd.DataFrame(rows, columns=DUPLICATE_COLUMNS)


def _add_issue(
    rows: list[dict[str, Any]],
    dataset: str,
    row_index: Any,
    ticker: Any,
    date_value: Any,
    issue_type: str,
    issue_detail: str,
    severity: str,
) -> None:
    rows.append(
        {
            "dataset": dataset,
            "row_index": row_index,
            "ticker": ticker,
            "date_or_disc_date": date_value,
            "issue_type": issue_type,
            "issue_detail": issue_detail,
            "severity": severity,
        }
    )


def _date_string(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def audit_financial_summary(path: Path) -> dict[str, Any]:
    df, warnings = load_csv_readonly(path)
    matrix = build_required_fields_matrix({"financial_summary_raw": list(df.columns)})
    ticker_col = _resolved(matrix, "financial_summary_raw", "ticker")
    date_col = _resolved(matrix, "financial_summary_raw", "disclosure_date")
    doc_col = _resolved(matrix, "financial_summary_raw", "document_type")

    working = df.copy()
    if ticker_col:
        working["_audit_ticker"] = normalize_ticker_series(working[ticker_col])
    else:
        working["_audit_ticker"] = np.nan
    if date_col:
        working["_audit_date"] = pd.to_datetime(working[date_col], errors="coerce")
    else:
        working["_audit_date"] = pd.NaT

    issues: list[dict[str, Any]] = []
    if len(working):
        bad_ticker = working["_audit_ticker"].isna()
        for idx, row in working[bad_ticker].head(1000).iterrows():
            _add_issue(issues, "financial_summary_raw", idx, None, row.get(date_col, None), "missing_ticker", "ticker is missing or unparsable", "high")
        bad_date = working["_audit_date"].isna()
        for idx, row in working[bad_date].head(1000).iterrows():
            _add_issue(issues, "financial_summary_raw", idx, row.get("_audit_ticker"), None, "missing_or_invalid_disc_date", "DiscDate is missing or unparsable", "high")

    core = {col for col in [ticker_col, date_col, doc_col] if col}
    missing_values = missing_values_frame("financial_summary_raw", df, core)
    duplicate = duplicate_keys_frame("financial_summary_raw", working, [c for c in ["_audit_ticker", "_audit_date", doc_col] if c])
    coverage_date = _coverage_by_date("financial_summary_raw", working, "_audit_date", "_audit_ticker", core)
    coverage_ticker = _coverage_by_ticker("financial_summary_raw", working, "_audit_ticker", "_audit_date", core)

    query_min = query_max = None
    if "query_date" in working.columns:
        query_dates = pd.to_datetime(working["query_date"], errors="coerce").dropna()
        if not query_dates.empty:
            query_min = query_dates.min().strftime("%Y-%m-%d")
            query_max = query_dates.max().strftime("%Y-%m-%d")

    numeric_candidates = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    forecast_candidates = [
        col for col in df.columns if any(token in col.lower() for token in ["forecast", "fdiv", "fsales", "fop", "fnp", "feps", "nxf"])
    ]
    important_missing = missing_values[missing_values["severity"].isin(["high", "medium"])].head(50).to_dict("records")

    summary = {
        "path": str(path),
        "exists": path.exists(),
        "rows": int(len(df)),
        "tickers": int(working["_audit_ticker"].dropna().nunique()) if "_audit_ticker" in working else 0,
        "disc_date_min": _min_date(working["_audit_date"]) if "_audit_date" in working else None,
        "disc_date_max": _max_date(working["_audit_date"]) if "_audit_date" in working else None,
        "query_date_min": query_min,
        "query_date_max": query_max,
        "columns": list(df.columns),
        "numeric_financial_column_candidates": numeric_candidates,
        "forecast_revision_dividend_column_candidates": forecast_candidates,
        "important_missing_values": important_missing,
        "warnings": warnings,
    }
    return {
        "summary": summary,
        "matrix": matrix[matrix["dataset"] == "financial_summary_raw"].copy(),
        "missing_values": missing_values,
        "duplicates": duplicate,
        "coverage_by_date": coverage_date,
        "coverage_by_ticker": coverage_ticker,
        "unusable_rows": pd.DataFrame(issues, columns=UNUSABLE_COLUMNS),
        "data": working,
        "ticker_col": ticker_col,
        "date_col": date_col,
    }


def audit_prices(path: Path) -> dict[str, Any]:
    df, warnings = load_csv_readonly(path)
    matrix = build_required_fields_matrix({"prices_daily_bars": list(df.columns)})
    ticker_col = _resolved(matrix, "prices_daily_bars", "ticker")
    date_col = _resolved(matrix, "prices_daily_bars", "price_date")

    working = df.copy()
    working["_audit_ticker"] = normalize_ticker_series(working[ticker_col]) if ticker_col else np.nan
    working["_audit_date"] = pd.to_datetime(working[date_col], errors="coerce") if date_col else pd.NaT
    for col in [
        "adj_close",
        "turnover_value",
        "volume",
        "adj_open",
        "adj_high",
        "adj_low",
        "adjustment_factor",
        "open",
        "high",
        "low",
        "close",
        "adj_volume",
    ]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    issues: list[dict[str, Any]] = []
    if len(working):
        for idx, row in working[working["_audit_ticker"].isna()].head(1000).iterrows():
            _add_issue(issues, "prices_daily_bars", idx, None, row.get(date_col, None), "missing_ticker", "ticker is missing or unparsable", "high")
        for idx, row in working[working["_audit_date"].isna()].head(1000).iterrows():
            _add_issue(issues, "prices_daily_bars", idx, row.get("_audit_ticker"), None, "missing_or_invalid_price_date", "date is missing or unparsable", "high")
        price_cols = [col for col in ["adj_close", "open", "high", "low", "close"] if col in working.columns]
        if price_cols:
            bad_price = pd.Series(False, index=working.index)
            for col in price_cols:
                bad_price = bad_price | (working[col].notna() & (working[col] <= 0))
            for idx, row in working[bad_price].head(1000).iterrows():
                _add_issue(issues, "prices_daily_bars", idx, row.get("_audit_ticker"), _fmt_date(row.get("_audit_date")), "non_positive_price", "one or more price fields are <= 0", "high")
        if {"high", "low"}.issubset(working.columns):
            bad_hilo = working["high"].notna() & working["low"].notna() & (working["high"] < working["low"])
            for idx, row in working[bad_hilo].head(1000).iterrows():
                _add_issue(issues, "prices_daily_bars", idx, row.get("_audit_ticker"), _fmt_date(row.get("_audit_date")), "high_less_than_low", "high is lower than low", "high")
        for col in ["volume", "adj_volume"]:
            if col in working.columns:
                bad_volume = working[col].notna() & (working[col] < 0)
                for idx, row in working[bad_volume].head(1000).iterrows():
                    _add_issue(issues, "prices_daily_bars", idx, row.get("_audit_ticker"), _fmt_date(row.get("_audit_date")), "negative_volume", f"{col} is negative", "high")

    core = {col for col in [ticker_col, date_col, "adj_close"] if col and col in df.columns}
    missing_values = missing_values_frame("prices_daily_bars", df, core)
    duplicate = duplicate_keys_frame("prices_daily_bars", working, ["_audit_ticker", "_audit_date"])
    coverage_date = _coverage_by_date("prices_daily_bars", working, "_audit_date", "_audit_ticker", core)
    coverage_ticker = _coverage_by_ticker("prices_daily_bars", working, "_audit_ticker", "_audit_date", core)

    latest_coverage = None
    if not coverage_date.empty:
        latest = coverage_date.sort_values("date").tail(1)
        latest_coverage = latest.to_dict("records")[0] if not latest.empty else None

    summary = {
        "path": str(path),
        "exists": path.exists(),
        "rows": int(len(df)),
        "tickers": int(working["_audit_ticker"].dropna().nunique()) if "_audit_ticker" in working else 0,
        "date_min": _min_date(working["_audit_date"]) if "_audit_date" in working else None,
        "date_max": _max_date(working["_audit_date"]) if "_audit_date" in working else None,
        "columns": list(df.columns),
        "latest_date_coverage": latest_coverage,
        "estimated_missing_ticker_date_cells": _estimate_missing_cells(working),
        "warnings": warnings,
    }
    return {
        "summary": summary,
        "matrix": matrix[matrix["dataset"] == "prices_daily_bars"].copy(),
        "missing_values": missing_values,
        "duplicates": duplicate,
        "coverage_by_date": coverage_date,
        "coverage_by_ticker": coverage_ticker,
        "unusable_rows": pd.DataFrame(issues, columns=UNUSABLE_COLUMNS),
        "data": working,
        "ticker_col": ticker_col,
        "date_col": date_col,
    }


def _resolved(matrix: pd.DataFrame, dataset: str, purpose: str) -> str:
    row = matrix[(matrix["dataset"] == dataset) & (matrix["field_purpose"] == purpose)]
    if row.empty:
        return ""
    return str(row.iloc[0]["resolved_column"] or "")


def _fmt_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _min_date(series: pd.Series) -> str | None:
    s = pd.to_datetime(series, errors="coerce").dropna()
    return None if s.empty else s.min().strftime("%Y-%m-%d")


def _max_date(series: pd.Series) -> str | None:
    s = pd.to_datetime(series, errors="coerce").dropna()
    return None if s.empty else s.max().strftime("%Y-%m-%d")


def _coverage_by_date(
    dataset: str, df: pd.DataFrame, date_col: str, ticker_col: str, core_columns: set[str]
) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return pd.DataFrame(columns=COVERAGE_DATE_COLUMNS)
    working = df.copy()
    working["_coverage_date"] = pd.to_datetime(working[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    missing_core = pd.Series(False, index=working.index)
    for col in core_columns:
        if col in working.columns:
            missing_core = missing_core | working[col].isna()
    working["_missing_core"] = missing_core
    rows = []
    for date, group in working.dropna(subset=["_coverage_date"]).groupby("_coverage_date", sort=True):
        rows.append(
            {
                "dataset": dataset,
                "date": date,
                "row_count": int(len(group)),
                "ticker_count": int(group[ticker_col].dropna().nunique()) if ticker_col in group.columns else 0,
                "missing_core_fields_count": int(group["_missing_core"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=COVERAGE_DATE_COLUMNS)


def _coverage_by_ticker(
    dataset: str, df: pd.DataFrame, ticker_col: str, date_col: str, core_columns: set[str]
) -> pd.DataFrame:
    if df.empty or ticker_col not in df.columns:
        return pd.DataFrame(columns=COVERAGE_TICKER_COLUMNS)
    working = df.copy()
    working["_coverage_date"] = pd.to_datetime(working[date_col], errors="coerce") if date_col in working else pd.NaT
    missing_core = pd.Series(False, index=working.index)
    for col in core_columns:
        if col in working.columns:
            missing_core = missing_core | working[col].isna()
    working["_missing_core"] = missing_core
    rows = []
    for ticker, group in working.dropna(subset=[ticker_col]).groupby(ticker_col, sort=True):
        row_count = int(len(group))
        missing = int(group["_missing_core"].sum())
        rows.append(
            {
                "dataset": dataset,
                "ticker": ticker,
                "row_count": row_count,
                "date_min": _min_date(group["_coverage_date"]),
                "date_max": _max_date(group["_coverage_date"]),
                "missing_core_fields_count": missing,
                "coverage_score": round(100.0 * (1.0 - missing / max(row_count, 1)), 2),
            }
        )
    return pd.DataFrame(rows, columns=COVERAGE_TICKER_COLUMNS)


def _estimate_missing_cells(prices: pd.DataFrame) -> int | None:
    if prices.empty or "_audit_ticker" not in prices or "_audit_date" not in prices:
        return None
    valid = prices.dropna(subset=["_audit_ticker", "_audit_date"])
    if valid.empty:
        return None
    possible = int(valid["_audit_ticker"].nunique() * valid["_audit_date"].nunique())
    return max(possible - int(len(valid.drop_duplicates(["_audit_ticker", "_audit_date"]))), 0)


def audit_existing_outputs(analysis_dir: Path) -> dict[str, Any]:
    ranking_summaries = {}
    warnings: list[str] = []
    required = ["ticker", "DiscDate", "liquidity_bucket", "abnormal_ret_1d"]
    for label, filename in RANKING_FILES.items():
        path = analysis_dir / filename
        if not path.exists():
            ranking_summaries[label] = {"path": str(path), "exists": False, "rows": 0, "tickers": 0, "missing_columns": required}
            warnings.append(f"missing ranking output: {path}")
            continue
        df, read_warnings = load_csv_readonly(path)
        missing = [col for col in required if col not in df.columns]
        ranking_summaries[label] = {
            "path": str(path),
            "exists": True,
            "rows": int(len(df)),
            "tickers": int(normalize_ticker_series(df["ticker"]).dropna().nunique()) if "ticker" in df.columns else 0,
            "missing_columns": missing,
            "has_ranking_score": "ranking_score" in df.columns or "commercial_candidate_score" in df.columns,
            "has_classification": any(col in df.columns for col in ["ranking_label", "dashboard_classification"]),
            "has_liquidity_bucket": "liquidity_bucket" in df.columns,
            "has_abnormal_return": any(col.startswith("abnormal_ret") for col in df.columns),
            "has_exclusion_reason": any(col in df.columns for col in ["exclusion_reasons", "unclassified_reason"]),
            "warnings": read_warnings,
        }
        warnings.extend(read_warnings)

    dashboard_audit, dashboard_warnings = load_json_readonly(analysis_dir / "commercial_daily_dashboard_audit.json")
    enriched_summary, enriched_warnings = load_json_readonly(analysis_dir / "commercial_quality_rankings_enriched_summary.json")
    warnings.extend(dashboard_warnings)
    warnings.extend(enriched_warnings)
    warnings.extend(dashboard_audit.get("warnings", []) if isinstance(dashboard_audit, dict) else [])
    warnings.extend(enriched_summary.get("warnings", []) if isinstance(enriched_summary, dict) else [])
    return {
        "analysis_dir": str(analysis_dir),
        "ranking_outputs": ranking_summaries,
        "dashboard_audit_warnings": dashboard_audit.get("warnings", []) if isinstance(dashboard_audit, dict) else [],
        "enriched_summary_warnings": enriched_summary.get("warnings", []) if isinstance(enriched_summary, dict) else [],
        "warnings": warnings,
    }


def compute_joinability(financial: pd.DataFrame, prices: pd.DataFrame) -> dict[str, Any]:
    if financial.empty or prices.empty or "_audit_ticker" not in financial or "_audit_ticker" not in prices:
        return {
            "events": int(len(financial)),
            "financial_ticker_in_price_ratio": None,
            "disc_date_or_after_price_ratio": None,
            "horizon_available_ratio": {"1d": None, "3d": None, "5d": None, "20d": None},
            "unusable_event_count": int(len(financial)),
        }
    events = financial.dropna(subset=["_audit_ticker", "_audit_date"]).copy()
    if events.empty:
        return {
            "events": 0,
            "financial_ticker_in_price_ratio": None,
            "disc_date_or_after_price_ratio": None,
            "horizon_available_ratio": {"1d": None, "3d": None, "5d": None, "20d": None},
            "unusable_event_count": int(len(financial)),
        }
    price_valid = prices.dropna(subset=["_audit_ticker", "_audit_date"]).sort_values(["_audit_ticker", "_audit_date"])
    price_tickers = set(price_valid["_audit_ticker"].dropna().unique())
    dates_by_ticker = {
        ticker: group["_audit_date"].drop_duplicates().sort_values().to_numpy(dtype="datetime64[ns]")
        for ticker, group in price_valid.groupby("_audit_ticker")
    }
    ticker_in_price = events["_audit_ticker"].isin(price_tickers)
    after_available = 0
    horizons = {1: 0, 3: 0, 5: 0, 20: 0}
    for _, row in events.iterrows():
        dates = dates_by_ticker.get(row["_audit_ticker"])
        if dates is None or len(dates) == 0:
            continue
        base = np.datetime64(pd.Timestamp(row["_audit_date"]).to_datetime64())
        pos = int(np.searchsorted(dates, base, side="left"))
        if pos < len(dates):
            after_available += 1
            for horizon in horizons:
                if pos + horizon < len(dates):
                    horizons[horizon] += 1
    total = int(len(events))
    unusable = total - after_available
    return {
        "events": total,
        "financial_ticker_in_price_ratio": float(ticker_in_price.mean()) if total else None,
        "disc_date_or_after_price_ratio": float(after_available / total) if total else None,
        "horizon_available_ratio": {f"{horizon}d": float(count / total) if total else None for horizon, count in horizons.items()},
        "price_reaction_unusable_event_count": int(unusable),
    }


def score_readiness(
    financial_summary: dict[str, Any],
    price_summary: dict[str, Any],
    required_matrix: pd.DataFrame,
    missing_values: pd.DataFrame,
    duplicates: pd.DataFrame,
    joinability: dict[str, Any],
) -> dict[str, Any]:
    fin_rows = max(int(financial_summary.get("rows", 0)), 1)
    price_rows = max(int(price_summary.get("rows", 0)), 1)
    fin_tickers = int(financial_summary.get("tickers", 0))
    price_tickers = int(price_summary.get("tickers", 0))
    financial_score = min(100.0, 100.0 * fin_tickers / max(price_tickers, 1)) if fin_tickers else 0.0
    price_score = 100.0
    if price_summary.get("estimated_missing_ticker_date_cells"):
        missing_cells = float(price_summary["estimated_missing_ticker_date_cells"] or 0)
        actual = price_rows
        price_score = max(0.0, 100.0 * actual / max(actual + missing_cells, 1))
    missing_required = required_matrix["status"].isin(["missing_should_exclude"]).sum()
    warning_missing = required_matrix["status"].isin(["missing_should_warn"]).sum()
    required_score = max(0.0, 100.0 - missing_required * 25.0 - warning_missing * 8.0)
    duplicate_count = int(duplicates["duplicate_count"].sum()) if not duplicates.empty else 0
    duplicate_score = max(0.0, 100.0 - 100.0 * duplicate_count / max(fin_rows + price_rows, 1))
    high_missing = missing_values[missing_values["severity"] == "high"]["missing_count"].sum() if not missing_values.empty else 0
    missing_score = max(0.0, 100.0 - 100.0 * float(high_missing) / max(fin_rows + price_rows, 1))
    join_ratio = joinability.get("disc_date_or_after_price_ratio")
    horizon_ratios = [
        value
        for value in (joinability.get("horizon_available_ratio") or {}).values()
        if value is not None
    ]
    join_components = [join_ratio, *horizon_ratios] if join_ratio is not None else horizon_ratios
    join_score = 0.0 if not join_components else 100.0 * float(min(join_components))
    extraction = float(np.mean([financial_score, price_score, required_score, duplicate_score, missing_score, join_score]))
    if duplicate_count > 0:
        extraction = min(extraction, 89.0)
    return {
        "financial_summary_coverage_score": round(financial_score, 2),
        "price_coverage_score": round(price_score, 2),
        "required_fields_score": round(required_score, 2),
        "duplicate_quality_score": round(duplicate_score, 2),
        "missing_values_score": round(missing_score, 2),
        "joinability_score": round(join_score, 2),
        "extraction_readiness_score": round(extraction, 2),
        "readiness_label": readiness_label(extraction),
    }


def readiness_label(score: float) -> str:
    if score >= 90:
        return "extraction_ready"
    if score >= 80:
        return "extraction_poc_ready"
    if score >= 70:
        return "extraction_needs_review"
    return "extraction_not_ready"


def _missing_columns_from_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    missing = matrix[matrix["status"].astype(str).str.startswith("missing")].copy()
    if missing.empty:
        return pd.DataFrame(columns=MISSING_COLUMNS_COLUMNS)
    return missing[
        ["dataset", "field_purpose", "expected_candidates", "status", "missing_policy", "impact_if_missing"]
    ].copy()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_df(path: Path, df: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if columns is not None:
        for col in columns:
            if col not in out.columns:
                out[col] = np.nan
        out = out[columns]
    out.to_csv(path, index=False)


def build_markdown(report: dict[str, Any]) -> str:
    scores = report["commercial_readiness_score"]
    major = report["major_issues"]
    files = report["input_files"]
    lines = [
        "# J-Quants Extraction Audit Report",
        "",
        "## 1. Summary",
        f"- generated_at_jst: {report['generated_at_jst']}",
        f"- extraction_readiness_score: {scores['extraction_readiness_score']}",
        f"- readiness_label: {scores['readiness_label']}",
        "",
        "## 2. Input files",
        f"- financial_summary_raw: {files['financial_summary_raw']}",
        f"- prices_daily_bars: {files['prices_daily_bars']}",
        f"- analysis_dir: {files['analysis_dir']}",
        "",
        "## 3. Required field resolution",
        f"- missing_columns: {report['output_counts']['missing_columns_rows']}",
        f"- required_fields_rows: {report['output_counts']['required_fields_rows']}",
        "",
        "## 4. Coverage",
        f"- financial rows: {report['financial_summary_raw']['rows']}",
        f"- financial tickers: {report['financial_summary_raw']['tickers']}",
        f"- financial DiscDate range: {report['financial_summary_raw']['disc_date_min']} to {report['financial_summary_raw']['disc_date_max']}",
        f"- price rows: {report['prices_daily_bars']['rows']}",
        f"- price tickers: {report['prices_daily_bars']['tickers']}",
        f"- price date range: {report['prices_daily_bars']['date_min']} to {report['prices_daily_bars']['date_max']}",
        "",
        "## 5. Missing values",
        f"- missing_values_rows: {report['output_counts']['missing_values_rows']}",
        "",
        "## 6. Duplicate keys",
        f"- duplicate_key_rows: {report['output_counts']['duplicate_key_rows']}",
        "",
        "## 7. Unusable rows",
        f"- unusable_rows: {report['output_counts']['unusable_rows']}",
        "",
        "## 8. Joinability",
        f"- financial_ticker_in_price_ratio: {report['joinability'].get('financial_ticker_in_price_ratio')}",
        f"- disc_date_or_after_price_ratio: {report['joinability'].get('disc_date_or_after_price_ratio')}",
        f"- horizon_available_ratio: {report['joinability'].get('horizon_available_ratio')}",
        "",
        "## 9. Existing output consistency",
        f"- existing output warnings: {len(report['existing_outputs'].get('warnings', []))}",
        "",
        "## 10. Extraction readiness score",
        *[f"- {key}: {value}" for key, value in scores.items()],
        "",
        "## 11. Issues to fix before analyzing analysis logic",
    ]
    if major:
        lines.extend(f"- {issue}" for issue in major)
    else:
        lines.append("- No blocking extraction issues detected by this audit.")
    lines.extend(
        [
            "",
            "## 12. Recommended next action",
            "- Review missing columns, duplicate keys, unusable rows, and joinability before changing analysis logic.",
            "- Keep raw J-Quants files read-only; fix extraction or mapping code only after confirming source coverage.",
            "",
        ]
    )
    return "\n".join(lines)


def build_extraction_audit(
    financial_summary: Path = DEFAULT_FINANCIAL_SUMMARY,
    prices: Path = DEFAULT_PRICES,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    output_dir: Path = DEFAULT_ANALYSIS_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    financial = audit_financial_summary(financial_summary)
    price = audit_prices(prices)
    existing = audit_existing_outputs(analysis_dir)

    required_matrix = pd.concat([financial["matrix"], price["matrix"]], ignore_index=True)
    missing_columns = _missing_columns_from_matrix(required_matrix)
    missing_values = pd.concat([financial["missing_values"], price["missing_values"]], ignore_index=True)
    duplicates = pd.concat([financial["duplicates"], price["duplicates"]], ignore_index=True)
    coverage_by_date = pd.concat([financial["coverage_by_date"], price["coverage_by_date"]], ignore_index=True)
    coverage_by_ticker = pd.concat([financial["coverage_by_ticker"], price["coverage_by_ticker"]], ignore_index=True)
    unusable_rows = pd.concat([financial["unusable_rows"], price["unusable_rows"]], ignore_index=True)
    joinability = compute_joinability(financial["data"], price["data"])
    scores = score_readiness(
        financial["summary"], price["summary"], required_matrix, missing_values, duplicates, joinability
    )
    major_issues = []
    if not missing_columns[missing_columns["status"] == "missing_should_exclude"].empty:
        major_issues.append("required extraction columns are missing")
    if not duplicates.empty:
        major_issues.append("duplicate ticker/date or ticker/date/document keys detected")
    if not unusable_rows[unusable_rows["severity"] == "high"].empty:
        major_issues.append("high severity unusable rows detected")
    if scores["joinability_score"] < 90:
        major_issues.append("event-to-price joinability or horizon availability is below 90")

    output_paths = {key: str(output_dir / filename) for key, filename in OUTPUT_FILES.items()}
    report = {
        "generated_at_jst": jst_now_iso(),
        "logic_version": LOGIC_VERSION,
        "input_files": {
            "financial_summary_raw": str(financial_summary),
            "prices_daily_bars": str(prices),
            "analysis_dir": str(analysis_dir),
        },
        "output_paths": output_paths,
        "financial_summary_raw": financial["summary"],
        "prices_daily_bars": price["summary"],
        "existing_outputs": existing,
        "joinability": joinability,
        "commercial_readiness_score": scores,
        "major_issues": major_issues,
        "missing_value_policy": {
            "ticker_missing": "write to unusable_rows and exclude from analysis",
            "DiscDate_or_date_missing": "write to unusable_rows and exclude from price reaction analysis",
            "price_missing": "price reaction unavailable; exclude or warn",
            "volume_or_turnover_missing": "liquidity unavailable; warn or exclude from liquidity screens",
            "financial_numeric_missing": "allowed, warn, or exclude depending on event type",
            "duplicate_ticker_date": "write to duplicate_keys; confirm dedupe rule before analysis",
            "missing_optional_columns": "report warning; pipeline may continue",
        },
        "output_counts": {
            "missing_columns_rows": int(len(missing_columns)),
            "missing_values_rows": int(len(missing_values)),
            "duplicate_key_rows": int(len(duplicates)),
            "coverage_by_date_rows": int(len(coverage_by_date)),
            "coverage_by_ticker_rows": int(len(coverage_by_ticker)),
            "unusable_rows": int(len(unusable_rows)),
            "required_fields_rows": int(len(required_matrix)),
        },
    }

    write_json(output_dir / OUTPUT_FILES["report_json"], report)
    (output_dir / OUTPUT_FILES["report_md"]).write_text(build_markdown(report), encoding="utf-8")
    write_df(output_dir / OUTPUT_FILES["missing_columns"], missing_columns, MISSING_COLUMNS_COLUMNS)
    write_df(output_dir / OUTPUT_FILES["missing_values"], missing_values, MISSING_VALUES_COLUMNS)
    write_df(output_dir / OUTPUT_FILES["duplicate_keys"], duplicates, DUPLICATE_COLUMNS)
    write_df(output_dir / OUTPUT_FILES["coverage_by_date"], coverage_by_date, COVERAGE_DATE_COLUMNS)
    write_df(output_dir / OUTPUT_FILES["coverage_by_ticker"], coverage_by_ticker, COVERAGE_TICKER_COLUMNS)
    write_df(output_dir / OUTPUT_FILES["unusable_rows"], unusable_rows, UNUSABLE_COLUMNS)
    write_df(output_dir / OUTPUT_FILES["required_fields_matrix"], required_matrix, REQUIRED_MATRIX_COLUMNS)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financial-summary", type=Path, default=DEFAULT_FINANCIAL_SUMMARY)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    args = parser.parse_args(argv)
    report = build_extraction_audit(args.financial_summary, args.prices, args.analysis_dir, args.output_dir)
    print(json.dumps(to_jsonable({
        "extraction_readiness_score": report["commercial_readiness_score"]["extraction_readiness_score"],
        "readiness_label": report["commercial_readiness_score"]["readiness_label"],
        "output_json": report["output_paths"]["report_json"],
    }), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

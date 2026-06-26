"""J-Quants daily bars ingestion, quality checks, and analysis-prep features."""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from jp_stock_analysis.data.incremental_prices import load_universe_file
from jp_stock_analysis.errors import DataValidationError, ProviderError

DAILY_BARS_CSV = "prices_daily_bars.csv"
DAILY_BARS_STATE = "daily_bars_fetch_state.json"
QUALITY_REPORT = "daily_bars_quality_report.json"
FIELD_COVERAGE_REPORT = "daily_bars_field_coverage_report.json"
DEFAULT_ADJUSTED_CLOSE_FILE = "/tmp/jquants_topix1000_price_store/prices_adjusted_close.csv"
DEFAULT_FEATURE_FILE = "/tmp/jquants_topix1000_price_store/daily_bars_analysis_features.csv"

OUTPUT_COLUMNS = [
    "ticker",
    "date",
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
    "upper_limit_flag",
    "lower_limit_flag",
    "source_fields_json",
]

NUMERIC_COLUMNS = [
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
]

FEATURE_COLUMNS = [
    "ticker",
    "date",
    "adj_close",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "intraday_return",
    "overnight_gap",
    "high_low_range",
    "rolling_vol_20d",
    "avg_volume_20d",
    "avg_turnover_20d",
    "turnover_rank_by_date",
    "volume_rank_by_date",
    "liquidity_rank_by_date",
    "upper_limit_flag",
    "lower_limit_flag",
    "adjustment_factor",
    "sector",
    "market",
    "new_index_category",
]

_FIELD_CANDIDATES = {
    "date": ("Date", "date"),
    "ticker": ("Code", "LocalCode", "code", "ticker"),
    "adj_close": ("AdjC", "AdjClose", "AdjClosePrice", "AdjustmentClose", "AdjClose", "AdjClose"),
    "turnover_value": ("Va", "TurnoverValue"),
    "volume": ("Vo", "Volume"),
    "adj_open": ("AdjO", "AdjOpen", "AdjustmentOpen"),
    "adj_high": ("AdjH", "AdjHigh", "AdjustmentHigh"),
    "adj_low": ("AdjL", "AdjLow", "AdjustmentLow"),
    "adjustment_factor": ("AdjFactor", "AdjustmentFactor"),
    "open": ("O", "Open"),
    "high": ("H", "High"),
    "low": ("L", "Low"),
    "close": ("C", "Close"),
    "adj_volume": ("AdjVo", "AdjVolume", "AdjustmentVolume"),
    "upper_limit_flag": ("UL", "UpperLimitFlag"),
    "lower_limit_flag": ("LL", "LowerLimitFlag"),
}


class _DailyBarsProvider(Protocol):
    api_version: str

    def fetch_daily_bar_rows_by_date(
        self, target_date: date | str, *, allow_network: bool = False
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class DailyBarsFetchResult:
    output_file: str
    state_file: str
    rows: int
    tickers: int
    added: int
    date_min: str | None
    date_max: str | None
    failed_dates: dict[str, str]
    empty_dates: list[str] = field(default_factory=list)


def normalize_jquants_code_to_ticker(code: str) -> str:
    token = str(code or "").strip()
    if len(token) == 5 and token.endswith("0") and token[:4].isdigit():
        return token[:4]
    if token.isdigit() and len(token) < 4:
        return token.zfill(4)
    return token


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _first_value(row: dict[str, object], names: Sequence[str]) -> object | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _format_value(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value)


def _to_date_string(value: object) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value.isoformat()
    token = str(value)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(token, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _flag_value(value: object) -> str:
    if value in (None, ""):
        return ""
    token = str(value).strip()
    if token.lower() in {"true", "t", "yes"}:
        return "1"
    if token.lower() in {"false", "f", "no"}:
        return "0"
    return token


def map_daily_bar_row(row: dict[str, object]) -> dict[str, object]:
    mapped: dict[str, object] = {}
    day = _to_date_string(_first_value(row, _FIELD_CANDIDATES["date"]))
    raw_ticker = _first_value(row, _FIELD_CANDIDATES["ticker"])
    ticker = normalize_jquants_code_to_ticker(str(raw_ticker or ""))
    if not day or not ticker:
        raise DataValidationError("daily bar row must include Date and Code")
    mapped["ticker"] = ticker
    mapped["date"] = day
    for column in NUMERIC_COLUMNS:
        mapped[column] = _to_float(_first_value(row, _FIELD_CANDIDATES[column]))
    mapped["upper_limit_flag"] = _flag_value(
        _first_value(row, _FIELD_CANDIDATES["upper_limit_flag"])
    )
    mapped["lower_limit_flag"] = _flag_value(
        _first_value(row, _FIELD_CANDIDATES["lower_limit_flag"])
    )
    source = {
        key: value
        for key, value in row.items()
        if value is not None
        and not any(marker in key.lower() for marker in ("api_key", "apikey", "secret", "token"))
    }
    mapped["source_fields_json"] = json.dumps(source, ensure_ascii=False, sort_keys=True)
    return {column: mapped.get(column, "") for column in OUTPUT_COLUMNS}


def load_daily_bars_csv(path: str | Path) -> list[dict[str, object]]:
    p = Path(path)
    if not p.is_file():
        return []
    with p.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and list(rows[0].keys()) != OUTPUT_COLUMNS:
        raise DataValidationError(f"daily bars CSV must have columns {OUTPUT_COLUMNS}")
    return rows


def merge_daily_bars(
    existing: Sequence[dict[str, object]], new: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    deduped: dict[tuple[str, str], dict[str, object]] = {}
    for row in [*existing, *new]:
        ticker = normalize_jquants_code_to_ticker(str(row.get("ticker", "")))
        day = str(row.get("date", "")).strip()
        if not ticker or not day:
            continue
        normalized = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
        normalized["ticker"] = ticker
        normalized["date"] = day
        deduped[(day, ticker)] = normalized
    return [deduped[key] for key in sorted(deduped)]


def write_daily_bars_csv(rows: Sequence[dict[str, object]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: _format_value(row.get(column, "")) for column in OUTPUT_COLUMNS}
            )


def summarize_daily_bars(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    dates = sorted({str(row.get("date", "")) for row in rows if row.get("date")})
    tickers = sorted({str(row.get("ticker", "")) for row in rows if row.get("ticker")})
    return {
        "rows": len(rows),
        "tickers": len(tickers),
        "date_min": dates[0] if dates else None,
        "date_max": dates[-1] if dates else None,
    }


def _date_range_weekdays(start: date, end: date) -> list[date]:
    if end < start:
        raise DataValidationError("--end-date must be on or after --start-date")
    out = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "http 429" in text or "rate limit" in text or "too many requests" in text


def _fetch_rows_with_retry(
    provider: _DailyBarsProvider,
    day: date,
    *,
    allow_network: bool,
    max_retries: int,
    sleep_seconds: float,
    sleep_fn: Callable[[float], None],
    retry_events: list[dict[str, object]],
) -> list[dict[str, object]]:
    attempt = 0
    wait = sleep_seconds
    while True:
        try:
            return provider.fetch_daily_bar_rows_by_date(day, allow_network=allow_network)
        except ProviderError as exc:
            if not _is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            retry_events.append(
                {
                    "date": day.isoformat(),
                    "attempt": attempt + 1,
                    "sleep_seconds": wait,
                    "reason": "rate_limited",
                }
            )
            sleep_fn(wait)
            attempt += 1


def fetch_jquants_daily_bars_incremental(
    provider: _DailyBarsProvider,
    *,
    universe_file: str | Path,
    store_dir: str | Path,
    start_date: str | date,
    end_date: str | date,
    allow_network: bool = False,
    sleep_seconds: float = 90.0,
    max_retries: int = 2,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> DailyBarsFetchResult:
    if not allow_network:
        raise DataValidationError("fetch-jquants-daily-bars-incremental requires --allow-network")
    start = start_date if isinstance(start_date, date) else date.fromisoformat(str(start_date))
    end = end_date if isinstance(end_date, date) else date.fromisoformat(str(end_date))
    universe = set(load_universe_file(universe_file))
    store = Path(store_dir)
    output_file = store / DAILY_BARS_CSV
    state_file = store / DAILY_BARS_STATE
    existing = load_daily_bars_csv(output_file)
    existing_keys = {(str(row.get("date")), str(row.get("ticker"))) for row in existing}
    new_rows: list[dict[str, object]] = []
    failed_dates: dict[str, str] = {}
    retry_events: list[dict[str, object]] = []
    empty_dates: list[str] = []
    successful_dates: list[str] = []

    for day in _date_range_weekdays(start, end):
        day_key = day.isoformat()
        try:
            raw_rows = _fetch_rows_with_retry(
                provider,
                day,
                allow_network=allow_network,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
                sleep_fn=sleep_fn,
                retry_events=retry_events,
            )
            if not raw_rows:
                empty_dates.append(day_key)
                successful_dates.append(day_key)
                continue
            for raw in raw_rows:
                mapped = map_daily_bar_row(raw)
                if mapped["ticker"] in universe:
                    new_rows.append(mapped)
            successful_dates.append(day_key)
        except (ProviderError, DataValidationError, ValueError) as exc:
            failed_dates[day_key] = str(exc)
            continue
        if day != end and sleep_seconds > 0:
            sleep_fn(sleep_seconds)

    combined = merge_daily_bars(existing, new_rows)
    write_daily_bars_csv(combined, output_file)
    summary = summarize_daily_bars(combined)
    after_keys = {(str(row.get("date")), str(row.get("ticker"))) for row in combined}
    added = len(after_keys - existing_keys)
    state = {
        "generated_at": _utc_now(),
        "last_successful_date": max(successful_dates) if successful_dates else None,
        "date_min": summary["date_min"],
        "date_max": summary["date_max"],
        "rows": summary["rows"],
        "tickers": summary["tickers"],
        "failed_dates": failed_dates,
        "empty_dates": empty_dates,
        "retry_events": retry_events,
        "universe_file": str(universe_file),
        "output_file": str(output_file),
        "api_key_status": "PRESENT" if os.environ.get("JQUANTS_API_KEY") else "MISSING",
        "secret_included": False,
    }
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return DailyBarsFetchResult(
        output_file=str(output_file),
        state_file=str(state_file),
        rows=int(summary["rows"]),
        tickers=int(summary["tickers"]),
        added=added,
        date_min=summary["date_min"],
        date_max=summary["date_max"],
        failed_dates=failed_dates,
        empty_dates=empty_dates,
    )


def _as_float(row: dict[str, object], field: str) -> float | None:
    return _to_float(row.get(field))


def _field_coverage(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    coverage = {}
    for column in OUTPUT_COLUMNS:
        non_null = sum(1 for row in rows if str(row.get(column, "")).strip() != "")
        null = len(rows) - non_null
        coverage[column] = {
            "non_null_count": non_null,
            "null_count": null,
            "non_null_ratio": non_null / len(rows) if rows else 0.0,
        }
    return coverage


def compare_daily_bars_adj_close_to_adjusted_close_store(
    daily_bars_rows: Sequence[dict[str, object]],
    adjusted_close_file: str | Path,
    *,
    tolerance: float = 1e-8,
) -> dict[str, object]:
    path = Path(adjusted_close_file)
    if not path.is_file():
        return {
            "adjusted_close_file": str(path),
            "overlap_rows": 0,
            "max_abs_diff": None,
            "mismatch_rows": [],
            "missing_file": True,
            "tolerance": tolerance,
        }
    with path.open(encoding="utf-8-sig", newline="") as handle:
        close_rows = list(csv.DictReader(handle))
    close_by_key = {}
    for row in close_rows:
        key = (
            normalize_jquants_code_to_ticker(str(row.get("ticker", ""))),
            str(row.get("date", "")),
        )
        close_by_key[key] = _to_float(row.get("close"))
    max_abs_diff = 0.0
    overlap = 0
    mismatches = []
    for row in daily_bars_rows:
        key = (
            normalize_jquants_code_to_ticker(str(row.get("ticker", ""))),
            str(row.get("date", "")),
        )
        expected = close_by_key.get(key)
        actual = _as_float(row, "adj_close")
        if expected is None or actual is None:
            continue
        overlap += 1
        diff = abs(actual - expected)
        max_abs_diff = max(max_abs_diff, diff)
        if diff > tolerance:
            mismatches.append(
                {
                    "ticker": key[0],
                    "date": key[1],
                    "daily_bars_adj_close": actual,
                    "adjusted_close_store_close": expected,
                    "abs_diff": diff,
                }
            )
    return {
        "adjusted_close_file": str(path),
        "overlap_rows": overlap,
        "max_abs_diff": max_abs_diff if overlap else None,
        "mismatch_rows": mismatches[:100],
        "mismatch_row_count": len(mismatches),
        "missing_file": False,
        "tolerance": tolerance,
    }


def validate_daily_bars_quality(
    rows: Sequence[dict[str, object]],
    *,
    universe_file: str | Path,
    input_file: str | Path,
    adjusted_close_file: str | Path = DEFAULT_ADJUSTED_CLOSE_FILE,
    tolerance: float = 1e-8,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    universe = set(load_universe_file(universe_file))
    summary = summarize_daily_bars(rows)
    keys = [
        (normalize_jquants_code_to_ticker(str(row.get("ticker", ""))), str(row.get("date", "")))
        for row in rows
    ]
    duplicate_count = len(keys) - len(set(keys))
    by_date: dict[str, set[str]] = {}
    for ticker, day in keys:
        by_date.setdefault(day, set()).add(ticker)
    missing_by_date = {}
    for day, tickers in sorted(by_date.items()):
        missing = universe - tickers
        if missing:
            missing_by_date[day] = sorted(missing)
    per_date_counts = [len(tickers) for tickers in by_date.values()]
    suspicious_counts = {
        "zero_or_negative_adjusted_prices": sum(
            1
            for row in rows
            for field in ("adj_open", "adj_high", "adj_low", "adj_close")
            if (value := _as_float(row, field)) is not None and value <= 0
        ),
        "zero_or_negative_raw_prices": sum(
            1
            for row in rows
            for field in ("open", "high", "low", "close")
            if (value := _as_float(row, field)) is not None and value <= 0
        ),
        "zero_volume": sum(
            1 for row in rows if (value := _as_float(row, "volume")) is not None and value <= 0
        ),
        "zero_turnover_value": sum(
            1
            for row in rows
            if (value := _as_float(row, "turnover_value")) is not None and value <= 0
        ),
    }
    adjustment_counter = Counter(str(row.get("adjustment_factor", "")) for row in rows)
    adjustment_not_one = sum(
        1
        for row in rows
        if (value := _as_float(row, "adjustment_factor")) is not None and value != 1.0
    )
    consistency = compare_daily_bars_adj_close_to_adjusted_close_store(
        rows, adjusted_close_file, tolerance=tolerance
    )
    field_coverage = _field_coverage(rows)
    report = {
        "generated_at": _utc_now(),
        "input_file": str(input_file),
        "universe_file": str(universe_file),
        "adjusted_close_file": str(adjusted_close_file),
        "rows": summary["rows"],
        "tickers": summary["tickers"],
        "date_min": summary["date_min"],
        "date_max": summary["date_max"],
        "duplicate_ticker_date_rows": duplicate_count,
        "per_date_ticker_count_min": min(per_date_counts) if per_date_counts else 0,
        "per_date_ticker_count_max": max(per_date_counts) if per_date_counts else 0,
        "missing_usable_tickers_by_date": missing_by_date,
        "suspicious_counts": suspicious_counts,
        "adjustment_factor_counts": {
            "not_equal_to_1": adjustment_not_one,
            "values": dict(sorted(adjustment_counter.items())),
        },
        "upper_limit_count": sum(1 for row in rows if str(row.get("upper_limit_flag", "")) == "1"),
        "lower_limit_count": sum(1 for row in rows if str(row.get("lower_limit_flag", "")) == "1"),
        "adjusted_close_consistency": consistency,
        "secret_included": False,
        "overall_status": "ok"
        if duplicate_count == 0 and consistency.get("mismatch_row_count", 0) == 0
        else "warning",
    }
    return report, field_coverage


def write_daily_bars_quality_reports(
    *,
    store_dir: str | Path,
    universe_file: str | Path,
    adjusted_close_file: str | Path = DEFAULT_ADJUSTED_CLOSE_FILE,
    output_report: str | Path | None = None,
    field_coverage_report: str | Path | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    store = Path(store_dir)
    input_file = store / DAILY_BARS_CSV
    rows = load_daily_bars_csv(input_file)
    report, coverage = validate_daily_bars_quality(
        rows,
        universe_file=universe_file,
        input_file=input_file,
        adjusted_close_file=adjusted_close_file,
    )
    report_path = Path(output_report) if output_report else store / QUALITY_REPORT
    coverage_path = (
        Path(field_coverage_report) if field_coverage_report else store / FIELD_COVERAGE_REPORT
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    coverage_path.write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report, coverage


def _load_coverage_file(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    rename = {}
    lowered = {str(col).lower(): col for col in df.columns}
    for target, candidates in {
        "ticker": ("ticker", "code", "symbol"),
        "sector": ("sector", "sector_33", "sector33", "s33nm"),
        "market": ("market", "market_name", "mktnm"),
        "new_index_category": ("new_index_category", "index_category", "newindexcategory"),
    }.items():
        for candidate in candidates:
            if candidate in lowered:
                rename[lowered[candidate]] = target
                break
    df = df.rename(columns=rename)
    if "ticker" not in df.columns:
        raise DataValidationError("coverage file must include ticker/code")
    for col in ("sector", "market", "new_index_category"):
        if col not in df.columns:
            df[col] = pd.NA
    df["ticker"] = df["ticker"].map(normalize_jquants_code_to_ticker)
    return df[["ticker", "sector", "market", "new_index_category"]].drop_duplicates("ticker")


def build_daily_bars_analysis_features(
    *,
    daily_bars_file: str | Path,
    coverage_file: str | Path | None = None,
    output_file: str | Path = DEFAULT_FEATURE_FILE,
    lookback_days: int = 20,
    min_average_turnover: float | None = None,
    include_partial_history: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(daily_bars_file, dtype={"ticker": str})
    if df.empty:
        out = pd.DataFrame(columns=FEATURE_COLUMNS)
        out.to_csv(output_file, index=False)
        return out
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ticker"] = df["ticker"].map(normalize_jquants_code_to_ticker)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker", sort=False)
    prev_adj_close = grouped["adj_close"].shift(1)
    df["ret_1d"] = df["adj_close"] / prev_adj_close - 1
    df["ret_5d"] = df["adj_close"] / grouped["adj_close"].shift(5) - 1
    df["ret_20d"] = df["adj_close"] / grouped["adj_close"].shift(20) - 1
    df["intraday_return"] = df["adj_close"] / df["adj_open"] - 1
    df["overnight_gap"] = df["adj_open"] / prev_adj_close - 1
    df["high_low_range"] = df["adj_high"] / df["adj_low"] - 1
    df["rolling_vol_20d"] = grouped["ret_1d"].transform(
        lambda values: values.rolling(lookback_days, min_periods=lookback_days).std()
    )
    df["avg_volume_20d"] = grouped["volume"].transform(
        lambda values: values.rolling(lookback_days, min_periods=lookback_days).mean()
    )
    df["avg_turnover_20d"] = grouped["turnover_value"].transform(
        lambda values: values.rolling(lookback_days, min_periods=lookback_days).mean()
    )
    df["turnover_rank_by_date"] = df.groupby("date")["turnover_value"].rank(
        ascending=False, method="min"
    )
    df["volume_rank_by_date"] = df.groupby("date")["volume"].rank(ascending=False, method="min")
    df["liquidity_rank_by_date"] = df.groupby("date")["avg_turnover_20d"].rank(
        ascending=False, method="min"
    )
    for col in ("sector", "market", "new_index_category"):
        df[col] = pd.NA
    if coverage_file:
        coverage = _load_coverage_file(coverage_file)
        df = df.drop(columns=["sector", "market", "new_index_category"]).merge(
            coverage, on="ticker", how="left"
        )
    if min_average_turnover is not None:
        df = df[df["avg_turnover_20d"] >= float(min_average_turnover)]
    if not include_partial_history:
        required = max(lookback_days, 20)
        df = df.groupby("ticker", group_keys=False).filter(lambda group: len(group) >= required + 1)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    out = df.sort_values(["date", "ticker"])[FEATURE_COLUMNS]
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    return out


__all__ = [
    "DAILY_BARS_CSV",
    "DAILY_BARS_STATE",
    "DEFAULT_ADJUSTED_CLOSE_FILE",
    "DEFAULT_FEATURE_FILE",
    "FIELD_COVERAGE_REPORT",
    "FEATURE_COLUMNS",
    "OUTPUT_COLUMNS",
    "QUALITY_REPORT",
    "DailyBarsFetchResult",
    "build_daily_bars_analysis_features",
    "compare_daily_bars_adj_close_to_adjusted_close_store",
    "fetch_jquants_daily_bars_incremental",
    "load_daily_bars_csv",
    "map_daily_bar_row",
    "merge_daily_bars",
    "normalize_jquants_code_to_ticker",
    "summarize_daily_bars",
    "validate_daily_bars_quality",
    "write_daily_bars_csv",
    "write_daily_bars_quality_reports",
]

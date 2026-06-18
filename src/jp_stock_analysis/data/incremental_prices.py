"""Incremental J-Quants adjusted-close price store (research-only).

The store is local, resumable, and explicit about adjusted-close usage. It never
falls back from adjusted close to raw close, never fabricates missing prices, and
does not emit trading or predictive claims.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from jp_stock_analysis.errors import DataValidationError, ProviderError
from jp_stock_analysis.schemas import PriceBar
from jp_stock_analysis.validation.jquants_prices import PriceField

STORE_CSV = "prices_adjusted_close.csv"
FETCH_STATE = "fetch_state.json"
COVERAGE_REPORT = "coverage_report.json"
ELIGIBILITY_REPORT = "eligibility_report.json"

RESEARCH_DISCLAIMER = (
    "This output is research-data infrastructure only. It is not personalized "
    "financial advice, contains no buy/sell recommendation, and makes no "
    "predictive-performance claim."
)


class _DatePriceProvider(Protocol):
    api_version: str

    def endpoint_url(self, dataset: str) -> str: ...

    def fetch_daily_bars_by_date(
        self, target_date: date | str, *, allow_network: bool = False
    ) -> list[PriceBar]: ...


@dataclass(frozen=True)
class IncrementalFetchResult:
    store_dir: str
    price_file: str
    fetched_dates: list[str]
    skipped_existing_dates: list[str]
    failed_dates: dict[str, str]
    rows_added: int
    row_count: int
    ticker_count: int
    date_min: str | None
    date_max: str | None
    coverage_report_path: str
    eligibility_report_path: str
    state_path: str
    warnings: list[str] = field(default_factory=list)


def normalize_ticker(value: str) -> str:
    token = str(value or "").strip()
    if len(token) == 5 and token.endswith("0") and token[:4].isdigit():
        return token[:4]
    if token.isdigit() and len(token) < 4:
        return token.zfill(4)
    return token


def load_universe_file(path: str | Path) -> list[str]:
    """Load tickers from CSV or plain text, preserving deterministic order."""
    p = Path(path)
    if not p.is_file():
        raise DataValidationError(f"universe file not found: {p}")
    if p.suffix.lower() == ".csv":
        with p.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            lowered = {h.lower(): h for h in headers}
            col = None
            for candidate in ("ticker", "code", "symbol"):
                if candidate in lowered:
                    col = lowered[candidate]
                    break
            if col is None:
                raise DataValidationError(
                    "universe CSV must contain one of: ticker, code, Code, symbol, Ticker"
                )
            values = [normalize_ticker(row.get(col, "")) for row in reader]
    else:
        values = []
        for line in p.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            values.append(normalize_ticker(token.split(",")[0]))
    tickers = [t for t in dict.fromkeys(values) if t]
    if not tickers:
        raise DataValidationError("universe file contains no tickers")
    return tickers


def _parse_date(value: str | date | None, *, name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise DataValidationError(f"invalid {name}: {value!r}") from exc


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


def _format_close(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return repr(float(value))


def _read_store(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = ["ticker", "date", "close"]
    if rows and list(rows[0]) != expected:
        raise DataValidationError(f"price store must have columns {expected}")
    return rows


def _write_store(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        ticker = normalize_ticker(row["ticker"])
        day = row["date"]
        deduped[(ticker, day)] = {"ticker": ticker, "date": day, "close": row["close"]}
    ordered = [deduped[key] for key in sorted(deduped)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "date", "close"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)


def _rows_by_date(rows: Sequence[dict[str, str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in rows:
        out.setdefault(row["date"], set()).add(normalize_ticker(row["ticker"]))
    return out


def _missing_dates(
    rows: Sequence[dict[str, str]], universe: Sequence[str], dates: Sequence[date]
) -> tuple[list[date], list[date]]:
    by_date = _rows_by_date(rows)
    universe_set = set(universe)
    missing: list[date] = []
    complete: list[date] = []
    for day in dates:
        present = by_date.get(day.isoformat(), set())
        if universe_set.issubset(present):
            complete.append(day)
        else:
            missing.append(day)
    return missing, complete


def _bar_to_store_row(bar: PriceBar, *, price_field: PriceField) -> dict[str, str]:
    if price_field == "adjusted_close":
        value = bar.adjusted_close
        if value is None:
            raise DataValidationError(
                f"adjusted_close requested but missing for {bar.ticker} {bar.date}; "
                "no raw close fallback is allowed"
            )
    elif price_field == "close":
        value = bar.close
    else:
        raise DataValidationError(f"invalid price_field {price_field!r}")
    return {
        "ticker": normalize_ticker(bar.ticker),
        "date": bar.date.isoformat(),
        "close": _format_close(float(value)),
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "http 429" in text or "rate limit" in text or "too many requests" in text


def _fetch_with_retry(
    provider: _DatePriceProvider,
    day: date,
    *,
    allow_network: bool,
    max_retries: int,
    sleep_seconds: float,
    backoff_multiplier: float,
    sleep_fn: Callable[[float], None],
    retry_events: list[dict[str, object]],
) -> list[PriceBar]:
    attempt = 0
    wait = sleep_seconds
    while True:
        try:
            return provider.fetch_daily_bars_by_date(day, allow_network=allow_network)
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
            wait *= backoff_multiplier
            attempt += 1


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def verify_price_store(
    store_dir: str | Path,
    *,
    universe_file: str | Path | None = None,
    horizons: Sequence[int] = (5, 20, 60),
) -> dict[str, object]:
    store = Path(store_dir)
    price_file = store / STORE_CSV
    rows = _read_store(price_file)
    universe = load_universe_file(universe_file) if universe_file else sorted(
        {normalize_ticker(row["ticker"]) for row in rows}
    )
    keys = [(normalize_ticker(row["ticker"]), row["date"]) for row in rows]
    duplicate_count = len(keys) - len(set(keys))
    by_ticker: dict[str, list[str]] = {ticker: [] for ticker in universe}
    for row in rows:
        ticker = normalize_ticker(row["ticker"])
        if ticker in by_ticker:
            by_ticker[ticker].append(row["date"])
    missing_universe = sorted(t for t, dates in by_ticker.items() if not dates)
    row_counts = {ticker: len(set(dates)) for ticker, dates in by_ticker.items()}
    dates = sorted({row["date"] for row in rows})
    latest: dict[str, str | None] = {}
    excluded: dict[str, list[dict[str, object]]] = {}
    for horizon in sorted({int(h) for h in horizons}):
        required = horizon + 1
        eligible_dates = []
        exclusions = []
        for ticker, ticker_dates in by_ticker.items():
            unique_dates = sorted(set(ticker_dates))
            if len(unique_dates) < required:
                exclusions.append(
                    {
                        "ticker": ticker,
                        "reason": "insufficient_total_rows",
                        "row_count": len(unique_dates),
                        "rows_required": required,
                    }
                )
        if not exclusions:
            for day in dates:
                if all(sum(1 for d in set(ds) if d > day) >= required for ds in by_ticker.values()):
                    eligible_dates.append(day)
        latest[f"h{horizon}"] = eligible_dates[-1] if eligible_dates else None
        excluded[f"h{horizon}"] = exclusions
    counts = list(row_counts.values())
    report = {
        "disclaimer": RESEARCH_DISCLAIMER,
        "research_only": True,
        "file_path": str(price_file),
        "rows": len(rows),
        "ticker_count": len({ticker for ticker, _ in keys}),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "duplicate_ticker_date_rows": duplicate_count,
        "universe_ticker_count": len(universe),
        "missing_universe_tickers": missing_universe,
        "per_ticker_row_count_summary": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "zero_count": sum(1 for c in counts if c == 0),
            "lt_6_count": sum(1 for c in counts if c < 6),
            "lt_21_count": sum(1 for c in counts if c < 21),
            "lt_61_count": sum(1 for c in counts if c < 61),
        },
        "latest_eligible_decision_dates": latest,
        "excluded_tickers": excluded,
    }
    return report


def write_reports(
    store_dir: str | Path,
    *,
    universe_file: str | Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    store = Path(store_dir)
    coverage = verify_price_store(store, universe_file=universe_file)
    eligibility = {
        "disclaimer": RESEARCH_DISCLAIMER,
        "research_only": True,
        "latest_eligible_decision_dates": coverage["latest_eligible_decision_dates"],
        "excluded_tickers": coverage["excluded_tickers"],
        "missing_universe_tickers": coverage["missing_universe_tickers"],
    }
    (store / COVERAGE_REPORT).write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (store / ELIGIBILITY_REPORT).write_text(
        json.dumps(eligibility, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return coverage, eligibility


def update_incremental_price_store(
    provider: _DatePriceProvider,
    *,
    universe_file: str | Path,
    store_dir: str | Path,
    start_date: str | date,
    end_date: str | date | None = None,
    price_field: PriceField = "adjusted_close",
    allow_network: bool = False,
    mode: Literal["date"] = "date",
    sleep_seconds: float = 13.0,
    max_retries: int = 8,
    backoff_multiplier: float = 2.0,
    continue_on_date_error: bool = False,
    universe_name: str | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> IncrementalFetchResult:
    if mode != "date":
        raise DataValidationError("only --mode date is supported")
    if price_field not in ("adjusted_close", "close"):
        raise DataValidationError("price_field must be adjusted_close or close")
    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    (store / "logs").mkdir(exist_ok=True)
    state_path = store / FETCH_STATE
    price_file = store / STORE_CSV
    universe = load_universe_file(universe_file)
    start = _parse_date(start_date, name="start_date")
    end = _parse_date(end_date, name="end_date") if end_date else date.today()

    existing_state = (
        json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    )
    if existing_state and existing_state.get("price_field") != price_field:
        raise DataValidationError(
            f"store price_field is {existing_state.get('price_field')!r}; "
            f"refusing to mix with {price_field!r}"
        )
    if price_field == "adjusted_close" and existing_state.get("raw_close_fallback") is True:
        raise DataValidationError("store state indicates raw_close_fallback=true; refusing")

    existing_rows = _read_store(price_file)
    candidate_dates = _date_range_weekdays(start, end)
    to_fetch, complete = _missing_dates(existing_rows, universe, candidate_dates)
    failed: dict[str, str] = {}
    retry_events: list[dict[str, object]] = []
    fetched_dates: list[str] = []
    new_rows: list[dict[str, str]] = []
    universe_set = set(universe)

    for day in to_fetch:
        try:
            bars = _fetch_with_retry(
                provider,
                day,
                allow_network=allow_network,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
                backoff_multiplier=backoff_multiplier,
                sleep_fn=sleep_fn,
                retry_events=retry_events,
            )
            for bar in bars:
                ticker = normalize_ticker(bar.ticker)
                if ticker in universe_set:
                    new_rows.append(_bar_to_store_row(bar, price_field=price_field))
            fetched_dates.append(day.isoformat())
        except (ProviderError, DataValidationError) as exc:
            failed[day.isoformat()] = str(exc)
            if not continue_on_date_error:
                break

    combined_rows = [*existing_rows, *new_rows]
    _write_store(price_file, combined_rows)
    all_rows = _read_store(price_file)
    coverage, _eligibility = write_reports(store, universe_file=universe_file)

    row_count = int(coverage["rows"])
    ticker_count = int(coverage["ticker_count"])
    state = {
        "universe_name": universe_name or Path(universe_file).stem,
        "universe_file": str(universe_file),
        "price_field": price_field,
        "start_date": start.isoformat(),
        "last_successful_date": (
            max(fetched_dates)
            if fetched_dates
            else existing_state.get("last_successful_date")
        ),
        "date_min": coverage["date_min"],
        "date_max": coverage["date_max"],
        "row_count": row_count,
        "ticker_count": ticker_count,
        "source": "jquants",
        "api_version": getattr(provider, "api_version", None),
        "generated_at": _utc_now(),
        "raw_close_fallback": False,
        "failed_dates": failed,
        "retry_events": retry_events,
    }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log_path = store / "logs" / f"fetch_{_utc_now().replace(':', '').replace('-', '')}.json"
    log_path.write_text(
        json.dumps(
            {
                "disclaimer": RESEARCH_DISCLAIMER,
                "fetched_dates": fetched_dates,
                "failed_dates": failed,
                "retry_events": retry_events,
                "rows_added": len(new_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    warnings = []
    if price_field == "close":
        warnings.append("raw close requested explicitly; adjusted-close guarantee does not apply")
    if failed:
        warnings.append("one or more dates failed; see fetch_state.json")
    return IncrementalFetchResult(
        store_dir=str(store),
        price_file=str(price_file),
        fetched_dates=fetched_dates,
        skipped_existing_dates=[d.isoformat() for d in complete],
        failed_dates=failed,
        rows_added=len(new_rows),
        row_count=len(all_rows),
        ticker_count=ticker_count,
        date_min=coverage["date_min"],
        date_max=coverage["date_max"],
        coverage_report_path=str(store / COVERAGE_REPORT),
        eligibility_report_path=str(store / ELIGIBILITY_REPORT),
        state_path=str(state_path),
        warnings=warnings,
    )


__all__ = [
    "COVERAGE_REPORT",
    "ELIGIBILITY_REPORT",
    "FETCH_STATE",
    "STORE_CSV",
    "IncrementalFetchResult",
    "load_universe_file",
    "normalize_ticker",
    "update_incremental_price_store",
    "verify_price_store",
    "write_reports",
]

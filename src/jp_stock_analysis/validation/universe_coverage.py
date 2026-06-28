"""TOPIX1000 universe coverage filtering utilities.

The coverage file joins a universe snapshot, J-Quants listed master metadata,
and the local price store. These helpers keep the default modeling universe to
tickers with full price-window coverage, while allowing explicit opt-in for
partial-history names.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from jp_stock_analysis.errors import DataValidationError

USABLE_FULL_WINDOW = "usable_full_window"
USABLE_PARTIAL_HISTORY = "usable_partial_history"

REQUIRED_COLUMNS = [
    "ticker",
    "name_universe",
    "universe_date",
    "new_index_category",
    "coverage_status",
    "reason",
]

OUTPUT_COLUMNS = [
    "ticker",
    "name_universe",
    "universe_date",
    "new_index_category",
    "coverage_status",
    "first_price_date",
    "last_price_date",
    "sector",
    "market",
]

EXCLUDED_REPORT_TICKER_COLUMNS = [
    "ticker",
    "name_universe",
    "new_index_category",
    "coverage_status",
    "reason",
]


@dataclass(frozen=True)
class CoverageFilterResult:
    """Summary returned by the TOPIX1000 coverage filter command."""

    coverage_file: str
    output_file: str
    excluded_report_file: str
    include_partial_history: bool
    included_count: int
    excluded_count: int
    included_statuses: dict[str, int]
    excluded_statuses: dict[str, int]


def load_universe_coverage(path: str | Path) -> list[dict[str, str]]:
    """Load and validate a TOPIX universe coverage CSV."""
    coverage_path = Path(path)
    with coverage_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise DataValidationError(
                f"coverage file missing required column(s): {', '.join(missing)}"
            )
        rows = []
        seen: set[str] = set()
        for row in reader:
            normalized = {key: (value or "").strip() for key, value in row.items()}
            ticker = normalized.get("ticker", "")
            if not ticker:
                continue
            if ticker in seen:
                raise DataValidationError(f"coverage file contains duplicate ticker: {ticker}")
            seen.add(ticker)
            rows.append(normalized)
    if not rows:
        raise DataValidationError("coverage file contains no ticker rows")
    return rows


def filter_usable_tickers(
    coverage: list[dict[str, str]],
    include_partial_history: bool = False,
) -> list[dict[str, str]]:
    """Return rows usable for modeling under the coverage policy."""
    allowed = {USABLE_FULL_WINDOW}
    if include_partial_history:
        allowed.add(USABLE_PARTIAL_HISTORY)
    return [row for row in coverage if row.get("coverage_status") in allowed]


def summarize_universe_coverage(coverage: list[dict[str, str]]) -> dict[str, object]:
    """Summarize coverage rows by status for reports and CLI output."""
    status_counts = Counter(row.get("coverage_status", "") for row in coverage)
    return {
        "universe_count": len(coverage),
        "coverage_statuses": dict(sorted(status_counts.items())),
    }


def write_excluded_tickers_report(
    *,
    coverage: list[dict[str, str]],
    coverage_file: str | Path,
    output_file: str | Path,
    include_partial_history: bool = False,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Write a JSON report for tickers excluded by the coverage policy."""
    included = filter_usable_tickers(coverage, include_partial_history)
    included_tickers = {row["ticker"] for row in included}
    excluded = [row for row in coverage if row["ticker"] not in included_tickers]

    payload: dict[str, object] = {
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "coverage_file": str(coverage_file),
        "include_partial_history": include_partial_history,
        "included_count": len(included),
        "excluded_count": len(excluded),
        "included_statuses": _status_counts(included),
        "excluded_statuses": _status_counts(excluded),
        "excluded_tickers": [
            {column: row.get(column, "") for column in EXCLUDED_REPORT_TICKER_COLUMNS}
            for row in excluded
        ],
    }

    report_path = Path(output_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def write_usable_tickers_csv(
    *,
    rows: list[dict[str, str]],
    output_file: str | Path,
) -> None:
    """Write the stable usable-ticker CSV consumed by downstream workflows."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def filter_topix_universe_by_coverage(
    *,
    coverage_file: str | Path,
    output_file: str | Path,
    excluded_report_file: str | Path,
    include_partial_history: bool = False,
) -> CoverageFilterResult:
    """Load coverage, write usable ticker CSV, and write excluded ticker report."""
    coverage = load_universe_coverage(coverage_file)
    included = filter_usable_tickers(coverage, include_partial_history)
    write_usable_tickers_csv(rows=included, output_file=output_file)
    report = write_excluded_tickers_report(
        coverage=coverage,
        coverage_file=coverage_file,
        output_file=excluded_report_file,
        include_partial_history=include_partial_history,
    )
    return CoverageFilterResult(
        coverage_file=str(coverage_file),
        output_file=str(output_file),
        excluded_report_file=str(excluded_report_file),
        include_partial_history=include_partial_history,
        included_count=int(report["included_count"]),
        excluded_count=int(report["excluded_count"]),
        included_statuses=dict(report["included_statuses"]),
        excluded_statuses=dict(report["excluded_statuses"]),
    )


def usable_ticker_set(
    coverage_file: str | Path,
    *,
    include_partial_history: bool = False,
) -> set[str]:
    """Return the usable ticker set from a coverage file."""
    return {
        row["ticker"]
        for row in filter_usable_tickers(
            load_universe_coverage(coverage_file),
            include_partial_history=include_partial_history,
        )
    }


def _status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("coverage_status", "") for row in rows).items()))

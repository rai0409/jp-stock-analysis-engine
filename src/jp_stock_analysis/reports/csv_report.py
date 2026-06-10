"""Screening CSV writer: one ranked row per ticker.

The ``screening_label`` and ``trade_signal`` columns are included only when
the active mode produced those values, so ``analysis_only`` output carries
neither labels nor signals.
"""

from __future__ import annotations

import csv
from pathlib import Path

from jp_stock_analysis.schemas import ScreeningResult, StockAnalysisResult

_BASE_COLUMNS = [
    "rank",
    "ticker",
    "company_name",
    "final_score",
    "quality_score",
    "growth_score",
    "valuation_score",
    "momentum_score",
    "disclosure_score",
    "risk_score",
    "confidence_score",
    "warnings_count",
]


def count_warnings(result: StockAnalysisResult) -> int:
    """Total warnings across the result and all of its components."""
    components = (
        result.fundamentals,
        result.valuation,
        result.momentum,
        result.disclosure,
        result.risks,
        result.score,
    )
    return len(result.warnings) + sum(
        len(component.warnings) for component in components if component is not None
    )


def write_screening_csv(
    results: list[StockAnalysisResult],
    screening: list[ScreeningResult],
    output_path: str | Path,
) -> Path:
    """Write the ranked screening CSV and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    by_ticker = {result.ticker: result for result in results}
    include_label = any(entry.screening_label is not None for entry in screening)
    include_signal = any(
        result.signal is not None for result in results
    )

    columns = list(_BASE_COLUMNS)
    if include_label:
        columns.insert(columns.index("confidence_score") + 1, "screening_label")
    if include_signal:
        columns.insert(columns.index("warnings_count"), "trade_signal")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for entry in screening:
            result = by_ticker.get(entry.ticker)
            score = result.score if result else None
            row: dict[str, object] = {
                "rank": entry.rank,
                "ticker": entry.ticker,
                "company_name": entry.company_name or "",
                "final_score": score.final_score if score else None,
                "quality_score": score.quality_score if score else None,
                "growth_score": score.growth_score if score else None,
                "valuation_score": score.valuation_score if score else None,
                "momentum_score": score.momentum_score if score else None,
                "disclosure_score": score.disclosure_score if score else None,
                "risk_score": score.risk_score if score else None,
                "confidence_score": entry.confidence_score,
                "warnings_count": count_warnings(result) if result else 0,
            }
            if include_label:
                row["screening_label"] = entry.screening_label or ""
            if include_signal:
                row["trade_signal"] = (
                    result.signal.label if result and result.signal is not None else ""
                )
            writer.writerow(row)
    return path

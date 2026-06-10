"""JSON report writer. Includes the disclaimer; signal/label fields appear
only when the corresponding mode produced them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jp_stock_analysis.config import AnalysisConfig
from jp_stock_analysis.schemas import ScreeningResult, StockAnalysisResult


def build_json_payload(
    results: list[StockAnalysisResult],
    screening: list[ScreeningResult],
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Build a JSON-serializable payload for all results."""
    result_dicts: list[dict[str, Any]] = []
    for result in results:
        data = result.model_dump(mode="json")
        if result.signal is None:
            data.pop("signal", None)
        if result.screening_label is None:
            data.pop("screening_label", None)
        result_dicts.append(data)

    screening_dicts: list[dict[str, Any]] = []
    for entry in screening:
        data = entry.model_dump(mode="json")
        if entry.screening_label is None:
            data.pop("screening_label", None)
        screening_dicts.append(data)

    return {
        "disclaimer": config.disclaimer,
        "signal_mode": config.signal_mode,
        "result_count": len(result_dicts),
        "screening": screening_dicts,
        "results": result_dicts,
    }


def write_json_report(
    results: list[StockAnalysisResult],
    screening: list[ScreeningResult],
    output_path: str | Path,
    config: AnalysisConfig,
) -> Path:
    """Write the JSON report and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_json_payload(results, screening, config)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path

"""Export J-Quants listed master metadata for a fixed universe.

The export is intentionally narrow and provider-backed. Real API access should
come from :class:`jp_stock_analysis.providers.jquants.JQuantsProvider`, so the
same endpoint configuration and ``x-api-key`` authentication path is reused.
Tests can pass a fake provider and never touch the network.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from jp_stock_analysis.errors import DataValidationError, ProviderError
from jp_stock_analysis.providers.jquants import ENV_API_KEY
from jp_stock_analysis.schemas import CompanyMetadata

OUTPUT_COLUMNS = [
    "ticker",
    "name_universe",
    "universe_date",
    "new_index_category",
    "matched",
    "company_name",
    "sector",
    "market",
    "source_metadata_json",
    "error",
    "raw_code",
    "company_name_en",
    "sector_17",
    "sector_33",
]


class _MetadataProvider(Protocol):
    """Minimal provider protocol used by the listed-master export."""

    def get_metadata(self, ticker: str) -> CompanyMetadata | None: ...


@dataclass(frozen=True)
class ListedMasterExportResult:
    """Deterministic summary of a listed master export run."""

    output_path: str
    report_path: str
    universe_count: int
    matched_count: int
    missing_count: int
    missing_tickers: list[str]
    warnings: list[str] = field(default_factory=list)


def export_jquants_listed_master_csv(
    provider: _MetadataProvider,
    *,
    universe_file: str | Path,
    output_file: str | Path,
    report_file: str | Path,
    sleep_seconds: float = 0.5,
    allow_network: bool = False,
    endpoint_url_for_listed_info: str = "",
    api_key_status: str | None = None,
) -> ListedMasterExportResult:
    """Write one listed-master row per universe ticker plus a JSON run report."""
    if sleep_seconds < 0:
        raise DataValidationError("--sleep-seconds must be non-negative")
    universe = _read_universe(universe_file)
    if not universe:
        raise DataValidationError("universe file contains no ticker rows")

    api_key_state = api_key_status or _api_key_status()
    metadata_by_ticker, warnings = _load_metadata(provider, universe, allow_network)
    use_per_ticker = metadata_by_ticker is None
    output_rows = []
    matched_by_category: Counter[str] = Counter()
    missing_by_category: Counter[str] = Counter()
    missing_tickers: list[str] = []

    for index, entry in enumerate(universe):
        ticker = entry["ticker"]
        error = ""
        metadata = None
        if metadata_by_ticker is not None:
            metadata = metadata_by_ticker.get(ticker)
        else:
            try:
                metadata = provider.get_metadata(ticker)
            except ProviderError as exc:
                error = str(exc)
            if sleep_seconds and index < len(universe) - 1:
                time.sleep(sleep_seconds)

        matched = metadata is not None
        category = entry["new_index_category"]
        if matched:
            matched_by_category[category] += 1
        else:
            missing_by_category[category] += 1
            missing_tickers.append(ticker)
            if not error:
                error = "missing_metadata"

        source_metadata = metadata.source_metadata if metadata else {}
        output_rows.append(
            {
                "ticker": ticker,
                "name_universe": entry["name_universe"],
                "universe_date": entry["universe_date"],
                "new_index_category": category,
                "matched": "true" if matched else "false",
                "company_name": metadata.company_name if metadata else "",
                "sector": metadata.sector if metadata else "",
                "market": metadata.market if metadata else "",
                "source_metadata_json": json.dumps(
                    source_metadata, ensure_ascii=False, sort_keys=True
                ),
                "error": error,
                "raw_code": source_metadata.get("raw_code", ""),
                "company_name_en": source_metadata.get("company_name_en", ""),
                "sector_17": source_metadata.get("sector_17", ""),
                "sector_33": source_metadata.get("sector_33", ""),
            }
        )

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "universe_file": str(universe_file),
        "output_file": str(output_file),
        "universe_count": len(universe),
        "matched_count": len(universe) - len(missing_tickers),
        "missing_count": len(missing_tickers),
        "missing_tickers": missing_tickers,
        "matched_by_new_index_category": dict(sorted(matched_by_category.items())),
        "missing_by_new_index_category": dict(sorted(missing_by_category.items())),
        "endpoint_url_for_listed_info": endpoint_url_for_listed_info,
        "api_key_status": api_key_state,
        "secret_included": False,
    }
    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return ListedMasterExportResult(
        output_path=str(out_path),
        report_path=str(report_path),
        universe_count=len(universe),
        matched_count=report["matched_count"],
        missing_count=report["missing_count"],
        missing_tickers=missing_tickers,
        warnings=warnings + (["used per-ticker listed_info fetches"] if use_per_ticker else []),
    )


def _read_universe(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ticker", "name", "date", "new_index_category"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise DataValidationError(
                f"universe file missing required column(s): {', '.join(missing)}"
            )
        for row in reader:
            ticker = (row.get("ticker") or "").strip()
            if not ticker:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "name_universe": (row.get("name") or "").strip(),
                    "universe_date": (row.get("date") or "").strip(),
                    "new_index_category": (row.get("new_index_category") or "").strip(),
                }
            )
    return rows


def _load_metadata(
    provider: _MetadataProvider,
    universe: Sequence[dict[str, str]],
    allow_network: bool,
) -> tuple[dict[str, CompanyMetadata] | None, list[str]]:
    get_all = getattr(provider, "get_all_metadata", None)
    if get_all is None:
        return None, []
    try:
        all_metadata = get_all(allow_network=allow_network)
    except ProviderError as exc:
        if ENV_API_KEY in str(exc):
            return {}, [f"full listed_info master fetch unavailable: {exc}"]
        return None, [f"full listed_info master fetch unavailable; falling back: {exc}"]

    requested = {entry["ticker"] for entry in universe}
    filtered = {
        ticker: metadata for ticker, metadata in all_metadata.items() if ticker in requested
    }
    return filtered, []


def _api_key_status() -> str:
    return "PRESENT" if os.environ.get(ENV_API_KEY) else "MISSING"


__all__ = [
    "ListedMasterExportResult",
    "OUTPUT_COLUMNS",
    "export_jquants_listed_master_csv",
]

"""Enrich commercial-quality screening CSVs with display metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jp_stock_analysis.analysis.jquants_commercial_common import (
    ANALYSIS_DIR,
    LOGIC_VERSION,
    ensure_parent,
    jst_now_iso,
    write_json,
)
from jp_stock_analysis.analysis.jquants_commercial_quality_rankings import (
    DEFAULT_EXCLUDED_AUDIT,
    DEFAULT_NEGATIVE_REACTED,
    DEFAULT_NEGATIVE_REVERSAL,
    DEFAULT_POSITIVE_REACTED,
    DEFAULT_POSITIVE_UNREACTED,
    DEFAULT_UNCLASSIFIED_AUDIT,
    EVIDENCE_COLUMNS,
)


DEFAULT_SUMMARY = ANALYSIS_DIR / "commercial_quality_rankings_enriched_summary.json"

INPUT_OUTPUT_FILES = {
    "positive_reacted": (
        DEFAULT_POSITIVE_REACTED,
        ANALYSIS_DIR / "commercial_quality_positive_reacted_ranking_enriched.csv",
    ),
    "positive_unreacted": (
        DEFAULT_POSITIVE_UNREACTED,
        ANALYSIS_DIR / "commercial_quality_positive_unreacted_ranking_enriched.csv",
    ),
    "negative_reacted": (
        DEFAULT_NEGATIVE_REACTED,
        ANALYSIS_DIR / "commercial_quality_negative_reacted_ranking_enriched.csv",
    ),
    "negative_reversal": (
        DEFAULT_NEGATIVE_REVERSAL,
        ANALYSIS_DIR / "commercial_quality_negative_reversal_ranking_enriched.csv",
    ),
    "excluded_audit": (
        DEFAULT_EXCLUDED_AUDIT,
        ANALYSIS_DIR / "commercial_quality_rankings_excluded_audit_enriched.csv",
    ),
    "unclassified_audit": (
        DEFAULT_UNCLASSIFIED_AUDIT,
        ANALYSIS_DIR / "commercial_quality_rankings_unclassified_audit_enriched.csv",
    ),
}

ADDED_COLUMNS = [
    "company_name",
    "company_name_en",
    "market_segment",
    "market_code",
    "sector33_code",
    "sector33_name",
    "sector17_code",
    "sector17_name",
    "topix_size_category",
    "topix_category",
    "universe_name",
    "is_topix1000_universe",
    "metadata_source",
    "metadata_missing",
]

DISPLAY_PREFIX = [
    "rank",
    "ranking_label",
    "ranking_reason",
    "DiscDate",
    "ticker",
    "company_name",
    "company_name_en",
    "market_segment",
    "sector33_name",
    "topix_size_category",
    "classification_confidence",
    "manual_review_required",
    "product_acceptability",
    "evidence_summary",
    "horizon_flags",
    "liquidity_flags",
    "product_risk_flags",
    "liquidity_bucket",
    "event_time_bucket",
    "commercial_signal",
    "candidate_bucket",
    "abnormal_ret_1d",
    "ret_1d",
    "universe_ret_1d",
    "avg_turnover_20d",
    "turnover_spike_ratio",
    "reaction_base_date",
]

METADATA_COLUMN_ALIASES = {
    "ticker": ("Code", "code", "ticker", "LocalCode", "local_code"),
    "company_name": ("CoName", "CompanyName", "company_name", "name", "Name"),
    "company_name_en": ("CoNameEn", "CompanyNameEnglish", "company_name_en"),
    "market_code": ("Mkt", "MarketCode", "market_code"),
    "market_segment": ("MktNm", "MarketCodeName", "market_segment", "market_name"),
    "sector33_code": ("S33", "Sector33Code", "sector33_code"),
    "sector33_name": ("S33Nm", "Sector33CodeName", "sector33_name"),
    "sector17_code": ("S17", "Sector17Code", "sector17_code"),
    "sector17_name": ("S17Nm", "Sector17CodeName", "sector17_name"),
    "topix_size_category": ("ScaleCat", "ScaleCategory", "topix_size_category", "new_index_category"),
    "topix_category": ("TOPIXCategory", "topix_category"),
    "universe_name": ("universe_name",),
}

METADATA_CANDIDATES = [
    Path(".cache/jquants/listed_info/_all.json"),
    Path("data/jquants_price_store/listed_info.csv"),
    Path("data/jquants_price_store/listed_master.csv"),
    Path("analysis/jquants_fin_summary/listed_info.csv"),
    Path("data/jquants_price_store.local_backup.20260628_154817/topix1000_usable_tickers.csv"),
    Path("data/jquants_price_store.local_backup.20260628_154817/topix1000_tickers.csv"),
]

TOPIX_UNIVERSE_CANDIDATES = [
    Path("data/jquants_price_store/topix1000_usable_tickers.csv"),
    Path("data/jquants_price_store.local_backup.20260628_154817/topix1000_usable_tickers.csv"),
    Path("data/jquants_price_store/topix1000_tickers.csv"),
    Path("data/jquants_price_store.local_backup.20260628_154817/topix1000_tickers.csv"),
]


def normalize_ticker(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) == 5 and text.endswith("0") and text[:4].isdigit():
        text = text[:4]
    if text.isdigit() and len(text) < 4:
        text = text.zfill(4)
    return text


def _read_csv_fallback(path: Path, nrows: int | None = None) -> pd.DataFrame:
    last_exc: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=encoding, nrows=nrows)
        except UnicodeDecodeError as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    return pd.read_csv(path, nrows=nrows)


def _read_metadata_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "info", "listed_info"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"metadata JSON must be a list or contain a list payload: {path}")
        return pd.DataFrame(payload)
    return _read_csv_fallback(path)


def _pick_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    exact = {col: col for col in df.columns}
    lowered = {str(col).lower(): col for col in df.columns}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _metadata_score(df: pd.DataFrame) -> int:
    score = 0
    for column in ("ticker", "company_name", "sector33_name", "market_segment"):
        if _pick_column(df, METADATA_COLUMN_ALIASES[column]) is not None:
            score += 1
    return score


def detect_metadata_path() -> Path | None:
    existing: list[tuple[int, int, Path]] = []
    for index, path in enumerate(METADATA_CANDIDATES):
        if not path.exists():
            continue
        try:
            df = _read_metadata_file(path)
        except Exception:
            continue
        score = _metadata_score(df)
        if score > 0:
            existing.append((score, -index, path))
    if not existing:
        return None
    existing.sort(reverse=True)
    return existing[0][2]


def detect_topix_universe_path() -> Path | None:
    for path in TOPIX_UNIVERSE_CANDIDATES:
        if path.exists():
            return path
    return None


def _normalize_metadata(
    raw: pd.DataFrame, source_path: Path, topix_path: Path | None = None
) -> tuple[pd.DataFrame, list[str]]:
    missing_columns: list[str] = []
    out = pd.DataFrame()
    ticker_col = _pick_column(raw, METADATA_COLUMN_ALIASES["ticker"])
    if ticker_col is None:
        return pd.DataFrame(columns=["ticker", *ADDED_COLUMNS[:-1]]), [
            key for key in METADATA_COLUMN_ALIASES
        ]
    out["ticker"] = raw[ticker_col].map(normalize_ticker)
    for target in ADDED_COLUMNS:
        if target in {"is_topix1000_universe", "metadata_source", "metadata_missing"}:
            continue
        col = _pick_column(raw, METADATA_COLUMN_ALIASES.get(target, (target,)))
        if col is None:
            out[target] = np.nan
            missing_columns.append(target)
        else:
            out[target] = raw[col]
    out["metadata_source"] = str(source_path)
    out = out.dropna(subset=["ticker"])
    out = out[out["ticker"].astype(str).str.len() > 0]
    out = out.drop_duplicates("ticker", keep="first")

    topix_tickers: set[str] = set()
    if topix_path is not None and topix_path.exists():
        try:
            topix = _read_csv_fallback(topix_path)
            topix_col = _pick_column(topix, METADATA_COLUMN_ALIASES["ticker"])
            if topix_col is not None:
                topix_tickers = set(topix[topix_col].map(normalize_ticker))
            topix_size_col = _pick_column(topix, METADATA_COLUMN_ALIASES["topix_size_category"])
            if topix_col is not None and topix_size_col is not None:
                topix_size = topix[[topix_col, topix_size_col]].copy()
                topix_size["ticker"] = topix_size[topix_col].map(normalize_ticker)
                topix_size = topix_size.drop_duplicates("ticker", keep="first")
                size_map = topix_size.set_index("ticker")[topix_size_col]
                out["topix_size_category"] = out["topix_size_category"].fillna(
                    out["ticker"].map(size_map)
                )
        except Exception:
            pass
    out["is_topix1000_universe"] = out["ticker"].isin(topix_tickers) if topix_tickers else np.nan
    if "universe_name" in out.columns:
        out["universe_name"] = out["universe_name"].fillna(
            "topix1000" if topix_tickers else "listed_info"
        )
    return out, sorted(set(missing_columns))


def load_metadata(
    metadata_path: Path | None = None, topix_universe_path: Path | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = metadata_path or detect_metadata_path()
    if selected is None:
        return pd.DataFrame(columns=["ticker", *ADDED_COLUMNS]), {
            "metadata_input_path": None,
            "metadata_source_type": "missing",
            "metadata_rows": 0,
            "metadata_unique_tickers": 0,
            "missing_metadata_columns": ADDED_COLUMNS[:-1],
            "warnings": ["metadata source not found"],
        }
    raw = _read_metadata_file(selected)
    topix_path = topix_universe_path if topix_universe_path is not None else detect_topix_universe_path()
    metadata, missing_columns = _normalize_metadata(raw, selected, topix_path)
    return metadata, {
        "metadata_input_path": str(selected),
        "metadata_source_type": "listed_info" if "listed_info" in str(selected) else "csv_universe",
        "metadata_rows": int(len(raw)),
        "metadata_unique_tickers": int(metadata["ticker"].nunique()) if "ticker" in metadata else 0,
        "missing_metadata_columns": missing_columns,
        "warnings": [],
    }


def _ordered_columns(df: pd.DataFrame) -> list[str]:
    prefix = [col for col in DISPLAY_PREFIX if col in df.columns]
    rest = [col for col in df.columns if col not in prefix]
    return [*prefix, *rest]


def _enrich_file(input_path: Path, output_path: Path, metadata: pd.DataFrame) -> tuple[int, int, list[str]]:
    df = _read_csv_fallback(input_path)
    if "ticker" not in df.columns:
        df["ticker"] = ""
    df["ticker"] = df["ticker"].map(normalize_ticker)
    for col in EVIDENCE_COLUMNS:
        if col not in df.columns:
            if col in {"mixed_event_flag", "manual_review_required", "duplicate_event_flag"}:
                df[col] = False
            else:
                df[col] = ""
    merged = df.merge(metadata, on="ticker", how="left", suffixes=("", "_metadata"))
    for col in ADDED_COLUMNS:
        if col not in merged.columns and col != "metadata_missing":
            merged[col] = np.nan
    metadata_cols = [col for col in ADDED_COLUMNS if col != "metadata_missing"]
    merged["metadata_missing"] = merged[metadata_cols].isna().all(axis=1)
    missing_tickers = sorted(merged.loc[merged["metadata_missing"], "ticker"].dropna().unique())
    ensure_parent(output_path)
    merged[_ordered_columns(merged)].to_csv(output_path, index=False)
    return int(len(df)), int(merged["metadata_missing"].sum()), [str(t) for t in missing_tickers]


def build_commercial_quality_enriched_rankings(
    metadata_path: Path | None = None,
    topix_universe_path: Path | None = None,
    summary_path: Path = DEFAULT_SUMMARY,
    input_output_files: dict[str, tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    files = input_output_files or INPUT_OUTPUT_FILES
    metadata, meta_summary = load_metadata(metadata_path, topix_universe_path)
    per_file_rows: dict[str, int] = {}
    per_file_missing: dict[str, int] = {}
    output_files: dict[str, str] = {}
    input_files: dict[str, str] = {}
    missing_tickers: set[str] = set()
    warnings = list(meta_summary.get("warnings", []))
    total_rows = 0
    total_missing = 0

    for label, (input_path, output_path) in files.items():
        input_files[label] = str(input_path)
        output_files[label] = str(output_path)
        if not input_path.exists():
            warnings.append(f"input ranking CSV missing: {input_path}")
            per_file_rows[label] = 0
            per_file_missing[label] = 0
            continue
        rows, missing, tickers = _enrich_file(input_path, output_path, metadata)
        output_rows = len(_read_csv_fallback(output_path))
        if output_rows != rows:
            warnings.append(f"output row count mismatch: {output_path}")
        per_file_rows[label] = rows
        per_file_missing[label] = missing
        total_rows += rows
        total_missing += missing
        missing_tickers.update(tickers)

    missing_metadata_columns = list(meta_summary.get("missing_metadata_columns", []))
    if total_missing > 0:
        warnings.append("metadata_missing_rows > 0")
    for required in ("company_name", "sector33_name", "topix_size_category"):
        if required in missing_metadata_columns:
            warnings.append(f"{required} could not be attached from metadata source")

    summary = {
        "generated_at_jst": jst_now_iso(),
        "logic_version": LOGIC_VERSION,
        "metadata_input_path": meta_summary.get("metadata_input_path"),
        "metadata_source_type": meta_summary.get("metadata_source_type"),
        "metadata_rows": meta_summary.get("metadata_rows"),
        "metadata_unique_tickers": meta_summary.get("metadata_unique_tickers"),
        "input_files": input_files,
        "output_files": output_files,
        "per_file_rows": per_file_rows,
        "per_file_metadata_missing_rows": per_file_missing,
        "total_rows": total_rows,
        "total_metadata_missing_rows": total_missing,
        "metadata_missing_tickers": sorted(missing_tickers),
        "added_columns": ADDED_COLUMNS,
        "missing_metadata_columns": missing_metadata_columns,
        "warnings": warnings,
    }
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--topix-universe", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    build_commercial_quality_enriched_rankings(args.metadata, args.topix_universe, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

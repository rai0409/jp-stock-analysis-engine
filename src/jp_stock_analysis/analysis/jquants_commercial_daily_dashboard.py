"""Build daily dashboard CSV/JSON outputs from commercial-quality screening files."""

from __future__ import annotations

import argparse
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


DEFAULT_PRICE_STORE = Path("/mnt/d/jquants_price_store/prices_daily_bars.csv")
DEFAULT_FINANCIAL_SUMMARY = Path("/mnt/d/jquants_price_store/financial_summary_raw.csv")

INPUT_FILES = {
    "positive_reacted": "commercial_quality_positive_reacted_ranking_enriched.csv",
    "positive_unreacted": "commercial_quality_positive_unreacted_ranking_enriched.csv",
    "negative_reacted": "commercial_quality_negative_reacted_ranking_enriched.csv",
    "negative_reversal": "commercial_quality_negative_reversal_ranking_enriched.csv",
    "excluded": "commercial_quality_rankings_excluded_audit_enriched.csv",
    "unclassified": "commercial_quality_rankings_unclassified_audit_enriched.csv",
}

OUTPUT_FILES = {
    "dashboard_summary_csv": "commercial_daily_dashboard_summary.csv",
    "dashboard_summary_json": "commercial_daily_dashboard_summary.json",
    "latest_snapshot_json": "commercial_daily_latest_snapshot.json",
    "sector_summary_csv": "commercial_daily_sector_summary.csv",
    "market_segment_summary_csv": "commercial_daily_market_segment_summary.csv",
    "topix_size_summary_csv": "commercial_daily_topix_size_summary.csv",
    "audit_json": "commercial_daily_dashboard_audit.json",
}

SUMMARY_COLUMNS = [
    "DiscDate",
    "total_classified_rows",
    "positive_reacted_rows",
    "positive_unreacted_rows",
    "negative_reacted_rows",
    "negative_reversal_rows",
    "excluded_rows",
    "unclassified_rows",
    "all_dashboard_rows",
    "unique_tickers",
    "high_liquidity_rows",
    "mid_liquidity_rows",
    "low_liquidity_rows",
    "unknown_liquidity_rows",
    "avg_abnormal_ret_1d",
    "median_abnormal_ret_1d",
    "avg_ret_1d",
    "median_ret_1d",
    "sector33_count",
    "market_segment_count",
    "topix_size_category_count",
    "latest_price_date",
    "latest_financial_summary_disc_date",
    "generated_at_jst",
]

TOP_COLUMNS = [
    "ticker",
    "company_name",
    "DiscDate",
    "first_time",
    "sector33_name",
    "market_segment",
    "topix_size_category",
    "ranking_score",
    "classification_confidence",
    "manual_review_required",
    "manual_review_reason",
    "mixed_event_flag",
    "ranking_score_components",
    "ranking_reason",
    "evidence_summary",
    "horizon_flags",
    "liquidity_flags",
    "duplicate_event_flag",
    "product_risk_flags",
    "product_acceptability",
    "abnormal_ret_1d",
    "ret_1d",
    "liquidity_bucket",
    "candidate_bucket",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ticker": str})


def _empty_dashboard_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["DiscDate", "ticker", "dashboard_classification"])


def _latest_date(path: Path, column: str) -> str | None:
    if not path.exists():
        return None
    try:
        values = pd.read_csv(path, usecols=[column])[column]
    except Exception:
        return None
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().strftime("%Y-%m-%d")


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _load_inputs(analysis_dir: Path) -> tuple[pd.DataFrame, dict[str, str], dict[str, int], list[str], dict[str, list[str]]]:
    frames: list[pd.DataFrame] = []
    input_files: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    missing_files: list[str] = []
    missing_columns: dict[str, list[str]] = {}
    required = ["DiscDate", "ticker", "liquidity_bucket", "abnormal_ret_1d", "ret_1d"]
    for classification, filename in INPUT_FILES.items():
        path = analysis_dir / filename
        input_files[classification] = str(path)
        if not path.exists():
            missing_files.append(str(path))
            row_counts[classification] = 0
            continue
        df = _read_csv(path)
        row_counts[classification] = int(len(df))
        missing = [col for col in required if col not in df.columns]
        if missing:
            missing_columns[classification] = missing
        df = _ensure_columns(df, ["DiscDate", "ticker"])
        df["dashboard_classification"] = classification
        if classification in {"excluded", "unclassified"}:
            df["ranking_label"] = classification
        elif "ranking_label" not in df.columns:
            df["ranking_label"] = classification
        frames.append(df)
    if not frames:
        return _empty_dashboard_frame(), input_files, row_counts, missing_files, missing_columns
    all_columns = sorted({col for frame in frames for col in frame.columns})
    frames = [_ensure_columns(frame, all_columns) for frame in frames]
    dashboard = pd.concat(frames, ignore_index=True)
    dashboard["ticker"] = dashboard["ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    dashboard["DiscDate"] = pd.to_datetime(dashboard["DiscDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["abnormal_ret_1d", "ret_1d", "commercial_candidate_score", "avg_turnover_20d"]:
        if col in dashboard.columns:
            dashboard[col] = pd.to_numeric(dashboard[col], errors="coerce")
    if "ranking_score" not in dashboard.columns:
        dashboard["ranking_score"] = dashboard.get("commercial_candidate_score", np.nan)
    return dashboard, input_files, row_counts, missing_files, missing_columns


def _count_liquidity(group: pd.DataFrame, bucket: str) -> int:
    if "liquidity_bucket" not in group.columns:
        return 0
    return int((group["liquidity_bucket"].fillna("unknown").astype(str).str.lower() == bucket).sum())


def _numeric_series(group: pd.DataFrame, column: str) -> pd.Series:
    if column not in group.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(group[column], errors="coerce")


def _daily_summary(
    dashboard: pd.DataFrame,
    latest_price_date: str | None,
    latest_financial_summary_disc_date: str | None,
    generated_at: str,
) -> pd.DataFrame:
    if dashboard.empty or "DiscDate" not in dashboard.columns:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for disc_date, group in dashboard.dropna(subset=["DiscDate"]).groupby("DiscDate", sort=True):
        counts = group["dashboard_classification"].value_counts()
        classified = sum(
            int(counts.get(label, 0))
            for label in ["positive_reacted", "positive_unreacted", "negative_reacted", "negative_reversal"]
        )
        row = {
            "DiscDate": disc_date,
            "total_classified_rows": classified,
            "positive_reacted_rows": int(counts.get("positive_reacted", 0)),
            "positive_unreacted_rows": int(counts.get("positive_unreacted", 0)),
            "negative_reacted_rows": int(counts.get("negative_reacted", 0)),
            "negative_reversal_rows": int(counts.get("negative_reversal", 0)),
            "excluded_rows": int(counts.get("excluded", 0)),
            "unclassified_rows": int(counts.get("unclassified", 0)),
            "all_dashboard_rows": int(len(group)),
            "unique_tickers": int(group["ticker"].nunique()) if "ticker" in group.columns else 0,
            "high_liquidity_rows": _count_liquidity(group, "high"),
            "mid_liquidity_rows": _count_liquidity(group, "mid"),
            "low_liquidity_rows": _count_liquidity(group, "low"),
            "unknown_liquidity_rows": _count_liquidity(group, "unknown"),
            "avg_abnormal_ret_1d": _numeric_series(group, "abnormal_ret_1d").mean(),
            "median_abnormal_ret_1d": _numeric_series(group, "abnormal_ret_1d").median(),
            "avg_ret_1d": _numeric_series(group, "ret_1d").mean(),
            "median_ret_1d": _numeric_series(group, "ret_1d").median(),
            "sector33_count": int(group["sector33_name"].dropna().nunique()) if "sector33_name" in group.columns else 0,
            "market_segment_count": int(group["market_segment"].dropna().nunique()) if "market_segment" in group.columns else 0,
            "topix_size_category_count": int(group["topix_size_category"].dropna().nunique()) if "topix_size_category" in group.columns else 0,
            "latest_price_date": latest_price_date,
            "latest_financial_summary_disc_date": latest_financial_summary_disc_date,
            "generated_at_jst": generated_at,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _group_summary(dashboard: pd.DataFrame, by_col: str, include_liquidity: bool = False) -> pd.DataFrame:
    base_cols = ["DiscDate", "dashboard_classification", by_col]
    out_cols = [
        *base_cols,
        "rows",
        "unique_tickers",
        "avg_abnormal_ret_1d",
        "median_abnormal_ret_1d",
    ]
    if include_liquidity:
        out_cols.extend(["high_liquidity_rows", "mid_liquidity_rows"])
    if dashboard.empty or by_col not in dashboard.columns:
        return pd.DataFrame(columns=out_cols)
    rows: list[dict[str, Any]] = []
    frame = dashboard.dropna(subset=["DiscDate"]).copy()
    frame[by_col] = frame[by_col].fillna("unknown")
    for keys, group in frame.groupby(base_cols, dropna=False, sort=True):
        row = dict(zip(base_cols, keys, strict=True))
        row.update(
            {
                "rows": int(len(group)),
                "unique_tickers": int(group["ticker"].nunique()) if "ticker" in group.columns else 0,
                "avg_abnormal_ret_1d": _numeric_series(group, "abnormal_ret_1d").mean(),
                "median_abnormal_ret_1d": _numeric_series(group, "abnormal_ret_1d").median(),
            }
        )
        if include_liquidity:
            row["high_liquidity_rows"] = _count_liquidity(group, "high")
            row["mid_liquidity_rows"] = _count_liquidity(group, "mid")
        rows.append(row)
    return pd.DataFrame(rows, columns=out_cols)


def _top_counts(frame: pd.DataFrame, column: str, limit: int = 10) -> list[dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return []
    counts = frame[column].fillna("unknown").value_counts().head(limit)
    return [{"label": str(k), "rows": int(v)} for k, v in counts.items()]


def _top_records(frame: pd.DataFrame, classification: str, latest_top_n: int) -> list[dict[str, Any]]:
    sub = frame[frame["dashboard_classification"] == classification].copy()
    if sub.empty:
        return []
    sort_cols = [col for col in ["rank", "ranking_score", "abnormal_ret_1d"] if col in sub.columns]
    if "rank" in sort_cols:
        sub["rank"] = pd.to_numeric(sub["rank"], errors="coerce")
        sub = sub.sort_values(["rank"], na_position="last")
    elif sort_cols:
        sub = sub.sort_values(sort_cols, ascending=False, na_position="last")
    cols = [col for col in TOP_COLUMNS if col in sub.columns]
    return sub.head(latest_top_n)[cols].replace({np.nan: None}).to_dict(orient="records")


def build_daily_dashboard(
    analysis_dir: Path = ANALYSIS_DIR,
    price_store: Path = DEFAULT_PRICE_STORE,
    financial_summary: Path = DEFAULT_FINANCIAL_SUMMARY,
    output_dir: Path = ANALYSIS_DIR,
    latest_top_n: int = 20,
) -> dict[str, Any]:
    generated_at = jst_now_iso()
    dashboard, input_files, input_row_counts, missing_files, missing_columns = _load_inputs(analysis_dir)
    latest_price_date = _latest_date(price_store, "date")
    latest_financial_disc_date = _latest_date(financial_summary, "DiscDate")
    warnings: list[str] = []
    if missing_files:
        warnings.append("one or more input ranking CSVs are missing")
    if missing_columns:
        warnings.append("one or more input CSVs are missing optional dashboard columns")
    if latest_price_date is None:
        warnings.append("latest_price_date unavailable")
    if latest_financial_disc_date is None:
        warnings.append("latest_financial_summary_disc_date unavailable")

    summary = _daily_summary(dashboard, latest_price_date, latest_financial_disc_date, generated_at)
    sector = _group_summary(dashboard, "sector33_name", include_liquidity=True)
    market = _group_summary(dashboard, "market_segment")
    topix = _group_summary(dashboard, "topix_size_category")
    output_paths = {key: str(output_dir / name) for key, name in OUTPUT_FILES.items()}
    ensure_parent(output_dir / OUTPUT_FILES["dashboard_summary_csv"])
    summary.to_csv(output_dir / OUTPUT_FILES["dashboard_summary_csv"], index=False)
    sector.to_csv(output_dir / OUTPUT_FILES["sector_summary_csv"], index=False)
    market.to_csv(output_dir / OUTPUT_FILES["market_segment_summary_csv"], index=False)
    topix.to_csv(output_dir / OUTPUT_FILES["topix_size_summary_csv"], index=False)

    latest_disc_date = None if summary.empty else str(summary["DiscDate"].max())
    latest_frame = dashboard[dashboard["DiscDate"] == latest_disc_date].copy() if latest_disc_date else dashboard.iloc[0:0]
    latest_counts = (
        latest_frame["dashboard_classification"].value_counts().astype(int).to_dict()
        if not latest_frame.empty
        else {}
    )
    total_counts = (
        dashboard["dashboard_classification"].value_counts().astype(int).to_dict()
        if not dashboard.empty
        else {}
    )
    total_dashboard_rows = int(len(dashboard))
    total_unique_tickers = int(dashboard["ticker"].nunique()) if "ticker" in dashboard.columns else 0
    latest_date_total_rows = int(len(latest_frame))
    latest_date_unique_tickers = (
        int(latest_frame["ticker"].nunique()) if "ticker" in latest_frame.columns else 0
    )
    snapshot = {
        "generated_at_jst": generated_at,
        "latest_disc_date": latest_disc_date,
        "latest_price_date": latest_price_date,
        "latest_financial_summary_disc_date": latest_financial_disc_date,
        "latest_date_counts": {str(k): int(v) for k, v in latest_counts.items()},
        "latest_date_total_rows": latest_date_total_rows,
        "latest_date_unique_tickers": latest_date_unique_tickers,
        "total_dashboard_rows": total_dashboard_rows,
        "total_unique_tickers": total_unique_tickers,
        "total_counts_by_classification": {str(k): int(v) for k, v in total_counts.items()},
        "top_sectors_latest": _top_counts(latest_frame, "sector33_name"),
        "top_market_segments_latest": _top_counts(latest_frame, "market_segment"),
        "top_topix_size_categories_latest": _top_counts(latest_frame, "topix_size_category"),
        "latest_positive_reacted_top": _top_records(latest_frame, "positive_reacted", latest_top_n),
        "latest_positive_unreacted_top": _top_records(latest_frame, "positive_unreacted", latest_top_n),
        "latest_negative_reacted_top": _top_records(latest_frame, "negative_reacted", latest_top_n),
        "latest_negative_reversal_top": _top_records(latest_frame, "negative_reversal", latest_top_n),
        "warnings": warnings,
        "input_files": input_files,
        "output_files": output_paths,
    }
    audit = {
        "generated_at_jst": generated_at,
        "input_files": input_files,
        "output_files": output_paths,
        "input_row_counts": input_row_counts,
        "output_row_counts": {
            "dashboard_summary": int(len(summary)),
            "sector_summary": int(len(sector)),
            "market_segment_summary": int(len(market)),
            "topix_size_summary": int(len(topix)),
        },
        "missing_input_files": missing_files,
        "missing_columns": missing_columns,
        "warnings": warnings,
        "latest_price_date": latest_price_date,
        "latest_financial_summary_disc_date": latest_financial_disc_date,
        "latest_disc_date": latest_disc_date,
        "total_dashboard_rows": total_dashboard_rows,
        "total_unique_tickers": total_unique_tickers,
    }
    summary_json = {
        "generated_at_jst": generated_at,
        "logic_version": LOGIC_VERSION,
        "rows": int(len(summary)),
        "latest_disc_date": latest_disc_date,
        "latest_price_date": latest_price_date,
        "latest_financial_summary_disc_date": latest_financial_disc_date,
        "total_dashboard_rows": total_dashboard_rows,
        "total_unique_tickers": total_unique_tickers,
        "total_counts_by_classification": snapshot["total_counts_by_classification"],
        "warnings": warnings,
        "input_files": input_files,
        "output_files": output_paths,
    }
    write_json(output_dir / OUTPUT_FILES["dashboard_summary_json"], summary_json)
    write_json(output_dir / OUTPUT_FILES["latest_snapshot_json"], snapshot)
    write_json(output_dir / OUTPUT_FILES["audit_json"], audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--price-store", type=Path, default=DEFAULT_PRICE_STORE)
    parser.add_argument("--financial-summary", type=Path, default=DEFAULT_FINANCIAL_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--latest-top-n", type=int, default=20)
    args = parser.parse_args(argv)
    build_daily_dashboard(
        args.analysis_dir,
        args.price_store,
        args.financial_summary,
        args.output_dir,
        args.latest_top_n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

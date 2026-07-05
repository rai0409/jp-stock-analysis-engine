"""Read-only data access helpers for the commercial screening dashboard."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


CLASSIFICATION_FILES = {
    "positive_reacted": "commercial_quality_positive_reacted_ranking_enriched.csv",
    "positive_unreacted": "commercial_quality_positive_unreacted_ranking_enriched.csv",
    "negative_reacted": "commercial_quality_negative_reacted_ranking_enriched.csv",
    "negative_reversal": "commercial_quality_negative_reversal_ranking_enriched.csv",
    "excluded": "commercial_quality_rankings_excluded_audit_enriched.csv",
    "unclassified": "commercial_quality_rankings_unclassified_audit_enriched.csv",
}

GROUP_FILES = {
    "sector": "commercial_daily_sector_summary.csv",
    "market_segment": "commercial_daily_market_segment_summary.csv",
    "topix_size": "commercial_daily_topix_size_summary.csv",
}

DOWNLOADABLE_FILES = {
    "commercial_daily_dashboard_summary.csv",
    "commercial_daily_dashboard_summary.json",
    "commercial_daily_latest_snapshot.json",
    "commercial_daily_sector_summary.csv",
    "commercial_daily_market_segment_summary.csv",
    "commercial_daily_topix_size_summary.csv",
    "commercial_daily_dashboard_audit.json",
    "commercial_quality_positive_reacted_ranking_enriched.csv",
    "commercial_quality_positive_unreacted_ranking_enriched.csv",
    "commercial_quality_negative_reacted_ranking_enriched.csv",
    "commercial_quality_negative_reversal_ranking_enriched.csv",
    "commercial_quality_rankings_excluded_audit_enriched.csv",
    "commercial_quality_rankings_unclassified_audit_enriched.csv",
    "commercial_quality_rankings_enriched_summary.json",
    "commercial_quality_improvement_final_summary.json",
}

EVIDENCE_FIELDS = [
    "abnormal_return_adjustment_basis",
    "abnormal_return_adjustment_quality",
    "abnormal_return_adjustment_flags",
    "commercial_abnormal_reaction_score",
    "sector_adjusted_return_1d",
    "sector_adjusted_return_3d",
    "sector_adjusted_return_5d",
    "sector_adjusted_return_20d",
    "size_adjusted_return_1d",
    "size_adjusted_return_3d",
    "size_adjusted_return_5d",
    "size_adjusted_return_20d",
    "sector_size_adjusted_return_1d",
    "sector_size_adjusted_return_3d",
    "sector_size_adjusted_return_5d",
    "sector_size_adjusted_return_20d",
    "sector_peer_count_1d",
    "size_peer_count_1d",
    "sector_size_peer_count_1d",
    "insufficient_sector_peer_flag",
    "insufficient_size_peer_flag",
    "insufficient_sector_size_peer_flag",
    "classification_confidence",
    "event_classification_detail",
    "event_direction_final",
    "event_reaction_pattern",
    "event_confidence_tier",
    "classification_basis_level",
    "classification_evidence_count",
    "classification_limitation_count",
    "financial_disclosure_available",
    "financial_disclosure_join_status",
    "financial_disclosure_match_count",
    "financial_disclosure_doc_types",
    "financial_disclosure_discnos",
    "financial_statement_event_flag",
    "earnings_forecast_revision_event_flag",
    "dividend_forecast_revision_event_flag",
    "financial_disclosure_event_flags",
    "financial_disclosure_event_summary",
    "financial_disclosure_confidence",
    "financial_disclosure_limitations",
    "earnings_forecast_revision_direction_v2",
    "earnings_forecast_revision_strength_v2",
    "earnings_forecast_revision_components",
    "dividend_revision_direction_v2",
    "dividend_revision_strength_v2",
    "dividend_revision_components",
    "financial_statement_direction",
    "financial_statement_strength",
    "financial_statement_components",
    "fundamental_event_type",
    "fundamental_event_direction",
    "fundamental_event_strength",
    "fundamental_event_confidence",
    "fundamental_event_components",
    "fundamental_event_flags",
    "fundamental_evidence_summary",
    "earnings_surprise_direction",
    "earnings_surprise_strength",
    "guidance_revision_direction",
    "guidance_revision_strength",
    "dividend_revision_direction",
    "dividend_revision_strength",
    "buyback_signal_direction",
    "buyback_signal_strength",
    "capital_policy_signal_direction",
    "capital_policy_signal_strength",
    "one_off_factor_flag",
    "one_off_factor_reason",
    "fundamental_mixed_signal_flag",
    "fundamental_mixed_signal_reason",
    "manual_review_required",
    "manual_review_reason",
    "manual_review_category",
    "manual_review_priority",
    "manual_review_resolution_hint",
    "auto_classification_safe_flag",
    "mixed_event_flag",
    "mixed_event_reason",
    "mixed_event_severity",
    "mixed_event_components",
    "contradiction_flags",
    "ranking_score",
    "ranking_score_components",
    "ranking_reason",
    "evidence_summary",
    "horizon_flags",
    "liquidity_flags",
    "duplicate_event_flag",
    "product_risk_flags",
    "product_acceptability",
]


def load_json(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    p = Path(path)
    if not p.exists():
        return {}, [f"missing file: {p}"]
    try:
        return json.loads(p.read_text(encoding="utf-8")), []
    except Exception as exc:
        return {}, [f"failed to load json {p}: {exc}"]


def load_csv(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(), [f"missing file: {p}"]
    try:
        return pd.read_csv(p, dtype={"ticker": str}), []
    except Exception as exc:
        return pd.DataFrame(), [f"failed to load csv {p}: {exc}"]


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def safe_records_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    records = []
    for row in df.to_dict(orient="records"):
        records.append({str(k): _json_value(v) for k, v in row.items()})
    return records


def _paginate(df: pd.DataFrame, limit: int, offset: int) -> tuple[pd.DataFrame, int]:
    total = int(len(df))
    limit = max(0, min(int(limit), 1000))
    offset = max(0, int(offset))
    return df.iloc[offset : offset + limit].copy(), total


def _filter_date_range(
    df: pd.DataFrame, start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    if df.empty or "DiscDate" not in df.columns:
        return df
    out = df.copy()
    dates = pd.to_datetime(out["DiscDate"], errors="coerce")
    if start_date:
        out = out[dates >= pd.Timestamp(start_date)]
        dates = pd.to_datetime(out["DiscDate"], errors="coerce")
    if end_date:
        out = out[dates <= pd.Timestamp(end_date)]
    return out


def _filter_exact(df: pd.DataFrame, column: str, value: str | None) -> pd.DataFrame:
    if not value or df.empty:
        return df
    if column not in df.columns:
        return df.iloc[0:0].copy()
    return df[df[column].fillna("").astype(str) == str(value)]


def _search(df: pd.DataFrame, query: str | None) -> pd.DataFrame:
    if not query or df.empty:
        return df
    cols = [col for col in ["ticker", "company_name", "sector33_name", "market_segment"] if col in df]
    if not cols:
        return df
    needle = query.lower()
    mask = pd.Series(False, index=df.index)
    for col in cols:
        mask = mask | df[col].fillna("").astype(str).str.lower().str.contains(needle, regex=False)
    return df[mask]


def _sort(df: pd.DataFrame, sort: str | None, order: str | None) -> pd.DataFrame:
    if not sort or df.empty or sort not in df.columns:
        return df
    ascending = (order or "asc").lower() == "asc"
    return df.sort_values(sort, ascending=ascending, na_position="last")


def _response(df: pd.DataFrame, total: int, limit: int, offset: int, warnings: list[str], **extra):
    return {
        **extra,
        "rows": safe_records_for_json(df),
        "total": total,
        "limit": limit,
        "offset": offset,
        "warnings": warnings,
    }


def get_snapshot(analysis_dir: str | Path) -> tuple[dict[str, Any], list[str]]:
    return load_json(Path(analysis_dir) / "commercial_daily_latest_snapshot.json")


def get_audit(analysis_dir: str | Path) -> tuple[dict[str, Any], list[str]]:
    return load_json(Path(analysis_dir) / "commercial_daily_dashboard_audit.json")


def get_daily_summary(
    analysis_dir: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> dict[str, Any]:
    df, warnings = load_csv(Path(analysis_dir) / "commercial_daily_dashboard_summary.csv")
    df = _filter_date_range(df, start_date, end_date)
    page, total = _paginate(df, limit, offset)
    return _response(page, total, limit, offset, warnings)


def get_rankings(
    analysis_dir: str | Path,
    classification: str,
    *,
    q: str | None = None,
    sector: str | None = None,
    market_segment: str | None = None,
    topix_size_category: str | None = None,
    disc_date: str | None = None,
    limit: int = 100,
    offset: int = 0,
    sort: str | None = None,
    order: str | None = None,
) -> dict[str, Any] | None:
    if classification not in CLASSIFICATION_FILES:
        return None
    df, warnings = load_csv(Path(analysis_dir) / CLASSIFICATION_FILES[classification])
    for col in EVIDENCE_FIELDS:
        if col not in df.columns:
            if col in {"manual_review_required", "mixed_event_flag", "duplicate_event_flag"}:
                df[col] = False
            else:
                df[col] = None
    df = _search(df, q)
    df = _filter_exact(df, "sector33_name", sector)
    df = _filter_exact(df, "market_segment", market_segment)
    df = _filter_exact(df, "topix_size_category", topix_size_category)
    df = _filter_exact(df, "DiscDate", disc_date)
    df = _sort(df, sort, order)
    page, total = _paginate(df, limit, offset)
    return _response(page, total, limit, offset, warnings, classification=classification)


def get_group_summary(
    analysis_dir: str | Path,
    kind: str,
    *,
    classification: str | None = None,
    disc_date: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> dict[str, Any] | None:
    if kind not in GROUP_FILES:
        return None
    df, warnings = load_csv(Path(analysis_dir) / GROUP_FILES[kind])
    df = _filter_exact(df, "dashboard_classification", classification)
    df = _filter_exact(df, "DiscDate", disc_date)
    page, total = _paginate(df, limit, offset)
    return _response(page, total, limit, offset, warnings, kind=kind)


def list_downloadable_files(analysis_dir: str | Path) -> list[dict[str, Any]]:
    base = Path(analysis_dir)
    rows = []
    for name in sorted(DOWNLOADABLE_FILES):
        path = base / name
        rows.append({"name": name, "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0})
    return rows


def resolve_download_file(analysis_dir: str | Path, name: str) -> Path | None:
    if "/" in name or "\\" in name or ".." in name:
        return None
    if name not in DOWNLOADABLE_FILES:
        return None
    path = Path(analysis_dir) / name
    return path if path.exists() else None

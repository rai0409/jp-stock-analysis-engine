"""Audit whether current J-Quants commercial analysis outputs are product-ready.

This module is intentionally read-only with respect to raw J-Quants inputs. It
reviews the existing analysis code/output shape and writes separate audit
artifacts under the requested output directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jp_stock_analysis.analysis.jquants_commercial_common import jst_now_iso, to_jsonable


DEFAULT_FINANCIAL_SUMMARY = Path("/mnt/d/jquants_price_store/financial_summary_raw.csv")
DEFAULT_PRICES = Path("/mnt/d/jquants_price_store/prices_daily_bars.csv")
DEFAULT_ANALYSIS_DIR = Path("analysis/jquants_fin_summary")

OUTPUT_FILES = {
    "report_json": "commercial_analysis_validity_report.json",
    "report_md": "commercial_analysis_validity_report.md",
    "code_review": "commercial_analysis_code_review_findings.csv",
    "classification": "commercial_event_classification_validation.csv",
    "price_reaction": "commercial_price_reaction_validation.csv",
    "abnormal_return": "commercial_abnormal_return_validation.csv",
    "liquidity": "commercial_liquidity_validation.csv",
    "ranking": "commercial_ranking_validation.csv",
    "signal": "commercial_signal_performance_by_bucket.csv",
    "manual_review": "commercial_manual_review_top_candidates.csv",
    "critical_issues": "commercial_analysis_critical_issues.csv",
    "scorecard": "commercial_product_readiness_scorecard.csv",
}

RANKING_FILES = {
    "positive_reacted": "commercial_quality_positive_reacted_ranking_enriched.csv",
    "positive_unreacted": "commercial_quality_positive_unreacted_ranking_enriched.csv",
    "negative_reacted": "commercial_quality_negative_reacted_ranking_enriched.csv",
    "negative_reversal": "commercial_quality_negative_reversal_ranking_enriched.csv",
}
AUDIT_FILES = {
    "excluded": "commercial_quality_rankings_excluded_audit_enriched.csv",
    "unclassified": "commercial_quality_rankings_unclassified_audit_enriched.csv",
}

CODE_REVIEW_COLUMNS = [
    "module",
    "function_or_area",
    "finding_type",
    "finding",
    "severity",
    "commercial_impact",
    "recommended_fix",
    "evidence",
]
CLASSIFICATION_COLUMNS = [
    "ticker",
    "DiscDate",
    "DocType",
    "ranking_file",
    "classification",
    "classification_confidence",
    "classification_reason",
    "event_classification_detail",
    "event_direction_final",
    "event_reaction_pattern",
    "event_confidence_tier",
    "classification_basis_level",
    "classification_evidence_count",
    "classification_limitation_count",
    "fundamental_event_type",
    "fundamental_event_direction",
    "fundamental_event_strength",
    "fundamental_event_confidence",
    "fundamental_mixed_signal_flag",
    "fundamental_mixed_signal_reason",
    "fundamental_event_flags",
    "mixed_event_flag",
    "mixed_event_severity",
    "manual_review_category",
    "manual_review_priority",
    "auto_classification_safe_flag",
    "classification_issue",
    "commercial_acceptability",
    "manual_review_required",
]
PRICE_COLUMNS = [
    "ticker",
    "DiscDate",
    "classification",
    "reaction_base_date",
    "reaction_start_date",
    "horizon_1d_available",
    "horizon_3d_available",
    "horizon_5d_available",
    "horizon_20d_available",
    "ret_1d_validity",
    "ret_3d_validity",
    "ret_5d_validity",
    "ret_20d_validity",
    "price_reaction_issue",
    "commercial_acceptability",
]
ABNORMAL_COLUMNS = [
    "ticker",
    "DiscDate",
    "classification",
    "raw_return_columns_detected",
    "abnormal_return_columns_detected",
    "market_adjustment_detected",
    "sector_adjustment_detected",
    "size_adjustment_detected",
    "abnormal_return_issue",
    "commercial_acceptability",
    "recommended_fix",
]
LIQUIDITY_COLUMNS = [
    "ticker",
    "DiscDate",
    "classification",
    "volume_fields_detected",
    "turnover_fields_detected",
    "adv_fields_detected",
    "liquidity_bucket",
    "liquidity_issue",
    "low_liquidity_in_top_rank_flag",
    "commercial_acceptability",
    "recommended_fix",
]
RANKING_COLUMNS = [
    "ranking_file",
    "ticker",
    "DiscDate",
    "rank",
    "classification",
    "ranking_score",
    "score_explainable",
    "score_components_detected",
    "duplicate_event_flag",
    "insufficient_horizon_flag",
    "low_liquidity_flag",
    "missing_reason_flag",
    "ranking_issue",
    "commercial_acceptability",
    "manual_review_required",
]
SIGNAL_COLUMNS = [
    "classification",
    "bucket",
    "count",
    "mean_ret_1d",
    "median_ret_1d",
    "hit_rate_1d",
    "mean_ret_3d",
    "median_ret_3d",
    "hit_rate_3d",
    "mean_ret_5d",
    "median_ret_5d",
    "hit_rate_5d",
    "mean_abnormal_ret_1d",
    "mean_abnormal_ret_3d",
    "rank_score_to_future_return_corr",
    "commercial_signal_assessment",
]
MANUAL_COLUMNS = [
    "review_priority",
    "ranking_file",
    "rank",
    "ticker",
    "DiscDate",
    "DocType",
    "classification",
    "company_name_if_available",
    "ranking_score",
    "primary_reason_detected",
    "risk_flags_detected",
    "data_quality_flags",
    "horizon_flags",
    "liquidity_flags",
    "analyst_review_questions",
    "commercial_risk_if_wrong",
    "source_rows_hint",
]
CRITICAL_COLUMNS = [
    "issue_id",
    "issue_type",
    "severity",
    "affected_module",
    "affected_file",
    "affected_rows_count",
    "description",
    "commercial_impact",
    "recommended_fix",
    "blocks_product_readiness",
]
SCORECARD_COLUMNS = ["area", "score", "max_score", "status", "evidence"]


def _read_csv(path: Path, *, dtype_ticker: bool = True) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    kwargs: dict[str, Any] = {}
    if dtype_ticker:
        kwargs["dtype"] = {"ticker": str, "Code": str}
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_csv(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    out = out[columns].replace({pd.NaT: np.nan})
    out.to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _json_clean(to_jsonable(payload))
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_clean(v) for v in value]
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _normalize_ticker_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True).str.zfill(4)


def _ensure_ticker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ticker" not in out.columns and "Code" in out.columns:
        out["ticker"] = out["Code"]
    if "ticker" in out.columns:
        out["ticker"] = _normalize_ticker_series(out["ticker"])
    return out


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _truthy(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _ranking_score(row: pd.Series) -> float | None:
    for col in ("ranking_score", "commercial_candidate_score", "commercial_revision_score"):
        if col in row.index:
            value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if not pd.isna(value):
                return float(value)
    return None


def _load_ranking_frames(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for label, filename in {**RANKING_FILES, **AUDIT_FILES}.items():
        frames[label] = _ensure_ticker(_read_csv(analysis_dir / filename))
    return frames


def _combined_rankings(frames: dict[str, pd.DataFrame], include_audits: bool = False) -> pd.DataFrame:
    labels = list(RANKING_FILES)
    if include_audits:
        labels.extend(AUDIT_FILES)
    parts = []
    for label in labels:
        df = frames.get(label, pd.DataFrame()).copy()
        if df.empty:
            continue
        df["ranking_file"] = label
        df["classification"] = label
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def _financial_lookup(financial_summary: pd.DataFrame) -> pd.DataFrame:
    fs = _ensure_ticker(financial_summary)
    if fs.empty or "DiscDate" not in fs.columns:
        return pd.DataFrame(columns=["ticker", "DiscDate", "DocType", "raw_doc_count", "raw_mixed_doctype_flag"])
    fs["DiscDate"] = pd.to_datetime(fs["DiscDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "DocType" not in fs.columns:
        fs["DocType"] = ""
    grouped = (
        fs.groupby(["ticker", "DiscDate"], dropna=False)
        .agg(
            DocType=("DocType", lambda s: "|".join(sorted({str(v) for v in s.dropna() if str(v)}))),
            raw_doc_count=("DocType", "size"),
            raw_unique_doctypes=("DocType", lambda s: int(s.dropna().nunique())),
        )
        .reset_index()
    )
    grouped["raw_mixed_doctype_flag"] = grouped["raw_unique_doctypes"] > 1
    return grouped


def _duplicate_keys(financial_summary: pd.DataFrame, extraction_duplicates: pd.DataFrame) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not extraction_duplicates.empty:
        text_cols = [col for col in extraction_duplicates.columns if col in {"key_value", "ticker", "DiscDate"}]
        if {"ticker", "DiscDate"}.issubset(extraction_duplicates.columns):
            dup = _ensure_ticker(extraction_duplicates)
            for _, row in dup.iterrows():
                keys.add((str(row.get("ticker")), str(row.get("DiscDate"))[:10]))
        elif "key_value" in text_cols:
            for value in extraction_duplicates["key_value"].dropna():
                text = str(value)
                parts = text.replace("(", "").replace(")", "").replace("'", "").split(",")
                if len(parts) >= 2:
                    ticker = "".join(ch for ch in parts[0] if ch.isdigit()).zfill(4)
                    keys.add((ticker, parts[1].strip()[:10]))
    fs = _ensure_ticker(financial_summary)
    if {"ticker", "DiscDate", "DocType"}.issubset(fs.columns):
        fs["DiscDate"] = pd.to_datetime(fs["DiscDate"], errors="coerce").dt.strftime("%Y-%m-%d")
        counts = fs.groupby(["ticker", "DiscDate", "DocType"], dropna=False).size()
        for ticker, date, _doctype in counts[counts > 1].index:
            keys.add((str(ticker), str(date)))
    return keys


def build_code_review_findings() -> pd.DataFrame:
    rows = [
        {
            "module": "jquants_price_reaction_v2.py",
            "function_or_area": "_future_return/build_price_reaction_v2",
            "finding_type": "calculation",
            "finding": "Returns are computed from adjusted close at reaction base date to adjusted close N trading rows later.",
            "severity": "info",
            "commercial_impact": "Adjusted close basis is appropriate, but intraday/open reaction is diagnostic only.",
            "recommended_fix": "Document close-to-close convention and event-time assumptions in product UI.",
            "evidence": "Uses base_adj_close and future adj_close; after_close/unknown_time starts at next trading day.",
        },
        {
            "module": "jquants_price_reaction_v2.py",
            "function_or_area": "event_time_bucket",
            "finding_type": "assumption",
            "finding": "Unknown disclosure time is treated like after-close and shifted to next trading day.",
            "severity": "medium",
            "commercial_impact": "Conservative but may understate same-day reactions for time-missing releases.",
            "recommended_fix": "Expose event_time_bucket and reaction_base_reason to users and add time-quality flags.",
            "evidence": "bucket in {'after_close', 'unknown_time'} uses dates > event_date.",
        },
        {
            "module": "jquants_abnormal_return.py",
            "function_or_area": "_build_universe_returns",
            "finding_type": "methodology",
            "finding": "Abnormal return is stock return minus equal-weighted available ticker universe return.",
            "severity": "high",
            "commercial_impact": "Market-wide adjustment exists, but sector, size and beta effects are not neutralized.",
            "recommended_fix": "Add sector/size/beta adjusted abnormal returns before commercial ranking claims.",
            "evidence": "Pivot by ticker adj_close; abnormal = ret_col - universe_ret.",
        },
        {
            "module": "jquants_liquidity.py",
            "function_or_area": "build_liquidity_features",
            "finding_type": "risk_control",
            "finding": "Liquidity uses prior 20 trading days of turnover and volume, with low bucket excluded downstream.",
            "severity": "info",
            "commercial_impact": "Turnover-based liquidity is present; no ADV60 or capacity/slippage model exists.",
            "recommended_fix": "Add ADV60, median spread proxy if available, and capacity flags for institutional use.",
            "evidence": "avg_turnover_20d, median_turnover_20d, avg_volume_20d are produced.",
        },
        {
            "module": "jquants_commercial_quality_rankings.py",
            "function_or_area": "_classification_masks/_sort_and_rank",
            "finding_type": "ranking",
            "finding": "Ranking classes are determined by commercial_signal direction and 1d abnormal return sign.",
            "severity": "high",
            "commercial_impact": "Classification is mechanically explainable but not a validated event taxonomy.",
            "recommended_fix": "Separate event-type classification from post-event price-response labels and persist score components.",
            "evidence": "positive & abnormal_ret_1d > 0, positive & abnormal_ret_1d < 0, negative analogues.",
        },
        {
            "module": "jquants_commercial_quality_rankings.py",
            "function_or_area": "ELIGIBLE_PRICE_STATUSES",
            "finding_type": "horizon",
            "finding": "Rows with 1d or 5d availability can enter rankings even when 20d is unavailable.",
            "severity": "medium",
            "commercial_impact": "Short-horizon rankings are usable, but 20d fields must be displayed as unavailable and not scored.",
            "recommended_fix": "Make horizon-specific eligibility explicit in ranking metadata and dashboard risk flags.",
            "evidence": "Eligible statuses include partial_5d_available and partial_1d_available.",
        },
        {
            "module": "commercial_dashboard_data.py",
            "function_or_area": "DOWNLOADABLE_FILES/TOP_COLUMNS",
            "finding_type": "explainability",
            "finding": "Dashboard exposes ranking rows and audit files but not this validity audit by default.",
            "severity": "medium",
            "commercial_impact": "Users may see rankings without full methodology/risk audit context.",
            "recommended_fix": "Add validity report artifacts to dashboard downloads after review.",
            "evidence": "Download allow-list contains quality outputs, not commercial_analysis_validity_* outputs.",
        },
    ]
    return pd.DataFrame(rows, columns=CODE_REVIEW_COLUMNS)


def validate_event_classification(
    combined: pd.DataFrame, financial_lookup: pd.DataFrame
) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame(columns=CLASSIFICATION_COLUMNS)
    df = combined.copy()
    df["DiscDate"] = pd.to_datetime(df.get("DiscDate"), errors="coerce").dt.strftime("%Y-%m-%d")
    merged = df.merge(financial_lookup, on=["ticker", "DiscDate"], how="left", suffixes=("", "_raw"))
    rows = []
    for _, row in merged.iterrows():
        label = _safe_text(row.get("ranking_file"))
        signal = _safe_text(row.get("commercial_signal")).lower()
        bucket = _safe_text(row.get("candidate_bucket")).lower()
        doctypes = _safe_text(row.get("doctypes") or row.get("DocType"))
        raw_doc = _safe_text(row.get("DocType"))
        mixed = "mixed" in signal or "mixed" in bucket or bool(row.get("raw_mixed_doctype_flag", False))
        positive = "upward" in signal or "increase" in signal or "positive" in bucket
        negative = "downward" in signal or "decrease" in signal or "negative" in bucket
        expected_positive = label.startswith("positive")
        expected_negative = label.startswith("negative")
        issue = ""
        persisted_confidence = pd.to_numeric(pd.Series([row.get("classification_confidence")]), errors="coerce").iloc[0]
        confidence = 0.9 if pd.isna(persisted_confidence) else float(persisted_confidence)
        tier = _safe_text(row.get("event_confidence_tier")) or ("high" if confidence >= 0.82 else "medium" if confidence >= 0.68 else "low")
        basis = _safe_text(row.get("classification_basis_level") or row.get("abnormal_return_adjustment_basis"))
        fundamental_direction = _safe_text(row.get("fundamental_event_direction"))
        fundamental_confidence = _safe_text(row.get("fundamental_event_confidence"))
        fundamental_mixed = _truthy(row.get("fundamental_mixed_signal_flag"))
        manual_category = _safe_text(row.get("manual_review_category"))
        manual_priority = _safe_text(row.get("manual_review_priority"))
        auto_safe = _truthy(row.get("auto_classification_safe_flag"))
        mixed_severity = _safe_text(row.get("mixed_event_severity")) or ("medium" if mixed else "none")
        if label == "unclassified":
            issue = _safe_text(row.get("unclassified_reason")) or "not_in_product_ranking"
            confidence = min(confidence, 0.4)
        elif label == "excluded":
            issue = _safe_text(row.get("exclusion_reasons")) or "excluded_from_ranking"
            confidence = min(confidence, 0.5)
        elif fundamental_mixed or fundamental_direction == "mixed":
            issue = "mixed_fundamental_event_requires_manual_review"
            confidence = min(confidence, 0.55)
        elif fundamental_direction == "insufficient_evidence" or fundamental_confidence == "not_evaluable":
            issue = "insufficient_fundamental_evidence"
            confidence = min(confidence, 0.62)
        elif mixed:
            issue = "mixed_event_requires_manual_review"
            confidence = min(confidence, 0.55)
        elif expected_positive and not positive:
            issue = "positive_ranking_without_positive_signal"
            confidence = min(confidence, 0.2)
        elif expected_negative and not negative:
            issue = "negative_ranking_without_negative_signal"
            confidence = min(confidence, 0.2)
        elif positive and negative:
            issue = "both_positive_and_negative_terms_detected"
            confidence = min(confidence, 0.45)
        if tier == "high" and basis == "universe_adjusted_only":
            issue = "universe_only_high_confidence"
        elif tier in {"manual_review", "not_evaluable"} and not issue:
            issue = "classification_requires_manual_review"
        elif manual_priority == "high" and not issue:
            issue = "high_priority_manual_review"
        accept = "acceptable" if not issue and confidence >= 0.75 and auto_safe else "manual_review"
        reason_parts = [
            f"signal={signal or 'missing'}",
            f"bucket={bucket or 'missing'}",
            f"doctypes={doctypes or raw_doc or 'missing'}",
            f"tier={tier or 'missing'}",
            f"basis={basis or 'missing'}",
            f"fundamental_direction={fundamental_direction or 'missing'}",
            f"fundamental_confidence={fundamental_confidence or 'missing'}",
        ]
        rows.append(
            {
                "ticker": row.get("ticker"),
                "DiscDate": row.get("DiscDate"),
                "DocType": raw_doc or doctypes,
                "ranking_file": label,
                "classification": row.get("classification"),
                "classification_confidence": confidence,
                "classification_reason": "; ".join(reason_parts),
                "event_classification_detail": row.get("event_classification_detail"),
                "event_direction_final": row.get("event_direction_final"),
                "event_reaction_pattern": row.get("event_reaction_pattern"),
                "event_confidence_tier": tier,
                "classification_basis_level": basis,
                "classification_evidence_count": row.get("classification_evidence_count"),
                "classification_limitation_count": row.get("classification_limitation_count"),
                "fundamental_event_type": row.get("fundamental_event_type"),
                "fundamental_event_direction": fundamental_direction,
                "fundamental_event_strength": row.get("fundamental_event_strength"),
                "fundamental_event_confidence": fundamental_confidence,
                "fundamental_mixed_signal_flag": fundamental_mixed,
                "fundamental_mixed_signal_reason": row.get("fundamental_mixed_signal_reason"),
                "fundamental_event_flags": row.get("fundamental_event_flags"),
                "mixed_event_flag": bool(mixed),
                "mixed_event_severity": mixed_severity,
                "manual_review_category": manual_category,
                "manual_review_priority": manual_priority,
                "auto_classification_safe_flag": auto_safe,
                "classification_issue": issue,
                "commercial_acceptability": accept,
                "manual_review_required": bool(issue or confidence < 0.75 or tier in {"manual_review", "not_evaluable"}),
            }
        )
    return pd.DataFrame(rows, columns=CLASSIFICATION_COLUMNS)


def validate_price_reaction(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    rows = []
    for _, row in combined.iterrows():
        issues = []
        validity: dict[int, str] = {}
        available: dict[int, bool] = {}
        for h in (1, 3, 5, 20):
            status = _safe_text(row.get(f"ret_{h}d_status"))
            ret = pd.to_numeric(pd.Series([row.get(f"ret_{h}d")]), errors="coerce").iloc[0]
            ok = status == "valid" and not pd.isna(ret)
            available[h] = bool(ok)
            validity[h] = "valid" if ok else (status or "missing_status")
            if h in (1, 3, 5) and not ok:
                issues.append(f"{h}d_return_unavailable")
        if not available[20]:
            issues.append("20d_return_unavailable")
        base = row.get("reaction_base_date")
        if pd.isna(base):
            issues.append("missing_reaction_base_date")
        issue_text = "|".join(sorted(set(issues)))
        accept = "acceptable" if not any(i for i in issues if i != "20d_return_unavailable") else "not_acceptable"
        rows.append(
            {
                "ticker": row.get("ticker"),
                "DiscDate": row.get("DiscDate"),
                "classification": row.get("classification"),
                "reaction_base_date": base,
                "reaction_start_date": base,
                "horizon_1d_available": available[1],
                "horizon_3d_available": available[3],
                "horizon_5d_available": available[5],
                "horizon_20d_available": available[20],
                "ret_1d_validity": validity[1],
                "ret_3d_validity": validity[3],
                "ret_5d_validity": validity[5],
                "ret_20d_validity": validity[20],
                "price_reaction_issue": issue_text,
                "commercial_acceptability": accept,
            }
        )
    return pd.DataFrame(rows, columns=PRICE_COLUMNS)


def validate_abnormal_return(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame(columns=ABNORMAL_COLUMNS)
    raw_cols = [col for col in ["ret_1d", "ret_3d", "ret_5d", "ret_20d"] if col in combined.columns]
    abnormal_cols = [col for col in ["abnormal_ret_1d", "abnormal_ret_3d", "abnormal_ret_5d", "abnormal_ret_20d"] if col in combined.columns]
    market_detected = any(col.startswith("universe_ret_") for col in combined.columns)
    sector_detected = any("sector_ret" in col or "sector_adjust" in col or "sector_adjusted" in col for col in combined.columns)
    size_detected = any("size_ret" in col or "size_adjust" in col or "size_adjusted" in col or "topix_size_ret" in col for col in combined.columns)
    rows = []
    for _, row in combined.iterrows():
        status = _safe_text(row.get("abnormal_ret_1d_status"))
        value = pd.to_numeric(pd.Series([row.get("abnormal_ret_1d")]), errors="coerce").iloc[0]
        issues = []
        if not abnormal_cols:
            issues.append("missing_abnormal_return_columns")
        if not market_detected:
            issues.append("market_adjustment_not_detected")
        if not sector_detected:
            issues.append("sector_adjustment_not_detected")
        if not size_detected:
            issues.append("size_adjustment_not_detected")
        if status and status != "valid":
            issues.append(f"abnormal_ret_1d_status={status}")
        if pd.isna(value):
            issues.append("abnormal_ret_1d_missing")
        basis = _safe_text(row.get("abnormal_return_adjustment_basis"))
        quality = _safe_text(row.get("abnormal_return_adjustment_quality"))
        if basis in {"sector_size_adjusted", "sector_and_size_adjusted", "sector_adjusted", "size_adjusted"} and quality in {"acceptable", "acceptable_with_limitations"}:
            accept = "acceptable"
        else:
            accept = "acceptable_with_limitations" if market_detected and not pd.isna(value) else "not_acceptable"
        rows.append(
            {
                "ticker": row.get("ticker"),
                "DiscDate": row.get("DiscDate"),
                "classification": row.get("classification"),
                "raw_return_columns_detected": "|".join(raw_cols),
                "abnormal_return_columns_detected": "|".join(abnormal_cols),
                "market_adjustment_detected": bool(market_detected),
                "sector_adjustment_detected": bool(sector_detected),
                "size_adjustment_detected": bool(size_detected),
                "abnormal_return_issue": "|".join(issues),
                "commercial_acceptability": accept,
                "recommended_fix": "Add sector/size adjusted abnormal returns and expose basis in ranking explanation.",
            }
        )
    return pd.DataFrame(rows, columns=ABNORMAL_COLUMNS)


def abnormal_return_v2_summary(combined: pd.DataFrame) -> dict[str, Any]:
    if combined.empty:
        return {
            "sector_adjusted_coverage_ratio": 0.0,
            "size_adjusted_coverage_ratio": 0.0,
            "sector_size_adjusted_coverage_ratio": 0.0,
            "robust_adjusted_score_coverage_ratio": 0.0,
            "insufficient_sector_peer_rows": 0,
            "insufficient_size_peer_rows": 0,
            "insufficient_sector_size_peer_rows": 0,
            "abnormal_return_adjustment_quality_counts": {},
            "abnormal_return_adjustment_basis_counts": {},
            "universe_adjusted_only_rows": 0,
            "adjustment_unavailable_rows": 0,
        }
    total = len(combined)

    def coverage(col: str) -> float:
        if col not in combined.columns:
            return 0.0
        return float(pd.to_numeric(combined[col], errors="coerce").notna().sum() / total)

    def flag_count(col: str) -> int:
        if col not in combined.columns:
            return 0
        return int(combined[col].map(_truthy).sum())

    basis = combined["abnormal_return_adjustment_basis"] if "abnormal_return_adjustment_basis" in combined.columns else pd.Series(dtype=object)
    quality = combined["abnormal_return_adjustment_quality"] if "abnormal_return_adjustment_quality" in combined.columns else pd.Series(dtype=object)
    return {
        "sector_adjusted_coverage_ratio": coverage("sector_adjusted_return_1d"),
        "size_adjusted_coverage_ratio": coverage("size_adjusted_return_1d"),
        "sector_size_adjusted_coverage_ratio": coverage("sector_size_adjusted_return_1d"),
        "robust_adjusted_score_coverage_ratio": coverage("commercial_abnormal_reaction_score"),
        "insufficient_sector_peer_rows": flag_count("insufficient_sector_peer_flag"),
        "insufficient_size_peer_rows": flag_count("insufficient_size_peer_flag"),
        "insufficient_sector_size_peer_rows": flag_count("insufficient_sector_size_peer_flag"),
        "abnormal_return_adjustment_quality_counts": quality.fillna("missing").value_counts(dropna=False).to_dict(),
        "abnormal_return_adjustment_basis_counts": basis.fillna("missing").value_counts(dropna=False).to_dict(),
        "universe_adjusted_only_rows": int((basis == "universe_adjusted_only").sum()) if not basis.empty else 0,
        "adjustment_unavailable_rows": int((basis == "adjustment_unavailable").sum()) if not basis.empty else 0,
    }


def validate_liquidity(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame(columns=LIQUIDITY_COLUMNS)
    volume_cols = [col for col in combined.columns if "volume" in col.lower()]
    turnover_cols = [col for col in combined.columns if "turnover" in col.lower()]
    adv_cols = [col for col in combined.columns if col in {"avg_volume_20d", "avg_turnover_20d", "median_turnover_20d"}]
    rows = []
    for _, row in combined.iterrows():
        rank = pd.to_numeric(pd.Series([row.get("rank")]), errors="coerce").iloc[0]
        bucket = _safe_text(row.get("liquidity_bucket")).lower() or "unknown"
        liquid_enough = _truthy(row.get("is_liquid_enough"))
        low_top = bucket in {"low", "unknown"} and (pd.isna(rank) or rank <= 50)
        issues = []
        if not turnover_cols:
            issues.append("turnover_fields_missing")
        if "avg_turnover_20d" not in combined.columns:
            issues.append("adv20_turnover_missing")
        if not any("60" in col for col in adv_cols):
            issues.append("adv60_missing")
        if bucket in {"low", "unknown"} or not liquid_enough:
            issues.append("low_or_unknown_liquidity")
        rows.append(
            {
                "ticker": row.get("ticker"),
                "DiscDate": row.get("DiscDate"),
                "classification": row.get("classification"),
                "volume_fields_detected": "|".join(volume_cols),
                "turnover_fields_detected": "|".join(turnover_cols),
                "adv_fields_detected": "|".join(adv_cols),
                "liquidity_bucket": bucket,
                "liquidity_issue": "|".join(issues),
                "low_liquidity_in_top_rank_flag": bool(low_top),
                "commercial_acceptability": "acceptable" if bucket in {"high", "mid"} and liquid_enough else "not_acceptable",
                "recommended_fix": "Add ADV60/capacity flags and keep low/unknown liquidity out of product rankings.",
            }
        )
    return pd.DataFrame(rows, columns=LIQUIDITY_COLUMNS)


def validate_ranking(combined: pd.DataFrame, duplicate_keys: set[tuple[str, str]]) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame(columns=RANKING_COLUMNS)
    rows = []
    for _, row in combined.iterrows():
        label = _safe_text(row.get("ranking_file"))
        rank = pd.to_numeric(pd.Series([row.get("rank")]), errors="coerce").iloc[0]
        date = _safe_text(row.get("DiscDate"))[:10]
        key = (_safe_text(row.get("ticker")), date)
        ranking_score = _ranking_score(row)
        components = [
            col
            for col in ["abnormal_ret_1d", "turnover_spike_ratio", "commercial_candidate_score", "avg_turnover_20d"]
            if col in row.index and not pd.isna(row.get(col))
        ]
        if _safe_text(row.get("ranking_score_components")):
            components.append("ranking_score_components")
        reason = _safe_text(row.get("ranking_reason"))
        duplicate = key in duplicate_keys
        insufficient = _safe_text(row.get("ret_20d_status")) != "valid"
        low_liq = _safe_text(row.get("liquidity_bucket")).lower() in {"low", "unknown"} or not _truthy(row.get("is_liquid_enough"))
        missing_reason = not reason
        issues = []
        if label in AUDIT_FILES:
            issues.append("audit_row_not_product_ranking")
        if ranking_score is None:
            issues.append("ranking_score_missing")
        if not components:
            issues.append("score_components_missing")
        if duplicate:
            issues.append("duplicate_event_key")
        if insufficient:
            issues.append("20d_horizon_unavailable")
        if low_liq:
            issues.append("low_or_unknown_liquidity")
        if missing_reason:
            issues.append("missing_ranking_reason")
        accept = "acceptable" if not any(i in issues for i in ["ranking_score_missing", "score_components_missing", "duplicate_event_key", "low_or_unknown_liquidity"]) else "manual_review"
        rows.append(
            {
                "ranking_file": label,
                "ticker": row.get("ticker"),
                "DiscDate": row.get("DiscDate"),
                "rank": None if pd.isna(rank) else int(rank),
                "classification": row.get("classification"),
                "ranking_score": ranking_score,
                "score_explainable": bool(ranking_score is not None and components and not missing_reason),
                "score_components_detected": "|".join(components),
                "duplicate_event_flag": bool(duplicate),
                "insufficient_horizon_flag": bool(insufficient),
                "low_liquidity_flag": bool(low_liq),
                "missing_reason_flag": bool(missing_reason),
                "ranking_issue": "|".join(issues),
                "commercial_acceptability": accept,
                "manual_review_required": bool(issues),
            }
        )
    return pd.DataFrame(rows, columns=RANKING_COLUMNS)


def _bucket_for_rank(rank: object, count: int) -> str:
    value = pd.to_numeric(pd.Series([rank]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "bottom"
    value = int(value)
    if value <= 10:
        return "top_10"
    if value <= 25:
        return "top_25"
    if value <= 50:
        return "top_50"
    if value <= 100:
        return "top_100"
    if count and value > max(100, int(count * 0.8)):
        return "bottom"
    return "middle"


def build_signal_performance(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    rows = []
    work = combined.copy()
    for col in ["rank", "ret_1d", "ret_3d", "ret_5d", "abnormal_ret_1d", "abnormal_ret_3d"]:
        work[col] = _safe_num(work, col)
    work["ranking_score_value"] = [_ranking_score(row) for _, row in work.iterrows()]
    for classification, group in work.groupby("classification", dropna=False):
        n = len(group)
        bucketed = group.copy()
        bucketed["bucket"] = [_bucket_for_rank(rank, n) for rank in bucketed["rank"]]
        for bucket in ["top_10", "top_25", "top_50", "top_100", "middle", "bottom"]:
            sub = bucketed[bucketed["bucket"] == bucket]
            if sub.empty:
                rows.append({"classification": classification, "bucket": bucket, "count": 0, "commercial_signal_assessment": "no_sample"})
                continue
            positive_class = str(classification).startswith("positive")
            direction = 1 if positive_class else -1
            hit_1d = (sub["ret_1d"] * direction > 0).mean(skipna=True)
            hit_3d = (sub["ret_3d"] * direction > 0).mean(skipna=True)
            hit_5d = (sub["ret_5d"] * direction > 0).mean(skipna=True)
            corr_frame = sub[["ranking_score_value", "ret_1d"]].dropna()
            corr = corr_frame["ranking_score_value"].corr(corr_frame["ret_1d"]) if len(corr_frame) >= 3 else np.nan
            assessment = "directionally_consistent" if not pd.isna(hit_1d) and hit_1d >= 0.5 else "weak_or_unproven"
            rows.append(
                {
                    "classification": classification,
                    "bucket": bucket,
                    "count": int(len(sub)),
                    "mean_ret_1d": sub["ret_1d"].mean(),
                    "median_ret_1d": sub["ret_1d"].median(),
                    "hit_rate_1d": hit_1d,
                    "mean_ret_3d": sub["ret_3d"].mean(),
                    "median_ret_3d": sub["ret_3d"].median(),
                    "hit_rate_3d": hit_3d,
                    "mean_ret_5d": sub["ret_5d"].mean(),
                    "median_ret_5d": sub["ret_5d"].median(),
                    "hit_rate_5d": hit_5d,
                    "mean_abnormal_ret_1d": sub["abnormal_ret_1d"].mean(),
                    "mean_abnormal_ret_3d": sub["abnormal_ret_3d"].mean(),
                    "rank_score_to_future_return_corr": corr,
                    "commercial_signal_assessment": assessment,
                }
            )
    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)


def build_manual_review(
    combined_all: pd.DataFrame,
    classification_validation: pd.DataFrame,
    ranking_validation: pd.DataFrame,
) -> pd.DataFrame:
    if combined_all.empty:
        return pd.DataFrame(columns=MANUAL_COLUMNS)
    issue_keys = set(
        zip(
            ranking_validation.loc[ranking_validation["manual_review_required"].astype(bool), "ranking_file"].astype(str),
            ranking_validation.loc[ranking_validation["manual_review_required"].astype(bool), "ticker"].astype(str),
            ranking_validation.loc[ranking_validation["manual_review_required"].astype(bool), "DiscDate"].astype(str),
        )
    )
    classification_issues = {
        (str(r.ranking_file), str(r.ticker), str(r.DiscDate)): str(r.classification_issue)
        for r in classification_validation.itertuples(index=False)
        if str(r.classification_issue)
    }
    rows = []
    for label, group in combined_all.groupby("ranking_file", dropna=False):
        work = group.copy()
        if "rank" in work.columns:
            work["rank_num"] = _safe_num(work, "rank")
            work = work.sort_values("rank_num", na_position="last")
        for _, row in work.head(50).iterrows():
            key = (str(label), str(row.get("ticker")), str(row.get("DiscDate")))
            data_flags = []
            horizon_flags = []
            liq_flags = []
            if key in issue_keys:
                data_flags.append("ranking_validation_issue")
            if classification_issues.get(key):
                data_flags.append(classification_issues[key])
            if _safe_text(row.get("ret_20d_status")) != "valid":
                horizon_flags.append("20d_unavailable")
            if _safe_text(row.get("liquidity_bucket")).lower() in {"low", "unknown"}:
                liq_flags.append("low_or_unknown_liquidity")
            review_priority = "high" if data_flags or horizon_flags or liq_flags or label in AUDIT_FILES else "standard"
            rows.append(
                {
                    "review_priority": review_priority,
                    "ranking_file": label,
                    "rank": row.get("rank"),
                    "ticker": row.get("ticker"),
                    "DiscDate": row.get("DiscDate"),
                    "DocType": row.get("DocType") or row.get("doctypes"),
                    "classification": row.get("classification"),
                    "company_name_if_available": row.get("company_name"),
                    "ranking_score": _ranking_score(row),
                    "primary_reason_detected": row.get("ranking_reason") or row.get("unclassified_reason") or row.get("exclusion_reasons"),
                    "risk_flags_detected": "|".join([*data_flags, *horizon_flags, *liq_flags]),
                    "data_quality_flags": "|".join(data_flags),
                    "horizon_flags": "|".join(horizon_flags),
                    "liquidity_flags": "|".join(liq_flags),
                    "analyst_review_questions": "Does the disclosure support the assigned direction, and is the price response horizon valid?",
                    "commercial_risk_if_wrong": "Misclassified event or invalid horizon can produce misleading client-facing rankings.",
                    "source_rows_hint": f"{label}:{row.get('ticker')}:{row.get('DiscDate')}",
                }
            )
    return pd.DataFrame(rows, columns=MANUAL_COLUMNS)


def build_critical_issues(
    combined_rankings: pd.DataFrame,
    combined_all: pd.DataFrame,
    classification_validation: pd.DataFrame,
    ranking_validation: pd.DataFrame,
    abnormal_validation: pd.DataFrame,
    financial_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(issue_type: str, severity: str, module: str, file: str, count: int, desc: str, impact: str, fix: str, blocks: bool = True) -> None:
        rows.append(
            {
                "issue_id": f"CAI-{len(rows) + 1:03d}",
                "issue_type": issue_type,
                "severity": severity,
                "affected_module": module,
                "affected_file": file,
                "affected_rows_count": int(count),
                "description": desc,
                "commercial_impact": impact,
                "recommended_fix": fix,
                "blocks_product_readiness": bool(blocks),
            }
        )

    if not combined_rankings.empty and "classification" in combined_rankings.columns:
        bad = combined_rankings[combined_rankings["classification"].isin(AUDIT_FILES)]
        if len(bad):
            add("excluded_or_unclassified_in_ranking", "critical", "ranking", "ranking CSVs", len(bad), "Excluded/unclassified rows are present in product ranking set.", "Client rankings can include rows explicitly marked non-product.", "Keep audit rows separate from product ranking inputs.")

    if not ranking_validation.empty:
        dup = int(ranking_validation["duplicate_event_flag"].astype(bool).sum())
        if dup:
            add("duplicate_event_ranked", "critical", "ranking", "commercial_quality_*_ranking_enriched.csv", dup, "Duplicate event keys appear in ranked candidates.", "A single disclosure can be overrepresented in client lists.", "Deduplicate by ticker/DiscDate/DocType before ranking or aggregate event bundles.")
        score_bad = int((~ranking_validation["score_explainable"].astype(bool)).sum())
        if score_bad:
            add("ranking_score_explainability_gap", "high", "ranking", "commercial_quality_*_ranking_enriched.csv", score_bad, "One or more ranking rows lack a fully explainable score/reason/component set.", "Users cannot audit why candidates are ranked.", "Persist explicit ranking_score and score component columns.")
        h20 = int(ranking_validation["insufficient_horizon_flag"].astype(bool).sum())
        if h20:
            add("insufficient_20d_horizon_in_rankings", "medium", "ranking", "commercial_quality_*_ranking_enriched.csv", h20, "Rankings include rows whose 20d return is unavailable.", "20d analytics cannot be interpreted for those rows.", "Display 20d risk flags and avoid 20d-dependent scoring.")
        low_top = ranking_validation[
            ranking_validation["low_liquidity_flag"].astype(bool)
            & (pd.to_numeric(ranking_validation["rank"], errors="coerce") <= 50)
        ]
        if len(low_top) >= 5:
            add("low_liquidity_top_rankings", "critical", "liquidity/ranking", "commercial_quality_*_ranking_enriched.csv", len(low_top), "Low/unknown liquidity rows are common in top-ranked candidates.", "Institutional users may see names unsuitable for execution.", "Tighten liquidity filters and add capacity constraints.")
        elif len(low_top):
            add("low_liquidity_top_rankings", "high", "liquidity/ranking", "commercial_quality_*_ranking_enriched.csv", len(low_top), "Low/unknown liquidity rows appear in top-ranked candidates.", "Execution suitability is unclear.", "Keep low/unknown liquidity out of product top lists.", blocks=False)

    class_bad = classification_validation[
        classification_validation["classification_issue"].fillna("").astype(str).str.len() > 0
        & ~classification_validation["ranking_file"].isin(AUDIT_FILES)
    ]
    if len(class_bad):
        add("classification_basis_issue", "high", "classification", "commercial_quality_*_ranking_enriched.csv", len(class_bad), "Some ranked rows need manual review for event direction or mixed-event context.", "Positive/negative labels may be oversimplified.", "Add explicit event taxonomy confidence and mixed-event handling.")
    basis_series = classification_validation["classification_basis_level"] if "classification_basis_level" in classification_validation else pd.Series("", index=classification_validation.index)
    tier_series = classification_validation["event_confidence_tier"] if "event_confidence_tier" in classification_validation else pd.Series("", index=classification_validation.index)
    universe_high = classification_validation[
        basis_series.fillna("").eq("universe_adjusted_only")
        & tier_series.fillna("").eq("high")
        & ~classification_validation["ranking_file"].isin(AUDIT_FILES)
    ]
    if len(universe_high):
        add("universe_only_high_confidence_classification", "high", "classification", "commercial_quality_*_ranking_enriched.csv", len(universe_high), "Universe-only adjusted evidence produced high-confidence classifications.", "Classification confidence can be overstated when peer-adjusted evidence is unavailable.", "Cap universe-only evidence below high confidence and require manual review or limitations.")

    if not abnormal_validation.empty:
        no_sector_size = abnormal_validation[
            (~abnormal_validation["sector_adjustment_detected"].astype(bool))
            | (~abnormal_validation["size_adjustment_detected"].astype(bool))
        ]
        if len(no_sector_size):
            add("abnormal_return_adjustment_incomplete", "high", "abnormal_return", "jquants_abnormal_return.py", len(no_sector_size), "Abnormal returns are market/universe-adjusted but not sector/size-adjusted.", "Market or sector-wide moves can be misattributed to disclosure-specific alpha.", "Implement sector and size neutral abnormal return variants.", blocks=False)

    fs = _ensure_ticker(financial_summary)
    raw_mutation_cols = [col for col in fs.columns if col.lower() in {"writeback", "mutated", "overwritten"}]
    if raw_mutation_cols:
        add("raw_data_mutation_suspected", "critical", "raw_inputs", "financial_summary_raw.csv", len(fs), "Columns suggesting raw mutation were detected.", "Raw lineage may be compromised.", "Separate raw immutable layer from derived outputs.")

    return pd.DataFrame(rows, columns=CRITICAL_COLUMNS)


def determine_product_status(
    critical_issues_count: int,
    high_issues_count: int,
    score: float,
    *,
    classification_explainable_ratio: float = 0.0,
    signal_direction_ok: bool = False,
    score_explainable_ratio: float = 0.0,
) -> str:
    if critical_issues_count >= 3 or score < 35:
        return "invalid"
    if critical_issues_count > 0:
        return "poc_only"
    if score < 60 or classification_explainable_ratio < 0.6 or score_explainable_ratio < 0.6:
        return "poc_only"
    if high_issues_count > 0 or not signal_direction_ok or score < 85:
        return "product_needs_corrections"
    return "product_ready"


def _scorecard(
    classification_validation: pd.DataFrame,
    price_validation: pd.DataFrame,
    abnormal_validation: pd.DataFrame,
    liquidity_validation: pd.DataFrame,
    ranking_validation: pd.DataFrame,
    signal: pd.DataFrame,
    critical: pd.DataFrame,
) -> pd.DataFrame:
    def ratio(mask: pd.Series, total: int) -> float:
        return float(mask.sum() / total) if total else 0.0

    class_ratio = ratio(classification_validation["classification_issue"].fillna("").eq(""), len(classification_validation)) if not classification_validation.empty else 0.0
    adjusted_supported_ratio = (
        ratio(
            classification_validation["classification_basis_level"].fillna("").isin(["sector_size_adjusted", "sector_adjusted", "size_adjusted"])
            & classification_validation["event_confidence_tier"].fillna("").isin(["high", "medium"]),
            len(classification_validation),
        )
        if not classification_validation.empty
        else 0.0
    )
    fundamental_supported_ratio = (
        ratio(classification_validation["fundamental_event_confidence"].fillna("").isin(["high", "medium"]), len(classification_validation))
        if not classification_validation.empty and "fundamental_event_confidence" in classification_validation
        else 0.0
    )
    price_ratio = ratio(price_validation["commercial_acceptability"].eq("acceptable"), len(price_validation)) if not price_validation.empty else 0.0
    abnormal_ratio = ratio(abnormal_validation["market_adjustment_detected"].astype(bool), len(abnormal_validation)) if not abnormal_validation.empty else 0.0
    liquidity_ratio = ratio(liquidity_validation["commercial_acceptability"].eq("acceptable"), len(liquidity_validation)) if not liquidity_validation.empty else 0.0
    ranking_ratio = ratio(ranking_validation["score_explainable"].astype(bool), len(ranking_validation)) if not ranking_validation.empty else 0.0
    signal_good = ratio(signal["commercial_signal_assessment"].eq("directionally_consistent"), len(signal[signal["count"] > 0])) if not signal.empty else 0.0
    blocker_penalty = min(1.0, len(critical[critical["blocks_product_readiness"].astype(bool)]) * 0.12) if not critical.empty else 0.0
    rows = [
        ("classification", class_ratio * 12, 12, "pass" if class_ratio >= 0.8 else "needs_work", f"clean_ratio={class_ratio:.3f}"),
        ("classification_quality_v2", adjusted_supported_ratio * 8, 8, "pass" if adjusted_supported_ratio >= 0.8 else "needs_work", f"adjusted_supported_ratio={adjusted_supported_ratio:.3f}"),
        ("fundamental_event_evidence", fundamental_supported_ratio * 8, 8, "pass" if fundamental_supported_ratio >= 0.7 else "needs_work", f"fundamental_supported_ratio={fundamental_supported_ratio:.3f}"),
        ("price_reaction", price_ratio * 15, 15, "pass" if price_ratio >= 0.8 else "needs_work", f"acceptable_ratio={price_ratio:.3f}"),
        ("abnormal_return", abnormal_ratio * 12, 12, "limited" if abnormal_ratio else "fail", "market adjusted only; sector/size absent"),
        ("liquidity", liquidity_ratio * 12, 12, "pass" if liquidity_ratio >= 0.9 else "needs_work", f"acceptable_ratio={liquidity_ratio:.3f}"),
        ("ranking_explainability", ranking_ratio * 18, 18, "pass" if ranking_ratio >= 0.8 else "needs_work", f"score_explainable_ratio={ranking_ratio:.3f}"),
        ("signal_bucket_quality", signal_good * 10, 10, "limited" if signal_good >= 0.5 else "unproven", f"directional_bucket_ratio={signal_good:.3f}"),
        ("critical_blockers", (1.0 - blocker_penalty) * 5, 5, "pass" if blocker_penalty == 0 else "blocked", f"blocker_penalty={blocker_penalty:.3f}"),
    ]
    return pd.DataFrame(rows, columns=SCORECARD_COLUMNS)


def _classification_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0}
    total = len(df)
    def col(name: str, default: object = "") -> pd.Series:
        return df[name] if name in df else pd.Series([default] * total, index=df.index)

    adjusted_supported = (
        col("classification_basis_level").fillna("").isin(["sector_size_adjusted", "sector_adjusted", "size_adjusted"])
        & col("event_confidence_tier").fillna("").isin(["high", "medium"])
    )
    auto_safe = col("auto_classification_safe_flag", False).map(_truthy)
    high_conf_auto = col("event_confidence_tier").fillna("").eq("high") & auto_safe
    high_priority = col("manual_review_priority").fillna("").eq("high")
    universe_high = col("classification_basis_level").fillna("").eq("universe_adjusted_only") & col("event_confidence_tier").fillna("").eq("high")
    fundamental_direction = col("fundamental_event_direction").fillna("")
    fundamental_confidence = col("fundamental_event_confidence").fillna("")
    fundamental_mixed = col("fundamental_mixed_signal_flag", False).map(_truthy)
    insufficient_fundamental = fundamental_direction.eq("insufficient_evidence") | fundamental_confidence.eq("not_evaluable")
    contradiction = col("classification_issue").fillna("").astype(str)
    price_conflict = contradiction.str.contains("fundamental|mixed|conflict", case=False, regex=True) & col("mixed_event_severity").fillna("").isin(["medium", "high"])
    final_direction = col("event_direction_final").fillna("")
    confirmed = fundamental_direction.isin(["positive", "negative"]) & final_direction.eq(fundamental_direction) & adjusted_supported
    supported_ratio = float(adjusted_supported.sum() / total) if total else 0.0
    auto_ratio = float(auto_safe.sum() / total) if total else 0.0
    fundamental_supported_ratio = float(fundamental_confidence.isin(["high", "medium"]).sum() / total) if total else 0.0
    confirmation_ratio = float(confirmed.sum() / total) if total else 0.0
    conflict_rows = int(price_conflict.sum())
    quality_score = round(
        max(
            0.0,
            min(
                100.0,
                supported_ratio * 55.0
                + auto_ratio * 25.0
                + float(col("event_confidence_tier").fillna("").isin(["high", "medium"]).sum() / total) * 20.0
                - float(high_priority.sum() / total) * 25.0
                - float(universe_high.sum() / total) * 40.0,
            ),
        ),
        2,
    )
    return {
        "rows": int(len(df)),
        "manual_review_required": int(df["manual_review_required"].astype(bool).sum()),
        "mixed_event_rows": int(df["mixed_event_flag"].astype(bool).sum()),
        "issue_counts": df["classification_issue"].replace("", "none").value_counts(dropna=False).to_dict(),
        "classification_confidence_tier_counts": col("event_confidence_tier").fillna("missing").value_counts(dropna=False).to_dict(),
        "event_direction_final_counts": col("event_direction_final").fillna("missing").value_counts(dropna=False).to_dict(),
        "event_reaction_pattern_counts": col("event_reaction_pattern").fillna("missing").value_counts(dropna=False).to_dict(),
        "manual_review_category_counts": col("manual_review_category").fillna("missing").value_counts(dropna=False).to_dict(),
        "manual_review_priority_counts": col("manual_review_priority").fillna("missing").value_counts(dropna=False).to_dict(),
        "fundamental_event_type_counts": col("fundamental_event_type").fillna("missing").value_counts(dropna=False).to_dict(),
        "fundamental_event_direction_counts": fundamental_direction.replace("", "missing").value_counts(dropna=False).to_dict(),
        "fundamental_event_confidence_counts": fundamental_confidence.replace("", "missing").value_counts(dropna=False).to_dict(),
        "fundamental_event_strength_counts": col("fundamental_event_strength").fillna("missing").value_counts(dropna=False).to_dict(),
        "fundamental_mixed_signal_rows": int(fundamental_mixed.sum()),
        "insufficient_fundamental_evidence_rows": int(insufficient_fundamental.sum()),
        "fundamental_price_confirmation_ratio": confirmation_ratio,
        "fundamental_price_conflict_rows": conflict_rows,
        "fundamental_evidence_quality_score": round(max(0.0, min(100.0, fundamental_supported_ratio * 60.0 + confirmation_ratio * 30.0 - float(insufficient_fundamental.sum() / total) * 20.0 - float(conflict_rows / total) * 25.0)), 2),
        "auto_classification_safe_ratio": auto_ratio,
        "high_confidence_auto_classified_rows": int(high_conf_auto.sum()),
        "high_priority_manual_review_rows": int(high_priority.sum()),
        "mixed_event_severity_counts": df["mixed_event_severity"].fillna("missing").value_counts(dropna=False).to_dict(),
        "adjusted_evidence_supported_classification_ratio": supported_ratio,
        "universe_only_high_confidence_rows": int(universe_high.sum()),
        "classification_quality_score": quality_score,
    }


def _simple_summary(df: pd.DataFrame, key: str = "commercial_acceptability") -> dict[str, Any]:
    if df.empty:
        return {"rows": 0}
    return {"rows": int(len(df)), f"{key}_counts": df[key].fillna("missing").value_counts(dropna=False).to_dict() if key in df else {}}


def _row_counts(frames: dict[str, pd.DataFrame], financial_summary: pd.DataFrame, prices: pd.DataFrame) -> dict[str, int]:
    out = {
        "financial_summary_raw": int(len(financial_summary)),
        "prices_daily_bars": int(len(prices)),
    }
    out.update({f"ranking_{label}": int(len(df)) for label, df in frames.items()})
    return out


def _markdown_report(report: dict[str, Any], code_review: pd.DataFrame, critical: pd.DataFrame) -> str:
    blockers = report.get("top_blockers", [])
    blocker_text = "\n".join(f"- {b}" for b in blockers) if blockers else "- None detected"
    high_findings = code_review[code_review["severity"].isin(["critical", "high"])]
    finding_text = "\n".join(
        f"- {r.severity}: {r.module} / {r.function_or_area}: {r.finding}"
        for r in high_findings.itertuples(index=False)
    ) or "- No critical/high code review findings"
    critical_text = "\n".join(
        f"- {r.severity}: {r.issue_type} ({r.affected_rows_count} rows): {r.description}"
        for r in critical.itertuples(index=False)
    ) or "- No critical product blockers detected"
    return f"""# Commercial Analysis Validity Audit

## 1. Executive Verdict
Verdict: **{report['verdict']}**

Status: **{report['commercial_product_analysis_status']}**

Commercial product readiness score: **{report['commercial_product_readiness_score']}**

## 2. Product vs PoC Judgment
This audit judges the current outputs as an analysis product, not as a runnable PoC. The current status is **{report['commercial_product_analysis_status']}**.

## 3. Data Extraction Preconditions
- Inputs: `{report['inputs'].get('financial_summary')}`, `{report['inputs'].get('prices')}`
- Row counts: `{report['row_counts']}`

## 4. Code Review Findings
{finding_text}

## 5. Event Classification Validity
`{report['classification_summary']}`

## 6. Price Reaction Validity
`{report['price_reaction_summary']}`

## 7. Abnormal Return Validity
`{report['abnormal_return_summary']}`

## 8. Liquidity Validity
`{report['liquidity_summary']}`

## 9. Ranking Validity
`{report['ranking_summary']}`

## 10. Signal Performance by Bucket
`{report['signal_performance_summary']}`

## 11. Manual Review Sample Summary
Manual review CSV rows: `{report.get('manual_review_rows', 0)}`

## 12. Critical Product Blockers
{critical_text}

## 13. Required Fixes Before Commercial Product Use
{blocker_text}

## 14. Recommended Roadmap
{chr(10).join(f"- {x}" for x in report.get('recommended_next_steps', []))}
"""


def build_commercial_analysis_validity_audit(
    financial_summary_path: Path = DEFAULT_FINANCIAL_SUMMARY,
    prices_path: Path = DEFAULT_PRICES,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    output_dir: Path = DEFAULT_ANALYSIS_DIR,
) -> dict[str, Any]:
    financial_summary = _read_csv(financial_summary_path)
    prices = _read_csv(prices_path)
    financial_summary = _ensure_ticker(financial_summary)
    prices = _ensure_ticker(prices)
    frames = _load_ranking_frames(analysis_dir)
    ranked = _combined_rankings(frames, include_audits=False)
    combined_all = _combined_rankings(frames, include_audits=True)
    financial = _financial_lookup(financial_summary)
    extraction_duplicates = _read_csv(analysis_dir / "extraction_duplicate_keys.csv")
    duplicate_keys = _duplicate_keys(financial_summary, extraction_duplicates)

    code_review = build_code_review_findings()
    classification = validate_event_classification(combined_all, financial)
    price = validate_price_reaction(combined_all)
    abnormal = validate_abnormal_return(combined_all)
    abnormal_v2 = abnormal_return_v2_summary(combined_all)
    liquidity = validate_liquidity(combined_all)
    ranking = validate_ranking(ranked, duplicate_keys)
    signal = build_signal_performance(ranked)
    manual = build_manual_review(combined_all, classification, ranking)
    critical = build_critical_issues(ranked, combined_all, classification, ranking, abnormal, financial_summary)
    scorecard = _scorecard(classification, price, abnormal, liquidity, ranking, signal, critical)
    classification_quality = _classification_summary(classification)

    for key, (df, cols) in {
        "code_review": (code_review, CODE_REVIEW_COLUMNS),
        "classification": (classification, CLASSIFICATION_COLUMNS),
        "price_reaction": (price, PRICE_COLUMNS),
        "abnormal_return": (abnormal, ABNORMAL_COLUMNS),
        "liquidity": (liquidity, LIQUIDITY_COLUMNS),
        "ranking": (ranking, RANKING_COLUMNS),
        "signal": (signal, SIGNAL_COLUMNS),
        "manual_review": (manual, MANUAL_COLUMNS),
        "critical_issues": (critical, CRITICAL_COLUMNS),
        "scorecard": (scorecard, SCORECARD_COLUMNS),
    }.items():
        _write_csv(output_dir / OUTPUT_FILES[key], df, cols)

    critical_count = int((critical["severity"] == "critical").sum()) if not critical.empty else 0
    high_count = int((critical["severity"] == "high").sum()) if not critical.empty else 0
    medium_count = int((critical["severity"] == "medium").sum()) if not critical.empty else 0
    readiness_score = round(float(scorecard["score"].sum()), 2) if not scorecard.empty else 0.0
    class_ratio = (
        float(classification["classification_issue"].fillna("").eq("").sum() / len(classification))
        if len(classification)
        else 0.0
    )
    score_ratio = (
        float(ranking["score_explainable"].astype(bool).sum() / len(ranking)) if len(ranking) else 0.0
    )
    signal_ok = bool(
        not signal.empty
        and signal.loc[signal["count"] > 0, "commercial_signal_assessment"].eq("directionally_consistent").mean()
        >= 0.5
    )
    status = determine_product_status(
        critical_count,
        high_count,
        readiness_score,
        classification_explainable_ratio=class_ratio,
        signal_direction_ok=signal_ok,
        score_explainable_ratio=score_ratio,
    )
    top_blockers = (
        critical.sort_values(["blocks_product_readiness", "severity"], ascending=[False, True])["description"].head(8).tolist()
        if not critical.empty
        else []
    )
    required_fixes = [
        "Persist explicit ranking_score and score component columns used for ordering.",
        "Add sector and size adjusted abnormal return metrics before commercial claims.",
        "Expose classification confidence, mixed-event flags, horizon flags, and liquidity flags in dashboard.",
        "Deduplicate or aggregate duplicate ticker/DiscDate/DocType events before ranking.",
    ]
    recommended = [
        "Run manual analyst review on top candidates and all rows flagged by this audit.",
        "Backtest ranking buckets out-of-sample by event family and disclosure time bucket.",
        "Add institutional liquidity capacity metrics such as ADV60 and trade-size buckets.",
    ]
    report = {
        "generated_at": jst_now_iso(),
        "inputs": {
            "financial_summary": str(financial_summary_path),
            "prices": str(prices_path),
            "analysis_dir": str(analysis_dir),
            "output_dir": str(output_dir),
            "extraction_audit_report": str(analysis_dir / "extraction_audit_report.json"),
        },
        "code_modules_reviewed": [
            "jquants_commercial_common.py",
            "jquants_commercial_quality_pipeline.py",
            "jquants_price_reaction_v2.py",
            "jquants_abnormal_return.py",
            "jquants_liquidity.py",
            "jquants_robust_summary.py",
            "jquants_commercial_quality_rankings.py",
            "jquants_commercial_quality_enrichment.py",
            "jquants_commercial_daily_dashboard.py",
            "jquants_extraction_audit.py",
            "commercial_dashboard_data.py",
        ],
        "row_counts": _row_counts(frames, financial_summary, prices),
        "classification_summary": classification_quality,
        "classification_confidence_tier_counts": classification_quality.get("classification_confidence_tier_counts", {}),
        "event_direction_final_counts": classification_quality.get("event_direction_final_counts", {}),
        "event_reaction_pattern_counts": classification_quality.get("event_reaction_pattern_counts", {}),
        "manual_review_category_counts": classification_quality.get("manual_review_category_counts", {}),
        "manual_review_priority_counts": classification_quality.get("manual_review_priority_counts", {}),
        "fundamental_event_type_counts": classification_quality.get("fundamental_event_type_counts", {}),
        "fundamental_event_direction_counts": classification_quality.get("fundamental_event_direction_counts", {}),
        "fundamental_event_confidence_counts": classification_quality.get("fundamental_event_confidence_counts", {}),
        "fundamental_event_strength_counts": classification_quality.get("fundamental_event_strength_counts", {}),
        "fundamental_mixed_signal_rows": classification_quality.get("fundamental_mixed_signal_rows", 0),
        "insufficient_fundamental_evidence_rows": classification_quality.get("insufficient_fundamental_evidence_rows", 0),
        "fundamental_price_confirmation_ratio": classification_quality.get("fundamental_price_confirmation_ratio", 0.0),
        "fundamental_price_conflict_rows": classification_quality.get("fundamental_price_conflict_rows", 0),
        "fundamental_evidence_quality_score": classification_quality.get("fundamental_evidence_quality_score", 0.0),
        "auto_classification_safe_ratio": classification_quality.get("auto_classification_safe_ratio", 0.0),
        "high_confidence_auto_classified_rows": classification_quality.get("high_confidence_auto_classified_rows", 0),
        "high_priority_manual_review_rows": classification_quality.get("high_priority_manual_review_rows", 0),
        "mixed_event_severity_counts": classification_quality.get("mixed_event_severity_counts", {}),
        "adjusted_evidence_supported_classification_ratio": classification_quality.get("adjusted_evidence_supported_classification_ratio", 0.0),
        "universe_only_high_confidence_rows": classification_quality.get("universe_only_high_confidence_rows", 0),
        "classification_quality_score": classification_quality.get("classification_quality_score", 0.0),
        "price_reaction_summary": _simple_summary(price),
        "abnormal_return_summary": _simple_summary(abnormal),
        "commercial_abnormal_return_v2_summary": abnormal_v2,
        **abnormal_v2,
        "liquidity_summary": _simple_summary(liquidity),
        "ranking_summary": {
            **_simple_summary(ranking, "commercial_acceptability"),
            "score_explainable_ratio": score_ratio,
            "duplicate_event_rows": int(ranking["duplicate_event_flag"].astype(bool).sum()) if not ranking.empty else 0,
            "insufficient_20d_rows": int(ranking["insufficient_horizon_flag"].astype(bool).sum()) if not ranking.empty else 0,
        },
        "signal_performance_summary": {
            "rows": int(len(signal)),
            "bucket_assessment_counts": signal["commercial_signal_assessment"].value_counts(dropna=False).to_dict()
            if not signal.empty
            else {},
        },
        "critical_issues_count": critical_count,
        "high_issues_count": high_count,
        "medium_issues_count": medium_count,
        "commercial_product_analysis_status": status,
        "commercial_product_readiness_score": readiness_score,
        "verdict": status,
        "top_blockers": top_blockers,
        "required_fixes_before_product": required_fixes if status != "product_ready" else [],
        "recommended_next_steps": recommended,
        "manual_review_rows": int(len(manual)),
        "output_files": {key: str(output_dir / filename) for key, filename in OUTPUT_FILES.items()},
        "raw_csv_modified": False,
    }
    _write_json(output_dir / OUTPUT_FILES["report_json"], report)
    (output_dir / OUTPUT_FILES["report_md"]).parent.mkdir(parents=True, exist_ok=True)
    (output_dir / OUTPUT_FILES["report_md"]).write_text(_markdown_report(report, code_review, critical), encoding="utf-8")
    return _json_clean(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financial-summary", type=Path, default=DEFAULT_FINANCIAL_SUMMARY)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    args = parser.parse_args(argv)
    build_commercial_analysis_validity_audit(
        financial_summary_path=args.financial_summary,
        prices_path=args.prices,
        analysis_dir=args.analysis_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

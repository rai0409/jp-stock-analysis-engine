"""Finance-grade validation diagnostics for commercial J-Quants signals.

This module reads derived commercial ranking outputs and writes separate
validation artifacts. It is read-only with respect to raw J-Quants inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jp_stock_analysis.analysis.jquants_commercial_common import (
    ANALYSIS_DIR,
    ensure_parent,
    jst_now_iso,
)


DEFAULT_ANALYSIS_DIR = ANALYSIS_DIR
DEFAULT_PRICES = Path("/mnt/d/jquants_price_store/prices_daily_bars.csv")

RANKING_FILES = {
    "positive_reacted": "commercial_quality_positive_reacted_ranking_enriched.csv",
    "positive_unreacted": "commercial_quality_positive_unreacted_ranking_enriched.csv",
    "negative_reacted": "commercial_quality_negative_reacted_ranking_enriched.csv",
    "negative_reversal": "commercial_quality_negative_reversal_ranking_enriched.csv",
}

OUTPUT_FILES = {
    "panel": "commercial_signal_validation_panel.csv",
    "summary": "commercial_signal_validation_summary.json",
    "report": "commercial_signal_validation_report.md",
    "rank_ic": "commercial_signal_validation_rank_ic.csv",
    "quantile_spread": "commercial_signal_validation_quantile_spread.csv",
    "event_study": "commercial_signal_validation_event_study.csv",
    "liquidity_cost": "commercial_signal_validation_liquidity_cost.csv",
    "sector_size_neutral": "commercial_signal_validation_sector_size_neutral.csv",
    "stability": "commercial_signal_validation_stability.csv",
    "leakage_safe_panel": "commercial_leakage_safe_validation_panel.csv",
    "leakage_safe_summary": "commercial_leakage_safe_validation_summary.json",
    "leakage_safe_report": "commercial_leakage_safe_validation_report.md",
    "walk_forward": "commercial_walk_forward_validation.csv",
    "date_level": "commercial_date_level_validation.csv",
    "score_component_audit": "commercial_score_component_contamination_audit.csv",
    "concentration_audit": "commercial_validation_concentration_audit.csv",
}

HORIZONS = (1, 3, 5, 20)
MIN_DATE_SAMPLE = 3
MIN_GROUP_SAMPLE = 8
MIN_QUANTILE_SAMPLE = 2
MIN_WALK_FORWARD_TRAIN_ROWS = 8
MIN_WALK_FORWARD_VALIDATION_ROWS = 8
SUSPICIOUS_SCORE_PATTERNS = (
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_20d",
    "abnormal_ret",
    "sector_adjusted_return",
    "sector_size_adjusted_return",
    "commercial_abnormal_reaction_score",
    "reaction_flag",
    "price_reaction",
    "fwd_ret",
    "future",
    "label",
)
SCORE_AUDIT_TEXT_FIELDS = (
    "ranking_score_components",
    "commercial_abnormal_reaction_score_components",
    "evidence_summary",
    "primary_signal",
    "secondary_signal",
    "ranking_reason",
    "product_risk_flags",
)

PANEL_COLUMNS = [
    "ticker",
    "DiscDate",
    "reaction_base_date",
    "ranking_label",
    "candidate_bucket",
    "ranking_score",
    "commercial_abnormal_reaction_score",
    "event_direction_final",
    "event_classification_detail",
    "event_confidence_tier",
    "classification_basis_level",
    "product_acceptability",
    "manual_review_required",
    "manual_review_priority",
    "mixed_event_severity",
    "auto_classification_safe_flag",
    "abnormal_return_adjustment_basis",
    "abnormal_return_adjustment_quality",
    "financial_disclosure_available",
    "financial_disclosure_join_status",
    "financial_disclosure_confidence",
    "earnings_forecast_revision_direction_v2",
    "dividend_revision_direction_v2",
    "liquidity_status",
    "avg_turnover_20d",
    "sector33_name",
    "topix_size_category",
    "horizon_flags",
    "product_risk_flags",
    "ranking_score_components",
    "commercial_abnormal_reaction_score_components",
    "evidence_summary",
    "primary_signal",
    "secondary_signal",
    "ranking_reason",
    "validation_asof_date",
    "feature_cutoff_date",
    "label_start_date",
    "label_end_date",
    "leakage_risk_flag",
    "validation_eligible_flag",
    "validation_exclusion_reason",
    "leakage_safe_flag",
    "leakage_safety_status",
    "leakage_safety_reason",
    "timing_safety_status",
    "same_day_reaction_risk_flag",
    "future_return_score_contamination_flag",
    "score_component_leakage_risk",
    "safe_label_start_date",
    "safe_label_end_date",
    "safe_horizon_available_flag",
    "walk_forward_eligible_flag",
    "walk_forward_exclusion_reason",
    "strict_validation_eligible_flag",
    "strict_validation_exclusion_reason",
    "estimated_cost_bps",
    "liquidity_capacity_bucket",
    "cost_model_basis",
]

for _h in HORIZONS:
    PANEL_COLUMNS.extend(
        [
            f"fwd_ret_{_h}d",
            f"fwd_abnormal_ret_{_h}d",
            f"fwd_sector_adjusted_ret_{_h}d",
            f"fwd_sector_size_adjusted_ret_{_h}d",
            f"fwd_ret_{_h}d_status",
        ]
    )


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _truthy(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_num(series: pd.DataFrame | pd.Series, col: str) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    if col not in series:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return pd.to_numeric(series[col], errors="coerce")


def _corr(x: pd.Series, y: pd.Series, *, method: str = "pearson") -> float:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(data) < 2:
        return np.nan
    if method == "spearman":
        data = data.rank(method="average")
    return float(data["x"].corr(data["y"], method="pearson"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"ticker": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _date_string_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    return pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _first_time_after_close(value: object) -> bool:
    text = _text(value).strip()
    if not text:
        return False
    if len(text) == 5:
        text = f"{text}:00"
    return text >= "15:00:00"


def _score_audit_text(row: pd.Series) -> str:
    return " | ".join(_text(row.get(col)) for col in SCORE_AUDIT_TEXT_FIELDS).lower()


def _matched_score_patterns(row: pd.Series) -> list[str]:
    text = _score_audit_text(row)
    return [pattern for pattern in SUSPICIOUS_SCORE_PATTERNS if pattern in text]


def _strict_exclusion(row: pd.Series) -> str:
    reasons: list[str] = []
    if row.get("leakage_safety_status") != "safe":
        reasons.append(str(row.get("leakage_safety_status") or "not_safe"))
    if _truthy(row.get("future_return_score_contamination_flag")):
        reasons.append("ranking_score_component_contamination")
    if _truthy(row.get("same_day_reaction_risk_flag")):
        reasons.append("same_day_reaction_risk")
    if pd.isna(pd.to_numeric(pd.Series([row.get("ranking_score")]), errors="coerce").iloc[0]):
        reasons.append("missing_ranking_score")
    return "|".join(dict.fromkeys(reasons))


def load_ranking_outputs(analysis_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, filename in RANKING_FILES.items():
        df = _read_csv(analysis_dir / filename)
        if df.empty:
            continue
        if "ranking_label" not in df:
            df["ranking_label"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _cost_proxy(row: pd.Series) -> tuple[float, str, str]:
    liquid = _truthy(row.get("is_liquid_enough"))
    turnover = pd.to_numeric(pd.Series([row.get("avg_turnover_20d")]), errors="coerce").iloc[0]
    status = str(row.get("liquidity_status") or "").lower()
    if liquid and not pd.isna(turnover) and turnover >= 1_000_000_000:
        return 5.0, "high_capacity", "turnover_and_liquidity_flag_proxy"
    if liquid or status == "valid":
        return 15.0, "medium_capacity", "turnover_and_liquidity_flag_proxy"
    return 35.0, "limited_or_unknown_capacity", "conservative_unknown_liquidity_proxy"


def build_event_validation_panel(rankings: pd.DataFrame, *, asof: str | None = None) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    out = rankings.copy()
    asof = asof or jst_now_iso()
    out["DiscDate"] = _date_string_col(out, "DiscDate")
    out["reaction_base_date"] = _date_string_col(out, "reaction_base_date")
    out["validation_asof_date"] = asof
    out["feature_cutoff_date"] = out["DiscDate"]
    out["label_start_date"] = out["reaction_base_date"].fillna(out["DiscDate"])
    base_dates = pd.to_datetime(out["label_start_date"], errors="coerce")
    out["label_end_date"] = (base_dates + pd.to_timedelta(20, unit="D")).dt.strftime("%Y-%m-%d")
    out["safe_label_start_date"] = out["label_start_date"]
    out["safe_label_end_date"] = out["label_end_date"]

    leakage_reasons: list[str] = []
    safety_statuses: list[str] = []
    timing_statuses: list[str] = []
    same_day_risks: list[bool] = []
    contamination_flags: list[bool] = []
    contamination_details: list[str] = []
    safe_horizon_flags: list[bool] = []
    for _, row in out.iterrows():
        reasons: list[str] = []
        disc = pd.to_datetime(row.get("DiscDate"), errors="coerce")
        react = pd.to_datetime(row.get("reaction_base_date"), errors="coerce")
        label_start = pd.to_datetime(row.get("label_start_date"), errors="coerce")
        after_close = _first_time_after_close(row.get("first_time") or row.get("DiscTime"))
        same_day = not pd.isna(disc) and not pd.isna(react) and react == disc
        if not pd.isna(disc) and not pd.isna(react) and react < disc:
            reasons.append("reaction_base_before_disclosure")
            status = "unsafe"
            timing_status = "unsafe"
        elif pd.isna(disc) or pd.isna(react):
            reasons.append("missing_or_invalid_disc_or_reaction_base_date")
            status = "insufficient_timing_metadata"
            timing_status = "insufficient_timing_metadata"
        elif after_close and same_day:
            reasons.append("timing_limited_after_close_same_day_base")
            status = "timing_limited"
            timing_status = "after_close_same_day_label"
        else:
            status = "safe"
            timing_status = "safe"
        if "20d" in str(row.get("horizon_flags") or ""):
            reasons.append("20d_horizon_unavailable")
        matched = _matched_score_patterns(row)
        if matched:
            reasons.append("possible_score_component_leakage")
            contamination_details.append("|".join(matched))
            contamination_flags.append(True)
            if status == "safe":
                status = "score_component_limited"
        else:
            contamination_details.append("")
            contamination_flags.append(False)
        same_day_risks.append(bool(after_close and same_day))
        safe_horizon_flags.append(bool(not pd.isna(label_start) and not pd.isna(disc) and label_start >= disc))
        safety_statuses.append(status)
        timing_statuses.append(timing_status)
        leakage_reasons.append("|".join(reasons))
    out["leakage_risk_flag"] = [bool(reason) for reason in leakage_reasons]
    out["leakage_safety_status"] = safety_statuses
    out["leakage_safety_reason"] = leakage_reasons
    out["timing_safety_status"] = timing_statuses
    out["same_day_reaction_risk_flag"] = same_day_risks
    out["future_return_score_contamination_flag"] = contamination_flags
    out["score_component_leakage_risk"] = contamination_details
    out["safe_horizon_available_flag"] = safe_horizon_flags

    exclusions: list[str] = []
    for reason, (_, row) in zip(leakage_reasons, out.iterrows(), strict=True):
        row_reasons: list[str] = []
        if reason:
            row_reasons.append(reason)
        if pd.isna(pd.to_numeric(pd.Series([row.get("ranking_score")]), errors="coerce").iloc[0]):
            row_reasons.append("missing_ranking_score")
        if not _truthy(row.get("auto_classification_safe_flag")) and str(row.get("manual_review_priority")) == "high":
            row_reasons.append("high_priority_manual_review")
        exclusions.append("|".join(row_reasons) if row_reasons else "")
    out["validation_exclusion_reason"] = exclusions
    out["validation_eligible_flag"] = [not bool(reason) for reason in exclusions]
    out["leakage_safe_flag"] = (
        out["leakage_safety_status"].isin(["safe", "score_component_limited"])
        & out["safe_horizon_available_flag"].astype(bool)
        & ~out["same_day_reaction_risk_flag"].astype(bool)
    )
    out["walk_forward_exclusion_reason"] = np.where(
        out["leakage_safe_flag"].astype(bool),
        "",
        out["leakage_safety_reason"].replace("", "not_leakage_safe"),
    )
    out["walk_forward_eligible_flag"] = out["leakage_safe_flag"].astype(bool)
    strict_reasons = [_strict_exclusion(row) for _, row in out.iterrows()]
    out["strict_validation_exclusion_reason"] = strict_reasons
    out["strict_validation_eligible_flag"] = [not bool(reason) for reason in strict_reasons]

    for h in HORIZONS:
        out[f"fwd_ret_{h}d"] = _safe_num(out, f"ret_{h}d")
        out[f"fwd_abnormal_ret_{h}d"] = _safe_num(out, f"abnormal_ret_{h}d")
        out[f"fwd_sector_adjusted_ret_{h}d"] = _safe_num(out, f"sector_adjusted_return_{h}d")
        out[f"fwd_sector_size_adjusted_ret_{h}d"] = _safe_num(out, f"sector_size_adjusted_return_{h}d")
        status = out.get(f"ret_{h}d_status", pd.Series("", index=out.index)).fillna("").astype(str)
        missing = out[f"fwd_ret_{h}d"].isna()
        out[f"fwd_ret_{h}d_status"] = np.where((status == "valid") & ~missing, "valid", np.where(missing, "missing", status))

    costs = [_cost_proxy(row) for _, row in out.iterrows()]
    out["estimated_cost_bps"] = [x[0] for x in costs]
    out["liquidity_capacity_bucket"] = [x[1] for x in costs]
    out["cost_model_basis"] = [x[2] for x in costs]

    for col in PANEL_COLUMNS:
        if col not in out:
            out[col] = np.nan
    return out[PANEL_COLUMNS]


def _ic_rows(panel: pd.DataFrame, group_col: str, label_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if panel.empty or group_col not in panel:
        return rows
    score = _safe_num(panel, "ranking_score")
    for group, data in panel.assign(_score=score).groupby(group_col, dropna=False):
        for h in HORIZONS:
            label = _safe_num(data, label_col.format(h=h))
            valid = data[["_score"]].assign(_label=label).dropna()
            if len(valid) < MIN_GROUP_SAMPLE:
                rows.append({"group_by": group_col, "group": group, "horizon": f"{h}d", "label": label_col.format(h=h), "sample_count": int(len(valid)), "status": "insufficient_sample"})
                continue
            pearson = _corr(valid["_score"], valid["_label"], method="pearson")
            spearman = _corr(valid["_score"], valid["_label"], method="spearman")
            daily_ics = []
            for _, date_group in data.assign(_label=label).dropna(subset=["_score", "_label"]).groupby("DiscDate", dropna=False):
                if len(date_group) >= MIN_DATE_SAMPLE:
                    daily_ics.append(_corr(date_group["_score"], date_group["_label"], method="spearman"))
            daily = pd.Series(daily_ics, dtype="float64").dropna()
            std = float(daily.std(ddof=1)) if len(daily) > 1 else np.nan
            rows.append(
                {
                    "group_by": group_col,
                    "group": group,
                    "horizon": f"{h}d",
                    "label": label_col.format(h=h),
                    "rank_ic_spearman": spearman,
                    "rank_ic_pearson": pearson,
                    "sample_count": int(len(valid)),
                    "valid_date_count": int(len(daily)),
                    "positive_ic_date_ratio": float((daily > 0).mean()) if len(daily) else np.nan,
                    "mean_daily_ic": float(daily.mean()) if len(daily) else np.nan,
                    "median_daily_ic": float(daily.median()) if len(daily) else np.nan,
                    "ic_std": std,
                    "ic_t_stat": float(daily.mean() / (std / np.sqrt(len(daily)))) if len(daily) > 1 and std and not np.isnan(std) else np.nan,
                    "status": "ok" if len(daily) else "insufficient_date_sample",
                }
            )
    return rows


def build_rank_ic(panel: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "ranking_label",
        "event_direction_final",
        "event_confidence_tier",
        "auto_classification_safe_flag",
        "product_acceptability",
        "sector33_name",
        "topix_size_category",
        "liquidity_status",
    ]
    rows: list[dict[str, Any]] = []
    for label_col in ["fwd_abnormal_ret_{h}d", "fwd_sector_size_adjusted_ret_{h}d"]:
        rows.extend(_ic_rows(panel, "overall", label_col) if "overall" in panel else [])
        work = panel.copy()
        work["overall"] = "overall"
        rows.extend(_ic_rows(work, "overall", label_col))
        for group in groups:
            rows.extend(_ic_rows(panel, group, label_col))
    return pd.DataFrame(rows)


def _monotonicity(means: pd.Series) -> float:
    vals = means.dropna().to_numpy()
    if len(vals) < 2:
        return np.nan
    diffs = np.diff(vals)
    return float((diffs >= 0).mean())


def build_quantile_spread(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for score_col in ["ranking_score", "commercial_abnormal_reaction_score"]:
        score = _safe_num(panel, score_col)
        for label_tpl, label_kind in [
            ("fwd_ret_{h}d", "gross"),
            ("fwd_abnormal_ret_{h}d", "abnormal"),
            ("fwd_sector_size_adjusted_ret_{h}d", "sector_size_adjusted"),
        ]:
            for h in HORIZONS:
                label_col = label_tpl.format(h=h)
                data = panel.assign(_score=score, _label=_safe_num(panel, label_col)).dropna(subset=["_score", "_label"])
                if len(data) < MIN_GROUP_SAMPLE:
                    rows.append({"score": score_col, "label": label_col, "horizon": f"{h}d", "return_basis": label_kind, "sample_count": int(len(data)), "status": "insufficient_sample"})
                    continue
                q = min(5, int(len(data)))
                data = data.copy()
                data["quantile"] = pd.qcut(data["_score"].rank(method="first"), q=q, labels=False, duplicates="drop") + 1
                grouped = data.groupby("quantile")["_label"]
                means = grouped.mean()
                top = data[data["quantile"] == data["quantile"].max()]["_label"]
                bottom = data[data["quantile"] == data["quantile"].min()]["_label"]
                spread = float(top.mean() - bottom.mean()) if len(top) and len(bottom) else np.nan
                spread_series = pd.concat([top.reset_index(drop=True), -bottom.reset_index(drop=True)], axis=1).sum(axis=1, min_count=2).dropna()
                std = spread_series.std(ddof=1) if len(spread_series) > 1 else np.nan
                rows.append(
                    {
                        "score": score_col,
                        "label": label_col,
                        "horizon": f"{h}d",
                        "return_basis": label_kind,
                        "sample_count": int(len(data)),
                        "quantile_count": int(data["quantile"].nunique()),
                        "top_quantile_mean_return": float(top.mean()) if len(top) else np.nan,
                        "bottom_quantile_mean_return": float(bottom.mean()) if len(bottom) else np.nan,
                        "top_bottom_spread": spread,
                        "top_bottom_spread_t_stat": float(spread / (std / np.sqrt(len(spread_series)))) if len(spread_series) > 1 and std and not np.isnan(std) else np.nan,
                        "top_bottom_hit_rate": float((top > bottom.mean()).mean()) if len(top) and len(bottom) else np.nan,
                        "monotonicity_score": _monotonicity(means),
                        "sample_count_by_quantile": json.dumps(grouped.size().to_dict(), sort_keys=True),
                        "status": "ok" if len(top) >= MIN_QUANTILE_SAMPLE and len(bottom) >= MIN_QUANTILE_SAMPLE else "insufficient_quantile_sample",
                    }
                )
    return pd.DataFrame(rows)


def build_event_study(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    buckets = [
        "ranking_label",
        "event_classification_detail",
        "event_direction_final",
        "event_confidence_tier",
        "earnings_forecast_revision_direction_v2",
        "dividend_revision_direction_v2",
        "manual_review_priority",
        "mixed_event_severity",
    ]
    for bucket in buckets:
        if bucket not in panel:
            continue
        for group, data in panel.groupby(bucket, dropna=False):
            for h in HORIZONS:
                label = _safe_num(data, f"fwd_abnormal_ret_{h}d").dropna()
                if len(label) < MIN_GROUP_SAMPLE:
                    rows.append({"bucket": bucket, "group": group, "window": f"0_{h}d", "sample_count": int(len(label)), "status": "insufficient_sample"})
                    continue
                std = label.std(ddof=1)
                rows.append(
                    {
                        "bucket": bucket,
                        "group": group,
                        "window": f"0_{h}d",
                        "sample_count": int(len(label)),
                        "aar": float(label.mean()),
                        "car": float(label.mean()),
                        "median_car": float(label.median()),
                        "win_rate": float((label > 0).mean()),
                        "t_stat": float(label.mean() / (std / np.sqrt(len(label)))) if std and not np.isnan(std) else np.nan,
                        "bootstrap_ci_low": float(label.sample(frac=1.0, replace=True, random_state=17).mean()) if len(label) else np.nan,
                        "bootstrap_ci_high": float(label.sample(frac=1.0, replace=True, random_state=29).mean()) if len(label) else np.nan,
                        "status": "ok",
                    }
                )
    return pd.DataFrame(rows)


def build_liquidity_cost(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket, data in panel.groupby("liquidity_capacity_bucket", dropna=False):
        for h in HORIZONS:
            gross = _safe_num(data, f"fwd_abnormal_ret_{h}d").dropna()
            if len(gross) < MIN_GROUP_SAMPLE:
                rows.append({"liquidity_capacity_bucket": bucket, "horizon": f"{h}d", "sample_count": int(len(gross)), "status": "insufficient_sample"})
                continue
            cost = _safe_num(data.loc[gross.index], "estimated_cost_bps") / 10000.0
            net = gross - cost
            rows.append(
                {
                    "liquidity_capacity_bucket": bucket,
                    "horizon": f"{h}d",
                    "sample_count": int(len(gross)),
                    "estimated_cost_bps_mean": float(cost.mean() * 10000.0),
                    "gross_validation_spread": float(gross.mean()),
                    "net_validation_spread": float(net.mean()),
                    "cost_model_basis": data["cost_model_basis"].mode().iloc[0] if "cost_model_basis" in data and not data["cost_model_basis"].mode().empty else "unknown",
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def build_sector_size_neutral(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for neutral_col, group_col in [
        ("fwd_sector_adjusted_ret_{h}d", "sector33_name"),
        ("fwd_sector_size_adjusted_ret_{h}d", "topix_size_category"),
        ("fwd_sector_size_adjusted_ret_{h}d", "sector33_name"),
    ]:
        for h in HORIZONS:
            work = panel.assign(_score=_safe_num(panel, "ranking_score"), _label=_safe_num(panel, neutral_col.format(h=h))).dropna(subset=["_score", "_label"])
            if len(work) < MIN_GROUP_SAMPLE or group_col not in work:
                rows.append({"neutralization": group_col, "horizon": f"{h}d", "sample_count": int(len(work)), "status": "insufficient_sample"})
                continue
            ics = []
            spreads = []
            sizes = work[group_col].value_counts(normalize=True)
            for _, data in work.groupby(group_col, dropna=False):
                if len(data) >= MIN_DATE_SAMPLE:
                    ics.append(_corr(data["_score"], data["_label"], method="spearman"))
                    hi = data[data["_score"] >= data["_score"].median()]["_label"].mean()
                    lo = data[data["_score"] < data["_score"].median()]["_label"].mean()
                    spreads.append(hi - lo)
            rows.append(
                {
                    "neutralization": group_col,
                    "horizon": f"{h}d",
                    "sample_count": int(len(work)),
                    "sector_neutral_rank_ic": float(pd.Series(ics).mean()) if group_col == "sector33_name" and ics else np.nan,
                    "size_neutral_rank_ic": float(pd.Series(ics).mean()) if group_col == "topix_size_category" and ics else np.nan,
                    "sector_size_neutral_rank_ic": float(pd.Series(ics).mean()) if neutral_col.startswith("fwd_sector_size") and ics else np.nan,
                    "within_group_quantile_spread": float(pd.Series(spreads).mean()) if spreads else np.nan,
                    "concentration_score": float((sizes**2).sum()) if len(sizes) else np.nan,
                    "status": "ok" if ics else "insufficient_group_sample",
                }
            )
    return pd.DataFrame(rows)


def build_stability(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = panel.copy()
    work["event_month"] = pd.to_datetime(work["DiscDate"], errors="coerce").dt.strftime("%Y-%m")
    for group_col in ["event_month", "sector33_name", "topix_size_category"]:
        for group, data in work.groupby(group_col, dropna=False):
            label = _safe_num(data, "fwd_abnormal_ret_5d")
            valid = data.assign(_label=label, _score=_safe_num(data, "ranking_score")).dropna(subset=["_label"])
            if len(valid) < MIN_DATE_SAMPLE:
                status = "insufficient_sample"
                rank_ic = np.nan
                spread = np.nan
            else:
                rank_ic = _corr(valid["_score"], valid["_label"], method="spearman") if valid["_score"].notna().sum() >= MIN_DATE_SAMPLE else np.nan
                spread = valid[valid["_score"] >= valid["_score"].median()]["_label"].mean() - valid[valid["_score"] < valid["_score"].median()]["_label"].mean()
                status = "ok"
            rows.append(
                {
                    "group_by": group_col,
                    "group": group,
                    "sample_count": int(len(valid)),
                    "average_score": float(_safe_num(valid, "ranking_score").mean()) if len(valid) else np.nan,
                    "average_forward_abnormal_return": float(valid["_label"].mean()) if len(valid) else np.nan,
                    "rank_ic": rank_ic,
                    "hit_rate": float((valid["_label"] > 0).mean()) if len(valid) else np.nan,
                    "spread": spread,
                    "drift_or_decay_flag": bool(not pd.isna(rank_ic) and rank_ic < 0),
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def build_score_component_contamination_audit(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = [
        "ticker",
        "DiscDate",
        "ranking_label",
        "matched_patterns",
        "audited_fields",
        "ranking_score_usable_for_pre_reaction_validation",
        "pre_reaction_score_available",
        "pre_reaction_score_basis",
        "status",
    ]
    if panel.empty:
        return pd.DataFrame(columns=columns)
    for _, row in panel.iterrows():
        matched = _matched_score_patterns(row)
        if not matched:
            continue
        rows.append(
            {
                "ticker": row.get("ticker"),
                "DiscDate": row.get("DiscDate"),
                "ranking_label": row.get("ranking_label"),
                "matched_patterns": "|".join(matched),
                "audited_fields": "|".join(SCORE_AUDIT_TEXT_FIELDS),
                "ranking_score_usable_for_pre_reaction_validation": False,
                "pre_reaction_score_available": False,
                "pre_reaction_score_basis": "not_available_without_post_event_reaction_fields",
                "status": "score_component_limited",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _top_bottom_spread(data: pd.DataFrame, score_col: str, label_col: str) -> float:
    valid = data.assign(_score=_safe_num(data, score_col), _label=_safe_num(data, label_col)).dropna(subset=["_score", "_label"])
    if len(valid) < MIN_DATE_SAMPLE:
        return np.nan
    median = valid["_score"].median()
    top = valid[valid["_score"] >= median]["_label"]
    bottom = valid[valid["_score"] < median]["_label"]
    return float(top.mean() - bottom.mean()) if len(top) and len(bottom) else np.nan


def build_date_level_validation(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = [
        "event_date",
        "row_count",
        "rank_ic_1d",
        "rank_ic_3d",
        "rank_ic_5d",
        "rank_ic_20d",
        "top_bottom_spread_1d",
        "top_bottom_spread_5d",
        "positive_ic_flag",
        "extreme_outlier_flag",
        "date_concentration_risk_flag",
        "status",
    ]
    if panel.empty:
        return pd.DataFrame(columns=columns)
    total = len(panel)
    work = panel.copy()
    work["event_date"] = pd.to_datetime(work["DiscDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    for event_date, data in work.groupby("event_date", dropna=False):
        row: dict[str, Any] = {"event_date": event_date, "row_count": int(len(data))}
        sufficient = len(data) >= MIN_DATE_SAMPLE
        for h in HORIZONS:
            label = f"fwd_abnormal_ret_{h}d"
            valid = data.assign(_score=_safe_num(data, "ranking_score"), _label=_safe_num(data, label)).dropna(subset=["_score", "_label"])
            row[f"rank_ic_{h}d"] = _corr(valid["_score"], valid["_label"], method="spearman") if sufficient and len(valid) >= MIN_DATE_SAMPLE else np.nan
        row["top_bottom_spread_1d"] = _top_bottom_spread(data, "ranking_score", "fwd_abnormal_ret_1d") if sufficient else np.nan
        row["top_bottom_spread_5d"] = _top_bottom_spread(data, "ranking_score", "fwd_abnormal_ret_5d") if sufficient else np.nan
        row["positive_ic_flag"] = bool(not pd.isna(row["rank_ic_1d"]) and row["rank_ic_1d"] > 0)
        row["extreme_outlier_flag"] = bool(any(abs(row.get(f"rank_ic_{h}d") or 0.0) >= 0.95 for h in HORIZONS if not pd.isna(row.get(f"rank_ic_{h}d"))))
        row["date_concentration_risk_flag"] = bool(total > 0 and len(data) / total >= 0.35)
        row["status"] = "ok" if sufficient else "insufficient_sample"
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_walk_forward_validation(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "split_id",
        "train_start_date",
        "train_end_date",
        "validation_start_date",
        "validation_end_date",
        "train_rows",
        "validation_rows",
        "validation_date_count",
        "validation_rank_ic_1d",
        "validation_rank_ic_3d",
        "validation_rank_ic_5d",
        "validation_rank_ic_20d",
        "validation_top_bottom_spread_1d",
        "validation_top_bottom_spread_3d",
        "validation_top_bottom_spread_5d",
        "validation_top_bottom_spread_20d",
        "validation_hit_rate",
        "status",
        "limitation_reason",
    ]
    if panel.empty:
        return pd.DataFrame(columns=columns)
    work = panel.copy()
    if "walk_forward_eligible_flag" in work:
        work = work[work["walk_forward_eligible_flag"].astype(bool)].copy()
    work["_event_date"] = pd.to_datetime(work["DiscDate"], errors="coerce")
    work = work.dropna(subset=["_event_date"]).sort_values("_event_date")
    dates = list(work["_event_date"].drop_duplicates().sort_values())
    rows: list[dict[str, Any]] = []
    if len(dates) < 2:
        rows.append({"split_id": "split_001", "status": "insufficient_sample", "limitation_reason": "fewer_than_two_event_dates"})
        return pd.DataFrame(rows, columns=columns)
    for idx in range(1, len(dates)):
        train_dates = dates[:idx]
        validation_dates = [dates[idx]]
        train = work[work["_event_date"].isin(train_dates)]
        validation = work[work["_event_date"].isin(validation_dates)]
        row: dict[str, Any] = {
            "split_id": f"split_{idx:03d}",
            "train_start_date": pd.Timestamp(train_dates[0]).strftime("%Y-%m-%d"),
            "train_end_date": pd.Timestamp(train_dates[-1]).strftime("%Y-%m-%d"),
            "validation_start_date": pd.Timestamp(validation_dates[0]).strftime("%Y-%m-%d"),
            "validation_end_date": pd.Timestamp(validation_dates[-1]).strftime("%Y-%m-%d"),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "validation_date_count": int(len(validation_dates)),
        }
        if len(train) < MIN_WALK_FORWARD_TRAIN_ROWS or len(validation) < MIN_WALK_FORWARD_VALIDATION_ROWS:
            row["status"] = "insufficient_sample"
            row["limitation_reason"] = "minimum_train_or_validation_rows_not_met"
        else:
            for h in HORIZONS:
                label = f"fwd_abnormal_ret_{h}d"
                valid = validation.assign(_score=_safe_num(validation, "ranking_score"), _label=_safe_num(validation, label)).dropna(subset=["_score", "_label"])
                row[f"validation_rank_ic_{h}d"] = _corr(valid["_score"], valid["_label"], method="spearman") if len(valid) >= MIN_DATE_SAMPLE else np.nan
                row[f"validation_top_bottom_spread_{h}d"] = _top_bottom_spread(validation, "ranking_score", label)
            label_1d = _safe_num(validation, "fwd_abnormal_ret_1d")
            row["validation_hit_rate"] = float((label_1d.dropna() > 0).mean()) if label_1d.notna().any() else np.nan
            row["status"] = "ok" if not pd.isna(row.get("validation_rank_ic_1d")) else "insufficient_sample"
            row["limitation_reason"] = "" if row["status"] == "ok" else "insufficient_valid_labels"
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _dominance_ratio(panel: pd.DataFrame, col: str) -> float:
    if panel.empty or col not in panel:
        return 0.0
    counts = panel[col].fillna("missing").astype(str).value_counts()
    return float(counts.iloc[0] / len(panel)) if len(counts) else 0.0


def _risk_level(ratio: float) -> str:
    if ratio <= 0:
        return "not_evaluable"
    if ratio >= 0.5:
        return "high"
    if ratio >= 0.3:
        return "medium"
    return "low"


def build_concentration_audit(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(
            [
                {
                    "top_date_contribution_ratio": 0.0,
                    "top_sector_contribution_ratio": 0.0,
                    "top_ticker_contribution_ratio": 0.0,
                    "repeated_ticker_count": 0,
                    "duplicate_event_count": 0,
                    "date_concentration_risk_level": "not_evaluable",
                    "sector_concentration_risk_level": "not_evaluable",
                    "ticker_concentration_risk_level": "not_evaluable",
                    "concentration_risk_level": "not_evaluable",
                }
            ]
        )
    work = panel.copy()
    work["_event_date"] = pd.to_datetime(work["DiscDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    date_ratio = _dominance_ratio(work, "_event_date")
    sector_ratio = _dominance_ratio(work, "sector33_name")
    ticker_ratio = _dominance_ratio(work, "ticker")
    repeated_tickers = int((work["ticker"].fillna("").astype(str).value_counts() > 1).sum()) if "ticker" in work else 0
    duplicate_events = int(work.duplicated([c for c in ["ticker", "DiscDate"] if c in work]).sum())
    levels = [_risk_level(date_ratio), _risk_level(sector_ratio), _risk_level(ticker_ratio)]
    overall = "high" if "high" in levels else "medium" if "medium" in levels else "low"
    return pd.DataFrame(
        [
            {
                "top_date_contribution_ratio": date_ratio,
                "top_sector_contribution_ratio": sector_ratio,
                "top_ticker_contribution_ratio": ticker_ratio,
                "repeated_ticker_count": repeated_tickers,
                "duplicate_event_count": duplicate_events,
                "date_concentration_risk_level": _risk_level(date_ratio),
                "sector_concentration_risk_level": _risk_level(sector_ratio),
                "ticker_concentration_risk_level": _risk_level(ticker_ratio),
                "concentration_risk_level": overall,
            }
        ]
    )


def _status_from_score(score: float, *, rows: int = 1) -> str:
    if rows == 0:
        return "not_evaluable"
    if score < 35:
        return "research_only"
    if score < 55:
        return "promising_but_unstable"
    if score < 75:
        return "usable_with_strict_limitations"
    return "finance_grade_candidate"


def score_leakage_safe_validation(
    panel: pd.DataFrame,
    leakage_safe_panel: pd.DataFrame,
    walk_forward: pd.DataFrame,
    date_level: pd.DataFrame,
    contamination: pd.DataFrame,
    concentration: pd.DataFrame,
) -> dict[str, Any]:
    rows = len(panel)
    safe_rows = len(leakage_safe_panel)
    unsafe_rows = int(panel["leakage_safety_status"].eq("unsafe").sum()) if "leakage_safety_status" in panel else 0
    timing_limited_rows = int(panel["leakage_safety_status"].eq("timing_limited").sum()) if "leakage_safety_status" in panel else 0
    contamination_rows = int(len(contamination))
    safe_ratio = float(safe_rows / rows) if rows else 0.0
    safe_score = min(100.0, safe_ratio * 55.0 + min(safe_rows / 250.0, 1.0) * 20.0 + (0 if unsafe_rows else 15.0) + (0 if timing_limited_rows else 10.0))
    if unsafe_rows:
        safe_score = min(safe_score, 60.0)
    if contamination_rows:
        safe_score = min(safe_score, 54.0)
    if safe_rows < 50:
        safe_score = min(safe_score, 45.0)
    ranking_usable = contamination_rows == 0
    pre_reaction_available = False
    if not ranking_usable:
        strict_score = 0.0
        strict_status = "not_evaluable"
        strict_reason = "ranking_score_contains_post_event_or_future_return_components"
    elif safe_rows < MIN_GROUP_SAMPLE:
        strict_score = 0.0
        strict_status = "not_evaluable"
        strict_reason = "insufficient_strict_safe_rows"
    else:
        strict_score = min(safe_score, 45.0)
        strict_status = _status_from_score(strict_score, rows=safe_rows)
        strict_reason = "pre_reaction_score_not_separately_available"
    ok_splits = int(walk_forward["status"].eq("ok").sum()) if not walk_forward.empty and "status" in walk_forward else 0
    insufficient_splits = int(walk_forward["status"].eq("insufficient_sample").sum()) if not walk_forward.empty and "status" in walk_forward else 0
    split_count = int(len(walk_forward))
    wf_score = min(100.0, ok_splits * 18.0 + max(0, split_count - insufficient_splits) * 4.0)
    if ok_splits < 3:
        wf_score = min(wf_score, 35.0)
    wf_status = _status_from_score(wf_score, rows=split_count if split_count else 0)
    sufficient_dates = int(date_level["status"].eq("ok").sum()) if not date_level.empty and "status" in date_level else 0
    date_count = int(len(date_level))
    concentration_level = concentration.iloc[0].get("concentration_risk_level", "not_evaluable") if not concentration.empty else "not_evaluable"
    integrity_score = min(100.0, safe_ratio * 35.0 + min(sufficient_dates / 8.0, 1.0) * 25.0 + min(ok_splits / 5.0, 1.0) * 25.0 + (15.0 if concentration_level in {"low", "medium"} else 0.0))
    if concentration_level == "high":
        integrity_score = min(integrity_score, 45.0)
    if not ranking_usable:
        integrity_score = min(integrity_score, 55.0)
    if unsafe_rows:
        integrity_score = min(integrity_score, 60.0)
    return {
        "leakage_safe_signal_validation_score": round(float(safe_score), 2),
        "leakage_safe_signal_validation_status": _status_from_score(safe_score, rows=safe_rows),
        "leakage_safe_panel_rows": int(safe_rows),
        "excluded_unsafe_rows": unsafe_rows,
        "timing_limited_rows": timing_limited_rows,
        "strict_pre_reaction_validation_score": round(float(strict_score), 2),
        "strict_pre_reaction_validation_status": strict_status,
        "strict_pre_reaction_not_evaluable_reason": strict_reason,
        "ranking_score_usable_for_pre_reaction_validation": ranking_usable,
        "pre_reaction_score_available": pre_reaction_available,
        "pre_reaction_score_basis": "not_available_without_clean_pre_reaction_numeric_score",
        "score_component_contamination_rows": contamination_rows,
        "score_component_contamination_examples": contamination.head(5).to_dict(orient="records") if not contamination.empty else [],
        "walk_forward_validation_score": round(float(wf_score), 2),
        "walk_forward_validation_status": wf_status,
        "walk_forward_split_count": split_count,
        "walk_forward_insufficient_sample_splits": insufficient_splits,
        "validation_integrity_score": round(float(integrity_score), 2),
        "validation_integrity_status": _status_from_score(integrity_score, rows=rows),
        "number_of_validation_dates": date_count,
        "dates_with_sufficient_sample": sufficient_dates,
        "median_date_level_ic": float(pd.to_numeric(date_level.get("rank_ic_1d", pd.Series(dtype=float)), errors="coerce").median()) if not date_level.empty else None,
        "worst_date_level_ic": float(pd.to_numeric(date_level.get("rank_ic_1d", pd.Series(dtype=float)), errors="coerce").min()) if not date_level.empty else None,
        "best_date_level_ic": float(pd.to_numeric(date_level.get("rank_ic_1d", pd.Series(dtype=float)), errors="coerce").max()) if not date_level.empty else None,
        "concentration_risk_score": float(concentration.iloc[0].get("top_date_contribution_ratio", 0.0)) if not concentration.empty else 0.0,
        "date_concentration_risk_level": concentration.iloc[0].get("date_concentration_risk_level", "not_evaluable") if not concentration.empty else "not_evaluable",
        "sector_concentration_risk_level": concentration.iloc[0].get("sector_concentration_risk_level", "not_evaluable") if not concentration.empty else "not_evaluable",
        "ticker_concentration_risk_level": concentration.iloc[0].get("ticker_concentration_risk_level", "not_evaluable") if not concentration.empty else "not_evaluable",
        "concentration_risk_level": concentration_level,
        "leakage_safe_validation_limitations": [
            item
            for item in [
                "unsafe_rows_present" if unsafe_rows else "",
                "timing_limited_rows_present" if timing_limited_rows else "",
                "score_component_contamination" if contamination_rows else "",
                "insufficient_walk_forward_splits" if ok_splits < 3 else "",
                "date_or_group_concentration" if concentration_level in {"high", "medium"} else "",
                "small_leakage_safe_sample" if safe_rows < 50 else "",
            ]
            if item
        ],
    }


def _first_ok(df: pd.DataFrame, metric: str) -> float | None:
    if df.empty or metric not in df:
        return None
    mask = df["status"].eq("ok") if "status" in df else pd.Series(True, index=df.index)
    vals = pd.to_numeric(df.loc[mask, metric], errors="coerce").dropna()
    return None if vals.empty else float(vals.iloc[0])


def score_signal_validation(panel: pd.DataFrame, rank_ic: pd.DataFrame, quantile: pd.DataFrame, event_study: pd.DataFrame, liquidity: pd.DataFrame, neutral: pd.DataFrame, stability: pd.DataFrame) -> dict[str, Any]:
    rows = len(panel)
    leakage = int(panel["leakage_risk_flag"].astype(bool).sum()) if "leakage_risk_flag" in panel else 0
    eligible_ratio = float(panel["validation_eligible_flag"].astype(bool).mean()) if rows else 0.0
    insufficient_sections = int(sum((not df.empty and "status" in df and df["status"].astype(str).str.contains("insufficient").any()) for df in [rank_ic, quantile, event_study, liquidity, neutral, stability]))
    best_ic = abs(_first_ok(rank_ic, "rank_ic_spearman") or 0.0)
    best_spread = abs(_first_ok(quantile, "top_bottom_spread") or 0.0)
    event_support = abs(_first_ok(event_study, "t_stat") or 0.0)
    net = _first_ok(liquidity, "net_validation_spread")
    neutral_ic = abs(_first_ok(neutral, "sector_size_neutral_rank_ic") or 0.0)
    stable_ok = 0.0 if stability.empty else float((stability.get("status", pd.Series(dtype=str)).eq("ok")).mean())
    score = (
        min(rows / 250.0, 1.0) * 15
        + eligible_ratio * 15
        + min(best_ic / 0.08, 1.0) * 18
        + min(best_spread / 0.02, 1.0) * 15
        + min(event_support / 2.0, 1.0) * 12
        + (1.0 if net is not None and net > 0 else 0.0) * 8
        + min(neutral_ic / 0.06, 1.0) * 10
        + stable_ok * 7
    )
    if rows < 50:
        score = min(score, 30.0)
    if leakage:
        score = min(score, 35.0)
    if insufficient_sections >= 4:
        score = min(score, 45.0)
    manual_ratio = float(panel["manual_review_required"].map(_truthy).mean()) if rows and "manual_review_required" in panel else 1.0
    if manual_ratio > 0.5:
        score = min(score, 55.0)
    score = round(float(max(0.0, min(100.0, score))), 2)
    if rows == 0:
        status = "not_evaluable"
    elif score < 35:
        status = "research_only"
    elif score < 55:
        status = "promising_but_unstable"
    elif score < 75:
        status = "usable_with_strict_limitations"
    else:
        status = "finance_grade_candidate"
    return {
        "finance_grade_signal_validation_score": score,
        "finance_grade_signal_validation_status": status,
        "validation_panel_rows": int(rows),
        "leakage_risk_rows": leakage,
        "validation_eligible_ratio": eligible_ratio,
        "insufficient_sample_sections": insufficient_sections,
        "manual_review_required_ratio": manual_ratio,
    }


def _summary_payload(
    panel: pd.DataFrame,
    rank_ic: pd.DataFrame,
    quantile: pd.DataFrame,
    event_study: pd.DataFrame,
    liquidity: pd.DataFrame,
    neutral: pd.DataFrame,
    stability: pd.DataFrame,
    leakage_safe_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = score_signal_validation(panel, rank_ic, quantile, event_study, liquidity, neutral, stability)
    return {
        "generated_at_jst": jst_now_iso(),
        **score,
        **(leakage_safe_summary or {}),
        "rank_ic_available": bool(not rank_ic.empty),
        "rank_ic_summary": rank_ic.head(5).to_dict(orient="records") if not rank_ic.empty else [],
        "quantile_spread_available": bool(not quantile.empty),
        "quantile_spread_summary": quantile.head(5).to_dict(orient="records") if not quantile.empty else [],
        "event_study_available": bool(not event_study.empty),
        "event_study_summary": event_study.head(5).to_dict(orient="records") if not event_study.empty else [],
        "net_spread_after_cost_available": bool(not liquidity.empty and "net_validation_spread" in liquidity),
        "liquidity_cost_summary": liquidity.head(5).to_dict(orient="records") if not liquidity.empty else [],
        "sector_size_neutral_validation_available": bool(not neutral.empty),
        "sector_size_neutral_summary": neutral.head(5).to_dict(orient="records") if not neutral.empty else [],
        "stability_validation_available": bool(not stability.empty),
        "stability_summary": stability.head(5).to_dict(orient="records") if not stability.empty else [],
    }


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(_json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _markdown(summary: dict[str, Any]) -> str:
    return f"""# Commercial Signal Validation

Status: **{summary.get('finance_grade_signal_validation_status')}**

Finance-grade signal validation score: **{summary.get('finance_grade_signal_validation_score')}**

- Panel rows: `{summary.get('validation_panel_rows')}`
- Leakage risk rows: `{summary.get('leakage_risk_rows')}`
- Insufficient sample sections: `{summary.get('insufficient_sample_sections')}`
- Rank IC available: `{summary.get('rank_ic_available')}`
- Quantile spread available: `{summary.get('quantile_spread_available')}`
- Event-study available: `{summary.get('event_study_available')}`
- Net spread after cost available: `{summary.get('net_spread_after_cost_available')}`
- Sector/size neutral validation available: `{summary.get('sector_size_neutral_validation_available')}`
- Stability validation available: `{summary.get('stability_validation_available')}`
- Leakage-safe status: `{summary.get('leakage_safe_signal_validation_status')}`
- Strict pre-reaction status: `{summary.get('strict_pre_reaction_validation_status')}`
- Walk-forward status: `{summary.get('walk_forward_validation_status')}`

This report is a research validation diagnostic only.
"""


def _leakage_safe_markdown(summary: dict[str, Any]) -> str:
    limitations = summary.get("leakage_safe_validation_limitations") or []
    limitation_lines = "\n".join(f"- `{item}`" for item in limitations) if limitations else "- `none_detected`"
    return f"""# Commercial Leakage-Safe Validation

Leakage-safe status: **{summary.get('leakage_safe_signal_validation_status')}**

Strict pre-reaction status: **{summary.get('strict_pre_reaction_validation_status')}**

Walk-forward status: **{summary.get('walk_forward_validation_status')}**

- Leakage-safe score: `{summary.get('leakage_safe_signal_validation_score')}`
- Leakage-safe panel rows: `{summary.get('leakage_safe_panel_rows')}`
- Excluded unsafe rows: `{summary.get('excluded_unsafe_rows')}`
- Timing-limited rows: `{summary.get('timing_limited_rows')}`
- Score component contamination rows: `{summary.get('score_component_contamination_rows')}`
- Walk-forward splits: `{summary.get('walk_forward_split_count')}`
- Insufficient walk-forward splits: `{summary.get('walk_forward_insufficient_sample_splits')}`
- Validation integrity status: `{summary.get('validation_integrity_status')}`

## Top Limitations

{limitation_lines}

This report is a conservative research validation diagnostic only.
"""


def build_commercial_signal_validation(
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    prices_path: Path = DEFAULT_PRICES,
    output_dir: Path = DEFAULT_ANALYSIS_DIR,
) -> dict[str, Any]:
    del prices_path  # Existing enriched ranking labels are reused to avoid redefining returns.
    rankings = load_ranking_outputs(analysis_dir)
    panel = build_event_validation_panel(rankings)
    rank_ic = build_rank_ic(panel)
    quantile = build_quantile_spread(panel)
    event_study = build_event_study(panel)
    liquidity = build_liquidity_cost(panel)
    neutral = build_sector_size_neutral(panel)
    stability = build_stability(panel)
    leakage_safe_panel = panel[panel["leakage_safe_flag"].astype(bool)].copy() if "leakage_safe_flag" in panel else panel.iloc[0:0].copy()
    contamination = build_score_component_contamination_audit(panel)
    walk_forward = build_walk_forward_validation(leakage_safe_panel)
    date_level = build_date_level_validation(leakage_safe_panel)
    concentration = build_concentration_audit(leakage_safe_panel)
    leakage_safe_summary = score_leakage_safe_validation(panel, leakage_safe_panel, walk_forward, date_level, contamination, concentration)
    summary = _summary_payload(panel, rank_ic, quantile, event_study, liquidity, neutral, stability, leakage_safe_summary)

    _write_csv(output_dir / OUTPUT_FILES["panel"], panel)
    _write_csv(output_dir / OUTPUT_FILES["rank_ic"], rank_ic)
    _write_csv(output_dir / OUTPUT_FILES["quantile_spread"], quantile)
    _write_csv(output_dir / OUTPUT_FILES["event_study"], event_study)
    _write_csv(output_dir / OUTPUT_FILES["liquidity_cost"], liquidity)
    _write_csv(output_dir / OUTPUT_FILES["sector_size_neutral"], neutral)
    _write_csv(output_dir / OUTPUT_FILES["stability"], stability)
    _write_csv(output_dir / OUTPUT_FILES["leakage_safe_panel"], leakage_safe_panel)
    _write_csv(output_dir / OUTPUT_FILES["walk_forward"], walk_forward)
    _write_csv(output_dir / OUTPUT_FILES["date_level"], date_level)
    _write_csv(output_dir / OUTPUT_FILES["score_component_audit"], contamination)
    _write_csv(output_dir / OUTPUT_FILES["concentration_audit"], concentration)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["leakage_safe_summary"], leakage_safe_summary)
    ensure_parent(output_dir / OUTPUT_FILES["report"])
    (output_dir / OUTPUT_FILES["report"]).write_text(_markdown(summary), encoding="utf-8")
    ensure_parent(output_dir / OUTPUT_FILES["leakage_safe_report"])
    (output_dir / OUTPUT_FILES["leakage_safe_report"]).write_text(_leakage_safe_markdown(leakage_safe_summary), encoding="utf-8")
    return _json_clean(summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    args = parser.parse_args(argv)
    build_commercial_signal_validation(args.analysis_dir, args.prices, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
}

HORIZONS = (1, 3, 5, 20)
MIN_DATE_SAMPLE = 3
MIN_GROUP_SAMPLE = 8
MIN_QUANTILE_SAMPLE = 2

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
    "validation_asof_date",
    "feature_cutoff_date",
    "label_start_date",
    "label_end_date",
    "leakage_risk_flag",
    "validation_eligible_flag",
    "validation_exclusion_reason",
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

    leakage_reasons: list[str] = []
    for _, row in out.iterrows():
        reasons: list[str] = []
        disc = pd.to_datetime(row.get("DiscDate"), errors="coerce")
        react = pd.to_datetime(row.get("reaction_base_date"), errors="coerce")
        if not pd.isna(disc) and not pd.isna(react) and react < disc:
            reasons.append("reaction_base_before_disclosure")
        if "20d" in str(row.get("horizon_flags") or ""):
            reasons.append("20d_horizon_unavailable")
        components = str(row.get("ranking_score_components") or "")
        if any(token in components for token in ["ret_20d", "ret_5d", "abnormal_ret_20d"]):
            reasons.append("possible_leakage_component")
        first_time = str(row.get("first_time") or "")
        if first_time >= "15:00:00" and not pd.isna(disc) and not pd.isna(react) and react == disc:
            reasons.append("timing_limited_after_close_same_day_base")
        leakage_reasons.append("|".join(reasons))
    out["leakage_risk_flag"] = [bool(reason) for reason in leakage_reasons]

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


def _summary_payload(panel: pd.DataFrame, rank_ic: pd.DataFrame, quantile: pd.DataFrame, event_study: pd.DataFrame, liquidity: pd.DataFrame, neutral: pd.DataFrame, stability: pd.DataFrame) -> dict[str, Any]:
    score = score_signal_validation(panel, rank_ic, quantile, event_study, liquidity, neutral, stability)
    return {
        "generated_at_jst": jst_now_iso(),
        **score,
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

This report is a research validation diagnostic only.
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
    summary = _summary_payload(panel, rank_ic, quantile, event_study, liquidity, neutral, stability)

    _write_csv(output_dir / OUTPUT_FILES["panel"], panel)
    _write_csv(output_dir / OUTPUT_FILES["rank_ic"], rank_ic)
    _write_csv(output_dir / OUTPUT_FILES["quantile_spread"], quantile)
    _write_csv(output_dir / OUTPUT_FILES["event_study"], event_study)
    _write_csv(output_dir / OUTPUT_FILES["liquidity_cost"], liquidity)
    _write_csv(output_dir / OUTPUT_FILES["sector_size_neutral"], neutral)
    _write_csv(output_dir / OUTPUT_FILES["stability"], stability)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    ensure_parent(output_dir / OUTPUT_FILES["report"])
    (output_dir / OUTPUT_FILES["report"]).write_text(_markdown(summary), encoding="utf-8")
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

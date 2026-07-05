"""Build final screening rankings from commercial-quality J-Quants liquidity rows."""

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
    counts,
    ensure_parent,
    is_negative_signal,
    is_positive_signal,
    jst_now_iso,
    load_candidates,
    write_json,
)
from jp_stock_analysis.analysis.jquants_liquidity import DEFAULT_OUTPUT as DEFAULT_LIQUIDITY


DEFAULT_POSITIVE_REACTED = ANALYSIS_DIR / "commercial_quality_positive_reacted_ranking.csv"
DEFAULT_POSITIVE_UNREACTED = ANALYSIS_DIR / "commercial_quality_positive_unreacted_ranking.csv"
DEFAULT_NEGATIVE_REACTED = ANALYSIS_DIR / "commercial_quality_negative_reacted_ranking.csv"
DEFAULT_NEGATIVE_REVERSAL = ANALYSIS_DIR / "commercial_quality_negative_reversal_ranking.csv"
DEFAULT_EXCLUDED_AUDIT = ANALYSIS_DIR / "commercial_quality_rankings_excluded_audit.csv"
DEFAULT_UNCLASSIFIED_AUDIT = (
    ANALYSIS_DIR / "commercial_quality_rankings_unclassified_audit.csv"
)
DEFAULT_SUMMARY = ANALYSIS_DIR / "commercial_quality_rankings_summary.json"

ELIGIBLE_PRICE_STATUSES = {
    "full_20d_available",
    "partial_5d_available",
    "partial_1d_available",
}

EVIDENCE_COLUMNS = [
    "classification_confidence",
    "classification_reason",
    "classification_reason_components",
    "mixed_event_flag",
    "mixed_event_reason",
    "manual_review_required",
    "manual_review_reason",
    "ranking_score",
    "ranking_score_components",
    "primary_signal",
    "secondary_signal",
    "evidence_summary",
    "horizon_flags",
    "liquidity_flags",
    "duplicate_event_flag",
    "data_quality_flags",
    "product_risk_flags",
    "product_acceptability",
]

ADJUSTED_ABNORMAL_COLUMNS = [
    "abnormal_return_adjustment_basis",
    "abnormal_return_adjustment_quality",
    "abnormal_return_adjustment_flags",
    "commercial_abnormal_reaction_score",
    "commercial_abnormal_reaction_score_components",
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
    "sector_adjusted_return_1d_robust_z",
    "sector_adjusted_return_3d_robust_z",
    "sector_adjusted_return_5d_robust_z",
    "sector_size_adjusted_return_1d_robust_z",
    "sector_size_adjusted_return_3d_robust_z",
    "sector_size_adjusted_return_5d_robust_z",
]

RANKING_COLUMNS = [
    "rank",
    "ranking_label",
    "ranking_reason",
    "DiscDate",
    "ticker",
    "first_time",
    "event_time_bucket",
    "reaction_base_date",
    "event_combo",
    "candidate_bucket",
    "commercial_signal",
    "commercial_candidate_score",
    "earnings_revision_direction",
    "dividend_revision_direction",
    "change_sales_pct",
    "change_op_pct",
    "change_np_pct",
    "change_eps_pct",
    "change_fdivann_pct",
    "ret_1d",
    "universe_ret_1d",
    "abnormal_ret_1d",
    "ret_3d",
    "universe_ret_3d",
    "abnormal_ret_3d",
    "ret_5d",
    "universe_ret_5d",
    "abnormal_ret_5d",
    "ret_20d",
    "universe_ret_20d",
    "abnormal_ret_20d",
    "ret_1d_status",
    "ret_5d_status",
    "ret_20d_status",
    "abnormal_ret_1d_status",
    "abnormal_ret_5d_status",
    "abnormal_ret_20d_status",
    "reaction_flag_v2",
    "abnormal_reaction_flag_1d",
    "abnormal_reaction_flag_5d",
    "abnormal_reaction_flag_20d",
    "avg_turnover_20d",
    "median_turnover_20d",
    "base_turnover_value",
    "turnover_spike_ratio",
    "avg_volume_20d",
    "median_volume_20d",
    "base_volume",
    "volume_spike_ratio",
    "liquidity_bucket",
    "liquidity_status",
    "is_liquid_enough",
    "price_reaction_status_v2",
    "doctypes",
    "discnos",
    *ADJUSTED_ABNORMAL_COLUMNS,
    *EVIDENCE_COLUMNS,
]

AUDIT_COLUMNS = [
    "DiscDate",
    "ticker",
    "commercial_signal",
    "candidate_bucket",
    "event_time_bucket",
    "reaction_base_date",
    "price_reaction_status_v2",
    "ret_1d_status",
    "abnormal_ret_1d_status",
    "liquidity_bucket",
    "liquidity_status",
    "is_liquid_enough",
    "abnormal_ret_1d",
    "exclusion_reasons",
    *ADJUSTED_ABNORMAL_COLUMNS,
    *EVIDENCE_COLUMNS,
]

UNCLASSIFIED_AUDIT_COLUMNS = [
    "DiscDate",
    "ticker",
    "first_time",
    "event_time_bucket",
    "reaction_base_date",
    "event_combo",
    "candidate_bucket",
    "commercial_signal",
    "commercial_candidate_score",
    "earnings_revision_direction",
    "dividend_revision_direction",
    "ret_1d",
    "universe_ret_1d",
    "abnormal_ret_1d",
    "ret_1d_status",
    "abnormal_ret_1d_status",
    "reaction_flag_v2",
    "abnormal_reaction_flag_1d",
    "liquidity_bucket",
    "liquidity_status",
    "is_liquid_enough",
    "price_reaction_status_v2",
    "unclassified_reason",
    "doctypes",
    "discnos",
    *ADJUSTED_ABNORMAL_COLUMNS,
    *EVIDENCE_COLUMNS,
]


def _truthy(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _is_review_signal(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().lower()
    return "needs_review" in text or "review_or_unknown" in text


def _missing_signal(value: object) -> bool:
    return value is None or pd.isna(value) or not str(value).strip()


def _stable_json(payload: dict[str, Any]) -> str:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and np.isnan(value):
            return None
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    return json.dumps(clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pipe(values: list[str]) -> str:
    unique = []
    seen = set()
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return "|".join(unique) if unique else "none"


def _as_float(value: object) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _is_valid_status(row: pd.Series, column: str) -> bool:
    return str(row.get(column, "")).strip() == "valid"


def _horizon_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []
    for horizon in ("1d", "3d", "5d", "20d"):
        ret_status_col = f"ret_{horizon}_status"
        abnormal_status_col = f"abnormal_ret_{horizon}_status"
        if ret_status_col in row.index and not _is_valid_status(row, ret_status_col):
            flags.append(f"ret_{horizon}_unavailable")
        if abnormal_status_col in row.index and not _is_valid_status(row, abnormal_status_col):
            flags.append(f"abnormal_ret_{horizon}_unavailable")
        if f"ret_{horizon}" in row.index and pd.isna(row.get(f"ret_{horizon}")):
            flags.append(f"ret_{horizon}_missing")
        if f"abnormal_ret_{horizon}" in row.index and pd.isna(row.get(f"abnormal_ret_{horizon}")):
            flags.append(f"abnormal_ret_{horizon}_missing")
    return flags


def _liquidity_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []
    bucket = str(row.get("liquidity_bucket", "")).strip().lower()
    if bucket == "low":
        flags.append("low_liquidity")
    elif bucket in {"", "nan", "none", "unknown"}:
        flags.append("unknown_liquidity")
    if not _truthy(row.get("is_liquid_enough")):
        flags.append("below_liquidity_threshold")
    if str(row.get("liquidity_status", "")).strip() != "valid":
        flags.append("invalid_liquidity_status")
    if "avg_turnover_20d" in row.index and pd.isna(row.get("avg_turnover_20d")):
        flags.append("missing_avg_turnover_20d")
    return flags


def _signal_conflict(row: pd.Series) -> tuple[bool, str]:
    signal = row.get("commercial_signal")
    positive = is_positive_signal(signal)
    negative = is_negative_signal(signal)
    reasons: list[str] = []
    if positive and negative:
        reasons.append("commercial_signal_contains_positive_and_negative_terms")
    earnings = str(row.get("earnings_revision_direction", "")).lower()
    dividend = str(row.get("dividend_revision_direction", "")).lower()
    earnings_pos = "upward" in earnings or "increase" in earnings
    earnings_neg = "downward" in earnings or "decrease" in earnings
    dividend_pos = "increase" in dividend or "upward" in dividend
    dividend_neg = "decrease" in dividend or "downward" in dividend
    if (earnings_pos and dividend_neg) or (earnings_neg and dividend_pos):
        reasons.append("earnings_and_dividend_directions_conflict")
    event_combo = str(row.get("event_combo", "")).lower()
    if "mixed" in event_combo or "mixed" in str(signal).lower():
        reasons.append("mixed_event_context")
    return bool(reasons), _pipe(reasons)


def _duplicate_keys(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if df.empty or "ticker" not in df.columns or "DiscDate" not in df.columns:
        return set()
    key_col = "doctypes" if "doctypes" in df.columns else "event_combo" if "event_combo" in df.columns else "commercial_signal"
    if key_col not in df.columns:
        key_col = "DiscDate"
    keys = (
        df[["ticker", "DiscDate", key_col]]
        .fillna("")
        .astype(str)
        .assign(ticker=lambda x: x["ticker"].str.replace(r"\.0$", "", regex=True).str.zfill(4))
    )
    dup_mask = keys.duplicated(["ticker", "DiscDate", key_col], keep=False)
    return {
        (row["ticker"], row["DiscDate"], row[key_col])
        for _, row in keys[dup_mask].iterrows()
    }


def _row_duplicate_flag(row: pd.Series, duplicate_keys: set[tuple[str, str, str]]) -> bool:
    key_value = row.get("doctypes", row.get("event_combo", row.get("commercial_signal", "")))
    ticker = str(row.get("ticker", "")).replace(".0", "").zfill(4)
    return (ticker, str(row.get("DiscDate", "")), str("" if pd.isna(key_value) else key_value)) in duplicate_keys


def _classification_confidence(
    row: pd.Series,
    *,
    mixed_event_flag: bool,
    duplicate_event_flag: bool,
    horizon_flags: list[str],
    liquidity_flags: list[str],
) -> float:
    confidence = 0.92
    if _missing_signal(row.get("commercial_signal")):
        confidence -= 0.35
    if not _is_valid_status(row, "abnormal_ret_1d_status"):
        confidence -= 0.20
    if mixed_event_flag:
        confidence -= 0.25
    if duplicate_event_flag:
        confidence -= 0.20
    if any(flag.startswith("ret_1d") or flag.startswith("abnormal_ret_1d") for flag in horizon_flags):
        confidence -= 0.20
    if liquidity_flags:
        confidence -= 0.10
    if "20d" in "|".join(horizon_flags):
        confidence -= 0.05
    return round(max(0.0, min(1.0, confidence)), 2)


def _build_evidence_columns(df: pd.DataFrame, duplicate_keys: set[tuple[str, str, str]]) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for col in EVIDENCE_COLUMNS:
            if col not in out.columns:
                out[col] = []
        return out
    rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        label = str(row.get("ranking_label", row.get("classification", "")) or "")
        ranking_reason = str(row.get("ranking_reason", row.get("exclusion_reasons", row.get("unclassified_reason", ""))) or "")
        positive = is_positive_signal(row.get("commercial_signal"))
        negative = is_negative_signal(row.get("commercial_signal"))
        mixed_event_flag, mixed_reason = _signal_conflict(row)
        duplicate_event_flag = _row_duplicate_flag(row, duplicate_keys)
        horizon = _horizon_flags(row)
        liquidity = _liquidity_flags(row)
        confidence = _classification_confidence(
            row,
            mixed_event_flag=mixed_event_flag,
            duplicate_event_flag=duplicate_event_flag,
            horizon_flags=horizon,
            liquidity_flags=liquidity,
        )
        adjusted_score = _as_float(row.get("commercial_abnormal_reaction_score"))
        score = adjusted_score
        if score is None:
            score = _as_float(row.get("commercial_candidate_score"))
        if score is None:
            abnormal = _as_float(row.get("abnormal_ret_1d"))
            score = None if abnormal is None else round(abs(abnormal), 8)
        score_components = {
            "commercial_candidate_score": _as_float(row.get("commercial_candidate_score")),
            "commercial_abnormal_reaction_score": adjusted_score,
            "commercial_abnormal_reaction_score_components": row.get("commercial_abnormal_reaction_score_components"),
            "abnormal_ret_1d": _as_float(row.get("abnormal_ret_1d")),
            "sector_adjusted_return_1d": _as_float(row.get("sector_adjusted_return_1d")),
            "size_adjusted_return_1d": _as_float(row.get("size_adjusted_return_1d")),
            "sector_size_adjusted_return_1d": _as_float(row.get("sector_size_adjusted_return_1d")),
            "abnormal_return_adjustment_basis": row.get("abnormal_return_adjustment_basis"),
            "abnormal_return_adjustment_quality": row.get("abnormal_return_adjustment_quality"),
            "turnover_spike_ratio": _as_float(row.get("turnover_spike_ratio")),
            "avg_turnover_20d": _as_float(row.get("avg_turnover_20d")),
            "sort_bucket": label,
        }
        components = {
            "commercial_signal": row.get("commercial_signal"),
            "positive_signal": positive,
            "negative_signal": negative,
            "abnormal_ret_1d": _as_float(row.get("abnormal_ret_1d")),
            "abnormal_ret_1d_status": row.get("abnormal_ret_1d_status"),
            "ret_1d_status": row.get("ret_1d_status"),
            "ret_20d_status": row.get("ret_20d_status"),
            "abnormal_ret_20d_status": row.get("abnormal_ret_20d_status"),
            "event_combo": row.get("event_combo"),
            "doctypes": row.get("doctypes"),
        }
        classification_reason = ranking_reason or "classification derived from signal direction and abnormal 1d reaction"
        if horizon:
            classification_reason = f"{classification_reason}; horizon_flags={_pipe(horizon)}"
        manual_reasons: list[str] = []
        if confidence < 0.70:
            manual_reasons.append("low_classification_confidence")
        if mixed_event_flag:
            manual_reasons.append(mixed_reason)
        if duplicate_event_flag:
            manual_reasons.append("duplicate_event_context_unresolved")
        if _missing_signal(row.get("commercial_signal")):
            manual_reasons.append("missing_commercial_signal")
        if not _is_valid_status(row, "abnormal_ret_1d_status"):
            manual_reasons.append("invalid_abnormal_ret_1d_status")
        if any(flag in liquidity for flag in ["low_liquidity", "unknown_liquidity", "below_liquidity_threshold"]):
            manual_reasons.append("liquidity_limitation_requires_review")
        risk_flags = []
        if horizon:
            risk_flags.extend([f"horizon:{flag}" for flag in horizon])
        if liquidity:
            risk_flags.extend([f"liquidity:{flag}" for flag in liquidity])
        if mixed_event_flag:
            risk_flags.append("mixed_event_context")
        if duplicate_event_flag:
            risk_flags.append("duplicate_event_context")
        data_flags = []
        if horizon:
            data_flags.append("horizon_limitation")
        if liquidity:
            data_flags.append("liquidity_limitation")
        if duplicate_event_flag:
            data_flags.append("duplicate_event_context")
        if _missing_signal(row.get("commercial_signal")):
            data_flags.append("missing_signal")
        manual_required = bool(manual_reasons)
        primary_signal = str(row.get("commercial_signal") or label or "unknown_signal")
        secondary_signal = str(row.get("abnormal_reaction_flag_1d") or row.get("reaction_flag_v2") or _pipe(risk_flags))
        evidence_parts = [
            f"classification={label or 'unclassified'}",
            f"signal={primary_signal}",
            f"abnormal_ret_1d={_as_float(row.get('abnormal_ret_1d'))}",
            f"liquidity={row.get('liquidity_bucket', 'unknown')}",
        ]
        if any("20d" in flag for flag in horizon):
            evidence_parts.append("long_horizon_unavailable")
        basis = str(row.get("abnormal_return_adjustment_basis") or "universe_adjusted_only")
        quality = str(row.get("abnormal_return_adjustment_quality") or "acceptable_with_limitations")
        evidence_parts.append(f"adjustment_basis={basis}")
        evidence_parts.append(f"adjustment_quality={quality}")
        adjustment_flags = str(row.get("abnormal_return_adjustment_flags") or "")
        if basis in {"universe_adjusted_only", "adjustment_unavailable"} or quality in {"insufficient_peers", "unavailable"}:
            risk_flags.append(f"adjustment:{basis}")
            if adjustment_flags and adjustment_flags != "none":
                risk_flags.append(f"adjustment_flags:{adjustment_flags}")
            if quality in {"insufficient_peers", "unavailable"}:
                manual_reasons.append("abnormal_return_adjustment_limited")
                manual_required = True
        if risk_flags:
            evidence_parts.append(f"risk_flags={_pipe(risk_flags)}")
        manual_required = bool(manual_reasons)
        if manual_required:
            acceptability = "manual_review_required"
        elif risk_flags:
            acceptability = "acceptable_with_limitations"
        else:
            acceptability = "acceptable"
        rows.append(
            {
                "classification_confidence": confidence,
                "classification_reason": classification_reason,
                "classification_reason_components": _stable_json(components),
                "mixed_event_flag": mixed_event_flag,
                "mixed_event_reason": mixed_reason if mixed_event_flag else "",
                "manual_review_required": manual_required,
                "manual_review_reason": _pipe(manual_reasons) if manual_required else "",
                "ranking_score": score,
                "ranking_score_components": _stable_json(score_components),
                "primary_signal": primary_signal,
                "secondary_signal": secondary_signal,
                "evidence_summary": "; ".join(evidence_parts),
                "horizon_flags": _pipe(horizon),
                "liquidity_flags": _pipe(liquidity),
                "duplicate_event_flag": duplicate_event_flag,
                "data_quality_flags": _pipe(data_flags),
                "product_risk_flags": _pipe(risk_flags),
                "product_acceptability": acceptability,
            }
        )
    evidence = pd.DataFrame(rows, index=out.index)
    for col in EVIDENCE_COLUMNS:
        out[col] = evidence[col]
    return out


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for col in [
        "abnormal_ret_1d",
        "commercial_candidate_score",
        "turnover_spike_ratio",
        "avg_turnover_20d",
        "commercial_abnormal_reaction_score",
    ]:
        if col not in prepared.columns:
            prepared[col] = np.nan
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    for col in [
        "liquidity_bucket",
        "liquidity_status",
        "ret_1d_status",
        "abnormal_ret_1d_status",
        "price_reaction_status_v2",
        "commercial_signal",
    ]:
        if col not in prepared.columns:
            prepared[col] = np.nan
    if "is_liquid_enough" not in prepared.columns:
        prepared["is_liquid_enough"] = False
    prepared["_is_liquid_enough_bool"] = prepared["is_liquid_enough"].map(_truthy)
    return prepared


def _exclusion_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    bucket = str(row.get("liquidity_bucket")).lower()
    if bucket == "low":
        reasons.append("low_liquidity")
    if bucket == "unknown":
        reasons.append("unknown_liquidity")
    if not row.get("_is_liquid_enough_bool", False):
        reasons.append("not_liquid_enough")
    if row.get("liquidity_status") != "valid":
        reasons.append("invalid_liquidity_status")
    if row.get("ret_1d_status") != "valid":
        reasons.append("invalid_ret_1d_status")
    if row.get("abnormal_ret_1d_status") != "valid":
        reasons.append("invalid_abnormal_ret_1d_status")
    if row.get("price_reaction_status_v2") not in ELIGIBLE_PRICE_STATUSES:
        reasons.append("insufficient_price_reaction_status")
    if pd.isna(row.get("abnormal_ret_1d")):
        reasons.append("missing_abnormal_ret_1d")
    signal = row.get("commercial_signal")
    if _is_review_signal(signal):
        reasons.append("review_signal")
    if _missing_signal(signal):
        reasons.append("missing_commercial_signal")
    return reasons


def _sort_and_rank(
    df: pd.DataFrame,
    label: str,
    reason: str,
    sort_columns: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    ranked = df.copy()
    ranked["ranking_label"] = label
    ranked["ranking_reason"] = reason
    for col in sort_columns:
        if col not in ranked.columns:
            ranked[col] = np.nan
    ranked = ranked.sort_values(sort_columns, ascending=ascending, na_position="last")
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked[[col for col in RANKING_COLUMNS if col in ranked.columns]]


def _classification_masks(eligible: pd.DataFrame) -> dict[str, pd.Series]:
    positive = eligible["commercial_signal"].map(is_positive_signal)
    negative = eligible["commercial_signal"].map(is_negative_signal)
    abnormal = pd.to_numeric(eligible["abnormal_ret_1d"], errors="coerce")
    return {
        "positive_reacted": positive & (abnormal > 0),
        "positive_unreacted": positive & (abnormal < 0),
        "negative_reacted": negative & (abnormal < 0),
        "negative_reversal": negative & (abnormal > 0),
    }


def _unclassified_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    positive = is_positive_signal(row.get("commercial_signal"))
    negative = is_negative_signal(row.get("commercial_signal"))
    abnormal = pd.to_numeric(pd.Series([row.get("abnormal_ret_1d")]), errors="coerce").iloc[0]
    if not positive and not negative:
        reasons.append("signal_not_positive_or_negative")
    if pd.isna(abnormal):
        reasons.append("abnormal_ret_1d_missing")
    elif abnormal == 0:
        reasons.append("abnormal_ret_1d_zero")
    if not reasons:
        reasons.append("unexpected_classification_gap")
    return "|".join(reasons)


def _count_pipe_reasons(series: pd.Series) -> dict[str, int]:
    reason_counts: dict[str, int] = {}
    for value in series.dropna():
        for reason in str(value).split("|"):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return dict(sorted(reason_counts.items()))


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)


def build_commercial_quality_rankings(
    input_path: Path = DEFAULT_LIQUIDITY,
    positive_reacted_path: Path = DEFAULT_POSITIVE_REACTED,
    positive_unreacted_path: Path = DEFAULT_POSITIVE_UNREACTED,
    negative_reacted_path: Path = DEFAULT_NEGATIVE_REACTED,
    negative_reversal_path: Path = DEFAULT_NEGATIVE_REVERSAL,
    excluded_audit_path: Path = DEFAULT_EXCLUDED_AUDIT,
    unclassified_audit_path: Path = DEFAULT_UNCLASSIFIED_AUDIT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, object]:
    source = load_candidates(input_path)
    generated_columns = {"rank", "ranking_label", "ranking_reason", "exclusion_reasons", "unclassified_reason", *EVIDENCE_COLUMNS}
    missing_output_columns = [
        col
        for col in [*RANKING_COLUMNS, *UNCLASSIFIED_AUDIT_COLUMNS]
        if col not in source.columns and col not in generated_columns and col != "unclassified_reason"
    ]
    missing_output_columns = sorted(set(missing_output_columns))
    df = _prepare(source)
    duplicate_keys = _duplicate_keys(df)

    reason_lists = [_exclusion_reasons(row) for _, row in df.iterrows()]
    df["_exclusion_reasons"] = ["|".join(reasons) for reasons in reason_lists]
    eligible = df[df["_exclusion_reasons"] == ""].copy()
    excluded = df[df["_exclusion_reasons"] != ""].copy()

    masks = _classification_masks(eligible)

    positive_reacted = _sort_and_rank(
        eligible[masks["positive_reacted"]],
        "positive_reacted",
        "positive_signal_and_positive_abnormal_1d",
        ["commercial_abnormal_reaction_score", "abnormal_ret_1d", "turnover_spike_ratio", "commercial_candidate_score", "avg_turnover_20d", "DiscDate"],
        [False, False, False, False, False, False],
    )
    positive_unreacted = _sort_and_rank(
        eligible[masks["positive_unreacted"]],
        "positive_unreacted",
        "positive_signal_but_negative_abnormal_1d",
        ["commercial_abnormal_reaction_score", "abnormal_ret_1d", "commercial_candidate_score", "turnover_spike_ratio", "avg_turnover_20d", "DiscDate"],
        [True, True, False, False, False, False],
    )
    negative_reacted = _sort_and_rank(
        eligible[masks["negative_reacted"]],
        "negative_reacted",
        "negative_signal_and_negative_abnormal_1d",
        ["commercial_abnormal_reaction_score", "abnormal_ret_1d", "turnover_spike_ratio", "commercial_candidate_score", "avg_turnover_20d", "DiscDate"],
        [True, True, False, False, False, False],
    )
    negative_reversal = _sort_and_rank(
        eligible[masks["negative_reversal"]],
        "negative_reversal",
        "negative_signal_but_positive_abnormal_1d",
        ["commercial_abnormal_reaction_score", "abnormal_ret_1d", "commercial_candidate_score", "turnover_spike_ratio", "avg_turnover_20d", "DiscDate"],
        [False, False, False, False, False, False],
    )
    positive_reacted = _build_evidence_columns(positive_reacted, duplicate_keys)
    positive_unreacted = _build_evidence_columns(positive_unreacted, duplicate_keys)
    negative_reacted = _build_evidence_columns(negative_reacted, duplicate_keys)
    negative_reversal = _build_evidence_columns(negative_reversal, duplicate_keys)
    classified_mask = (
        masks["positive_reacted"]
        | masks["positive_unreacted"]
        | masks["negative_reacted"]
        | masks["negative_reversal"]
    )
    # Excluded audit covers rows that fail eligible checks. Unclassified audit covers
    # eligible rows that do not fall into the four screening classifications.
    unclassified = eligible[~classified_mask].copy()
    unclassified["unclassified_reason"] = [
        _unclassified_reason(row) for _, row in unclassified.iterrows()
    ]

    _write_csv(positive_reacted_path, positive_reacted)
    _write_csv(positive_unreacted_path, positive_unreacted)
    _write_csv(negative_reacted_path, negative_reacted)
    _write_csv(negative_reversal_path, negative_reversal)

    audit = excluded.copy()
    audit["exclusion_reasons"] = audit["_exclusion_reasons"]
    for col in AUDIT_COLUMNS:
        if col not in audit.columns:
            audit[col] = np.nan
    audit = _build_evidence_columns(audit, duplicate_keys)
    _write_csv(excluded_audit_path, audit[[col for col in AUDIT_COLUMNS if col in audit.columns]])
    for col in UNCLASSIFIED_AUDIT_COLUMNS:
        if col not in unclassified.columns:
            unclassified[col] = np.nan
    unclassified = _build_evidence_columns(unclassified, duplicate_keys)
    _write_csv(unclassified_audit_path, unclassified[[col for col in UNCLASSIFIED_AUDIT_COLUMNS if col in unclassified.columns]])

    reason_counts: dict[str, int] = {}
    for reasons in reason_lists:
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    ranking_counts = {
        "positive_reacted": int(len(positive_reacted)),
        "positive_unreacted": int(len(positive_unreacted)),
        "negative_reacted": int(len(negative_reacted)),
        "negative_reversal": int(len(negative_reversal)),
    }
    unclassified_reason_counts = _count_pipe_reasons(unclassified["unclassified_reason"])
    warnings: list[str] = []
    if len(excluded) > 0:
        warnings.append(f"{len(excluded)} rows excluded from regular rankings")
    if missing_output_columns:
        warnings.append("missing_output_columns present")
    for label, count in ranking_counts.items():
        if count == 0:
            warnings.append(f"{label} ranking has zero rows")
    input_liquidity_counts = counts(df["liquidity_bucket"])
    if input_liquidity_counts.get("low", 0) or input_liquidity_counts.get("unknown", 0):
        warnings.append("low_or_unknown_liquidity_present")
    price_status_counts = counts(df["price_reaction_status_v2"])
    if price_status_counts.get("event_day_only_or_insufficient", 0):
        warnings.append("event_day_only_or_insufficient_present")
    if len(unclassified) > 0:
        warnings.append("unclassified_eligible_rows_exist")

    summary = {
        "generated_at_jst": jst_now_iso(),
        "logic_version": LOGIC_VERSION,
        "input_path": str(input_path),
        "input_rows": int(len(df)),
        "eligible_rows": int(len(eligible)),
        "excluded_rows": int(len(excluded)),
        "unclassified_rows": int(len(unclassified)),
        "unclassified_output_path": str(unclassified_audit_path),
        "unclassified_reason_counts": unclassified_reason_counts,
        "output_paths": {
            "positive_reacted": str(positive_reacted_path),
            "positive_unreacted": str(positive_unreacted_path),
            "negative_reacted": str(negative_reacted_path),
            "negative_reversal": str(negative_reversal_path),
            "excluded_audit": str(excluded_audit_path),
            "unclassified_audit": str(unclassified_audit_path),
            "summary": str(summary_path),
        },
        "ranking_counts": ranking_counts,
        "liquidity_bucket_counts_input": input_liquidity_counts,
        "liquidity_bucket_counts_eligible": counts(eligible["liquidity_bucket"])
        if not eligible.empty
        else {},
        "candidate_bucket_counts_eligible": counts(eligible["candidate_bucket"])
        if "candidate_bucket" in eligible.columns and not eligible.empty
        else {},
        "commercial_signal_counts_eligible": counts(eligible["commercial_signal"])
        if "commercial_signal" in eligible.columns and not eligible.empty
        else {},
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "missing_output_columns": missing_output_columns,
        "warnings": warnings,
    }
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--positive-reacted", type=Path, default=DEFAULT_POSITIVE_REACTED)
    parser.add_argument("--positive-unreacted", type=Path, default=DEFAULT_POSITIVE_UNREACTED)
    parser.add_argument("--negative-reacted", type=Path, default=DEFAULT_NEGATIVE_REACTED)
    parser.add_argument("--negative-reversal", type=Path, default=DEFAULT_NEGATIVE_REVERSAL)
    parser.add_argument("--excluded-audit", type=Path, default=DEFAULT_EXCLUDED_AUDIT)
    parser.add_argument("--unclassified-audit", type=Path, default=DEFAULT_UNCLASSIFIED_AUDIT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    build_commercial_quality_rankings(
        args.input,
        args.positive_reacted,
        args.positive_unreacted,
        args.negative_reacted,
        args.negative_reversal,
        args.excluded_audit,
        args.unclassified_audit,
        args.summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

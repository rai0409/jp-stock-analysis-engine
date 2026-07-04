"""Build final screening rankings from commercial-quality J-Quants liquidity rows."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for col in [
        "abnormal_ret_1d",
        "commercial_candidate_score",
        "turnover_spike_ratio",
        "avg_turnover_20d",
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
    generated_columns = {"rank", "ranking_label", "ranking_reason"}
    missing_output_columns = [
        col
        for col in [*RANKING_COLUMNS, *UNCLASSIFIED_AUDIT_COLUMNS]
        if col not in source.columns and col not in generated_columns and col != "unclassified_reason"
    ]
    missing_output_columns = sorted(set(missing_output_columns))
    df = _prepare(source)

    reason_lists = [_exclusion_reasons(row) for _, row in df.iterrows()]
    df["_exclusion_reasons"] = ["|".join(reasons) for reasons in reason_lists]
    eligible = df[df["_exclusion_reasons"] == ""].copy()
    excluded = df[df["_exclusion_reasons"] != ""].copy()

    masks = _classification_masks(eligible)

    positive_reacted = _sort_and_rank(
        eligible[masks["positive_reacted"]],
        "positive_reacted",
        "positive_signal_and_positive_abnormal_1d",
        ["abnormal_ret_1d", "turnover_spike_ratio", "commercial_candidate_score", "avg_turnover_20d", "DiscDate"],
        [False, False, False, False, False],
    )
    positive_unreacted = _sort_and_rank(
        eligible[masks["positive_unreacted"]],
        "positive_unreacted",
        "positive_signal_but_negative_abnormal_1d",
        ["abnormal_ret_1d", "commercial_candidate_score", "turnover_spike_ratio", "avg_turnover_20d", "DiscDate"],
        [True, False, False, False, False],
    )
    negative_reacted = _sort_and_rank(
        eligible[masks["negative_reacted"]],
        "negative_reacted",
        "negative_signal_and_negative_abnormal_1d",
        ["abnormal_ret_1d", "turnover_spike_ratio", "commercial_candidate_score", "avg_turnover_20d", "DiscDate"],
        [True, False, False, False, False],
    )
    negative_reversal = _sort_and_rank(
        eligible[masks["negative_reversal"]],
        "negative_reversal",
        "negative_signal_but_positive_abnormal_1d",
        ["abnormal_ret_1d", "commercial_candidate_score", "turnover_spike_ratio", "avg_turnover_20d", "DiscDate"],
        [False, False, False, False, False],
    )
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
    _write_csv(excluded_audit_path, audit[AUDIT_COLUMNS])
    for col in UNCLASSIFIED_AUDIT_COLUMNS:
        if col not in unclassified.columns:
            unclassified[col] = np.nan
    _write_csv(unclassified_audit_path, unclassified[UNCLASSIFIED_AUDIT_COLUMNS])

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

"""Build robust grouped summaries for commercial J-Quants price reactions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jp_stock_analysis.analysis.jquants_commercial_common import (
    ANALYSIS_DIR,
    counts,
    ensure_parent,
    load_candidates,
    sample_size_bucket,
    sample_warning,
    write_json,
)
from jp_stock_analysis.analysis.jquants_liquidity import DEFAULT_OUTPUT as DEFAULT_LIQUIDITY


DEFAULT_OUTPUT = ANALYSIS_DIR / "commercial_price_reaction_summary_robust.csv"
DEFAULT_SUMMARY = ANALYSIS_DIR / "commercial_price_reaction_summary_robust_summary.json"
GROUP_COLUMNS = [
    "candidate_bucket",
    "commercial_signal",
    "reaction_flag_v2",
    "abnormal_reaction_flag_1d",
    "liquidity_bucket",
]
METRICS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_20d",
    "abnormal_ret_1d",
    "abnormal_ret_3d",
    "abnormal_ret_5d",
    "abnormal_ret_20d",
]


def trimmed_mean_10pct(values: pd.Series) -> float | None:
    s = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if s.empty:
        return None
    trim = int(len(s) * 0.1)
    if trim > 0 and len(s) > trim * 2:
        s = s.iloc[trim:-trim]
    return float(s.mean()) if not s.empty else None


def metric_stats(values: pd.Series) -> dict[str, object]:
    s = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {
            "available": 0,
            "mean": np.nan,
            "median": np.nan,
            "trimmed_mean_10pct": np.nan,
            "std": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "min": np.nan,
            "max": np.nan,
            "positive_rate": np.nan,
        }
    return {
        "available": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "trimmed_mean_10pct": trimmed_mean_10pct(s),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
        "positive_rate": float((s > 0).mean()),
    }


def build_robust_summary(
    liquidity_path: Path = DEFAULT_LIQUIDITY,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, object]:
    df = load_candidates(liquidity_path)
    for col in GROUP_COLUMNS:
        if col not in df.columns:
            df[col] = "missing"
        df[col] = df[col].fillna("missing")
    for col in METRICS:
        if col not in df.columns:
            df[col] = np.nan

    rows: list[dict[str, object]] = []
    grouped = df.groupby(GROUP_COLUMNS, dropna=False, sort=True)
    for keys, group in grouped:
        row = dict(zip(GROUP_COLUMNS, keys, strict=True))
        n = int(len(group))
        row["count"] = n
        bucket = sample_size_bucket(n)
        row["sample_size_bucket"] = bucket
        row["sample_warning"] = sample_warning(bucket)
        for metric in METRICS:
            stats = metric_stats(group[metric])
            row[f"{metric}_count"] = n
            for name, value in stats.items():
                row[f"{metric}_{name}"] = value
        rows.append(row)

    output = pd.DataFrame(rows)
    ensure_parent(output_path)
    output.to_csv(output_path, index=False)
    summary = {
        "rows": int(len(df)),
        "groups": int(len(output)),
        "sample_size_bucket_counts": counts(output["sample_size_bucket"])
        if not output.empty
        else {},
        "output_path": str(output_path),
    }
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    build_robust_summary(args.liquidity, args.output, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

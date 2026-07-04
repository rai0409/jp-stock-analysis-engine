"""Attach pre-event liquidity and tradability features to J-Quants event rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jp_stock_analysis.analysis.jquants_abnormal_return import DEFAULT_OUTPUT as DEFAULT_ABNORMAL
from jp_stock_analysis.analysis.jquants_commercial_common import (
    ANALYSIS_DIR,
    DEFAULT_PRICES,
    counts,
    ensure_parent,
    load_candidates,
    load_prices,
    write_json,
)


DEFAULT_OUTPUT = ANALYSIS_DIR / "commercial_revision_candidates_with_liquidity.csv"
DEFAULT_SUMMARY = ANALYSIS_DIR / "commercial_revision_candidates_with_liquidity_summary.json"


def liquidity_bucket(avg_turnover: object) -> str:
    if pd.isna(avg_turnover):
        return "unknown"
    if avg_turnover >= 1_000_000_000:
        return "high"
    if avg_turnover >= 100_000_000:
        return "mid"
    return "low"


def build_liquidity_features(
    abnormal_path: Path = DEFAULT_ABNORMAL,
    prices_path: Path = DEFAULT_PRICES,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, object]:
    df = load_candidates(abnormal_path)
    prices = load_prices(prices_path)
    prices_by_ticker = {ticker: frame.reset_index(drop=True) for ticker, frame in prices.groupby("ticker")}

    feature_rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker")).zfill(4)
        base_date = pd.to_datetime(row.get("reaction_base_date"), errors="coerce")
        ticker_prices = prices_by_ticker.get(ticker)
        features: dict[str, object] = {
            "avg_turnover_20d": np.nan,
            "median_turnover_20d": np.nan,
            "avg_volume_20d": np.nan,
            "median_volume_20d": np.nan,
            "avg_adj_volume_20d": np.nan,
            "median_adj_volume_20d": np.nan,
            "turnover_spike_ratio": np.nan,
            "volume_spike_ratio": np.nan,
            "liquidity_bucket": "unknown",
            "liquidity_status": "missing_price",
            "is_liquid_enough": False,
        }
        if ticker_prices is None or pd.isna(base_date):
            feature_rows.append(features)
            continue
        prior = ticker_prices[ticker_prices["date"] < base_date].tail(20)
        if len(prior) < 20:
            features["liquidity_status"] = "insufficient_lookback"
            feature_rows.append(features)
            continue
        turnover = pd.to_numeric(prior.get("turnover_value"), errors="coerce")
        volume = pd.to_numeric(prior.get("volume"), errors="coerce")
        adj_volume = pd.to_numeric(prior.get("adj_volume"), errors="coerce")
        if turnover.dropna().empty:
            features["liquidity_status"] = "missing_turnover"
            feature_rows.append(features)
            continue
        avg_turnover = float(turnover.mean())
        avg_volume = float(volume.mean()) if not volume.dropna().empty else np.nan
        avg_adj_volume = float(adj_volume.mean()) if not adj_volume.dropna().empty else np.nan
        bucket = liquidity_bucket(avg_turnover)
        base_turnover = row.get("base_turnover_value")
        base_volume = row.get("base_volume")
        features.update(
            {
                "avg_turnover_20d": avg_turnover,
                "median_turnover_20d": float(turnover.median()),
                "avg_volume_20d": avg_volume,
                "median_volume_20d": float(volume.median()) if not volume.dropna().empty else np.nan,
                "avg_adj_volume_20d": avg_adj_volume,
                "median_adj_volume_20d": (
                    float(adj_volume.median()) if not adj_volume.dropna().empty else np.nan
                ),
                "turnover_spike_ratio": (
                    float(base_turnover / avg_turnover)
                    if not pd.isna(base_turnover) and avg_turnover != 0.0
                    else np.nan
                ),
                "volume_spike_ratio": (
                    float(base_volume / avg_volume)
                    if not pd.isna(base_volume) and not pd.isna(avg_volume) and avg_volume != 0.0
                    else np.nan
                ),
                "liquidity_bucket": bucket,
                "liquidity_status": "valid",
                "is_liquid_enough": bucket in {"high", "mid"},
            }
        )
        feature_rows.append(features)

    features = pd.DataFrame(feature_rows)
    for col in ["base_turnover_value", "base_volume", "base_adj_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    output = pd.concat([df.reset_index(drop=True), features], axis=1)
    ensure_parent(output_path)
    output.to_csv(output_path, index=False)

    median_by_bucket = (
        output.groupby("liquidity_bucket")["avg_turnover_20d"].median().dropna().to_dict()
        if "liquidity_bucket" in output.columns
        else {}
    )
    summary = {
        "input_rows": int(len(df)),
        "output_rows": int(len(output)),
        "liquidity_bucket_counts": counts(output["liquidity_bucket"]),
        "is_liquid_enough_counts": counts(output["is_liquid_enough"]),
        "avg_turnover_20d_median_by_bucket": {
            str(k): float(v) for k, v in median_by_bucket.items()
        },
        "liquidity_status_counts": counts(output["liquidity_status"]),
        "output_path": str(output_path),
    }
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abnormal", type=Path, default=DEFAULT_ABNORMAL)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    build_liquidity_features(args.abnormal, args.prices, args.output, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

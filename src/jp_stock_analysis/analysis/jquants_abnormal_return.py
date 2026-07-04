"""Attach universe-relative abnormal returns to J-Quants price reaction rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jp_stock_analysis.analysis.jquants_commercial_common import (
    ANALYSIS_DIR,
    DEFAULT_PRICES,
    counts,
    ensure_parent,
    is_negative_signal,
    is_positive_signal,
    load_candidates,
    load_prices,
    numeric_summary,
    write_json,
)
from jp_stock_analysis.analysis.jquants_price_reaction_v2 import DEFAULT_OUTPUT as DEFAULT_REACTION


DEFAULT_OUTPUT = ANALYSIS_DIR / "commercial_revision_candidates_with_abnormal_return.csv"
DEFAULT_SUMMARY = ANALYSIS_DIR / "commercial_revision_candidates_with_abnormal_return_summary.json"
HORIZONS = (1, 3, 5, 20)


def _build_universe_returns(
    prices: pd.DataFrame, required_pairs: set[tuple[str, str]]
) -> dict[tuple[str, str], float]:
    pivot = prices.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
    cache: dict[tuple[str, str], float] = {}
    for base_text, future_text in required_pairs:
        base_date = pd.to_datetime(base_text, errors="coerce")
        future_date = pd.to_datetime(future_text, errors="coerce")
        if pd.isna(base_date) or pd.isna(future_date):
            continue
        if base_date not in pivot.index or future_date not in pivot.index:
            continue
        base = pivot.loc[base_date]
        future = pivot.loc[future_date]
        returns = future / base - 1.0
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if not returns.empty:
            cache[(base_text, future_text)] = float(returns.mean())
    return cache


def _flag(signal: object, abnormal_ret: object, suffix: str) -> str:
    if pd.isna(abnormal_ret):
        return "review_or_unknown"
    positive = is_positive_signal(signal)
    negative = is_negative_signal(signal)
    if positive and abnormal_ret > 0:
        return f"positive_event_abnormal_positive_{suffix}"
    if positive and abnormal_ret < 0:
        return f"positive_event_abnormal_negative_{suffix}"
    if negative and abnormal_ret < 0:
        return f"negative_event_abnormal_negative_{suffix}"
    if negative and abnormal_ret > 0:
        return f"negative_event_abnormal_positive_{suffix}"
    return "review_or_unknown"


def build_abnormal_return(
    reaction_path: Path = DEFAULT_REACTION,
    prices_path: Path = DEFAULT_PRICES,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, object]:
    df = load_candidates(reaction_path)
    prices = load_prices(prices_path)
    required_pairs: set[tuple[str, str]] = set()
    for horizon in HORIZONS:
        date_col = f"ret_{horizon}d_date"
        status_col = f"ret_{horizon}d_status"
        if date_col not in df.columns or status_col not in df.columns:
            continue
        valid = df[status_col] == "valid"
        required_pairs.update(
            (str(row["reaction_base_date"]), str(row[date_col]))
            for _, row in df.loc[valid, ["reaction_base_date", date_col]].dropna().iterrows()
        )
    universe_returns = _build_universe_returns(prices, required_pairs)

    for horizon in HORIZONS:
        ret_col = f"ret_{horizon}d"
        date_col = f"ret_{horizon}d_date"
        status_col = f"ret_{horizon}d_status"
        universe_col = f"universe_ret_{horizon}d"
        abnormal_col = f"abnormal_ret_{horizon}d"
        abnormal_status_col = f"abnormal_ret_{horizon}d_status"
        universes: list[float] = []
        abnormals: list[float] = []
        statuses: list[str] = []
        for _, row in df.iterrows():
            if row.get(status_col) != "valid":
                universes.append(np.nan)
                abnormals.append(np.nan)
                statuses.append("invalid_source_return")
                continue
            base_date = str(row.get("reaction_base_date"))
            future_date = str(row.get(date_col))
            universe_ret = universe_returns.get((base_date, future_date))
            if universe_ret is None or pd.isna(row.get(ret_col)):
                universes.append(np.nan)
                abnormals.append(np.nan)
                statuses.append("missing_universe_return")
                continue
            abnormal = float(row.get(ret_col) - universe_ret)
            universes.append(universe_ret)
            abnormals.append(abnormal)
            statuses.append("valid")
        df[universe_col] = universes
        df[abnormal_col] = abnormals
        df[abnormal_status_col] = statuses

    df["abnormal_reaction_flag_1d"] = [
        _flag(signal, value, "1d")
        for signal, value in zip(df.get("commercial_signal", pd.Series(dtype=object)), df["abnormal_ret_1d"])
    ]
    df["abnormal_reaction_flag_5d"] = [
        _flag(signal, value, "5d")
        for signal, value in zip(df.get("commercial_signal", pd.Series(dtype=object)), df["abnormal_ret_5d"])
    ]
    df["abnormal_reaction_flag_20d"] = [
        _flag(signal, value, "20d")
        for signal, value in zip(df.get("commercial_signal", pd.Series(dtype=object)), df["abnormal_ret_20d"])
    ]

    ensure_parent(output_path)
    df.to_csv(output_path, index=False)
    summary = {
        "input_rows": int(len(df)),
        "output_rows": int(len(df)),
        "abnormal_ret_status_counts": {
            f"abnormal_ret_{h}d_status": counts(df[f"abnormal_ret_{h}d_status"]) for h in HORIZONS
        },
        "abnormal_reaction_flag_counts": {
            "abnormal_reaction_flag_1d": counts(df["abnormal_reaction_flag_1d"]),
            "abnormal_reaction_flag_5d": counts(df["abnormal_reaction_flag_5d"]),
            "abnormal_reaction_flag_20d": counts(df["abnormal_reaction_flag_20d"]),
        },
        "valid_abnormal_return_summary": {
            f"abnormal_ret_{h}d": numeric_summary(
                df.loc[df[f"abnormal_ret_{h}d_status"] == "valid", f"abnormal_ret_{h}d"]
            )
            for h in (1, 5, 20)
        },
        "output_path": str(output_path),
    }
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction", type=Path, default=DEFAULT_REACTION)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    build_abnormal_return(args.reaction, args.prices, args.output, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

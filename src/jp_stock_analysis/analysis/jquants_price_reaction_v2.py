"""Build event-time aware price reaction features for J-Quants revision candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jp_stock_analysis.analysis.jquants_commercial_common import (
    ANALYSIS_DIR,
    DEFAULT_CANDIDATES,
    DEFAULT_PRICES,
    counts,
    ensure_parent,
    event_time_bucket,
    is_negative_signal,
    is_positive_signal,
    load_candidates,
    load_prices,
    write_json,
)


DEFAULT_OUTPUT = ANALYSIS_DIR / "commercial_revision_candidates_with_price_reaction_v2.csv"
DEFAULT_SUMMARY = ANALYSIS_DIR / "commercial_revision_candidates_with_price_reaction_v2_summary.json"
HORIZONS = {1: 5, 3: 10, 5: 14, 20: 45}


def _future_return(
    ticker_prices: pd.DataFrame | None,
    base_pos: int | None,
    base_adj_close: float | None,
    horizon: int,
    max_calendar_gap: int,
) -> dict[str, object]:
    prefix = f"ret_{horizon}d"
    if ticker_prices is None or base_pos is None or pd.isna(base_adj_close):
        return {
            prefix: np.nan,
            f"{prefix}_date": pd.NaT,
            f"{prefix}_status": "missing_base_price",
            f"{prefix}_calendar_gap_days": np.nan,
        }
    future_pos = base_pos + horizon
    if future_pos >= len(ticker_prices):
        return {
            prefix: np.nan,
            f"{prefix}_date": pd.NaT,
            f"{prefix}_status": "insufficient_future_price",
            f"{prefix}_calendar_gap_days": np.nan,
        }
    base_date = ticker_prices.iloc[base_pos]["date"]
    future = ticker_prices.iloc[future_pos]
    future_close = future.get("adj_close")
    gap = int((future["date"] - base_date).days)
    if gap > max_calendar_gap:
        return {
            prefix: np.nan,
            f"{prefix}_date": future["date"],
            f"{prefix}_status": "stale_calendar_gap",
            f"{prefix}_calendar_gap_days": gap,
        }
    if pd.isna(future_close):
        return {
            prefix: np.nan,
            f"{prefix}_date": future["date"],
            f"{prefix}_status": "insufficient_future_price",
            f"{prefix}_calendar_gap_days": gap,
        }
    return {
        prefix: float(future_close / base_adj_close - 1.0),
        f"{prefix}_date": future["date"],
        f"{prefix}_status": "valid",
        f"{prefix}_calendar_gap_days": gap,
    }


def _reaction_status(row: dict[str, object], has_base: bool) -> str:
    if not has_base:
        return "no_price_after_event"
    if row.get("ret_20d_status") == "valid":
        return "full_20d_available"
    if row.get("ret_5d_status") == "valid":
        return "partial_5d_available"
    if row.get("ret_1d_status") == "valid":
        return "partial_1d_available"
    return "event_day_only_or_insufficient"


def _reaction_flag(signal: object, row: dict[str, object]) -> str:
    positive = is_positive_signal(signal)
    negative = is_negative_signal(signal)
    ret_1d = row.get("ret_1d")
    ret_5d = row.get("ret_5d")
    ret_1d_valid = row.get("ret_1d_status") == "valid" and not pd.isna(ret_1d)
    ret_5d_valid = row.get("ret_5d_status") == "valid" and not pd.isna(ret_5d)
    if positive and ret_1d_valid and ret_1d > 0:
        return "positive_event_reacted_positive_1d"
    if positive and ret_1d_valid and ret_1d < 0:
        return "positive_event_but_1d_negative"
    if positive and not ret_1d_valid and ret_5d_valid and ret_5d > 0:
        return "positive_event_reacted_positive_5d"
    if positive and not ret_1d_valid and ret_5d_valid and ret_5d < 0:
        return "positive_event_but_5d_negative"
    if negative and ret_1d_valid and ret_1d < 0:
        return "negative_event_reacted_negative_1d"
    if negative and ret_1d_valid and ret_1d > 0:
        return "negative_event_but_1d_positive"
    if negative and not ret_1d_valid and ret_5d_valid and ret_5d < 0:
        return "negative_event_reacted_negative_5d"
    if negative and not ret_1d_valid and ret_5d_valid and ret_5d > 0:
        return "negative_event_but_5d_positive"
    return "review_or_unknown"


def build_price_reaction_v2(
    candidates_path: Path = DEFAULT_CANDIDATES,
    prices_path: Path = DEFAULT_PRICES,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, object]:
    candidates = load_candidates(candidates_path)
    prices = load_prices(prices_path)
    prices_by_ticker = {ticker: frame.reset_index(drop=True) for ticker, frame in prices.groupby("ticker")}

    rows: list[dict[str, object]] = []
    for _, source in candidates.iterrows():
        out = source.to_dict()
        event_date = pd.to_datetime(source.get("DiscDate"), errors="coerce")
        bucket, close_text = event_time_bucket(event_date, source.get("first_time"))
        out["event_time_bucket"] = bucket
        out["market_close_time_used"] = close_text

        ticker = str(source.get("ticker")).zfill(4)
        ticker_prices = prices_by_ticker.get(ticker)
        base_pos: int | None = None
        base_row = None
        if ticker_prices is not None and not pd.isna(event_date):
            dates = ticker_prices["date"]
            if bucket in {"after_close", "unknown_time"}:
                mask = dates > event_date
                reason = f"{bucket}_next_trading_day"
            else:
                mask = dates >= event_date
                reason = f"{bucket}_same_or_next_trading_day"
            positions = np.flatnonzero(mask.to_numpy())
            if len(positions):
                base_pos = int(positions[0])
                base_row = ticker_prices.iloc[base_pos]
        else:
            reason = "missing_ticker_prices_or_event_date"

        if base_row is None:
            out.update(
                {
                    "reaction_base_date": pd.NaT,
                    "reaction_base_reason": reason,
                    "price_gap_days": np.nan,
                    "base_open": np.nan,
                    "base_close": np.nan,
                    "base_adj_open": np.nan,
                    "base_adj_close": np.nan,
                    "base_turnover_value": np.nan,
                    "base_volume": np.nan,
                    "base_adj_volume": np.nan,
                    "intraday_ret": np.nan,
                }
            )
            has_base = False
        else:
            base_adj_open = base_row.get("adj_open")
            base_adj_close = base_row.get("adj_close")
            out.update(
                {
                    "reaction_base_date": base_row["date"],
                    "reaction_base_reason": reason,
                    "price_gap_days": int((base_row["date"] - event_date).days),
                    "base_open": base_row.get("open", np.nan),
                    "base_close": base_row.get("close", np.nan),
                    "base_adj_open": base_adj_open,
                    "base_adj_close": base_adj_close,
                    "base_turnover_value": base_row.get("turnover_value", np.nan),
                    "base_volume": base_row.get("volume", np.nan),
                    "base_adj_volume": base_row.get("adj_volume", np.nan),
                    "intraday_ret": (
                        float(base_adj_close / base_adj_open - 1.0)
                        if not pd.isna(base_adj_open)
                        and not pd.isna(base_adj_close)
                        and float(base_adj_open) != 0.0
                        else np.nan
                    ),
                }
            )
            has_base = not pd.isna(base_adj_close)

        for horizon, max_gap in HORIZONS.items():
            out.update(
                _future_return(
                    ticker_prices,
                    base_pos,
                    out.get("base_adj_close"),
                    horizon,
                    max_gap,
                )
            )
        out["price_reaction_status_v2"] = _reaction_status(out, has_base)
        out["reaction_flag_v2"] = _reaction_flag(source.get("commercial_signal"), out)
        rows.append(out)

    output = pd.DataFrame(rows)
    for col in [
        "reaction_base_date",
        "ret_1d_date",
        "ret_3d_date",
        "ret_5d_date",
        "ret_20d_date",
    ]:
        if col in output.columns:
            output[col] = pd.to_datetime(output[col], errors="coerce").dt.strftime("%Y-%m-%d")

    ensure_parent(output_path)
    output.to_csv(output_path, index=False)
    event_dates = pd.to_datetime(candidates.get("DiscDate"), errors="coerce")
    summary = {
        "input_rows": int(len(candidates)),
        "output_rows": int(len(output)),
        "tickers": int(candidates["ticker"].nunique()) if "ticker" in candidates.columns else 0,
        "event_date_min": None if event_dates.dropna().empty else event_dates.min().strftime("%Y-%m-%d"),
        "event_date_max": None if event_dates.dropna().empty else event_dates.max().strftime("%Y-%m-%d"),
        "price_date_min": prices["date"].min().strftime("%Y-%m-%d") if not prices.empty else None,
        "price_date_max": prices["date"].max().strftime("%Y-%m-%d") if not prices.empty else None,
        "event_time_bucket_counts": counts(output["event_time_bucket"]),
        "price_reaction_status_v2_counts": counts(output["price_reaction_status_v2"]),
        "reaction_flag_v2_counts": counts(output["reaction_flag_v2"]),
        "horizon_status_counts": {
            f"ret_{h}d_status": counts(output[f"ret_{h}d_status"]) for h in HORIZONS
        },
        "output_path": str(output_path),
    }
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    build_price_reaction_v2(args.candidates, args.prices, args.output, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

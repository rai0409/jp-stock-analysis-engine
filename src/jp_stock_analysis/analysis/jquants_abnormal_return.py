"""Attach universe-relative abnormal returns to J-Quants price reaction rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
MIN_PEERS = 5
ACCEPTABLE_PEERS = 10

ABNORMAL_V2_COLUMNS = [
    "abnormal_return_adjustment_basis",
    "abnormal_return_adjustment_quality",
    "abnormal_return_adjustment_flags",
    "size_bucket",
    "insufficient_sector_peer_flag",
    "insufficient_size_peer_flag",
    "insufficient_sector_size_peer_flag",
    "commercial_abnormal_reaction_score",
]
for _basis in ("sector", "size", "sector_size"):
    for _metric in ("peer_return", "adjusted_return", "peer_count"):
        for _horizon in HORIZONS:
            ABNORMAL_V2_COLUMNS.append(f"{_basis}_{_metric}_{_horizon}d")
for _col in [
    "sector_adjusted_return_1d_robust_z",
    "sector_adjusted_return_3d_robust_z",
    "sector_adjusted_return_5d_robust_z",
    "sector_size_adjusted_return_1d_robust_z",
    "sector_size_adjusted_return_3d_robust_z",
    "sector_size_adjusted_return_5d_robust_z",
]:
    ABNORMAL_V2_COLUMNS.append(_col)


def _stable_json(payload: dict[str, Any]) -> str:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    return json.dumps(clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pipe(values: list[str]) -> str:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return "|".join(out) if out else "none"


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


def _load_peer_metadata(tickers: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
    from jp_stock_analysis.analysis.jquants_commercial_quality_enrichment import load_metadata

    metadata, _summary = load_metadata()
    if metadata.empty:
        metadata = pd.DataFrame({"ticker": sorted(set(tickers.dropna().astype(str)))})
    keep_cols = [
        col
        for col in [
            "ticker",
            "sector33_name",
            "sector33_code",
            "topix_size_category",
            "market_segment",
        ]
        if col in metadata.columns
    ]
    meta = metadata[keep_cols].copy() if keep_cols else pd.DataFrame({"ticker": []})
    if "ticker" not in meta.columns:
        meta["ticker"] = []
    meta["ticker"] = meta["ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    if "sector33_name" not in meta.columns:
        meta["sector33_name"] = np.nan
    if "topix_size_category" not in meta.columns:
        meta["topix_size_category"] = np.nan

    all_tickers = pd.DataFrame({"ticker": sorted(set(tickers.dropna().astype(str)))})
    out = all_tickers.merge(meta.drop_duplicates("ticker"), on="ticker", how="left")
    out["size_bucket"] = out["topix_size_category"]
    out["size_bucket"] = out["size_bucket"].astype(object)
    missing_size = out["size_bucket"].isna() | (out["size_bucket"].astype(str).str.len() == 0)
    if missing_size.any():
        fallback = _derive_turnover_size_bucket(prices)
        out.loc[missing_size, "size_bucket"] = out.loc[missing_size, "ticker"].map(fallback)
    return out


def _derive_turnover_size_bucket(prices: pd.DataFrame) -> dict[str, str]:
    if "turnover_value" not in prices.columns:
        return {}
    work = prices.copy()
    work["turnover_value"] = pd.to_numeric(work["turnover_value"], errors="coerce")
    avg = work.groupby("ticker")["turnover_value"].median().dropna()
    if avg.empty:
        return {}
    ranks = avg.rank(method="first", pct=True)
    buckets = pd.Series("size_mid", index=avg.index)
    buckets[ranks >= 2 / 3] = "size_large"
    buckets[ranks < 1 / 3] = "size_small"
    return {str(k): str(v) for k, v in buckets.items()}


def _build_pair_returns(prices: pd.DataFrame, required_pairs: set[tuple[str, str]]) -> dict[tuple[str, str], pd.Series]:
    pivot = prices.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
    cache: dict[tuple[str, str], pd.Series] = {}
    for base_text, future_text in required_pairs:
        base_date = pd.to_datetime(base_text, errors="coerce")
        future_date = pd.to_datetime(future_text, errors="coerce")
        if pd.isna(base_date) or pd.isna(future_date):
            continue
        if base_date not in pivot.index or future_date not in pivot.index:
            continue
        returns = pivot.loc[future_date] / pivot.loc[base_date] - 1.0
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if not returns.empty:
            cache[(base_text, future_text)] = returns
    return cache


def _peer_return(
    pair_returns: dict[tuple[str, str], pd.Series],
    peer_map: dict[str, set[str]],
    ticker: str,
    base_date: str,
    future_date: str,
) -> tuple[float | None, int]:
    returns = pair_returns.get((base_date, future_date))
    if returns is None:
        return None, 0
    peers = set(peer_map.get(ticker, set()))
    peers.discard(ticker)
    available = returns.loc[[peer for peer in peers if peer in returns.index]].dropna()
    if available.empty:
        return None, 0
    return float(available.mean()), int(len(available))


def _peer_maps(metadata: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    maps: dict[str, dict[str, set[str]]] = {"sector": {}, "size": {}, "sector_size": {}}
    by_ticker = metadata.set_index("ticker", drop=False)
    for ticker, row in by_ticker.iterrows():
        sector = row.get("sector33_name")
        size = row.get("size_bucket")
        if pd.notna(sector) and str(sector).strip():
            maps["sector"][ticker] = set(metadata.loc[metadata["sector33_name"].astype(str) == str(sector), "ticker"])
        if pd.notna(size) and str(size).strip():
            maps["size"][ticker] = set(metadata.loc[metadata["size_bucket"].astype(str) == str(size), "ticker"])
        if pd.notna(sector) and str(sector).strip() and pd.notna(size) and str(size).strip():
            mask = (metadata["sector33_name"].astype(str) == str(sector)) & (metadata["size_bucket"].astype(str) == str(size))
            maps["sector_size"][ticker] = set(metadata.loc[mask, "ticker"])
    return maps


def _peer_quality(counts_: list[int]) -> str:
    valid_counts = [count for count in counts_ if count > 0]
    if not valid_counts:
        return "unavailable"
    min_count = min(valid_counts)
    if min_count >= ACCEPTABLE_PEERS:
        return "acceptable"
    if min_count >= MIN_PEERS:
        return "acceptable_with_limitations"
    return "insufficient_peers"


def _robust_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if len(valid) < MIN_PEERS:
        return pd.Series(np.nan, index=series.index)
    median = valid.median()
    mad = (valid - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index)
    z = (values - median) / (1.4826 * mad)
    return z.clip(-5, 5)


def _attach_peer_adjusted_returns(df: pd.DataFrame, prices: pd.DataFrame, required_pairs: set[tuple[str, str]]) -> pd.DataFrame:
    out = df.copy()
    metadata = _load_peer_metadata(prices["ticker"], prices)
    meta_cols = ["ticker", "sector33_name", "size_bucket"]
    out = out.merge(metadata[meta_cols].drop_duplicates("ticker"), on="ticker", how="left")
    pair_returns = _build_pair_returns(prices, required_pairs)
    maps = _peer_maps(metadata)

    for basis in ("sector", "size", "sector_size"):
        for horizon in HORIZONS:
            out[f"{basis}_peer_return_{horizon}d"] = np.nan
            out[f"{basis}_adjusted_return_{horizon}d"] = np.nan
            out[f"{basis}_peer_count_{horizon}d"] = 0

    row_flags: list[list[str]] = []
    for idx, row in out.iterrows():
        ticker = str(row.get("ticker", "")).zfill(4)
        flags: list[str] = []
        if pd.isna(row.get("sector33_name")):
            flags.append("missing_sector")
        if pd.isna(row.get("size_bucket")):
            flags.append("missing_size_bucket")
        for basis in ("sector", "size", "sector_size"):
            for horizon in HORIZONS:
                ret_col = f"ret_{horizon}d"
                date_col = f"ret_{horizon}d_date"
                status_col = f"ret_{horizon}d_status"
                if row.get(status_col) != "valid" or pd.isna(row.get(ret_col)):
                    flags.append(f"{basis}_{horizon}d_source_unavailable")
                    continue
                peer_ret, peer_count = _peer_return(
                    pair_returns,
                    maps[basis],
                    ticker,
                    str(row.get("reaction_base_date")),
                    str(row.get(date_col)),
                )
                out.at[idx, f"{basis}_peer_count_{horizon}d"] = peer_count
                if peer_count < MIN_PEERS or peer_ret is None:
                    flags.append(f"insufficient_{basis}_peers_{horizon}d")
                    continue
                out.at[idx, f"{basis}_peer_return_{horizon}d"] = peer_ret
                out.at[idx, f"{basis}_adjusted_return_{horizon}d"] = float(row.get(ret_col) - peer_ret)
        row_flags.append(flags)

    out["insufficient_sector_peer_flag"] = [
        any(flag.startswith("insufficient_sector_peers") for flag in flags) or "missing_sector" in flags
        for flags in row_flags
    ]
    out["insufficient_size_peer_flag"] = [
        any(flag.startswith("insufficient_size_peers") for flag in flags) or "missing_size_bucket" in flags
        for flags in row_flags
    ]
    out["insufficient_sector_size_peer_flag"] = [
        any(flag.startswith("insufficient_sector_size_peers") for flag in flags) or "missing_sector" in flags or "missing_size_bucket" in flags
        for flags in row_flags
    ]

    for col in [
        "sector_adjusted_return_1d",
        "sector_adjusted_return_3d",
        "sector_adjusted_return_5d",
        "sector_size_adjusted_return_1d",
        "sector_size_adjusted_return_3d",
        "sector_size_adjusted_return_5d",
    ]:
        out[f"{col}_robust_z"] = _robust_z(out[col]) if col in out.columns else np.nan

    out["abnormal_return_adjustment_flags"] = [_pipe(flags) for flags in row_flags]
    basis_values: list[str] = []
    quality_values: list[str] = []
    scores: list[float] = []
    for _, row in out.iterrows():
        has_sector_size = not pd.isna(row.get("sector_size_adjusted_return_1d"))
        has_sector = not pd.isna(row.get("sector_adjusted_return_1d"))
        has_size = not pd.isna(row.get("size_adjusted_return_1d"))
        if has_sector_size:
            basis = "sector_size_adjusted"
            quality = _peer_quality([int(row.get(f"sector_size_peer_count_{h}d", 0)) for h in (1, 3, 5) if int(row.get(f"sector_size_peer_count_{h}d", 0)) > 0])
        elif has_sector and has_size:
            basis = "sector_and_size_adjusted"
            quality = min(
                _peer_quality([int(row.get(f"sector_peer_count_{h}d", 0)) for h in (1, 3, 5) if int(row.get(f"sector_peer_count_{h}d", 0)) > 0]),
                _peer_quality([int(row.get(f"size_peer_count_{h}d", 0)) for h in (1, 3, 5) if int(row.get(f"size_peer_count_{h}d", 0)) > 0]),
                key=["unavailable", "insufficient_peers", "acceptable_with_limitations", "acceptable"].index,
            )
        elif has_sector:
            basis = "sector_adjusted"
            quality = _peer_quality([int(row.get(f"sector_peer_count_{h}d", 0)) for h in (1, 3, 5) if int(row.get(f"sector_peer_count_{h}d", 0)) > 0])
        elif has_size:
            basis = "size_adjusted"
            quality = _peer_quality([int(row.get(f"size_peer_count_{h}d", 0)) for h in (1, 3, 5) if int(row.get(f"size_peer_count_{h}d", 0)) > 0])
        elif not pd.isna(row.get("abnormal_ret_1d")):
            basis = "universe_adjusted_only"
            quality = "acceptable_with_limitations"
        else:
            basis = "adjustment_unavailable"
            quality = "unavailable"
        basis_values.append(basis)
        quality_values.append(quality)
        score, _components = _commercial_score(row)
        scores.append(score)
    out["abnormal_return_adjustment_basis"] = basis_values
    out["abnormal_return_adjustment_quality"] = quality_values
    out["commercial_abnormal_reaction_score"] = scores
    return out


def _commercial_score(row: pd.Series) -> tuple[float, dict[str, Any]]:
    def value(col: str, fallback: float = 0.0) -> float:
        parsed = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        return fallback if pd.isna(parsed) else float(parsed)

    z1 = value("sector_size_adjusted_return_1d_robust_z", value("sector_adjusted_return_1d_robust_z"))
    z3 = value("sector_size_adjusted_return_3d_robust_z", value("sector_adjusted_return_3d_robust_z"))
    z5 = value("sector_adjusted_return_5d_robust_z")
    short = value("abnormal_ret_1d") * 10.0
    quality = 0.5 if str(row.get("abnormal_return_adjustment_quality")) == "acceptable" else 0.25
    insufficient_penalty = 0.0
    for col in ["insufficient_sector_peer_flag", "insufficient_size_peer_flag", "insufficient_sector_size_peer_flag"]:
        if bool(row.get(col)):
            insufficient_penalty += 0.15
    score = 0.35 * z1 + 0.25 * z3 + 0.20 * z5 + 0.10 * short + 0.10 * quality - insufficient_penalty
    components = {
        "sector_size_adjusted_return_1d_robust_z": z1,
        "sector_size_adjusted_return_3d_robust_z": z3,
        "sector_adjusted_return_5d_robust_z": z5,
        "short_horizon_universe_evidence": short,
        "quality_component": quality,
        "insufficient_peer_penalty": insufficient_penalty,
    }
    return round(float(score), 6), components


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

    df = _attach_peer_adjusted_returns(df, prices, required_pairs)
    score_components = []
    scores = []
    for _, row in df.iterrows():
        score, components = _commercial_score(row)
        scores.append(score)
        score_components.append(_stable_json(components))
    df["commercial_abnormal_reaction_score"] = scores
    df["commercial_abnormal_reaction_score_components"] = score_components

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
        "abnormal_return_adjustment_basis_counts": counts(df["abnormal_return_adjustment_basis"]),
        "abnormal_return_adjustment_quality_counts": counts(df["abnormal_return_adjustment_quality"]),
        "adjusted_return_coverage": {
            "sector_adjusted_coverage_ratio": float(df["sector_adjusted_return_1d"].notna().mean()) if len(df) else 0.0,
            "size_adjusted_coverage_ratio": float(df["size_adjusted_return_1d"].notna().mean()) if len(df) else 0.0,
            "sector_size_adjusted_coverage_ratio": float(df["sector_size_adjusted_return_1d"].notna().mean()) if len(df) else 0.0,
            "robust_adjusted_score_coverage_ratio": float(df["commercial_abnormal_reaction_score"].notna().mean()) if len(df) else 0.0,
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

"""Shared helpers for offline J-Quants commercial-quality analysis outputs."""

from __future__ import annotations

import json
from datetime import datetime, time, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path("analysis/jquants_fin_summary")
DEFAULT_CANDIDATES = ANALYSIS_DIR / "commercial_revision_candidates.csv"
DEFAULT_PRICES = Path("data/jquants_price_store/prices_daily_bars.csv")
LOGIC_VERSION = "commercial_quality_v2.0"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def jst_now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=9))).replace(microsecond=0).isoformat()


def normalize_ticker_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ticker" in df.columns:
        df["ticker"] = normalize_ticker_series(df["ticker"])
    return df


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ticker" not in df.columns or "date" not in df.columns:
        raise ValueError("prices CSV must contain ticker and date columns")
    df["ticker"] = normalize_ticker_series(df["ticker"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in [
        "adj_close",
        "turnover_value",
        "volume",
        "adj_open",
        "adj_high",
        "adj_low",
        "open",
        "high",
        "low",
        "close",
        "adj_volume",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.fillna("missing").value_counts(dropna=False).items()}


def numeric_summary(series: pd.Series) -> dict[str, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": None, "median": None}
    return {"mean": float(s.mean()), "median": float(s.median())}


def is_positive_signal(value: object) -> bool:
    text = str(value).lower()
    return "upward" in text or "increase" in text


def is_negative_signal(value: object) -> bool:
    text = str(value).lower()
    return "downward" in text or "decrease" in text


def parse_time(value: object) -> time | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            pass
    return None


def market_close_for_date(event_date: pd.Timestamp) -> time:
    if pd.isna(event_date):
        return time(15, 30)
    return time(15, 30) if event_date.date() >= datetime(2024, 11, 5).date() else time(15, 0)


def event_time_bucket(event_date: pd.Timestamp, first_time: object) -> tuple[str, str]:
    parsed = parse_time(first_time)
    close_time = market_close_for_date(event_date)
    close_text = close_time.strftime("%H:%M")
    if parsed is None:
        return "unknown_time", close_text
    if parsed < time(9, 0):
        return "before_open", close_text
    if parsed < time(11, 30):
        return "morning_session", close_text
    if parsed < time(12, 30):
        return "lunch_break", close_text
    if parsed < close_time:
        return "afternoon_session", close_text
    return "after_close", close_text


def sample_size_bucket(n: int) -> str:
    if n < 5:
        return "very_low_sample"
    if n < 20:
        return "low_sample"
    if n < 50:
        return "medium_sample"
    return "usable_sample"


def sample_warning(bucket: str) -> str:
    return {
        "very_low_sample": "very_low_sample_do_not_overinterpret",
        "low_sample": "low_sample_use_caution",
        "medium_sample": "medium_sample",
        "usable_sample": "usable_sample",
    }[bucket]

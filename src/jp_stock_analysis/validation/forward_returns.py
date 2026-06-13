"""Offline forward-return validation harness.

Goal
----
Measure realized forward returns from a point-in-time ``screening.json`` and a
*later* local prices CSV, then group those returns by the engine's screening
fields. This is the evidence needed to judge whether ``screening_score``,
``reliability_grade`` and ``screening_eligible`` are more informative than the
raw ``final_score`` -- without any network access, paid API, or trading logic.

What this is NOT
----------------
This module is strictly analytical / research-only. It does not:

- emit buy / sell / hold instructions or any trade signal,
- construct a portfolio or weight positions,
- size positions, apply leverage, or model derivatives,
- give personalized financial advice.

It only reports realized forward returns and descriptive statistics grouped by
the screening fields the engine already produced.

Forward-return definition
--------------------------
For each ticker:

1. ``analysis_date`` is taken from the matching entry in the report's
   ``results`` list (the engine writes it there). If absent for a ticker, the
   caller-supplied ``--analysis-date`` override is required.
2. Price rows for the ticker are sorted ascending by date. The **base** price is
   the first row *strictly after* ``analysis_date`` (never on or before it, so
   there is no look-ahead onto the decision date itself).
3. For horizon ``N`` the **target** price is the row ``N`` trading rows after the
   base row (index ``N`` into the strictly-after sequence). The forward return is
   ``(target_close / base_close - 1) * 100`` expressed in percent, matching the
   repository's percent convention.
4. Prices are never interpolated. If fewer than ``N + 1`` rows exist strictly
   after ``analysis_date`` the horizon is marked missing.

Determinism
-----------
Tickers are processed in sorted order, horizons in ascending order, and group
keys in a fixed dimension order, so JSON / CSV / Markdown output is byte-stable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from jp_stock_analysis.config import DEFAULT_DISCLAIMER
from jp_stock_analysis.providers.local_csv import load_prices_csv
from jp_stock_analysis.schemas import PriceBar

FORWARD_RETURN_DISCLAIMER = (
    f"{DEFAULT_DISCLAIMER} This forward-return validation is research-only: it "
    "reports realized historical returns grouped by screening fields and emits no "
    "trading signals, no portfolio construction, and no position sizing."
)

# Status codes attached to each (ticker, horizon) cell.
STATUS_OK = "ok"
STATUS_NO_PRICE_DATA = "no_price_data"
STATUS_NO_BASE_PRICE = "no_base_price"
STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
STATUS_INVALID_BASE_PRICE = "invalid_base_price"

# Fixed-width bucketing for the two score dimensions. Fixed (rather than
# sample-dependent decile) edges keep group keys stable regardless of how many
# tickers are present, which matters for reproducible output.
SCORE_BUCKET_WIDTH = 10.0

DEFAULT_HORIZONS = (5, 20, 60)


def _parse_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = value.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _score_bucket(score: float | None) -> str:
    """Map a 0-100 score onto a fixed-width bucket label such as ``70-80``.

    ``None`` scores group under ``"none"``. Scores are clamped into [0, 100] so
    out-of-range inputs still land in a deterministic bucket.
    """
    if score is None:
        return "none"
    clamped = max(0.0, min(100.0, float(score)))
    lower = int(clamped // SCORE_BUCKET_WIDTH) * int(SCORE_BUCKET_WIDTH)
    if lower >= 100:  # a score of exactly 100 belongs to the top bucket
        lower = 100 - int(SCORE_BUCKET_WIDTH)
    upper = lower + int(SCORE_BUCKET_WIDTH)
    return f"{lower}-{upper}"


@dataclass(frozen=True)
class HorizonReturn:
    """Forward return for one ticker at one horizon."""

    horizon: int
    status: str
    forward_return: float | None = None
    base_date: date | None = None
    base_price: float | None = None
    target_date: date | None = None
    target_price: float | None = None

    @property
    def available(self) -> bool:
        return self.status == STATUS_OK and self.forward_return is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "status": self.status,
            "available": self.available,
            "forward_return": self.forward_return,
            "base_date": self.base_date.isoformat() if self.base_date else None,
            "base_price": self.base_price,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "target_price": self.target_price,
        }


@dataclass(frozen=True)
class TickerForwardReturns:
    """All horizon returns plus carried screening fields for one ticker."""

    ticker: str
    analysis_date: date | None
    analysis_date_source: str
    final_score: float | None
    screening_score: float | None
    confidence_score: float | None
    data_coverage_score: float | None
    screening_eligible: bool | None
    reliability_grade: str | None
    returns: list[HorizonReturn]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "analysis_date": self.analysis_date.isoformat() if self.analysis_date else None,
            "analysis_date_source": self.analysis_date_source,
            "final_score": self.final_score,
            "screening_score": self.screening_score,
            "confidence_score": self.confidence_score,
            "data_coverage_score": self.data_coverage_score,
            "screening_eligible": self.screening_eligible,
            "reliability_grade": self.reliability_grade,
            "returns": [r.to_dict() for r in self.returns],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class GroupHorizonSummary:
    """Descriptive stats for one (dimension, group, horizon) cell."""

    dimension: str
    group: str
    horizon: int
    count: int
    available_horizon_count: int
    missing_horizon_count: int
    mean_forward_return: float | None
    median_forward_return: float | None
    hit_rate_positive: float | None
    min_forward_return: float | None
    max_forward_return: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "group": self.group,
            "horizon": self.horizon,
            "count": self.count,
            "available_horizon_count": self.available_horizon_count,
            "missing_horizon_count": self.missing_horizon_count,
            "mean_forward_return": self.mean_forward_return,
            "median_forward_return": self.median_forward_return,
            "hit_rate_positive": self.hit_rate_positive,
            "min_forward_return": self.min_forward_return,
            "max_forward_return": self.max_forward_return,
        }


@dataclass(frozen=True)
class ForwardReturnReport:
    """Full validation report: per-ticker rows plus grouped summaries."""

    horizons: list[int]
    disclaimer: str
    per_ticker_forward_returns: list[TickerForwardReturns]
    grouped_summary: list[GroupHorizonSummary]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "horizons": list(self.horizons),
            "ticker_count": len(self.per_ticker_forward_returns),
            "warnings": list(self.warnings),
            "per_ticker_forward_returns": [
                t.to_dict() for t in self.per_ticker_forward_returns
            ],
            "grouped_summary": [g.to_dict() for g in self.grouped_summary],
        }


def _load_screening_entries(
    screening_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return screening entries and a ticker -> analysis_date map from results."""
    screening = screening_payload.get("screening")
    if not isinstance(screening, list):
        raise ValueError("screening.json must contain a 'screening' list")

    analysis_dates: dict[str, str] = {}
    for result in screening_payload.get("results", []) or []:
        ticker = result.get("ticker")
        analysis_date = result.get("analysis_date")
        if ticker and analysis_date:
            analysis_dates[str(ticker)] = analysis_date
    return screening, analysis_dates


def _compute_horizon(
    bars: list[PriceBar], analysis_date: date, horizon: int
) -> HorizonReturn:
    """Compute one forward return with strict no-look-ahead semantics."""
    future = [bar for bar in bars if bar.date > analysis_date]
    if not bars:
        return HorizonReturn(horizon=horizon, status=STATUS_NO_PRICE_DATA)
    if not future:
        return HorizonReturn(horizon=horizon, status=STATUS_NO_BASE_PRICE)

    base = future[0]
    if base.close is None or base.close <= 0:
        return HorizonReturn(
            horizon=horizon,
            status=STATUS_INVALID_BASE_PRICE,
            base_date=base.date,
            base_price=base.close,
        )
    if horizon >= len(future):
        return HorizonReturn(
            horizon=horizon,
            status=STATUS_INSUFFICIENT_HISTORY,
            base_date=base.date,
            base_price=base.close,
        )

    target = future[horizon]
    forward_return = round((target.close / base.close - 1.0) * 100.0, 6)
    return HorizonReturn(
        horizon=horizon,
        status=STATUS_OK,
        forward_return=forward_return,
        base_date=base.date,
        base_price=base.close,
        target_date=target.date,
        target_price=target.close,
    )


def _build_ticker_returns(
    entry: dict[str, Any],
    bars: list[PriceBar],
    analysis_date: date | None,
    analysis_date_source: str,
    horizons: list[int],
    extra_warnings: list[str],
) -> TickerForwardReturns:
    warnings = list(extra_warnings)
    if not bars:
        warnings.append("no price rows in prices CSV for this ticker")

    if analysis_date is None:
        returns = [
            HorizonReturn(horizon=h, status=STATUS_NO_BASE_PRICE) for h in horizons
        ]
    else:
        returns = [_compute_horizon(bars, analysis_date, h) for h in horizons]

    return TickerForwardReturns(
        ticker=str(entry.get("ticker")),
        analysis_date=analysis_date,
        analysis_date_source=analysis_date_source,
        final_score=entry.get("final_score"),
        screening_score=entry.get("screening_score"),
        confidence_score=entry.get("confidence_score"),
        data_coverage_score=entry.get("data_coverage_score"),
        screening_eligible=entry.get("screening_eligible"),
        reliability_grade=entry.get("reliability_grade"),
        returns=returns,
        warnings=warnings,
    )


def _group_key(ticker: TickerForwardReturns, dimension: str) -> str:
    if dimension == "screening_eligible":
        if ticker.screening_eligible is None:
            return "none"
        return "true" if ticker.screening_eligible else "false"
    if dimension == "reliability_grade":
        return ticker.reliability_grade or "none"
    if dimension == "screening_score_bucket":
        return _score_bucket(ticker.screening_score)
    if dimension == "final_score_bucket":
        return _score_bucket(ticker.final_score)
    raise ValueError(f"unknown grouping dimension: {dimension}")


_DIMENSIONS = (
    "screening_eligible",
    "reliability_grade",
    "screening_score_bucket",
    "final_score_bucket",
)


def _summarize_group(
    dimension: str,
    group: str,
    members: list[TickerForwardReturns],
    horizons: list[int],
) -> list[GroupHorizonSummary]:
    summaries: list[GroupHorizonSummary] = []
    for horizon in horizons:
        values: list[float] = []
        missing = 0
        for member in members:
            cell = next((r for r in member.returns if r.horizon == horizon), None)
            if cell is not None and cell.available:
                values.append(cell.forward_return)  # type: ignore[arg-type]
            else:
                missing += 1
        available = len(values)
        if values:
            positive = sum(1 for v in values if v > 0)
            summaries.append(
                GroupHorizonSummary(
                    dimension=dimension,
                    group=group,
                    horizon=horizon,
                    count=len(members),
                    available_horizon_count=available,
                    missing_horizon_count=missing,
                    mean_forward_return=round(mean(values), 6),
                    median_forward_return=round(median(values), 6),
                    hit_rate_positive=round(positive / available, 6),
                    min_forward_return=round(min(values), 6),
                    max_forward_return=round(max(values), 6),
                )
            )
        else:
            summaries.append(
                GroupHorizonSummary(
                    dimension=dimension,
                    group=group,
                    horizon=horizon,
                    count=len(members),
                    available_horizon_count=0,
                    missing_horizon_count=missing,
                    mean_forward_return=None,
                    median_forward_return=None,
                    hit_rate_positive=None,
                    min_forward_return=None,
                    max_forward_return=None,
                )
            )
    return summaries


def _build_grouped_summary(
    tickers: list[TickerForwardReturns], horizons: list[int]
) -> list[GroupHorizonSummary]:
    summaries: list[GroupHorizonSummary] = []
    for dimension in _DIMENSIONS:
        groups: dict[str, list[TickerForwardReturns]] = {}
        for ticker in tickers:
            groups.setdefault(_group_key(ticker, dimension), []).append(ticker)
        for group in sorted(groups):
            summaries.extend(
                _summarize_group(dimension, group, groups[group], horizons)
            )
    return summaries


def build_forward_return_report(
    screening_payload: dict[str, Any],
    prices: dict[str, list[PriceBar]],
    horizons: list[int],
    analysis_date_override: date | None = None,
) -> ForwardReturnReport:
    """Build a forward-return report from a parsed screening payload and prices.

    ``prices`` is the ``ticker -> sorted PriceBar list`` mapping returned by
    :func:`jp_stock_analysis.providers.local_csv.load_prices_csv`.
    """
    if not horizons:
        raise ValueError("at least one horizon is required")
    horizons = sorted(set(int(h) for h in horizons))
    for horizon in horizons:
        if horizon < 1:
            raise ValueError("horizons must be positive trading-row offsets")

    entries, analysis_dates = _load_screening_entries(screening_payload)

    # Deterministic processing order, independent of the report's rank order.
    entries = sorted(entries, key=lambda e: str(e.get("ticker")))

    report_warnings: list[str] = []
    missing_date_tickers: list[str] = []
    per_ticker: list[TickerForwardReturns] = []

    for entry in entries:
        ticker = str(entry.get("ticker"))
        extra_warnings: list[str] = []

        raw_date = analysis_dates.get(ticker)
        analysis_date = _parse_date(raw_date)
        if analysis_date is not None:
            source = "result"
        elif analysis_date_override is not None:
            analysis_date = analysis_date_override
            source = "override"
            extra_warnings.append(
                "analysis_date missing from results; used --analysis-date override"
            )
        else:
            source = "missing"
            missing_date_tickers.append(ticker)
            extra_warnings.append(
                "analysis_date missing from results and no --analysis-date override; "
                "forward returns not computed"
            )

        per_ticker.append(
            _build_ticker_returns(
                entry,
                prices.get(ticker, []),
                analysis_date,
                source,
                horizons,
                extra_warnings,
            )
        )

    if missing_date_tickers:
        report_warnings.append(
            "no analysis_date available for tickers "
            f"{', '.join(sorted(missing_date_tickers))}; pass --analysis-date to compute "
            "their forward returns"
        )

    grouped = _build_grouped_summary(per_ticker, horizons)
    return ForwardReturnReport(
        horizons=horizons,
        disclaimer=FORWARD_RETURN_DISCLAIMER,
        per_ticker_forward_returns=per_ticker,
        grouped_summary=grouped,
        warnings=report_warnings,
    )


def load_forward_return_report(
    screening_json_path: str | Path,
    prices_path: str | Path,
    horizons: list[int],
    analysis_date_override: date | None = None,
) -> ForwardReturnReport:
    """Load inputs from disk and build the report. No network access."""
    payload = json.loads(Path(screening_json_path).read_text(encoding="utf-8"))
    prices = load_prices_csv(prices_path)
    return build_forward_return_report(
        payload, prices, horizons, analysis_date_override
    )


def _write_json(report: ForwardReturnReport, path: Path) -> Path:
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(report: ForwardReturnReport, path: Path) -> Path:
    import csv

    columns = [
        "ticker",
        "analysis_date",
        "analysis_date_source",
        "horizon",
        "status",
        "available",
        "forward_return",
        "base_date",
        "base_price",
        "target_date",
        "target_price",
        "final_score",
        "screening_score",
        "confidence_score",
        "data_coverage_score",
        "screening_eligible",
        "reliability_grade",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for ticker in report.per_ticker_forward_returns:
            for cell in ticker.returns:
                writer.writerow(
                    {
                        "ticker": ticker.ticker,
                        "analysis_date": (
                            ticker.analysis_date.isoformat()
                            if ticker.analysis_date
                            else ""
                        ),
                        "analysis_date_source": ticker.analysis_date_source,
                        "horizon": cell.horizon,
                        "status": cell.status,
                        "available": str(cell.available).lower(),
                        "forward_return": (
                            "" if cell.forward_return is None else cell.forward_return
                        ),
                        "base_date": cell.base_date.isoformat() if cell.base_date else "",
                        "base_price": "" if cell.base_price is None else cell.base_price,
                        "target_date": (
                            cell.target_date.isoformat() if cell.target_date else ""
                        ),
                        "target_price": (
                            "" if cell.target_price is None else cell.target_price
                        ),
                        "final_score": (
                            "" if ticker.final_score is None else ticker.final_score
                        ),
                        "screening_score": (
                            ""
                            if ticker.screening_score is None
                            else ticker.screening_score
                        ),
                        "confidence_score": (
                            ""
                            if ticker.confidence_score is None
                            else ticker.confidence_score
                        ),
                        "data_coverage_score": (
                            ""
                            if ticker.data_coverage_score is None
                            else ticker.data_coverage_score
                        ),
                        "screening_eligible": (
                            ""
                            if ticker.screening_eligible is None
                            else str(ticker.screening_eligible).lower()
                        ),
                        "reliability_grade": ticker.reliability_grade or "",
                    }
                )
    return path


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _write_markdown(report: ForwardReturnReport, path: Path) -> Path:
    lines: list[str] = []
    lines.append("# Forward-Return Validation")
    lines.append("")
    lines.append(report.disclaimer)
    lines.append("")
    lines.append(
        "This report measures realized historical forward returns grouped by the "
        "engine's screening fields. It contains no trading signals, no portfolio "
        "construction, no position sizing, and is not financial advice."
    )
    lines.append("")
    lines.append(f"Horizons (trading rows after analysis date): {report.horizons}")
    lines.append("")
    for warning in report.warnings:
        lines.append(f"> warning: {warning}")
    if report.warnings:
        lines.append("")

    lines.append("## Grouped summary")
    lines.append("")
    lines.append(
        "| dimension | group | horizon | count | available | missing | mean | "
        "median | hit_rate_positive | min | max |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g in report.grouped_summary:
        hit = "—" if g.hit_rate_positive is None else f"{g.hit_rate_positive:.4f}"
        lines.append(
            f"| {g.dimension} | {g.group} | {g.horizon} | {g.count} | "
            f"{g.available_horizon_count} | {g.missing_horizon_count} | "
            f"{_fmt(g.mean_forward_return)} | {_fmt(g.median_forward_return)} | "
            f"{hit} | {_fmt(g.min_forward_return)} | {_fmt(g.max_forward_return)} |"
        )
    lines.append("")

    lines.append("## Per-ticker forward returns")
    lines.append("")
    lines.append(
        "| ticker | analysis_date | horizon | status | forward_return | "
        "screening_eligible | reliability_grade |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for ticker in report.per_ticker_forward_returns:
        ad = ticker.analysis_date.isoformat() if ticker.analysis_date else "—"
        eligible = (
            "—"
            if ticker.screening_eligible is None
            else str(ticker.screening_eligible).lower()
        )
        for cell in ticker.returns:
            lines.append(
                f"| {ticker.ticker} | {ad} | {cell.horizon} | {cell.status} | "
                f"{_fmt(cell.forward_return)} | {eligible} | "
                f"{ticker.reliability_grade or '—'} |"
            )
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_forward_return_outputs(
    report: ForwardReturnReport,
    output_dir: str | Path,
    write_markdown: bool = True,
) -> dict[str, Path]:
    """Write JSON + CSV (+ optional Markdown) outputs and return their paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json_path": _write_json(report, out_dir / "forward_returns.json"),
        "csv_path": _write_csv(report, out_dir / "forward_returns.csv"),
    }
    if write_markdown:
        paths["markdown_path"] = _write_markdown(
            report, out_dir / "forward_returns.md"
        )
    return paths

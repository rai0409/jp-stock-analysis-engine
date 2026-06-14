"""JPX-style long-short ranking evaluation (research-only).

Given a per-observation *prediction score* and a realised *forward return*, this
builds a cross-sectional **top-N long / bottom-N short** portfolio per decision
date and summarises the resulting **spread-return series** the way long-short
competition scoring does: a Sharpe-like mean/std of the spread, an equity curve,
max drawdown, hit rate, turnover, and an optional simplified transaction-cost
drag.

This is a research metric, **not** a trading system. It emits no buy/sell
signal, claims no predictive performance, and does not simulate execution. It
operates on forward-return labels already produced upstream (dataset/validation)
and never fetches prices — real inputs must be adjusted-close-derived upstream.
Degenerate cross-sections return an explicit status, never an exception, and are
never fabricated.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from jp_stock_analysis.modeling.ranking_metrics import RESEARCH_DISCLAIMER, ScoredObservation

MODE_COUNT = "count"
MODE_QUANTILE = "quantile"

STATUS_OK = "ok"
STATUS_INSUFFICIENT_NAMES = "insufficient_names"
STATUS_CONSTANT_PREDICTIONS = "constant_predictions"
STATUS_NO_VALID_DATES = "no_valid_dates"
STATUS_DEGENERATE_SERIES = "degenerate_series"

WEIGHT_EQUAL = "equal"
WEIGHT_RANK = "rank_weighted"


@dataclass(frozen=True)
class PortfolioObservation:
    """One (ticker, decision_date) prediction + realised forward return."""

    decision_date: date
    ticker: str
    score: float | None
    forward_return: float | None
    sector: str | None = None
    weight: float | None = None


@dataclass(frozen=True)
class DateSpread:
    """Long-short outcome for one decision date."""

    decision_date: date
    status: str
    spread_return: float | None
    long_leg_return: float | None
    short_leg_return: float | None
    count_long: int
    count_short: int
    universe_count: int
    long_weights: dict[str, float] = field(default_factory=dict)
    short_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_date": self.decision_date.isoformat(),
            "status": self.status,
            "spread_return": self.spread_return,
            "long_leg_return": self.long_leg_return,
            "short_leg_return": self.short_leg_return,
            "count_long": self.count_long,
            "count_short": self.count_short,
            "universe_count": self.universe_count,
        }


def observations_from_scored(
    scored: Sequence[ScoredObservation], horizon: int
) -> list[PortfolioObservation]:
    """Adapt ranking ``ScoredObservation``s into portfolio observations."""
    label_key = f"forward_return_h{horizon}"
    out: list[PortfolioObservation] = []
    for obs in scored:
        out.append(
            PortfolioObservation(
                decision_date=obs.decision_date,
                ticker=obs.ticker,
                score=obs.score,
                forward_return=obs.labels.get(label_key),
                sector=obs.sector,
            )
        )
    return out


def _leg_weights(
    members: list[PortfolioObservation], *, rank_weighted: bool, ascending: bool
) -> dict[str, float]:
    """Normalised positive weights for one leg.

    ``rank_weighted`` gives the most-favourable name the largest weight via
    linearly decreasing weights (n, n-1, …, 1); equal weight otherwise. An
    optional per-observation ``weight`` multiplies the base weight. ``ascending``
    orders the short leg so its lowest-score name gets the largest weight.
    """
    ordered = sorted(members, key=lambda o: (o.score, o.ticker), reverse=not ascending)
    n = len(ordered)
    weights: dict[str, float] = {}
    for index, obs in enumerate(ordered):
        base = float(n - index) if rank_weighted else 1.0
        multiplier = obs.weight if obs.weight is not None else 1.0
        weights[obs.ticker] = base * multiplier
    total = sum(weights.values())
    if total <= 0:
        # all multipliers zero: fall back to equal weights so the leg is defined
        return {obs.ticker: 1.0 / n for obs in ordered}
    return {ticker: value / total for ticker, value in weights.items()}


def _leg_return(
    members: list[PortfolioObservation], weights: Mapping[str, float]
) -> float:
    return sum(weights[o.ticker] * float(o.forward_return) for o in members)  # type: ignore[arg-type]


def _select_counts(
    universe: int, *, mode: str, top_n: int, bottom_n: int, top_q: float, bottom_q: float
) -> tuple[int, int]:
    if mode == MODE_COUNT:
        return top_n, bottom_n
    long_n = max(1, math.floor(universe * top_q))
    short_n = max(1, math.floor(universe * bottom_q))
    return long_n, short_n


def evaluate_date_spread(
    observations: Sequence[PortfolioObservation],
    *,
    mode: str = MODE_QUANTILE,
    top_n: int = 1,
    bottom_n: int = 1,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    rank_weighted: bool = False,
) -> DateSpread:
    """Compute the long-short spread for a single decision date's cross-section."""
    decision_date = observations[0].decision_date if observations else None
    usable = [
        o for o in observations if o.score is not None and o.forward_return is not None
    ]
    universe = len(usable)
    base = DateSpread(
        decision_date=decision_date,  # type: ignore[arg-type]
        status=STATUS_INSUFFICIENT_NAMES,
        spread_return=None,
        long_leg_return=None,
        short_leg_return=None,
        count_long=0,
        count_short=0,
        universe_count=universe,
    )
    if decision_date is None or universe < 2:
        return base
    scores = [float(o.score) for o in usable]  # type: ignore[arg-type]
    if max(scores) == min(scores):
        return DateSpread(
            decision_date=decision_date,
            status=STATUS_CONSTANT_PREDICTIONS,
            spread_return=None,
            long_leg_return=None,
            short_leg_return=None,
            count_long=0,
            count_short=0,
            universe_count=universe,
        )

    long_n, short_n = _select_counts(
        universe,
        mode=mode,
        top_n=top_n,
        bottom_n=bottom_n,
        top_q=top_quantile,
        bottom_q=bottom_quantile,
    )
    if long_n < 1 or short_n < 1 or long_n + short_n > universe:
        return base  # would overlap or empty a leg

    ranked = sorted(usable, key=lambda o: (o.score, o.ticker), reverse=True)
    long_members = ranked[:long_n]
    short_members = ranked[-short_n:]

    long_weights = _leg_weights(long_members, rank_weighted=rank_weighted, ascending=False)
    short_weights = _leg_weights(short_members, rank_weighted=rank_weighted, ascending=True)
    long_return = _leg_return(long_members, long_weights)
    short_return = _leg_return(short_members, short_weights)
    return DateSpread(
        decision_date=decision_date,
        status=STATUS_OK,
        spread_return=long_return - short_return,
        long_leg_return=long_return,
        short_leg_return=short_return,
        count_long=len(long_members),
        count_short=len(short_members),
        universe_count=universe,
        long_weights=long_weights,
        short_weights=short_weights,
    )


# --------------------------------------------------------------------------- #
# Series-level summaries
# --------------------------------------------------------------------------- #
def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


@dataclass(frozen=True)
class SpreadSeriesSummary:
    observation_count: int
    mean_spread: float | None
    std_spread: float | None
    sharpe_like: float | None
    annualized_sharpe_like: float | None
    hit_rate: float | None
    max_drawdown: float | None
    worst_period: dict[str, Any] | None
    best_period: dict[str, Any] | None
    equity_curve: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "mean_spread": self.mean_spread,
            "std_spread": self.std_spread,
            "sharpe_like": self.sharpe_like,
            "annualized_sharpe_like": self.annualized_sharpe_like,
            "hit_rate": self.hit_rate,
            "max_drawdown": self.max_drawdown,
            "worst_period": self.worst_period,
            "best_period": self.best_period,
            "equity_curve": self.equity_curve,
        }


def summarize_spread_series(
    spreads: Sequence[DateSpread], *, periods_per_year: int | None = None
) -> SpreadSeriesSummary:
    """Summarise the spread-return series (Sharpe-like, equity curve, drawdown)."""
    valid = [s for s in spreads if s.status == STATUS_OK and s.spread_return is not None]
    valid.sort(key=lambda s: s.decision_date)
    series = [float(s.spread_return) for s in valid]  # type: ignore[arg-type]
    mean = _mean(series)
    std = _std(series)
    sharpe = mean / std if (mean is not None and std not in (None, 0.0)) else None
    annualized = (
        sharpe * math.sqrt(periods_per_year)
        if (sharpe is not None and periods_per_year)
        else None
    )
    hit_rate = (
        sum(1 for v in series if v > 0) / len(series) if series else None
    )

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    curve: list[dict[str, Any]] = []
    for spread in valid:
        equity *= 1.0 + float(spread.spread_return) / 100.0  # type: ignore[arg-type]
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, equity / peak - 1.0)
        curve.append({"decision_date": spread.decision_date.isoformat(), "equity": equity})

    worst = best = None
    if valid:
        worst_spread = min(valid, key=lambda s: s.spread_return)  # type: ignore[arg-type,return-value]
        best_spread = max(valid, key=lambda s: s.spread_return)  # type: ignore[arg-type,return-value]
        worst = {
            "decision_date": worst_spread.decision_date.isoformat(),
            "spread_return": worst_spread.spread_return,
        }
        best = {
            "decision_date": best_spread.decision_date.isoformat(),
            "spread_return": best_spread.spread_return,
        }
    return SpreadSeriesSummary(
        observation_count=len(valid),
        mean_spread=mean,
        std_spread=std,
        sharpe_like=sharpe,
        annualized_sharpe_like=annualized,
        hit_rate=hit_rate,
        max_drawdown=max_dd * 100.0 if valid else None,
        worst_period=worst,
        best_period=best,
        equity_curve=curve,
    )


# --------------------------------------------------------------------------- #
# Turnover + (optional, simplified) transaction cost
# --------------------------------------------------------------------------- #
def _leg_turnover(prev: Mapping[str, float], curr: Mapping[str, float]) -> float:
    """One-sided turnover: 0.5 * sum |w_curr - w_prev| over the union of names."""
    names = set(prev) | set(curr)
    return 0.5 * sum(abs(curr.get(n, 0.0) - prev.get(n, 0.0)) for n in names)


@dataclass(frozen=True)
class TurnoverSummary:
    per_date: list[dict[str, Any]]
    average_turnover: float | None
    max_turnover: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_date": self.per_date,
            "average_turnover": self.average_turnover,
            "max_turnover": self.max_turnover,
            "note": (
                "two-sided total_turnover = long_turnover + short_turnover (range "
                "0..2). Initial-build turnover on the first date is excluded."
            ),
        }


def compute_turnover(spreads: Sequence[DateSpread]) -> TurnoverSummary:
    """Long/short/total turnover between consecutive valid decision dates."""
    valid = [s for s in spreads if s.status == STATUS_OK]
    valid.sort(key=lambda s: s.decision_date)
    per_date: list[dict[str, Any]] = []
    totals: list[float] = []
    for prev, curr in zip(valid, valid[1:], strict=False):
        long_t = _leg_turnover(prev.long_weights, curr.long_weights)
        short_t = _leg_turnover(prev.short_weights, curr.short_weights)
        total = long_t + short_t
        totals.append(total)
        per_date.append(
            {
                "decision_date": curr.decision_date.isoformat(),
                "long_turnover": long_t,
                "short_turnover": short_t,
                "total_turnover": total,
            }
        )
    return TurnoverSummary(
        per_date=per_date,
        average_turnover=_mean(totals),
        max_turnover=max(totals) if totals else None,
    )


@dataclass(frozen=True)
class TransactionCostSummary:
    transaction_cost_bps: float
    gross_mean_spread: float | None
    net_mean_spread: float | None
    net_sharpe_like: float | None
    per_date: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_cost_bps": self.transaction_cost_bps,
            "gross_mean_spread": self.gross_mean_spread,
            "net_mean_spread": self.net_mean_spread,
            "net_sharpe_like": self.net_sharpe_like,
            "per_date": self.per_date,
            "note": (
                "SIMPLIFIED research approximation, NOT execution simulation: "
                "net_spread = gross_spread - turnover * cost_bps / 100 (returns in "
                "percent; equivalently gross_fraction - turnover*cost_bps/10000)."
            ),
        }


def apply_transaction_cost(
    spreads: Sequence[DateSpread],
    turnover: TurnoverSummary,
    *,
    transaction_cost_bps: float,
) -> TransactionCostSummary:
    """Apply a simplified turnover-based cost. Default upstream is 0 bps."""
    valid = sorted(
        (s for s in spreads if s.status == STATUS_OK and s.spread_return is not None),
        key=lambda s: s.decision_date,
    )
    turnover_by_date = {row["decision_date"]: row["total_turnover"] for row in turnover.per_date}
    gross = [float(s.spread_return) for s in valid]  # type: ignore[arg-type]
    net_series: list[float] = []
    per_date: list[dict[str, Any]] = []
    for spread in valid:
        iso = spread.decision_date.isoformat()
        turn = turnover_by_date.get(iso, 0.0)  # first date has no prior -> 0 cost
        cost = turn * transaction_cost_bps / 100.0
        net = float(spread.spread_return) - cost  # type: ignore[arg-type]
        net_series.append(net)
        per_date.append(
            {"decision_date": iso, "gross_spread": spread.spread_return, "net_spread": net}
        )
    net_mean = _mean(net_series)
    net_std = _std(net_series)
    net_sharpe = (
        net_mean / net_std if (net_mean is not None and net_std not in (None, 0.0)) else None
    )
    return TransactionCostSummary(
        transaction_cost_bps=transaction_cost_bps,
        gross_mean_spread=_mean(gross),
        net_mean_spread=net_mean,
        net_sharpe_like=net_sharpe,
        per_date=per_date,
    )


@dataclass(frozen=True)
class PortfolioReport:
    model_label: str
    horizon: int
    is_synthetic: bool
    config: dict[str, Any]
    per_date: list[DateSpread]
    series: SpreadSeriesSummary
    turnover: TurnoverSummary
    transaction_cost: TransactionCostSummary | None
    disclaimer: str = RESEARCH_DISCLAIMER

    @property
    def status(self) -> str:
        if not any(s.status == STATUS_OK for s in self.per_date):
            return STATUS_NO_VALID_DATES
        if self.series.sharpe_like is None:
            return STATUS_DEGENERATE_SERIES
        return STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "research_only": True,
            "model_label": self.model_label,
            "horizon": self.horizon,
            "is_synthetic": self.is_synthetic,
            "synthetic_warning": (
                "SYNTHETIC FIXTURE RESULTS — not real market evidence."
                if self.is_synthetic
                else None
            ),
            "status": self.status,
            "config": self.config,
            "spread_series": self.series.to_dict(),
            "turnover": self.turnover.to_dict(),
            "transaction_cost": (
                self.transaction_cost.to_dict() if self.transaction_cost else None
            ),
            "per_date": [s.to_dict() for s in self.per_date],
        }


def evaluate_portfolio(
    observations: Sequence[PortfolioObservation],
    *,
    horizon: int,
    model_label: str = "baseline_factor_ranker",
    is_synthetic: bool = False,
    mode: str = MODE_QUANTILE,
    top_n: int = 1,
    bottom_n: int = 1,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    rank_weighted: bool = False,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int | None = None,
) -> PortfolioReport:
    """Full JPX-style long-short evaluation over all decision dates."""
    if mode not in (MODE_COUNT, MODE_QUANTILE):
        raise ValueError(f"unknown mode {mode!r}")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")

    by_date: dict[date, list[PortfolioObservation]] = {}
    for obs in observations:
        by_date.setdefault(obs.decision_date, []).append(obs)

    per_date = [
        evaluate_date_spread(
            by_date[d],
            mode=mode,
            top_n=top_n,
            bottom_n=bottom_n,
            top_quantile=top_quantile,
            bottom_quantile=bottom_quantile,
            rank_weighted=rank_weighted,
        )
        for d in sorted(by_date)
    ]
    series = summarize_spread_series(per_date, periods_per_year=periods_per_year)
    turnover = compute_turnover(per_date)
    cost_summary = (
        apply_transaction_cost(per_date, turnover, transaction_cost_bps=transaction_cost_bps)
        if transaction_cost_bps > 0
        else None
    )
    config = {
        "mode": mode,
        "weighting": WEIGHT_RANK if rank_weighted else WEIGHT_EQUAL,
        "top_n": top_n,
        "bottom_n": bottom_n,
        "top_quantile": top_quantile,
        "bottom_quantile": bottom_quantile,
        "transaction_cost_bps": transaction_cost_bps,
        "periods_per_year": periods_per_year,
    }
    return PortfolioReport(
        model_label=model_label,
        horizon=horizon,
        is_synthetic=is_synthetic,
        config=config,
        per_date=per_date,
        series=series,
        turnover=turnover,
        transaction_cost=cost_summary,
    )


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def portfolio_markdown(report: PortfolioReport) -> str:
    """Markdown summary (research-only; synthetic clearly labelled)."""
    lines = ["# JPX-Style Long-Short Spread Evaluation", "", report.disclaimer, ""]
    lines.append(
        "Research metric only — inspired by long-short spread competition scoring; "
        "it does not claim exchange/execution realism and produces no trading signal."
    )
    lines.append("")
    if report.is_synthetic:
        lines += ["> **SYNTHETIC FIXTURE RESULTS — not real market evidence.**", ""]
    s = report.series
    lines += [
        f"- Model: `{report.model_label}`  |  horizon: {report.horizon}  |  "
        f"status: **{report.status}**",
        f"- Config: {report.config}",
        f"- Valid periods: {s.observation_count}",
        f"- Mean spread: {_fmt(s.mean_spread)}  |  Std: {_fmt(s.std_spread)}  |  "
        f"**Sharpe-like: {_fmt(s.sharpe_like)}**  |  annualized: {_fmt(s.annualized_sharpe_like)}",
        f"- Hit rate (spread > 0): {_fmt(s.hit_rate)}",
        f"- Max drawdown: {_fmt(s.max_drawdown)}  |  worst: {s.worst_period}  |  "
        f"best: {s.best_period}",
        f"- Turnover avg/max: {_fmt(report.turnover.average_turnover)} / "
        f"{_fmt(report.turnover.max_turnover)}",
    ]
    if report.transaction_cost is not None:
        tc = report.transaction_cost
        lines.append(
            f"- Transaction cost {tc.transaction_cost_bps}bps -> net mean spread "
            f"{_fmt(tc.net_mean_spread)} (gross {_fmt(tc.gross_mean_spread)}), "
            f"net Sharpe-like {_fmt(tc.net_sharpe_like)}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_portfolio_outputs(
    report: PortfolioReport, output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    """Write portfolio_metrics.json / .csv (+ optional .md)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "portfolio_metrics.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = out_dir / "portfolio_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "decision_date",
                "status",
                "spread_return",
                "long_leg_return",
                "short_leg_return",
                "count_long",
                "count_short",
                "universe_count",
            ]
        )
        for s in report.per_date:
            writer.writerow(
                [
                    s.decision_date.isoformat(),
                    s.status,
                    "" if s.spread_return is None else f"{s.spread_return:.6f}",
                    "" if s.long_leg_return is None else f"{s.long_leg_return:.6f}",
                    "" if s.short_leg_return is None else f"{s.short_leg_return:.6f}",
                    s.count_long,
                    s.count_short,
                    s.universe_count,
                ]
            )
    paths = {"json_path": json_path, "csv_path": csv_path}
    if write_markdown:
        md_path = out_dir / "portfolio_metrics.md"
        md_path.write_text(portfolio_markdown(report), encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


__all__ = [
    "MODE_COUNT",
    "MODE_QUANTILE",
    "STATUS_CONSTANT_PREDICTIONS",
    "STATUS_INSUFFICIENT_NAMES",
    "STATUS_NO_VALID_DATES",
    "STATUS_OK",
    "DateSpread",
    "PortfolioObservation",
    "PortfolioReport",
    "SpreadSeriesSummary",
    "TransactionCostSummary",
    "TurnoverSummary",
    "apply_transaction_cost",
    "compute_turnover",
    "evaluate_date_spread",
    "evaluate_portfolio",
    "observations_from_scored",
    "portfolio_markdown",
    "summarize_spread_series",
    "write_portfolio_outputs",
]

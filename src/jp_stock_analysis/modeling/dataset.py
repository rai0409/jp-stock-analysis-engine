"""Offline modeling-dataset builder.

Assembles model-ready observations from fundamentals, metadata, optional
disclosure narrative, and forward-return labels (when later prices exist). The
build is deterministic and enforces no-look-ahead guardrails:

- a fundamentals row is usable at a decision date only if its disclosure date is
  **on or before** that decision date (else excluded: ``disclosure_after_decision``);
- ``non_consolidated`` rows are excluded by default and ``unknown`` rows flagged,
  so consolidated and parent-only figures are never silently pooled;
- forward-return labels use only prices **strictly after** the decision date
  (price-axis no-look-ahead, matching ``validation/no_lookahead.py``);
- synthetic observations carry an explicit ``is_synthetic`` flag and are never
  presented as real market evidence.

Nothing here makes a predictive or trading claim. Labels are realised forward
returns measured for research validation only.
"""

from __future__ import annotations

import bisect
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from jp_stock_analysis.modeling.factors import ALL_FACTORS, compute_factors
from jp_stock_analysis.modeling.trading_features import (
    TRADING_FEATURES,
    adjusted_ohlc,
    compute_trading_features,
    equal_weight_market_returns,
    traded_value_20d,
)
from jp_stock_analysis.schemas import (
    CompanyMetadata,
    DisclosureDocument,
    FinancialStatement,
    ListingStatus,
    MarginBalance,
    PriceBar,
)

BASIS_CONSOLIDATED = "consolidated"
BASIS_NON_CONSOLIDATED = "non_consolidated"
BASIS_MIXED = "mixed"
BASIS_UNKNOWN = "unknown"

EXCLUDE_NO_FUNDAMENTALS = "no_fundamentals"
EXCLUDE_DISCLOSURE_AFTER_DECISION = "disclosure_after_decision"
EXCLUDE_NON_CONSOLIDATED = "non_consolidated_basis"

DEFAULT_HORIZONS = (5, 20, 60)
MIN_SECTOR_PEERS_FOR_EXCESS = 2


ENTRY_NEXT_CLOSE = "next_close"
ENTRY_NEXT_OPEN = "next_open"
ENTRY_MODES = (ENTRY_NEXT_CLOSE, ENTRY_NEXT_OPEN)

FEATURE_SET_CORE = "core"
FEATURE_SET_EXTENDED = "extended"
FEATURE_SETS = (FEATURE_SET_CORE, FEATURE_SET_EXTENDED)

EXCLUDE_ILLIQUID = "illiquid"
EXCLUDE_NOT_YET_LISTED = "not_yet_listed"
EXCLUDE_DELISTED = "delisted"


def _adjusted_open(bar: PriceBar) -> float | None:
    if bar.open is None:
        return None
    if bar.adjusted_close is not None and bar.close:
        return float(bar.open) * float(bar.adjusted_close) / float(bar.close)
    return float(bar.open)


# Trailing bars used for as-of feature computation. The longest look-back is
# ~253 rows (12-1 momentum, 52-week high, 1-year drawdown); 400 rows (~1.6
# years) also covers post-disclosure drift for statements up to ~1.5 years old.
FEATURE_WINDOW_ROWS = 400
# next_open exits: if the exit session is locked limit-up/down (no range or no
# volume) the position is sold at the first tradable open within this many
# further sessions; purging adds the same allowance to every label window.
MAX_EXIT_DELAY_ROWS = 5


def _unfillable(bar: PriceBar) -> bool:
    """Locked limit (high == low) or zero volume: an open order would not fill."""
    if bar.volume is not None and bar.volume == 0:
        return True
    return bar.high is not None and bar.low is not None and bar.high == bar.low


def _forward_return_after(
    after: Sequence[PriceBar], horizon: int, entry: str
) -> float | None:
    """Forward return from the bars strictly after the decision date (sorted)."""
    if len(after) < horizon + 1:
        return None
    if entry == ENTRY_NEXT_OPEN:
        if _unfillable(after[0]):
            return None  # cannot buy at a locked open
        base = _adjusted_open(after[0])
        exit_bar = None
        for k in range(horizon, min(len(after), horizon + MAX_EXIT_DELAY_ROWS + 1)):
            if not _unfillable(after[k]):
                exit_bar = after[k]
                break
        target = _adjusted_open(exit_bar) if exit_bar is not None else None
        if base in (None, 0) or target is None:
            return None
        return (float(target) / float(base) - 1.0) * 100.0  # type: ignore[arg-type]
    use_adjusted = all(b.adjusted_close is not None for b in after[: horizon + 1])
    base = after[0].adjusted_close if use_adjusted else after[0].close
    target = after[horizon].adjusted_close if use_adjusted else after[horizon].close
    if base in (None, 0):
        return None
    return (float(target) / float(base) - 1.0) * 100.0  # type: ignore[arg-type]


def _adjust_factor(bar: PriceBar) -> float:
    if bar.adjusted_close is not None and bar.close:
        return float(bar.adjusted_close) / float(bar.close)
    return 1.0


def _delisting_adjusted_return(
    after: Sequence[PriceBar],
    horizon: int,
    entry: str,
    listing: ListingStatus,
    penalty_pct: float | None,
) -> tuple[float | None, str | None]:
    """Forward return when the ticker delists inside the holding window.

    Returns ``(value, flag)``; ``flag`` is None when the delisting does not
    affect this label (the normal label applies). Exit value precedence:
    ``final_price`` (cash-out, e.g. TOB) > last traded close x
    (1 + ``delisting_return_pct``) > last traded close x (1 + ``penalty_pct``,
    only if explicitly configured) > last traded close. Nothing is invented:
    without any source value the exit is the last real trade.
    """
    delist = listing.delisting_date
    if delist is None:
        return None, None
    trading = [b for b in after if b.date < delist]
    window_rows = horizon + 1 + (MAX_EXIT_DELAY_ROWS if entry == ENTRY_NEXT_OPEN else 0)
    if len(trading) >= window_rows:
        return None, None  # delisting is beyond the label window: normal label
    if len(trading) > horizon and _forward_return_after(trading, horizon, entry) is not None:
        return None, None  # the normal exit happens before the delisting
    if not trading:
        return None, "delisted_before_entry"
    first, last = trading[0], trading[-1]
    if entry == ENTRY_NEXT_OPEN:
        if _unfillable(first):
            return None, "delisted_unfillable_entry"
        base = _adjusted_open(first)
    else:
        base = first.adjusted_close if first.adjusted_close is not None else first.close
    if base in (None, 0):
        return None, "delisted_missing_price"
    last_value = float(last.adjusted_close if last.adjusted_close is not None else last.close)
    if listing.final_price is not None:
        exit_value = float(listing.final_price) * _adjust_factor(last)
        source = "final_price"
    elif listing.delisting_return_pct is not None:
        exit_value = last_value * (1.0 + listing.delisting_return_pct / 100.0)
        source = "delisting_return"
    elif penalty_pct is not None:
        exit_value = last_value * (1.0 + penalty_pct / 100.0)
        source = "configured_penalty"
    else:
        exit_value = last_value
        source = "last_trade"
    return (exit_value / float(base) - 1.0) * 100.0, f"delisting_adjusted:{source}"


def forward_return(
    bars: Sequence[PriceBar],
    decision_date: date,
    horizon: int,
    *,
    entry: str = ENTRY_NEXT_CLOSE,
) -> float | None:
    """Realised forward return (percent) for ``horizon`` trading rows.

    ``entry="next_close"`` (legacy): base = close of the first row strictly
    after ``decision_date``; target = close ``horizon`` rows later.
    ``entry="next_open"`` (tradable): buy at the next session's **open** and
    sell at the open ``horizon`` sessions later — the fill a decision taken
    after the close can actually get. A locked entry session (limit up/down,
    ``high == low``, or zero volume) is unfillable -> ``None``; a locked exit
    session rolls to the first tradable open within ``MAX_EXIT_DELAY_ROWS``.
    Split-adjusted prices are used when available. ``None`` when rows/prices
    are missing.
    """
    if entry not in ENTRY_MODES:
        raise ValueError(f"unknown entry mode {entry!r}; expected {ENTRY_MODES}")
    ordered = sorted(bars, key=lambda b: b.date)
    after = ordered[bisect.bisect_right([b.date for b in ordered], decision_date):]
    return _forward_return_after(after, horizon, entry)


@dataclass(frozen=True)
class ModelingObservation:
    """One (ticker, decision_date) model row."""

    ticker: str
    decision_date: date
    disclosure_date: date | None
    horizons: tuple[int, ...]
    accounting_basis: str
    sector: str | None
    market: str | None
    features: dict[str, float | None]
    labels: dict[str, float | None]
    included: bool
    exclusion_reason: str | None
    is_synthetic: bool
    missing_feature_count: int
    # per-label provenance (e.g. "forward_return_h20" -> "delisting_adjusted:...")
    label_flags: dict[str, str] = field(default_factory=dict)

    def label_window_end(self, bars: Sequence[PriceBar], horizon: int) -> date | None:
        """Date the ``horizon`` forward return realises (for purge/embargo)."""
        ordered = sorted(bars, key=lambda b: b.date)
        after = [b for b in ordered if b.date > self.decision_date]
        if len(after) < horizon + 1:
            return None
        return after[horizon].date

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "ticker": self.ticker,
            "decision_date": self.decision_date.isoformat(),
            "disclosure_date": self.disclosure_date.isoformat()
            if self.disclosure_date
            else None,
            "accounting_basis": self.accounting_basis,
            "sector": self.sector,
            "market": self.market,
            "included": self.included,
            "exclusion_reason": self.exclusion_reason,
            "is_synthetic": self.is_synthetic,
            "missing_feature_count": self.missing_feature_count,
        }
        for name in ALL_FACTORS:
            row[name] = self.features.get(name)
        for name in TRADING_FEATURES:
            if name in self.features:
                row[name] = self.features.get(name)
        for key, value in sorted(self.labels.items()):
            row[key] = value
        if self.label_flags:
            row["label_flags"] = ";".join(f"{k}={v}" for k, v in sorted(self.label_flags.items()))
        return row


@dataclass(frozen=True)
class ModelingDataset:
    """A built, deterministic modeling dataset."""

    observations: list[ModelingObservation]
    horizons: tuple[int, ...]
    decision_dates: tuple[date, ...]
    is_synthetic: bool
    feature_names: tuple[str, ...] = ALL_FACTORS
    entry: str = ENTRY_NEXT_CLOSE
    survivorship: dict[str, object] = field(default_factory=dict)
    disclaimer: str = (
        "This output is for analytical and self-directed research purposes. "
        "It is not personalized financial advice."
    )

    def included(self) -> list[ModelingObservation]:
        return [o for o in self.observations if o.included]

    def exclusion_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for obs in self.observations:
            if obs.exclusion_reason:
                counts[obs.exclusion_reason] = counts.get(obs.exclusion_reason, 0) + 1
        return dict(sorted(counts.items()))

    def basis_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for obs in self.observations:
            counts[obs.accounting_basis] = counts.get(obs.accounting_basis, 0) + 1
        return dict(sorted(counts.items()))

    def feature_coverage(self) -> dict[str, int]:
        """Count of included observations with a non-missing value per factor."""
        coverage = dict.fromkeys(self.feature_names, 0)
        for obs in self.included():
            for name in self.feature_names:
                if obs.features.get(name) is not None:
                    coverage[name] += 1
        return coverage

    def label_coverage(self) -> dict[str, int]:
        coverage: dict[str, int] = {}
        for obs in self.included():
            for key, value in obs.labels.items():
                coverage.setdefault(key, 0)
                if value is not None:
                    coverage[key] += 1
        return dict(sorted(coverage.items()))

    def to_rows(self) -> list[dict[str, object]]:
        return [obs.to_row() for obs in self.observations]


def _resolve_basis(statements: Sequence[FinancialStatement]) -> str:
    bases = {
        (s.accounting_basis or BASIS_UNKNOWN).strip().lower()
        for s in statements
        if s is not None
    }
    bases.discard("")
    known = bases - {BASIS_UNKNOWN}
    if not known:
        return BASIS_UNKNOWN
    if len(known) > 1:
        return BASIS_MIXED
    return known.pop()


def _disclosure_date_for(
    ticker: str,
    disclosure_dates: Mapping[str, date] | None,
    bundle_disclosure_date: date | None,
) -> date | None:
    if disclosure_dates and ticker in disclosure_dates:
        return disclosure_dates[ticker]
    return bundle_disclosure_date


def _available_date(
    statement: FinancialStatement, fallback: date | None
) -> date | None:
    return statement.disclosure_date or fallback


def _point_in_time_statements(
    statements: Sequence[FinancialStatement],
    fallback_disclosure: date | None,
    decision_date: date,
) -> list[FinancialStatement]:
    """Statements public on or before ``decision_date``, oldest -> newest.

    Ordered by (fiscal_year, disclosure date) so a late-filed older period never
    overrides a newer one. A statement with no known disclosure date (neither
    its own nor a fallback) is treated as NOT available — never assumed public.
    """
    visible: list[tuple[int, date, FinancialStatement]] = []
    for statement in statements:
        public_on = _available_date(statement, fallback_disclosure)
        if public_on is None or public_on > decision_date:
            continue
        visible.append((statement.fiscal_year or 0, public_on, statement))
    visible.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in visible]


def _prior_year_statement(
    available: Sequence[FinancialStatement], latest: FinancialStatement | None
) -> FinancialStatement | None:
    """Same-period statement one fiscal year earlier (for YoY growth).

    Comparing a quarter to a full year (or two arbitrary rows) is not a YoY
    growth rate, so the prior row must match ``fiscal_period`` and be exactly
    ``fiscal_year - 1``. Without fiscal years the previous visible row is used
    only when periods match.
    """
    if latest is None:
        return None
    candidates = [s for s in available if s is not latest]
    if latest.fiscal_year is not None:
        matched = [
            s
            for s in candidates
            if s.fiscal_year == latest.fiscal_year - 1
            and (s.fiscal_period or None) == (latest.fiscal_period or None)
        ]
        return matched[-1] if matched else None
    same_period = [
        s for s in candidates if (s.fiscal_period or None) == (latest.fiscal_period or None)
    ]
    return same_period[-1] if same_period else None


def build_modeling_dataset(
    fundamentals: Mapping[str, Sequence[FinancialStatement]],
    prices: Mapping[str, Sequence[PriceBar]],
    metadata: Mapping[str, CompanyMetadata] | None = None,
    narratives: Mapping[str, DisclosureDocument] | None = None,
    *,
    decision_dates: Sequence[date],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    bundle_disclosure_date: date | None = None,
    disclosure_dates: Mapping[str, date] | None = None,
    include_non_consolidated: bool = False,
    is_synthetic: bool = False,
    feature_set: str = FEATURE_SET_CORE,
    entry: str = ENTRY_NEXT_CLOSE,
    earnings_dates: Mapping[str, Sequence[date]] | None = None,
    margin: Mapping[str, Sequence[MarginBalance]] | None = None,
    min_traded_value: float | None = None,
    listing_status: Mapping[str, ListingStatus] | None = None,
    delisting_penalty_pct: float | None = None,
) -> ModelingDataset:
    """Build a modeling dataset from offline inputs. No network, deterministic.

    ``feature_set="extended"`` adds the trading feature families
    (``trading_features.TRADING_FEATURES``) to the 18 core factors.
    ``entry="next_open"`` labels returns as next-open -> open ``h`` sessions
    later (a fill a post-close decision can actually get).
    ``min_traded_value`` excludes names whose 20-day mean traded value (yen) at
    the decision date is below the threshold (``illiquid``) — untradable names
    never enter training or evaluation.
    ``listing_status`` (survivorship control, see ``providers.listing_status``)
    excludes decisions before listing / on-or-after delisting and replaces
    labels whose holding window crosses a delisting with a delisting-adjusted
    return (cash-out price > source delisting return > explicitly configured
    ``delisting_penalty_pct`` > last traded price).
    """
    if not decision_dates:
        raise ValueError("at least one decision date is required")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set {feature_set!r}; expected {FEATURE_SETS}")
    if entry not in ENTRY_MODES:
        raise ValueError(f"unknown entry {entry!r}; expected {ENTRY_MODES}")
    horizons_tuple = tuple(sorted({int(h) for h in horizons}))
    if any(h < 1 for h in horizons_tuple):
        raise ValueError("horizons must be positive trading-row offsets")
    metadata = metadata or {}
    narratives = narratives or {}
    universe = sorted(set(fundamentals) | set(prices) | set(metadata))
    ordered_decisions = tuple(sorted(set(decision_dates)))
    market_returns = (
        equal_weight_market_returns(prices) if feature_set == FEATURE_SET_EXTENDED else None
    )

    # per-ticker preprocessing runs ONCE (sorted bars, date index, split-
    # adjusted OHLC); each observation then slices a trailing window by bisect.
    sorted_bars = {t: sorted(prices.get(t, []), key=lambda b: b.date) for t in universe}
    bar_dates = {t: [b.date for b in rows] for t, rows in sorted_bars.items()}
    adjusted_rows = (
        {t: adjusted_ohlc(rows) for t, rows in sorted_bars.items()}
        if feature_set == FEATURE_SET_EXTENDED
        else {}
    )

    observations: list[ModelingObservation] = []
    for decision_date in ordered_decisions:
        for ticker in universe:
            end = bisect.bisect_right(bar_dates[ticker], decision_date)
            start = max(0, end - FEATURE_WINDOW_ROWS)
            observations.append(
                _build_observation(
                    ticker=ticker,
                    decision_date=decision_date,
                    statements=list(fundamentals.get(ticker, [])),
                    bars_as_of=sorted_bars[ticker][start:end],
                    bars_after=sorted_bars[ticker][end:],
                    adjusted_window=(
                        adjusted_rows[ticker][start:end] if adjusted_rows else None
                    ),
                    company=metadata.get(ticker),
                    narrative=narratives.get(ticker),
                    horizons=horizons_tuple,
                    bundle_disclosure_date=bundle_disclosure_date,
                    disclosure_dates=disclosure_dates,
                    include_non_consolidated=include_non_consolidated,
                    is_synthetic=is_synthetic,
                    feature_set=feature_set,
                    entry=entry,
                    market_returns=market_returns,
                    earnings_dates=(earnings_dates or {}).get(ticker),
                    margin=(margin or {}).get(ticker),
                    min_traded_value=min_traded_value,
                    listing=(listing_status or {}).get(ticker),
                    delisting_penalty_pct=delisting_penalty_pct,
                )
            )

    _attach_excess_returns(observations, horizons_tuple)
    feature_names = (
        ALL_FACTORS + TRADING_FEATURES if feature_set == FEATURE_SET_EXTENDED else ALL_FACTORS
    )
    return ModelingDataset(
        observations=observations,
        horizons=horizons_tuple,
        decision_dates=ordered_decisions,
        is_synthetic=is_synthetic,
        feature_names=feature_names,
        entry=entry,
        survivorship=_survivorship_summary(
            observations, listing_status, delisting_penalty_pct
        ),
    )


def _build_observation(
    *,
    ticker: str,
    decision_date: date,
    statements: list[FinancialStatement],
    bars_as_of: list[PriceBar],
    bars_after: list[PriceBar],
    adjusted_window: list | None,
    company: CompanyMetadata | None,
    narrative: DisclosureDocument | None,
    horizons: tuple[int, ...],
    bundle_disclosure_date: date | None,
    disclosure_dates: Mapping[str, date] | None,
    include_non_consolidated: bool,
    is_synthetic: bool,
    feature_set: str = FEATURE_SET_CORE,
    entry: str = ENTRY_NEXT_CLOSE,
    market_returns: Mapping[date, float] | None = None,
    earnings_dates: Sequence[date] | None = None,
    margin: Sequence[MarginBalance] | None = None,
    min_traded_value: float | None = None,
    listing: ListingStatus | None = None,
    delisting_penalty_pct: float | None = None,
) -> ModelingObservation:
    fallback_disclosure = _disclosure_date_for(ticker, disclosure_dates, bundle_disclosure_date)
    sector = company.sector if company else None
    market = company.market if company else None

    # Point-in-time selection: only statements already public at the decision
    # date are visible. Each statement uses its own ``disclosure_date``; when the
    # source did not provide one, the ticker/bundle-level date is the fallback.
    available = _point_in_time_statements(statements, fallback_disclosure, decision_date)
    latest = available[-1] if available else None
    prior = _prior_year_statement(available, latest)
    if latest is not None:
        disclosure_date: date | None = _available_date(latest, fallback_disclosure)
    else:
        disclosure_date = fallback_disclosure
    basis_source = available or statements
    basis = _resolve_basis(basis_source) if basis_source else BASIS_UNKNOWN

    # A date alone cannot establish availability before a same-day decision.
    # Do not infer availability from document metadata or statement dates.
    if (
        narrative is None
        or not narrative.text.strip()
        or narrative.available_date is None
        or narrative.available_date >= decision_date
    ):
        narrative = None
    factor_result = compute_factors(latest, prior, bars_as_of, company, narrative)
    features = dict(factor_result.features)
    if feature_set == FEATURE_SET_EXTENDED:
        latest_public = _available_date(latest, fallback_disclosure) if latest else None
        earlier = [
            s
            for s in sorted(
                available,
                key=lambda st: (_available_date(st, fallback_disclosure), st.fiscal_year or 0),
            )
            if s is not latest
            and latest_public is not None
            and (_available_date(s, fallback_disclosure) or latest_public) <= latest_public
        ]
        features.update(
            compute_trading_features(
                statement=latest,
                statement_public_date=latest_public,
                earlier_disclosures=earlier,
                bars_as_of=bars_as_of,
                decision_date=decision_date,
                market_returns=market_returns,
                earnings_dates=earnings_dates,
                margin=margin,
                adjusted_rows=adjusted_window,
            )
        )

    labels: dict[str, float | None] = {}
    label_flags: dict[str, str] = {}
    for horizon in horizons:
        key = f"forward_return_h{horizon}"
        labels[key] = _forward_return_after(bars_after, horizon, entry)
        if listing is not None and listing.delisting_date is not None:
            value, flag = _delisting_adjusted_return(
                bars_after, horizon, entry, listing, delisting_penalty_pct
            )
            if flag is not None:
                labels[key] = value
                label_flags[key] = flag
        labels[f"excess_return_h{horizon}"] = None  # filled cross-sectionally later

    exclusion_reason: str | None = None
    if not statements:
        exclusion_reason = EXCLUDE_NO_FUNDAMENTALS
    elif latest is None:
        # statements exist, but none was public yet at this decision date
        exclusion_reason = EXCLUDE_DISCLOSURE_AFTER_DECISION
    elif basis == BASIS_NON_CONSOLIDATED and not include_non_consolidated:
        exclusion_reason = EXCLUDE_NON_CONSOLIDATED
    elif listing is not None and listing.listing_date and decision_date < listing.listing_date:
        exclusion_reason = EXCLUDE_NOT_YET_LISTED
    elif (
        listing is not None and listing.delisting_date
        and decision_date >= listing.delisting_date
    ):
        exclusion_reason = EXCLUDE_DELISTED
    elif min_traded_value is not None:
        liquidity = traded_value_20d(bars_as_of)
        if liquidity is None or liquidity < min_traded_value:
            exclusion_reason = EXCLUDE_ILLIQUID

    return ModelingObservation(
        ticker=ticker,
        decision_date=decision_date,
        disclosure_date=disclosure_date,
        horizons=horizons,
        accounting_basis=basis,
        sector=sector,
        market=market,
        features=features,
        labels=labels,
        included=exclusion_reason is None,
        exclusion_reason=exclusion_reason,
        is_synthetic=is_synthetic,
        missing_feature_count=sum(1 for v in features.values() if v is None),
        label_flags=label_flags,
    )


def _survivorship_summary(
    observations: Sequence[ModelingObservation],
    listing_status: Mapping[str, ListingStatus] | None,
    penalty_pct: float | None,
) -> dict[str, object]:
    if not listing_status:
        return {}
    flags: dict[str, int] = {}
    for obs in observations:
        for flag in obs.label_flags.values():
            flags[flag] = flags.get(flag, 0) + 1
    delisted = [s for s in listing_status.values() if s.delisting_date]
    return {
        "listing_status_tickers": len(listing_status),
        "delisted_tickers": len(delisted),
        "delisting_penalty_pct": penalty_pct,
        "delisting_adjusted_labels": dict(sorted(flags.items())),
    }


def _attach_excess_returns(
    observations: list[ModelingObservation], horizons: tuple[int, ...]
) -> None:
    """Excess return = raw return minus same-sector mean (or universe mean).

    Computed cross-sectionally per decision date over *included* observations,
    so it is a deterministic sector/market-neutral framing — not a vendor
    benchmark. Sectors with too few peers fall back to the universe mean.
    """
    by_date: dict[date, list[ModelingObservation]] = {}
    for obs in observations:
        if obs.included:
            by_date.setdefault(obs.decision_date, []).append(obs)

    for group in by_date.values():
        for horizon in horizons:
            key = f"forward_return_h{horizon}"
            excess_key = f"excess_return_h{horizon}"
            present = [o for o in group if o.labels.get(key) is not None]
            if not present:
                continue
            universe_mean = sum(o.labels[key] for o in present) / len(present)  # type: ignore[misc]
            sector_means: dict[str, float] = {}
            sector_groups: dict[str, list[ModelingObservation]] = {}
            for obs in present:
                if obs.sector:
                    sector_groups.setdefault(obs.sector, []).append(obs)
            for sector, members in sector_groups.items():
                if len(members) >= MIN_SECTOR_PEERS_FOR_EXCESS:
                    sector_means[sector] = sum(
                        o.labels[key] for o in members  # type: ignore[misc]
                    ) / len(members)
            for obs in present:
                baseline = (
                    sector_means.get(obs.sector, universe_mean)
                    if obs.sector
                    else universe_mean
                )
                obs.labels[excess_key] = obs.labels[key] - baseline  # type: ignore[operator]


def write_dataset_outputs(
    dataset: ModelingDataset, output_dir: str | Path
) -> dict[str, Path]:
    """Write the dataset rows CSV and a coverage-summary JSON. Returns paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = dataset.to_rows()
    csv_path = out_dir / "modeling_dataset.csv"
    if rows:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n", restval=""
            )
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    summary = {
        "disclaimer": dataset.disclaimer,
        "is_synthetic": dataset.is_synthetic,
        "synthetic_warning": (
            "SYNTHETIC FIXTURE RESULTS — not real market evidence."
            if dataset.is_synthetic
            else None
        ),
        "decision_dates": [d.isoformat() for d in dataset.decision_dates],
        "horizons": list(dataset.horizons),
        "entry": dataset.entry,
        "feature_count": len(dataset.feature_names),
        "total_observations": len(dataset.observations),
        "eligible_observations": len(dataset.included()),
        "accounting_basis_distribution": dataset.basis_counts(),
        "exclusions": dataset.exclusion_counts(),
        "feature_coverage": dataset.feature_coverage(),
        "label_coverage": dataset.label_coverage(),
        **({"survivorship": dataset.survivorship} if dataset.survivorship else {}),
    }
    summary_path = out_dir / "modeling_dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"csv_path": csv_path, "summary_path": summary_path}


__all__ = [
    "ENTRY_MODES",
    "ENTRY_NEXT_CLOSE",
    "ENTRY_NEXT_OPEN",
    "EXCLUDE_DELISTED",
    "EXCLUDE_ILLIQUID",
    "EXCLUDE_NOT_YET_LISTED",
    "FEATURE_SETS",
    "FEATURE_SET_CORE",
    "FEATURE_SET_EXTENDED",
    "BASIS_CONSOLIDATED",
    "BASIS_MIXED",
    "BASIS_NON_CONSOLIDATED",
    "BASIS_UNKNOWN",
    "DEFAULT_HORIZONS",
    "EXCLUDE_DISCLOSURE_AFTER_DECISION",
    "EXCLUDE_NON_CONSOLIDATED",
    "EXCLUDE_NO_FUNDAMENTALS",
    "ModelingDataset",
    "ModelingObservation",
    "build_modeling_dataset",
    "forward_return",
    "write_dataset_outputs",
]

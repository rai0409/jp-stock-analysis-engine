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

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from jp_stock_analysis.modeling.factors import ALL_FACTORS, compute_factors
from jp_stock_analysis.schemas import (
    CompanyMetadata,
    DisclosureDocument,
    FinancialStatement,
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


def forward_return(bars: Sequence[PriceBar], decision_date: date, horizon: int) -> float | None:
    """Realised forward return (percent) for ``horizon`` trading rows.

    Base = first price row strictly after ``decision_date``; target = the row
    ``horizon`` positions later. ``None`` when there are too few later rows or a
    zero base. Uses adjusted close when the whole series has it, else close.
    """
    ordered = sorted(bars, key=lambda b: b.date)
    after = [b for b in ordered if b.date > decision_date]
    if len(after) < horizon + 1:
        return None
    use_adjusted = all(b.adjusted_close is not None for b in after)
    base = after[0].adjusted_close if use_adjusted else after[0].close
    target = after[horizon].adjusted_close if use_adjusted else after[horizon].close
    if base in (None, 0):
        return None
    return (float(target) / float(base) - 1.0) * 100.0  # type: ignore[arg-type]


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
        for key, value in sorted(self.labels.items()):
            row[key] = value
        return row


@dataclass(frozen=True)
class ModelingDataset:
    """A built, deterministic modeling dataset."""

    observations: list[ModelingObservation]
    horizons: tuple[int, ...]
    decision_dates: tuple[date, ...]
    is_synthetic: bool
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
        coverage = dict.fromkeys(ALL_FACTORS, 0)
        for obs in self.included():
            for name in ALL_FACTORS:
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
) -> ModelingDataset:
    """Build a modeling dataset from offline inputs. No network, deterministic."""
    if not decision_dates:
        raise ValueError("at least one decision date is required")
    horizons_tuple = tuple(sorted({int(h) for h in horizons}))
    if any(h < 1 for h in horizons_tuple):
        raise ValueError("horizons must be positive trading-row offsets")
    metadata = metadata or {}
    narratives = narratives or {}
    universe = sorted(set(fundamentals) | set(prices) | set(metadata))
    ordered_decisions = tuple(sorted(set(decision_dates)))

    observations: list[ModelingObservation] = []
    for decision_date in ordered_decisions:
        for ticker in universe:
            observations.append(
                _build_observation(
                    ticker=ticker,
                    decision_date=decision_date,
                    statements=list(fundamentals.get(ticker, [])),
                    bars=list(prices.get(ticker, [])),
                    company=metadata.get(ticker),
                    narrative=narratives.get(ticker),
                    horizons=horizons_tuple,
                    bundle_disclosure_date=bundle_disclosure_date,
                    disclosure_dates=disclosure_dates,
                    include_non_consolidated=include_non_consolidated,
                    is_synthetic=is_synthetic,
                )
            )

    _attach_excess_returns(observations, horizons_tuple)
    return ModelingDataset(
        observations=observations,
        horizons=horizons_tuple,
        decision_dates=ordered_decisions,
        is_synthetic=is_synthetic,
    )


def _build_observation(
    *,
    ticker: str,
    decision_date: date,
    statements: list[FinancialStatement],
    bars: list[PriceBar],
    company: CompanyMetadata | None,
    narrative: DisclosureDocument | None,
    horizons: tuple[int, ...],
    bundle_disclosure_date: date | None,
    disclosure_dates: Mapping[str, date] | None,
    include_non_consolidated: bool,
    is_synthetic: bool,
) -> ModelingObservation:
    disclosure_date = _disclosure_date_for(ticker, disclosure_dates, bundle_disclosure_date)
    basis = _resolve_basis(statements) if statements else BASIS_UNKNOWN
    sector = company.sector if company else None
    market = company.market if company else None

    ordered_statements = sorted(
        statements, key=lambda s: (s.fiscal_year is None, s.fiscal_year or 0)
    )
    latest = ordered_statements[-1] if ordered_statements else None
    prior = ordered_statements[-2] if len(ordered_statements) > 1 else None

    bars_as_of = [b for b in bars if b.date <= decision_date]
    factor_result = compute_factors(latest, prior, bars_as_of, company, narrative)
    features = factor_result.features

    labels: dict[str, float | None] = {}
    for horizon in horizons:
        labels[f"forward_return_h{horizon}"] = forward_return(bars, decision_date, horizon)
        labels[f"excess_return_h{horizon}"] = None  # filled cross-sectionally later

    exclusion_reason: str | None = None
    if latest is None:
        exclusion_reason = EXCLUDE_NO_FUNDAMENTALS
    elif disclosure_date is None or disclosure_date > decision_date:
        exclusion_reason = EXCLUDE_DISCLOSURE_AFTER_DECISION
    elif basis == BASIS_NON_CONSOLIDATED and not include_non_consolidated:
        exclusion_reason = EXCLUDE_NON_CONSOLIDATED

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
        missing_feature_count=len(factor_result.missing_factors),
    )


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
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
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
        "total_observations": len(dataset.observations),
        "eligible_observations": len(dataset.included()),
        "accounting_basis_distribution": dataset.basis_counts(),
        "exclusions": dataset.exclusion_counts(),
        "feature_coverage": dataset.feature_coverage(),
        "label_coverage": dataset.label_coverage(),
    }
    summary_path = out_dir / "modeling_dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"csv_path": csv_path, "summary_path": summary_path}


__all__ = [
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

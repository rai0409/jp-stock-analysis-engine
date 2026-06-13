"""Deterministic synthetic fixtures for the modeling layer.

SYNTHETIC ONLY. Everything here is generated from fixed formulae (a small
integer LCG — no RNG seeding surprises, identical on every platform). It exists
so the whole modeling pipeline is testable before any real fundamentals/prices
arrive. It is **not** market data and its results are **not** market evidence.

The bundle deliberately contains:

- 12 tickers across 3 sectors,
- 3 decision dates with enough later price rows for the 5/20/60 horizons,
- a mix of available/missing fundamentals (one single-statement ticker -> no
  growth factors; one ticker missing several figures),
- one ``non_consolidated`` ticker (excluded by default),
- realised forward-return labels for every horizon.

A faint, intentional link between quality and forward drift keeps the ranking
metrics non-degenerate; it is a fixture artefact, never a finding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from jp_stock_analysis.schemas import (
    CompanyMetadata,
    DisclosureDocument,
    FinancialStatement,
    PriceBar,
)

SYNTHETIC_SOURCE = "synthetic_fixture"
N_TICKERS = 12
SECTORS = ("technology", "industrials", "consumer")
N_PRICE_ROWS = 160
DECISION_INDICES = (30, 60, 90)
HORIZONS = (5, 20, 60)
NON_CONSOLIDATED_INDEX = 11
SINGLE_STATEMENT_INDEX = 5
MISSING_FIELDS_INDEX = 7


def _tickers() -> list[str]:
    return [f"{1301 + i:04d}" for i in range(N_TICKERS)]


def _business_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:  # Mon-Fri
            days.append(current)
        current += timedelta(days=1)
    return days


class _Lcg:
    """Tiny deterministic LCG -> uniform floats in [0, 1)."""

    def __init__(self, seed: int) -> None:
        self._state = (seed * 2654435761 + 1013904223) & 0x7FFFFFFF

    def uniform(self) -> float:
        self._state = (1103515245 * self._state + 12345) & 0x7FFFFFFF
        return self._state / 0x7FFFFFFF


def _quality_score(index: int) -> float:
    """A per-ticker latent quality in [0, 1] driving both ROE and price drift."""
    return ((index * 7 + 3) % N_TICKERS) / (N_TICKERS - 1)


def _price_series(index: int, dates: Sequence[date]) -> list[PriceBar]:
    ticker = _tickers()[index]
    rng = _Lcg(index + 1)
    quality = _quality_score(index)
    drift = 0.0003 + quality * 0.0008  # higher quality -> faint positive drift
    vol = 0.010 + (index % 4) * 0.004
    price = 800.0 + index * 120.0
    bars: list[PriceBar] = []
    for d in dates:
        shock = (rng.uniform() - 0.5) * 2.0 * vol
        price = max(1.0, price * (1.0 + drift + shock))
        rounded = round(price, 2)
        bars.append(
            PriceBar(
                ticker=ticker,
                date=d,
                close=rounded,
                adjusted_close=rounded,
                volume=float(1000 + index * 100),
            )
        )
    return bars


def _statements(index: int) -> list[FinancialStatement]:
    ticker = _tickers()[index]
    quality = _quality_score(index)
    revenue = 50_000.0 + index * 9_000.0
    net_income = revenue * (0.02 + quality * 0.08)
    equity = revenue * (0.4 + (index % 3) * 0.1)
    total_assets = equity * (1.8 + (index % 4) * 0.3)
    operating_income = net_income * 1.4
    shares = 1_000.0 + index * 50.0
    basis = (
        "non_consolidated" if index == NON_CONSOLIDATED_INDEX else "consolidated"
    )

    current = FinancialStatement(
        ticker=ticker,
        fiscal_year=2025,
        fiscal_period="FY",
        accounting_basis=basis,
        revenue=revenue,
        operating_income=operating_income,
        net_income=net_income,
        equity=equity,
        total_assets=total_assets,
        shares_outstanding=shares,
    )
    if index == MISSING_FIELDS_INDEX:
        # exercise missing-value handling: drop several figures
        current = current.model_copy(
            update={"operating_income": None, "equity": None, "total_assets": None}
        )
    if index == SINGLE_STATEMENT_INDEX:
        return [current]  # no prior year -> growth factors unavailable

    prior = FinancialStatement(
        ticker=ticker,
        fiscal_year=2024,
        fiscal_period="FY",
        accounting_basis=basis,
        revenue=revenue * 0.92,
        operating_income=operating_income * 0.9,
        net_income=net_income * 0.88,
        equity=equity * 0.95,
        total_assets=total_assets * 0.97,
        shares_outstanding=shares,
    )
    return [prior, current]


@dataclass(frozen=True)
class SyntheticBundle:
    fundamentals: dict[str, list[FinancialStatement]]
    prices: dict[str, list[PriceBar]]
    metadata: dict[str, CompanyMetadata]
    narratives: dict[str, DisclosureDocument]
    decision_dates: list[date]
    bundle_disclosure_date: date
    horizons: tuple[int, ...]


def build_synthetic_bundle() -> SyntheticBundle:
    """Build the in-memory synthetic input bundle. Deterministic, offline."""
    tickers = _tickers()
    dates = _business_days(date(2025, 1, 1), N_PRICE_ROWS)
    decision_dates = [dates[i] for i in DECISION_INDICES]
    # all fundamentals public the day before the first decision date
    bundle_disclosure_date = decision_dates[0] - timedelta(days=1)

    fundamentals: dict[str, list[FinancialStatement]] = {}
    prices: dict[str, list[PriceBar]] = {}
    metadata: dict[str, CompanyMetadata] = {}
    narratives: dict[str, DisclosureDocument] = {}
    for index, ticker in enumerate(tickers):
        fundamentals[ticker] = _statements(index)
        prices[ticker] = _price_series(index, dates)
        metadata[ticker] = CompanyMetadata(
            ticker=ticker,
            company_name=f"Synthetic Co {ticker}",
            sector=SECTORS[index % len(SECTORS)],
            market="synthetic_market",
        )
        # narrative not attempted (matches the export contract default)
        narratives[ticker] = DisclosureDocument(
            ticker=ticker,
            text="",
            document_type="synthetic",
            source=SYNTHETIC_SOURCE,
            warnings=["synthetic narrative: extraction not attempted"],
            source_metadata={"extraction_status": "not_attempted", "synthetic": "true"},
        )
    return SyntheticBundle(
        fundamentals=fundamentals,
        prices=prices,
        metadata=metadata,
        narratives=narratives,
        decision_dates=decision_dates,
        bundle_disclosure_date=bundle_disclosure_date,
        horizons=HORIZONS,
    )


def write_synthetic_csv_fixtures(output_dir) -> dict[str, object]:
    """Write deterministic CSV fixtures (prices/fundamentals/metadata) + meta.

    Returns a dict of written paths plus the decision dates / disclosure date so
    a CLI run can reproduce the in-memory bundle from files. SYNTHETIC ONLY.
    """
    import csv
    import json
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_synthetic_bundle()

    prices_path = out_dir / "prices.csv"
    with prices_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["ticker", "date", "close", "adjusted_close", "volume"])
        for ticker in sorted(bundle.prices):
            for bar in bundle.prices[ticker]:
                writer.writerow(
                    [bar.ticker, bar.date.isoformat(), bar.close, bar.adjusted_close, bar.volume]
                )

    fundamentals_path = out_dir / "fundamentals.csv"
    with fundamentals_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "ticker",
                "fiscal_year",
                "accounting_basis",
                "revenue",
                "operating_income",
                "net_income",
                "equity",
                "total_assets",
                "shares_outstanding",
            ]
        )
        for ticker in sorted(bundle.fundamentals):
            for s in bundle.fundamentals[ticker]:
                writer.writerow(
                    [
                        s.ticker,
                        s.fiscal_year,
                        s.accounting_basis or "",
                        _blank(s.revenue),
                        _blank(s.operating_income),
                        _blank(s.net_income),
                        _blank(s.equity),
                        _blank(s.total_assets),
                        _blank(s.shares_outstanding),
                    ]
                )

    metadata_path = out_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["ticker", "company_name", "sector", "market"])
        for ticker in sorted(bundle.metadata):
            company = bundle.metadata[ticker]
            writer.writerow(
                [company.ticker, company.company_name, company.sector, company.market]
            )

    meta_path = out_dir / "synthetic_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "synthetic": True,
                "warning": "SYNTHETIC FIXTURE — not real market data.",
                "decision_dates": [d.isoformat() for d in bundle.decision_dates],
                "bundle_disclosure_date": bundle.bundle_disclosure_date.isoformat(),
                "horizons": list(bundle.horizons),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "prices_path": prices_path,
        "fundamentals_path": fundamentals_path,
        "metadata_path": metadata_path,
        "meta_path": meta_path,
        "decision_dates": bundle.decision_dates,
        "bundle_disclosure_date": bundle.bundle_disclosure_date,
        "horizons": bundle.horizons,
    }


def _blank(value: float | None) -> str:
    return "" if value is None else repr(value)


__all__ = [
    "HORIZONS",
    "SYNTHETIC_SOURCE",
    "SyntheticBundle",
    "build_synthetic_bundle",
    "write_synthetic_csv_fixtures",
]

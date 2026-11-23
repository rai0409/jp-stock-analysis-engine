"""Typed data models for the Japanese stock analysis engine.

Conventions:

- Percentages are expressed as percent values (``10.0`` means 10%).
- Confidence scores and analysis scores use a 0-100 scale.
- ``max_drawdown`` is a negative percent (``-25.0`` means a 25% peak-to-trough
  decline).
- Missing data is represented as ``None`` plus an entry in ``warnings``.
  Values are never fabricated.
"""

from __future__ import annotations

from datetime import date as date_type
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SignalMode = Literal["analysis_only", "screening", "trade_signal"]

ScreeningLabel = Literal[
    "strong_candidate",
    "candidate",
    "watchlist",
    "avoid_candidate",
    "insufficient_data",
]

TradeSignalLabel = Literal[
    "buy_signal",
    "hold_signal",
    "sell_signal",
    "watch_signal",
    "avoid_signal",
    "insufficient_data",
]

RiskSeverity = Literal["low", "medium", "high", "critical"]

ReliabilityGrade = Literal["high", "medium", "low"]

FindingCategory = Literal[
    "positive_factor",
    "negative_factor",
    "risk_factor",
    "growth_driver",
    "management_outlook",
    "business_environment",
    "guidance_revision",
    "one_time_factor",
    "uncertainty",
]

ValuationClassification = Literal["cheap", "fair", "expensive", "unavailable"]

VolumeTrend = Literal["increasing", "decreasing", "flat"]


class SchemaBase(BaseModel):
    """Base model: tolerant of partial data, ignores unknown fields."""

    model_config = ConfigDict(extra="ignore")


class PriceBar(SchemaBase):
    """One daily OHLCV observation. Only ``close`` is required."""

    ticker: str
    date: date_type
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    adjusted_close: float | None = None
    volume: float | None = None


class FinancialStatement(SchemaBase):
    """One fiscal-period financial statement. All figures are optional."""

    ticker: str
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    # consolidated / non_consolidated / mixed / unknown — the basis the figures
    # below are reported on. Never mix bases silently; the modeling layer filters
    # on this (see modeling/dataset.py). None when the source did not label it.
    accounting_basis: str | None = None
    # Date this statement became public (TDnet / J-Quants ``DisclosedDate``).
    # Used for point-in-time feature construction: a statement is usable at a
    # decision date only when ``disclosure_date <= decision_date``. ``None`` means
    # unknown; the modeling layer then falls back to a ticker/bundle-level date.
    disclosure_date: date_type | None = None
    revenue: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    eps: float | None = None
    bps: float | None = None
    dividends_per_share: float | None = None
    shares_outstanding: float | None = None
    total_assets: float | None = None
    equity: float | None = None
    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None
    gross_profit: float | None = None
    # Company guidance disclosed WITH this statement (決算短信の会社予想).
    # ``forecast_fiscal_year`` is the fiscal year the guidance refers to (a FY
    # results release usually guides the NEXT year). Used for guidance-revision
    # and earnings-surprise features; never treated as realised figures.
    forecast_fiscal_year: int | None = None
    forecast_revenue: float | None = None
    forecast_operating_income: float | None = None
    forecast_net_income: float | None = None
    forecast_eps: float | None = None
    forecast_dividends_per_share: float | None = None
    source_metadata: dict[str, str] = Field(default_factory=dict)


class MarginBalance(SchemaBase):
    """Weekly margin-trading balance (信用取引残高) for one ticker.

    ``published_date`` is when the figure became public (JPX publishes the
    week's balance a few business days later). Features only use rows whose
    publication is on or before the decision date; when it is unknown a
    conservative lag is applied by the modeling layer.
    """

    ticker: str
    date: date_type
    long_balance: float | None = None
    short_balance: float | None = None
    published_date: date_type | None = None


class ListingStatus(SchemaBase):
    """Listing / delisting record for one ticker (survivorship-bias control).

    - ``listing_date``: first day the ticker is eligible (None = unknown/old).
    - ``delisting_date``: first day it no longer trades (None = still listed).
    - ``delisting_reason``: free text / code from the source (e.g. TOB, merger,
      bankruptcy, 上場廃止基準抵触) — recorded, never interpreted as a return.
    - ``final_price``: cash-out / last settlement price per share in the same
      (unadjusted) units as ``PriceBar.close`` when known (e.g. TOB price).
    - ``delisting_return_pct``: return from the last traded close to the final
      proceeds when the source provides it (e.g. CRSP-style DLRET).
    Neither value is ever fabricated: when both are missing the label exits at
    the last traded price, and any penalty must be configured explicitly.
    """

    ticker: str
    listing_date: date_type | None = None
    delisting_date: date_type | None = None
    delisting_reason: str | None = None
    final_price: float | None = None
    delisting_return_pct: float | None = None
    source: str | None = None


class CompanyMetadata(SchemaBase):
    ticker: str
    company_name: str | None = None
    sector: str | None = None
    market: str | None = None
    source_metadata: dict[str, str] = Field(default_factory=dict)


class DisclosureDocument(SchemaBase):
    """Raw disclosure text (earnings summary, securities report excerpt, etc.).

    ``text`` may be empty for metadata-only documents (e.g. a topix1000 export
    entry whose text extraction has not run yet); analyzers must then degrade
    to low confidence instead of fabricating content. ``warnings`` carries
    provider-side caveats and ``source_metadata`` provenance (doc ids, source
    archive paths) that analyzers propagate into their results.
    """

    ticker: str
    text: str = ""
    document_type: str | None = None
    fiscal_year: int | None = None
    # Only a verified public-availability date; never inferred from metadata.
    available_date: date_type | None = None
    source: str | None = None
    doc_id: str | None = None
    edinet_code: str | None = None
    company_name: str | None = None
    warnings: list[str] = Field(default_factory=list)
    source_metadata: dict[str, str] = Field(default_factory=dict)


class ResultBase(SchemaBase):
    """Shared fields for analysis result models."""

    warnings: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    source_metadata: dict[str, str] = Field(default_factory=dict)


class FundamentalMetrics(ResultBase):
    """Derived fundamental metrics for the latest statement of one ticker."""

    ticker: str
    fiscal_year: int | None = None
    latest_eps: float | None = None
    revenue_growth_yoy: float | None = None
    operating_income_growth_yoy: float | None = None
    net_income_growth_yoy: float | None = None
    eps_growth_yoy: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    roe: float | None = None
    roa: float | None = None
    equity_ratio: float | None = None
    fcf_margin: float | None = None
    dividend_payout_ratio: float | None = None


class ValuationMetrics(ResultBase):
    ticker: str
    market_price: float | None = None
    per: float | None = None
    pbr: float | None = None
    psr: float | None = None
    dividend_yield: float | None = None
    peg: float | None = None
    market_cap: float | None = None
    valuation_classification: ValuationClassification = "unavailable"


class MomentumMetrics(ResultBase):
    ticker: str
    observations: int = 0
    return_1m: float | None = None
    return_3m: float | None = None
    return_6m: float | None = None
    return_12m: float | None = None
    moving_average_20d: float | None = None
    moving_average_60d: float | None = None
    moving_average_120d: float | None = None
    moving_average_200d: float | None = None
    volatility_annualized: float | None = None
    max_drawdown: float | None = None
    volume_trend: VolumeTrend | None = None


class DisclosureFinding(SchemaBase):
    """One evidence-backed finding extracted from disclosure text."""

    category: FindingCategory
    summary: str
    evidence_text: str
    severity: RiskSeverity = "low"
    confidence: float = 0.0
    rule_id: str


class DisclosureAnalysisResult(ResultBase):
    ticker: str
    # provenance propagated from the source DisclosureDocument; stable export
    # paths for future RAG ingestion (see docs/future_rag_integration_separate_project.md)
    document_type: str | None = None
    fiscal_year: int | None = None
    analyzer: str = "rule_based"
    findings: list[DisclosureFinding] = Field(default_factory=list)
    positive_count: int = 0
    negative_count: int = 0
    risk_count: int = 0
    uncertainty_count: int = 0
    tone_score: float = 0.0


class RiskFlag(SchemaBase):
    risk_id: str
    severity: RiskSeverity
    explanation: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class RiskMetrics(ResultBase):
    ticker: str
    flags: list[RiskFlag] = Field(default_factory=list)
    risk_score: float = 0.0


class ScoreBreakdown(ResultBase):
    """Integrated 0-100 sub-scores and final score. ``None`` = not computable."""

    ticker: str
    quality_score: float | None = None
    growth_score: float | None = None
    valuation_score: float | None = None
    momentum_score: float | None = None
    disclosure_score: float | None = None
    risk_score: float | None = None
    final_score: float | None = None
    reasons: dict[str, str] = Field(default_factory=dict)


class SectorRelativeMetrics(ResultBase):
    """Percentile ranks versus same-sector peers in the analyzed universe.

    100 always means most favorable within the sector: lower-is-better
    metrics (PER, PBR, risk score) are inverted before ranking.
    ``sector_relative_score`` is the mean of available percentiles and is
    kept separate from ``final_score``. ``peer_count`` is the number of
    same-sector companies in the universe, including this one.
    """

    ticker: str
    sector: str
    peer_count: int = 0
    per_percentile: float | None = None
    pbr_percentile: float | None = None
    revenue_growth_percentile: float | None = None
    operating_margin_percentile: float | None = None
    roe_percentile: float | None = None
    momentum_percentile: float | None = None
    risk_percentile: float | None = None
    sector_relative_score: float | None = None


class ScreeningResult(SchemaBase):
    """One ranked screening row.

    ``final_score`` stays the raw weighted score for transparency. The
    reliability fields exist so a high ``final_score`` computed from very
    little data is never presented as a strong screening winner:

    - ``data_coverage_score``: 20 points per covered analysis component
      (fundamentals, valuation, momentum, disclosure, risk).
    - ``screening_score``: reliability-adjusted ranking score
      (``final_score`` scaled by confidence and coverage).
    - ``screening_eligible`` / ``reliability_grade``: deterministic guard
      computed in ``analysis/reliability.py``; ineligible results are always
      graded ``low`` and ranked below all eligible results.
    """

    ticker: str
    company_name: str | None = None
    rank: int | None = None
    final_score: float | None = None
    confidence_score: float = 0.0
    data_coverage_score: float = 0.0
    screening_score: float | None = None
    screening_eligible: bool = False
    reliability_grade: ReliabilityGrade = "low"
    screening_label: ScreeningLabel | None = None
    warnings: list[str] = Field(default_factory=list)


class SignalResult(SchemaBase):
    """Self-directed research trade signal. Only produced in trade_signal mode."""

    ticker: str
    label: TradeSignalLabel
    confidence: float = 0.0
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    blocking_risks: list[str] = Field(default_factory=list)
    supporting_factors: list[str] = Field(default_factory=list)
    thresholds_used: dict[str, float] = Field(default_factory=dict)
    disclaimer: str = ""


class StockAnalysisResult(ResultBase):
    """Full per-ticker analysis bundle."""

    ticker: str
    company_name: str | None = None
    analysis_date: date_type
    signal_mode: SignalMode = "analysis_only"
    fundamentals: FundamentalMetrics | None = None
    valuation: ValuationMetrics | None = None
    momentum: MomentumMetrics | None = None
    disclosure: DisclosureAnalysisResult | None = None
    risks: RiskMetrics | None = None
    score: ScoreBreakdown | None = None
    sector_relative: SectorRelativeMetrics | None = None
    screening_label: ScreeningLabel | None = None
    signal: SignalResult | None = None

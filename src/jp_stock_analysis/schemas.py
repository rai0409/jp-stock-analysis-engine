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
    source_metadata: dict[str, str] = Field(default_factory=dict)


class CompanyMetadata(SchemaBase):
    ticker: str
    company_name: str | None = None
    sector: str | None = None
    market: str | None = None
    source_metadata: dict[str, str] = Field(default_factory=dict)


class DisclosureDocument(SchemaBase):
    """Raw disclosure text (earnings summary, securities report excerpt, etc.)."""

    ticker: str
    text: str
    document_type: str | None = None
    fiscal_year: int | None = None
    source: str | None = None


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
    ticker: str
    company_name: str | None = None
    rank: int | None = None
    final_score: float | None = None
    confidence_score: float = 0.0
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

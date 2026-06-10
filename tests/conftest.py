"""Shared deterministic test helpers. No network, no randomness."""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import pytest

from jp_stock_analysis.schemas import (
    FinancialStatement,
    PriceBar,
    RiskFlag,
    RiskMetrics,
    ScoreBreakdown,
    StockAnalysisResult,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


def make_price_bars(
    ticker: str = "7203",
    days: int = 300,
    start_price: float = 1000.0,
    daily_drift: float = 0.001,
    amplitude: float = 5.0,
    volume_base: float = 1_000_000.0,
    volume_slope: float = 0.0,
) -> list[PriceBar]:
    """Generate a deterministic weekday price series from a closed-form formula."""
    bars: list[PriceBar] = []
    day = date(2024, 1, 1)
    count = 0
    while count < days:
        if day.weekday() < 5:
            price = start_price * (1 + daily_drift) ** count + amplitude * math.sin(count / 5.0)
            bars.append(
                PriceBar(
                    ticker=ticker,
                    date=day,
                    close=round(price, 4),
                    volume=volume_base + volume_slope * count,
                )
            )
            count += 1
        day += timedelta(days=1)
    return bars


def make_statement(
    ticker: str = "7203", fiscal_year: int = 2024, **overrides
) -> FinancialStatement:
    values = {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "fiscal_period": "FY",
        "revenue": 1000.0,
        "operating_income": 100.0,
        "net_income": 70.0,
        "eps": 70.0,
        "bps": 700.0,
        "dividends_per_share": 20.0,
        "shares_outstanding": 1.0,
        "total_assets": 2000.0,
        "equity": 1000.0,
        "operating_cash_flow": 120.0,
        "capital_expenditure": 40.0,
    }
    values.update(overrides)
    return FinancialStatement(**values)


def make_score(
    ticker: str = "7203",
    final_score: float | None = 70.0,
    confidence_score: float = 80.0,
    quality_score: float | None = 70.0,
    growth_score: float | None = 70.0,
    valuation_score: float | None = 70.0,
    momentum_score: float | None = 70.0,
    disclosure_score: float | None = 70.0,
    risk_score: float | None = 10.0,
) -> ScoreBreakdown:
    return ScoreBreakdown(
        ticker=ticker,
        final_score=final_score,
        confidence_score=confidence_score,
        quality_score=quality_score,
        growth_score=growth_score,
        valuation_score=valuation_score,
        momentum_score=momentum_score,
        disclosure_score=disclosure_score,
        risk_score=risk_score,
    )


def make_result(
    ticker: str = "7203",
    signal_mode: str = "analysis_only",
    score: ScoreBreakdown | None = None,
    risk_score: float = 10.0,
    risk_flags: list[RiskFlag] | None = None,
    company_name: str | None = None,
) -> StockAnalysisResult:
    breakdown = score if score is not None else make_score(ticker=ticker, risk_score=risk_score)
    return StockAnalysisResult(
        ticker=ticker,
        company_name=company_name,
        analysis_date=date(2025, 6, 30),
        signal_mode=signal_mode,
        score=breakdown,
        risks=RiskMetrics(
            ticker=ticker,
            flags=risk_flags or [],
            risk_score=breakdown.risk_score if breakdown.risk_score is not None else risk_score,
            confidence_score=90.0,
        ),
        confidence_score=breakdown.confidence_score,
    )

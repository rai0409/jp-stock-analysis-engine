"""Configuration: disclaimer, score weights, signal thresholds, analysis config."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from jp_stock_analysis.schemas import SignalMode

DEFAULT_DISCLAIMER = (
    "This output is for analytical and self-directed research purposes. "
    "It is not personalized financial advice."
)


class ScoreWeights(BaseModel):
    """Relative weights of sub-scores in the final score.

    ``risk_adjustment`` is the fraction of ``risk_score`` subtracted from the
    weighted base score (0.10 means up to -10 points at risk_score=100).
    """

    model_config = ConfigDict(extra="forbid")

    quality_score: float = 0.25
    growth_score: float = 0.20
    valuation_score: float = 0.20
    momentum_score: float = 0.15
    disclosure_score: float = 0.10
    risk_adjustment: float = 0.10

    @field_validator("*")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("score weights must be non-negative")
        return value


class SignalThresholds(BaseModel):
    """0-100 score thresholds used by screening and the trade-signal engine.

    The ``sector_support_*`` fields gate the sector-relative entry in
    ``supporting_factors`` only. They are evidence-only and applied after the
    signal label is decided, so they never influence label selection and are
    deliberately not reported in ``thresholds_used``.
    """

    model_config = ConfigDict(extra="forbid")

    strong_candidate_threshold: float = 80.0
    candidate_threshold: float = 65.0
    watchlist_threshold: float = 50.0
    avoid_threshold: float = 35.0
    buy_signal_threshold: float = 78.0
    sell_signal_threshold: float = 35.0
    min_confidence_for_signal: float = 55.0
    max_risk_score_for_buy_signal: float = 45.0
    sector_support_score_threshold: float = 70.0
    sector_support_min_peers: int = 4
    sector_support_min_confidence: float = 50.0

    @field_validator("sector_support_min_peers")
    @classmethod
    def _positive_peers(cls, value: int) -> int:
        if value < 1:
            raise ValueError("sector_support_min_peers must be a positive integer")
        return value

    @field_validator("*")
    @classmethod
    def _in_range(cls, value: float, info: ValidationInfo) -> float:
        if info.field_name == "sector_support_min_peers":
            return value  # validated separately as a positive integer
        if not 0 <= value <= 100:
            raise ValueError("thresholds must be between 0 and 100")
        return value


class AnalysisConfig(BaseModel):
    """Top-level engine configuration. Default mode is ``analysis_only``."""

    model_config = ConfigDict(extra="forbid")

    signal_mode: SignalMode = "analysis_only"
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    thresholds: SignalThresholds = Field(default_factory=SignalThresholds)
    disclaimer: str = DEFAULT_DISCLAIMER

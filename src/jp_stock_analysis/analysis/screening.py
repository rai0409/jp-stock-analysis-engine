"""Screening: rank stocks and assign candidate labels.

Labels are only assigned in ``screening`` and ``trade_signal`` modes;
``analysis_only`` results keep ``screening_label=None``. Borderline scores
between the avoid and watchlist thresholds are conservatively labelled
``watchlist``.
"""

from __future__ import annotations

from jp_stock_analysis.config import AnalysisConfig
from jp_stock_analysis.schemas import (
    ScoreBreakdown,
    ScreeningLabel,
    ScreeningResult,
    StockAnalysisResult,
)

_LABELLED_MODES = ("screening", "trade_signal")


def assign_screening_label(
    score_breakdown: ScoreBreakdown | None, config: AnalysisConfig
) -> ScreeningLabel:
    """Map a score breakdown onto a screening label."""
    thresholds = config.thresholds
    if (
        score_breakdown is None
        or score_breakdown.final_score is None
        or score_breakdown.confidence_score < thresholds.min_confidence_for_signal
    ):
        return "insufficient_data"
    final = score_breakdown.final_score
    if final >= thresholds.strong_candidate_threshold:
        return "strong_candidate"
    if final >= thresholds.candidate_threshold:
        return "candidate"
    if final >= thresholds.watchlist_threshold:
        return "watchlist"
    if final <= thresholds.avoid_threshold:
        return "avoid_candidate"
    return "watchlist"


def screen_stocks(
    results: list[StockAnalysisResult], config: AnalysisConfig
) -> list[ScreeningResult]:
    """Rank results by final score (descending) and label them when enabled."""
    def sort_key(result: StockAnalysisResult) -> tuple[bool, float]:
        final = result.score.final_score if result.score else None
        return (final is None, -(final if final is not None else 0.0))

    label_enabled = config.signal_mode in _LABELLED_MODES
    screening: list[ScreeningResult] = []
    for rank, result in enumerate(sorted(results, key=sort_key), start=1):
        score = result.score
        screening.append(
            ScreeningResult(
                ticker=result.ticker,
                company_name=result.company_name,
                rank=rank,
                final_score=score.final_score if score else None,
                confidence_score=score.confidence_score if score else 0.0,
                screening_label=assign_screening_label(score, config) if label_enabled else None,
                warnings=list(score.warnings) if score else ["no score available"],
            )
        )
    return screening

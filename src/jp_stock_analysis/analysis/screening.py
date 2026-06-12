"""Screening: rank stocks and assign candidate labels.

Labels are only assigned in ``screening`` and ``trade_signal`` modes;
``analysis_only`` results keep ``screening_label=None``. Borderline scores
between the avoid and watchlist thresholds are conservatively labelled
``watchlist``.

Ranking is reliability-aware (see ``analysis/reliability.py``): eligible
results rank first by ``screening_score`` (confidence/coverage-adjusted),
then ineligible results; ``final_score`` is only a tie-breaker. A high
``final_score`` built from a single sub-score can therefore never outrank a
well-covered result. The raw ``final_score`` is still reported unchanged.
"""

from __future__ import annotations

from jp_stock_analysis.analysis.reliability import assess_reliability
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
    """Rank results (eligible first, then screening_score, then final score).

    Reliability fields and warnings from ``assess_reliability`` are attached
    to every entry; labels are assigned only when the mode enables them.
    """
    assessments = {
        id(result): assess_reliability(result, config.thresholds) for result in results
    }

    def sort_key(result: StockAnalysisResult) -> tuple:
        assessment = assessments[id(result)]
        final = result.score.final_score if result.score else None
        return (
            not assessment.screening_eligible,
            assessment.screening_score is None,
            -(assessment.screening_score or 0.0),
            final is None,
            -(final or 0.0),
            result.ticker,
        )

    label_enabled = config.signal_mode in _LABELLED_MODES
    screening: list[ScreeningResult] = []
    for rank, result in enumerate(sorted(results, key=sort_key), start=1):
        score = result.score
        assessment = assessments[id(result)]
        warnings = list(score.warnings) if score else ["no score available"]
        warnings.extend(assessment.warnings)
        screening.append(
            ScreeningResult(
                ticker=result.ticker,
                company_name=result.company_name,
                rank=rank,
                final_score=score.final_score if score else None,
                confidence_score=score.confidence_score if score else 0.0,
                data_coverage_score=assessment.data_coverage_score,
                screening_score=assessment.screening_score,
                screening_eligible=assessment.screening_eligible,
                reliability_grade=assessment.reliability_grade,
                screening_label=assign_screening_label(score, config) if label_enabled else None,
                warnings=warnings,
            )
        )
    return screening

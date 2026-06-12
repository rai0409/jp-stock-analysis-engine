"""Confidence-aware screening reliability guard.

A single available sub-score (e.g. momentum derived from only a few price
bars) can produce a high ``final_score`` while almost every analysis
dimension is missing. ``final_score`` is kept untouched for transparency;
this module derives deterministic reliability fields so screening never
ranks or presents such results as strong candidates.

Rules (see docs/confidence_aware_screening.md):

- ``data_coverage_score`` — 20 points per covered component out of
  fundamentals, valuation, momentum, disclosure, risk. A component counts as
  covered only when its analysis exists AND reported non-zero confidence, so
  a metadata-only disclosure with no extractable text does not count.
- ``screening_score`` — the reliability-adjusted ranking score:
  ``final_score x (confidence/100) x (coverage/100)``.
- ``screening_eligible`` — False when ``confidence_score`` is below
  ``thresholds.screening_min_confidence``, coverage is below
  ``thresholds.screening_min_coverage``, fewer than
  ``thresholds.screening_min_subscores`` of the five major sub-scores are
  available, or there is no final score.
- ``reliability_grade`` — ``low`` whenever ineligible; ``high`` when
  confidence >= 70 and coverage >= 80; otherwise ``medium``.

Nothing here mutates scores, screening labels, or trade-signal behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jp_stock_analysis.config import SignalThresholds
from jp_stock_analysis.schemas import ReliabilityGrade, StockAnalysisResult

COMPONENT_COUNT = 5  # fundamentals, valuation, momentum, disclosure, risk

# eligible results are graded "high" at or above both of these, else "medium"
HIGH_GRADE_MIN_CONFIDENCE = 70.0
HIGH_GRADE_MIN_COVERAGE = 80.0

_SUBSCORE_NAMES = (
    "quality_score",
    "growth_score",
    "valuation_score",
    "momentum_score",
    "disclosure_score",
)


@dataclass(frozen=True)
class ReliabilityAssessment:
    """Deterministic screening reliability verdict for one analysis result."""

    data_coverage_score: float
    available_subscores: int
    screening_score: float | None
    screening_eligible: bool
    reliability_grade: ReliabilityGrade
    warnings: list[str] = field(default_factory=list)


def assess_reliability(
    result: StockAnalysisResult, thresholds: SignalThresholds
) -> ReliabilityAssessment:
    """Assess how much the result's ``final_score`` can be trusted for ranking."""
    components = (
        ("fundamentals", result.fundamentals),
        ("valuation", result.valuation),
        ("momentum", result.momentum),
        ("disclosure", result.disclosure),
        ("risk", result.risks),
    )
    covered = [
        name
        for name, component in components
        if component is not None and component.confidence_score > 0
    ]
    coverage = round(len(covered) / COMPONENT_COUNT * 100.0, 1)

    score = result.score
    final = score.final_score if score else None
    confidence = score.confidence_score if score else 0.0
    available = [
        name
        for name in _SUBSCORE_NAMES
        if score is not None and getattr(score, name) is not None
    ]

    screening_score = (
        None
        if final is None
        else round(final * (confidence / 100.0) * (coverage / 100.0), 1)
    )

    eligible = (
        final is not None
        and confidence >= thresholds.screening_min_confidence
        and coverage >= thresholds.screening_min_coverage
        and len(available) >= thresholds.screening_min_subscores
    )

    warnings: list[str] = []
    if final is None:
        warnings.append("screening reliability: no final score; not eligible for ranking")
    if confidence < thresholds.screening_min_confidence:
        warnings.append(
            f"screening reliability: confidence {confidence:.1f} is below the minimum "
            f"{thresholds.screening_min_confidence:.0f}; not eligible as a candidate"
        )
    if coverage < thresholds.screening_min_coverage:
        warnings.append(
            f"screening reliability: data coverage {coverage:.1f} is below the minimum "
            f"{thresholds.screening_min_coverage:.0f} "
            f"(covered components: {', '.join(covered) or 'none'})"
        )
    if len(available) < thresholds.screening_min_subscores:
        warnings.append(
            f"screening reliability: only {len(available)} of {COMPONENT_COUNT} sub-scores "
            f"available ({', '.join(available) or 'none'}); minimum is "
            f"{thresholds.screening_min_subscores}"
        )

    if not eligible and final is not None and final >= thresholds.candidate_threshold:
        warnings.append(
            f"screening reliability: high final_score {final:.1f} rests on limited data "
            f"(coverage {coverage:.1f}/100, confidence {confidence:.1f}/100) and must not "
            "be interpreted as a strong candidate"
        )

    if not eligible:
        grade: ReliabilityGrade = "low"
    elif confidence >= HIGH_GRADE_MIN_CONFIDENCE and coverage >= HIGH_GRADE_MIN_COVERAGE:
        grade = "high"
    else:
        grade = "medium"

    return ReliabilityAssessment(
        data_coverage_score=coverage,
        available_subscores=len(available),
        screening_score=screening_score,
        screening_eligible=eligible,
        reliability_grade=grade,
        warnings=warnings,
    )

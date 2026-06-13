"""Offline-ready, commercial-grade stock-analysis modeling infrastructure.

This package builds model-ready datasets, computes explainable factor features,
validates cross-sectional ranking (Rank IC / quantiles), runs domain-aware
walk-forward and purged/embargo splits, scores a transparent baseline ranker,
and optionally wraps LightGBM / CatBoost — all offline and deterministic.

It is research infrastructure only: it emits no buy/sell signals, makes no
predictive claim, keeps ``analysis_only`` semantics, and treats synthetic
fixtures as non-evidence. See ``docs/modeling_pipeline.md``.
"""

from __future__ import annotations

__all__ = [
    "baseline_ranker",
    "dataset",
    "factors",
    "fixtures",
    "ml_models",
    "purged",
    "ranking_metrics",
    "report",
    "walk_forward",
]

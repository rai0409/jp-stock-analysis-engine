"""Offline forward-return validation harness.

Measures realized forward returns from a point-in-time ``screening.json`` and a
later local prices CSV, then groups them by the engine's screening fields so the
research value of ``screening_score`` / ``reliability_grade`` /
``screening_eligible`` can be compared against the raw ``final_score``.

This package is research-only. It produces no trading signals, no portfolio
construction, and no position sizing (see ``forward_returns.py``).
"""

from jp_stock_analysis.validation.forward_returns import (
    FORWARD_RETURN_DISCLAIMER,
    ForwardReturnReport,
    build_forward_return_report,
    write_forward_return_outputs,
)

__all__ = [
    "FORWARD_RETURN_DISCLAIMER",
    "ForwardReturnReport",
    "build_forward_return_report",
    "write_forward_return_outputs",
]

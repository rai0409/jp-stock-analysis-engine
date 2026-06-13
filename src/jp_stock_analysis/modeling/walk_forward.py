"""Domain-aware walk-forward validation framework.

Generates deterministic train/test folds over an ordered list of decision-date
*periods* — expanding or rolling window — similar in spirit to a time-series
split, but built for forward-return studies: folds are by decision date, a
minimum training history is enforced, and folds are horizon-aware so a downstream
purge/embargo (see ``modeling/purged.py``) can drop training labels whose
forward-return window overlaps the test period.

No look-ahead is preserved: a fold's training periods are always strictly before
its test periods. Output is a stable, inspectable fold table.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

MODE_EXPANDING = "expanding"
MODE_ROLLING = "rolling"


@dataclass(frozen=True)
class WalkForwardFold:
    """One train/test split over decision-date periods."""

    fold_index: int
    train_periods: tuple[date, ...]
    test_periods: tuple[date, ...]

    @property
    def train_start(self) -> date:
        return self.train_periods[0]

    @property
    def train_end(self) -> date:
        return self.train_periods[-1]

    @property
    def test_start(self) -> date:
        return self.test_periods[0]

    @property
    def test_end(self) -> date:
        return self.test_periods[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "train_period_count": len(self.train_periods),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "test_period_count": len(self.test_periods),
            "train_periods": [d.isoformat() for d in self.train_periods],
            "test_periods": [d.isoformat() for d in self.test_periods],
        }


def generate_folds(
    periods: Sequence[date],
    *,
    min_train_periods: int = 1,
    test_periods: int = 1,
    mode: str = MODE_EXPANDING,
    train_periods: int | None = None,
) -> list[WalkForwardFold]:
    """Generate non-overlapping walk-forward folds over sorted unique periods.

    - ``mode=expanding`` — training set grows to include all prior periods.
    - ``mode=rolling``   — training set is the last ``train_periods`` (defaults
      to ``min_train_periods``) periods before the test window.
    """
    if mode not in (MODE_EXPANDING, MODE_ROLLING):
        raise ValueError(f"unknown mode {mode!r}")
    if min_train_periods < 1:
        raise ValueError("min_train_periods must be >= 1")
    if test_periods < 1:
        raise ValueError("test_periods must be >= 1")
    window = train_periods if train_periods is not None else min_train_periods
    if mode == MODE_ROLLING and window < min_train_periods:
        raise ValueError("rolling train_periods must be >= min_train_periods")

    ordered = sorted(set(periods))
    folds: list[WalkForwardFold] = []
    fold_index = 0
    test_start_idx = min_train_periods
    while test_start_idx + test_periods <= len(ordered):
        test_slice = ordered[test_start_idx : test_start_idx + test_periods]
        if mode == MODE_EXPANDING:
            train_slice = ordered[:test_start_idx]
        else:
            train_slice = ordered[max(0, test_start_idx - window) : test_start_idx]
        if len(train_slice) >= min_train_periods:
            folds.append(
                WalkForwardFold(
                    fold_index=fold_index,
                    train_periods=tuple(train_slice),
                    test_periods=tuple(test_slice),
                )
            )
            fold_index += 1
        test_start_idx += test_periods
    return folds


@dataclass(frozen=True)
class WalkForwardPlan:
    mode: str
    min_train_periods: int
    test_periods: int
    horizons: tuple[int, ...]
    folds: list[WalkForwardFold]
    disclaimer: str = (
        "This output is for analytical and self-directed research purposes. It "
        "is not personalized financial advice."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "mode": self.mode,
            "min_train_periods": self.min_train_periods,
            "test_periods": self.test_periods,
            "horizons": list(self.horizons),
            "fold_count": len(self.folds),
            "folds": [f.to_dict() for f in self.folds],
        }


def build_walk_forward_plan(
    periods: Sequence[date],
    *,
    horizons: Sequence[int],
    min_train_periods: int = 1,
    test_periods: int = 1,
    mode: str = MODE_EXPANDING,
    train_periods: int | None = None,
) -> WalkForwardPlan:
    folds = generate_folds(
        periods,
        min_train_periods=min_train_periods,
        test_periods=test_periods,
        mode=mode,
        train_periods=train_periods,
    )
    return WalkForwardPlan(
        mode=mode,
        min_train_periods=min_train_periods,
        test_periods=test_periods,
        horizons=tuple(sorted({int(h) for h in horizons})),
        folds=folds,
    )


def write_walk_forward_outputs(
    plan: WalkForwardPlan, output_dir: str | Path, *, write_markdown: bool = True
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "walk_forward.json"
    json_path.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths = {"json_path": json_path}
    if write_markdown:
        md_path = out_dir / "walk_forward.md"
        lines = ["# Walk-Forward Validation Folds", "", plan.disclaimer, ""]
        lines.append(f"- Mode: `{plan.mode}`")
        lines.append(f"- Minimum train periods: {plan.min_train_periods}")
        lines.append(f"- Test periods per fold: {plan.test_periods}")
        lines.append(f"- Horizons: {list(plan.horizons)}")
        lines.append(f"- Folds: {len(plan.folds)}")
        lines += [
            "",
            "| fold | train_start | train_end | train_n | test_start | test_end | test_n |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for f in plan.folds:
            lines.append(
                f"| {f.fold_index} | {f.train_start} | {f.train_end} | "
                f"{len(f.train_periods)} | {f.test_start} | {f.test_end} | "
                f"{len(f.test_periods)} |"
            )
        lines.append("")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths["markdown_path"] = md_path
    return paths


__all__ = [
    "MODE_EXPANDING",
    "MODE_ROLLING",
    "WalkForwardFold",
    "WalkForwardPlan",
    "build_walk_forward_plan",
    "generate_folds",
    "write_walk_forward_outputs",
]

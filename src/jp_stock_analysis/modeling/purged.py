"""Purged / embargoed splitting for forward-return labels.

Forward-return labels span a window (decision date -> the date the horizon return
realises). If a training sample's label window overlaps the test period, the
training and test sets share information and the validation leaks. This module
removes that overlap deterministically, with no dependency on mlfinlab:

- **purge**   — drop training samples whose ``[label_start, label_end]`` overlaps
  the test window ``[test_start, test_end]`` (touching counts as overlap);
- **embargo** — additionally drop training samples that *start* within
  ``embargo_days`` after the test window ends, guarding against serial
  correlation right after the test set.

Lightweight, internal, and fully testable offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

PURGE_OVERLAP = "purged_overlap"
EMBARGO_AFTER = "embargo_after_test"


@dataclass(frozen=True)
class LabeledSample:
    """A training/test sample with its forward-return label window."""

    key: Any
    label_start_date: date
    label_end_date: date

    def __post_init__(self) -> None:
        if self.label_end_date < self.label_start_date:
            raise ValueError(
                f"label_end_date {self.label_end_date} precedes "
                f"label_start_date {self.label_start_date}"
            )


def _overlaps(a_start: date, a_end: date, b_start: date, b_end: date) -> bool:
    """Closed-interval overlap (touching endpoints count as overlapping)."""
    return a_start <= b_end and b_start <= a_end


@dataclass(frozen=True)
class PurgeResult:
    kept: list[LabeledSample]
    dropped: list[tuple[LabeledSample, str]]

    @property
    def kept_keys(self) -> list[Any]:
        return [s.key for s in self.kept]

    @property
    def drop_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _sample, reason in self.dropped:
            counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items()))


def purge_embargo_split(
    train_samples: Sequence[LabeledSample],
    test_start: date,
    test_end: date,
    *,
    embargo_days: int = 0,
) -> PurgeResult:
    """Purge overlapping and embargoed training samples for one test window."""
    if test_end < test_start:
        raise ValueError("test_end precedes test_start")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")
    embargo_cutoff = test_end + timedelta(days=embargo_days)

    kept: list[LabeledSample] = []
    dropped: list[tuple[LabeledSample, str]] = []
    for sample in train_samples:
        if _overlaps(sample.label_start_date, sample.label_end_date, test_start, test_end):
            dropped.append((sample, PURGE_OVERLAP))
        elif embargo_days > 0 and test_end < sample.label_start_date <= embargo_cutoff:
            dropped.append((sample, EMBARGO_AFTER))
        else:
            kept.append(sample)
    return PurgeResult(kept=kept, dropped=dropped)


__all__ = [
    "EMBARGO_AFTER",
    "PURGE_OVERLAP",
    "LabeledSample",
    "PurgeResult",
    "purge_embargo_split",
]

"""Walk-forward, purged out-of-sample (OOS) training and prediction.

Every model score that feeds an evaluation metric (Rank IC, spreads, ensembles,
MMC-style deltas) must be produced by a model that never saw that row's label.
This module is the single place that guarantees it:

1. **Walk-forward folds** over decision dates (``walk_forward.generate_folds``):
   training periods are always strictly before the test periods.
2. **Purging** (López de Prado, *Advances in Financial Machine Learning*, ch. 7):
   a training row whose forward-return window ``[decision_date, label_end]``
   overlaps the test window is dropped, because its label is partly realised
   inside the test period. ``embargo_days`` is passed through for completeness.
3. **Cross-sectional transforms**: features are converted to centred
   within-date ranks and the target to a centred within-date rank (or a
   within-date demeaned return). The model then learns *relative* ordering,
   which is what a cross-sectional ranking strategy trades, instead of fitting
   the market-wide level of returns on each date.

Ranks use only same-date information, so the transform introduces no
look-ahead. When per-ticker prices are unavailable the label end date is
estimated conservatively from calendar days (over-purging, never under-purging).
"""

from __future__ import annotations

import bisect
import math
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from jp_stock_analysis.modeling.dataset import ModelingDataset, ModelingObservation
from jp_stock_analysis.modeling.factors import ALL_FACTORS
from jp_stock_analysis.modeling.purged import LabeledSample, purge_embargo_split
from jp_stock_analysis.modeling.ranking_metrics import ScoredObservation, _ranks
from jp_stock_analysis.modeling.walk_forward import MODE_EXPANDING, generate_folds
from jp_stock_analysis.schemas import PriceBar

VALIDATION_SCHEME = "walk_forward_purged_oos"
TARGET_RANK = "rank"
TARGET_DEMEANED = "demeaned"

STATUS_OK = "ok"
STATUS_NO_FOLDS = "no_folds"
STATUS_INSUFFICIENT_TRAIN = "insufficient_train_rows"
STATUS_NO_TEST_ROWS = "no_test_rows"

# fit_predict(train_rows, test_rows, label_key) -> one score per test row
FitPredict = Callable[
    [Sequence[ModelingObservation], Sequence[ModelingObservation], str],
    Sequence[float],
]


# --------------------------------------------------------------------------- #
# Cross-sectional transforms (same-date information only)
# --------------------------------------------------------------------------- #
def _centered_ranks(values: Sequence[float | None]) -> list[float | None]:
    """Map present values to evenly spaced ranks in [-0.5, 0.5]; None stays None."""
    present_idx = [i for i, v in enumerate(values) if v is not None]
    out: list[float | None] = [None] * len(values)
    if not present_idx:
        return out
    if len(present_idx) == 1:
        out[present_idx[0]] = 0.0
        return out
    ranks = _ranks([float(values[i]) for i in present_idx])  # type: ignore[arg-type]
    denom = len(present_idx) - 1
    for position, index in enumerate(present_idx):
        out[index] = (ranks[position] - 1.0) / denom - 0.5
    return out


def cross_sectional_rank_matrix(
    observations: Sequence[ModelingObservation],
    feature_names: Sequence[str] = ALL_FACTORS,
) -> list[list[float | None]]:
    """Per-date centred rank of every feature, aligned to ``observations``."""
    by_date: dict[date, list[int]] = {}
    for index, obs in enumerate(observations):
        by_date.setdefault(obs.decision_date, []).append(index)
    matrix: list[list[float | None]] = [[None] * len(feature_names) for _ in observations]
    for indices in by_date.values():
        for j, name in enumerate(feature_names):
            ranked = _centered_ranks([observations[i].features.get(name) for i in indices])
            for position, index in enumerate(indices):
                matrix[index][j] = ranked[position]
    return matrix


def cross_sectional_target(
    observations: Sequence[ModelingObservation],
    label_key: str,
    *,
    mode: str = TARGET_RANK,
) -> list[float]:
    """Per-date centred rank (default) or demeaned forward return."""
    if mode not in (TARGET_RANK, TARGET_DEMEANED):
        raise ValueError(f"unknown target mode {mode!r}")
    by_date: dict[date, list[int]] = {}
    for index, obs in enumerate(observations):
        by_date.setdefault(obs.decision_date, []).append(index)
    target = [0.0] * len(observations)
    for indices in by_date.values():
        values = [float(observations[i].labels[label_key]) for i in indices]  # type: ignore[arg-type]
        if mode == TARGET_RANK:
            transformed = _centered_ranks(values)
        else:
            mean = sum(values) / len(values)
            transformed = [v - mean for v in values]
        for position, index in enumerate(indices):
            target[index] = float(transformed[position] or 0.0)
    return target


# --------------------------------------------------------------------------- #
# Label windows for purging
# --------------------------------------------------------------------------- #
def conservative_calendar_label_end(decision_date: date, horizon: int) -> date:
    """Upper-bound calendar estimate of the date an ``horizon``-row label realises.

    ``horizon`` trading rows (+1 entry row) map to at most ~7/5 calendar days
    each; a 10-day buffer covers Japanese holiday clusters (Golden Week, New
    Year). Overestimating only purges more, which is the safe direction.
    """
    return decision_date + timedelta(days=math.ceil((horizon + 1) * 7 / 5) + 10)


class LabelEndIndex:
    """Memoised label end dates: one sort per ticker, then O(log n) lookups.

    ``extra_rows`` widens every window (next-open labels may roll a locked exit
    forward by up to ``MAX_EXIT_DELAY_ROWS`` sessions, so purging must cover
    that too). Unknown / too-short price histories fall back to the
    conservative calendar estimate.
    """

    def __init__(
        self, prices: Mapping[str, Sequence[PriceBar]] | None, *, extra_rows: int = 0
    ) -> None:
        self._prices = prices
        self.extra_rows = extra_rows
        self._dates: dict[str, list[date]] = {}
        self._memo: dict[tuple[str, date, int], date] = {}

    def _ticker_dates(self, ticker: str) -> list[date] | None:
        if self._prices is None or ticker not in self._prices:
            return None
        if ticker not in self._dates:
            self._dates[ticker] = sorted(b.date for b in self._prices[ticker])
        return self._dates[ticker]

    def end(self, obs: ModelingObservation, horizon: int) -> date:
        key = (obs.ticker, obs.decision_date, horizon)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        dates = self._ticker_dates(obs.ticker)
        result: date | None = None
        if dates:
            first_after = bisect.bisect_right(dates, obs.decision_date)
            if first_after + horizon < len(dates):
                target = first_after + horizon + self.extra_rows
                result = dates[min(target, len(dates) - 1)]
        if result is None:
            result = conservative_calendar_label_end(
                obs.decision_date, horizon + self.extra_rows
            )
        self._memo[key] = result
        return result


def label_end_index_for(
    dataset: ModelingDataset, prices: Mapping[str, Sequence[PriceBar]] | None
) -> LabelEndIndex:
    """Index with the exit-delay allowance matching the dataset's label entry."""
    from jp_stock_analysis.modeling.dataset import ENTRY_NEXT_OPEN, MAX_EXIT_DELAY_ROWS

    extra = MAX_EXIT_DELAY_ROWS if dataset.entry == ENTRY_NEXT_OPEN else 0
    return LabelEndIndex(prices, extra_rows=extra)


def label_end_date(
    obs: ModelingObservation,
    horizon: int,
    prices: Mapping[str, Sequence[PriceBar]] | None,
    index: LabelEndIndex | None = None,
) -> date:
    """Date the ``horizon`` label realises (pass ``index`` to reuse its cache)."""
    return (index or LabelEndIndex(prices)).end(obs, horizon)


# --------------------------------------------------------------------------- #
# Ranked-feature cache (same-date ranks are fold-independent)
# --------------------------------------------------------------------------- #
_RANK_CACHE: dict[tuple[int, tuple[str, ...]], dict[tuple[str, date], list[float | None]]] = {}


def ranked_feature_rows(
    dataset: ModelingDataset, feature_names: Sequence[str] | None = None
) -> dict[tuple[str, date], list[float | None]]:
    """Within-date centred ranks for every included row, computed once per dataset.

    Ranks use only rows of the same decision date (all included names), so the
    same lookup serves every walk-forward fold without look-ahead.
    """
    names = tuple(feature_names or dataset.feature_names)
    key = (id(dataset), names)
    cached = _RANK_CACHE.get(key)
    if cached is not None:
        return cached
    rows = sorted(dataset.included(), key=lambda o: (o.decision_date, o.ticker))
    matrix = cross_sectional_rank_matrix(rows, names)
    lookup = {(o.ticker, o.decision_date): row for o, row in zip(rows, matrix, strict=True)}
    _RANK_CACHE[key] = lookup
    weakref.finalize(dataset, _RANK_CACHE.pop, key, None)
    return lookup


def ranked_matrix(
    dataset: ModelingDataset,
    rows: Sequence[ModelingObservation],
    feature_names: Sequence[str] | None = None,
) -> list[list[float | None]]:
    lookup = ranked_feature_rows(dataset, feature_names)
    width = len(tuple(feature_names or dataset.feature_names))
    return [lookup.get((o.ticker, o.decision_date), [None] * width) for o in rows]


# --------------------------------------------------------------------------- #
# Walk-forward OOS driver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OOSFold:
    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_rows_before_purge: int
    purged_rows: int
    train_rows: int
    test_rows: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_rows_before_purge": self.train_rows_before_purge,
            "purged_rows": self.purged_rows,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "status": self.status,
        }


@dataclass(frozen=True)
class OOSResult:
    horizon: int
    scored: list[ScoredObservation]
    folds: list[OOSFold] = field(default_factory=list)
    status: str = STATUS_OK
    message: str = ""
    validation: str = VALIDATION_SCHEME

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and bool(self.scored)

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation": self.validation,
            "horizon": self.horizon,
            "status": self.status,
            "message": self.message,
            "oos_scored_count": len(self.scored),
            "folds": [f.to_dict() for f in self.folds],
        }


def labelled_observations(
    dataset: ModelingDataset, horizon: int
) -> tuple[list[ModelingObservation], str]:
    label_key = f"forward_return_h{horizon}"
    rows = [o for o in dataset.included() if o.labels.get(label_key) is not None]
    rows.sort(key=lambda o: (o.decision_date, o.ticker))
    return rows, label_key


def walk_forward_oos(
    dataset: ModelingDataset,
    horizon: int,
    fit_predict: FitPredict,
    *,
    prices: Mapping[str, Sequence[PriceBar]] | None = None,
    min_train_periods: int = 1,
    mode: str = MODE_EXPANDING,
    train_periods: int | None = None,
    embargo_days: int = 0,
    min_train_rows: int = 5,
    purge_horizon: int | None = None,
    refit_every: int = 1,
) -> OOSResult:
    """Produce purged walk-forward OOS scores for every testable decision date.

    ``purge_horizon`` (default ``horizon``) sets the label window used for
    purging; a multi-horizon model trained on labels up to ``h_max`` must pass
    ``purge_horizon=h_max`` so no training label overlaps the test window.

    ``refit_every`` (decision dates per fold) refits the model every N decision
    dates instead of every date — e.g. 4 with weekly decisions = monthly refits,
    the usual production cadence. Purging is relative to each fold's first test
    date, so a larger value never loosens leakage protection.

    Test rows are all *included* observations on the test dates (labelled or
    not, so a live date can be scored); training rows are labelled observations
    on earlier dates whose label window does not overlap the test window.
    """
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    purge_h = max(horizon, purge_horizon or horizon)
    ends = label_end_index_for(dataset, prices)
    labelled, label_key = labelled_observations(dataset, horizon)
    included = sorted(dataset.included(), key=lambda o: (o.decision_date, o.ticker))
    periods = sorted({o.decision_date for o in included})
    folds = generate_folds(
        periods,
        min_train_periods=min_train_periods,
        test_periods=refit_every,
        mode=mode,
        train_periods=train_periods,
    )
    if not folds:
        return OOSResult(
            horizon=horizon,
            scored=[],
            status=STATUS_NO_FOLDS,
            message=(
                f"{len(periods)} decision date(s); walk-forward OOS needs more than "
                f"{min_train_periods} to form a train/test split"
            ),
        )

    scored: list[ScoredObservation] = []
    fold_records: list[OOSFold] = []
    for fold in folds:
        train_dates = set(fold.train_periods)
        test_dates = set(fold.test_periods)
        candidates = [o for o in labelled if o.decision_date in train_dates]
        test_rows = [o for o in included if o.decision_date in test_dates]
        samples = [
            LabeledSample(
                key=index,
                label_start_date=obs.decision_date,
                label_end_date=ends.end(obs, purge_h),
            )
            for index, obs in enumerate(candidates)
        ]
        test_window_end = max(
            [ends.end(o, purge_h) for o in test_rows] or [fold.test_end]
        )
        purge = purge_embargo_split(
            samples, fold.test_start, test_window_end, embargo_days=embargo_days
        )
        train_rows = [candidates[k] for k in purge.kept_keys]

        if not test_rows:
            status = STATUS_NO_TEST_ROWS
        elif len(train_rows) < min_train_rows:
            status = STATUS_INSUFFICIENT_TRAIN
        else:
            status = STATUS_OK
            predictions = list(fit_predict(train_rows, test_rows, label_key))
            if len(predictions) != len(test_rows):
                raise ValueError("fit_predict must return one score per test row")
            for obs, prediction in zip(test_rows, predictions, strict=True):
                scored.append(
                    ScoredObservation(
                        decision_date=obs.decision_date,
                        ticker=obs.ticker,
                        score=float(prediction),
                        sector=obs.sector,
                        labels=obs.labels,
                    )
                )
        fold_records.append(
            OOSFold(
                fold_index=fold.fold_index,
                train_start=fold.train_start,
                train_end=fold.train_end,
                test_start=fold.test_start,
                test_end=fold.test_end,
                train_rows_before_purge=len(candidates),
                purged_rows=len(purge.dropped),
                train_rows=len(train_rows),
                test_rows=len(test_rows),
                status=status,
            )
        )

    if not scored:
        return OOSResult(
            horizon=horizon,
            scored=[],
            folds=fold_records,
            status=STATUS_INSUFFICIENT_TRAIN,
            message=(
                "no fold had enough purged training rows "
                f"(min_train_rows={min_train_rows}); label windows of horizon "
                f"{horizon} overlap the test periods"
            ),
        )
    return OOSResult(horizon=horizon, scored=scored, folds=fold_records)


# --------------------------------------------------------------------------- #
# Linear-model adapter
# --------------------------------------------------------------------------- #
@dataclass
class LinearOOSRun:
    """OOS result plus the last fold's model/test matrix (for OOS importance)."""

    result: OOSResult
    last_model: Any = None
    last_test_matrix: list[list[float | None]] = field(default_factory=list)
    last_test_rows: list[ModelingObservation] = field(default_factory=list)


def linear_walk_forward_oos(
    dataset: ModelingDataset,
    horizon: int,
    model_factory: Callable[[], Any],
    *,
    prices: Mapping[str, Sequence[PriceBar]] | None = None,
    target_mode: str = TARGET_RANK,
    min_train_periods: int = 1,
    embargo_days: int = 0,
    min_train_rows: int = 5,
    refit_every: int = 1,
) -> LinearOOSRun:
    """Walk-forward OOS for a ``fit(X, y, names)`` / ``predict(X)`` linear ranker.

    A fresh model is built per fold from ``model_factory`` so no state leaks
    across folds. Features are within-date ranks; the target is a within-date
    rank (default) or demeaned return.
    """
    run = LinearOOSRun(result=OOSResult(horizon=horizon, scored=[]))
    names = list(dataset.feature_names)

    def fit_predict(
        train: Sequence[ModelingObservation],
        test: Sequence[ModelingObservation],
        label_key: str,
    ) -> list[float]:
        model = model_factory()
        x_train = ranked_matrix(dataset, train, names)
        y_train = cross_sectional_target(train, label_key, mode=target_mode)
        model.fit(x_train, y_train, names)
        x_test = ranked_matrix(dataset, test, names)
        run.last_model = model
        run.last_test_matrix = x_test
        run.last_test_rows = list(test)
        return [float(p) for p in model.predict(x_test)]

    run.result = walk_forward_oos(
        dataset,
        horizon,
        fit_predict,
        prices=prices,
        min_train_periods=min_train_periods,
        embargo_days=embargo_days,
        min_train_rows=min_train_rows,
        refit_every=refit_every,
    )
    return run


__all__ = [
    "STATUS_INSUFFICIENT_TRAIN",
    "STATUS_NO_FOLDS",
    "STATUS_OK",
    "TARGET_DEMEANED",
    "TARGET_RANK",
    "VALIDATION_SCHEME",
    "LinearOOSRun",
    "OOSFold",
    "OOSResult",
    "conservative_calendar_label_end",
    "cross_sectional_rank_matrix",
    "cross_sectional_target",
    "label_end_date",
    "labelled_observations",
    "linear_walk_forward_oos",
    "walk_forward_oos",
]

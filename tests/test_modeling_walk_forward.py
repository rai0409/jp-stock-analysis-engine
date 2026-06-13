"""Tests for walk-forward fold generation."""

from __future__ import annotations

from datetime import date

import pytest

from jp_stock_analysis.modeling.walk_forward import (
    MODE_EXPANDING,
    MODE_ROLLING,
    build_walk_forward_plan,
    generate_folds,
    write_walk_forward_outputs,
)

PERIODS = [date(2025, m, 1) for m in range(1, 6)]  # 5 monthly periods


def test_expanding_window_grows_training_set():
    folds = generate_folds(PERIODS, min_train_periods=2, test_periods=1, mode=MODE_EXPANDING)
    # test starts at index 2,3,4 -> 3 folds
    assert len(folds) == 3
    assert folds[0].train_periods == (PERIODS[0], PERIODS[1])
    assert folds[0].test_periods == (PERIODS[2],)
    assert folds[2].train_periods == tuple(PERIODS[:4])  # expands
    # train always strictly before test (no look-ahead)
    for fold in folds:
        assert fold.train_end < fold.test_start


def test_rolling_window_is_fixed_length():
    folds = generate_folds(
        PERIODS, min_train_periods=2, test_periods=1, mode=MODE_ROLLING, train_periods=2
    )
    assert len(folds) == 3
    for fold in folds:
        assert len(fold.train_periods) == 2  # fixed rolling window


def test_test_periods_greater_than_one():
    folds = generate_folds(PERIODS, min_train_periods=1, test_periods=2)
    # test windows [1,2], [3,4] -> 2 folds (index 0 trains alone is min_train=1)
    assert len(folds) == 2
    assert folds[0].test_periods == (PERIODS[1], PERIODS[2])


def test_insufficient_periods_yields_no_folds():
    folds = generate_folds([date(2025, 1, 1)], min_train_periods=2, test_periods=1)
    assert folds == []


def test_dedup_and_sort_periods():
    messy = [PERIODS[2], PERIODS[0], PERIODS[0], PERIODS[1]]
    folds = generate_folds(messy, min_train_periods=1, test_periods=1)
    assert folds[0].train_periods == (PERIODS[0],)
    assert folds[0].test_periods == (PERIODS[1],)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        generate_folds(PERIODS, mode="bogus")
    with pytest.raises(ValueError):
        generate_folds(PERIODS, min_train_periods=0)
    with pytest.raises(ValueError):
        generate_folds(PERIODS, test_periods=0)


def test_plan_outputs_written(tmp_path):
    plan = build_walk_forward_plan(PERIODS, horizons=[5, 20], min_train_periods=2)
    payload = plan.to_dict()
    assert payload["fold_count"] == len(plan.folds)
    assert "not personalized financial advice" in payload["disclaimer"]
    paths = write_walk_forward_outputs(plan, tmp_path / "out")
    assert paths["json_path"].exists()
    assert paths["markdown_path"].exists()

"""Tests for purged / embargo splitting of forward-return labels."""

from __future__ import annotations

from datetime import date

import pytest

from jp_stock_analysis.modeling.purged import (
    EMBARGO_AFTER,
    PURGE_OVERLAP,
    LabeledSample,
    purge_embargo_split,
)

TEST_START = date(2025, 3, 1)
TEST_END = date(2025, 3, 31)


def _sample(key: str, start: date, end: date) -> LabeledSample:
    return LabeledSample(key=key, label_start_date=start, label_end_date=end)


def test_no_purge_when_label_window_before_test():
    train = [_sample("a", date(2025, 1, 1), date(2025, 2, 1))]
    result = purge_embargo_split(train, TEST_START, TEST_END)
    assert result.kept_keys == ["a"]
    assert result.dropped == []


def test_overlapping_train_sample_is_purged():
    # label window ends inside the test window -> overlap
    train = [
        _sample("clean", date(2025, 1, 1), date(2025, 2, 1)),
        _sample("overlap", date(2025, 2, 20), date(2025, 3, 10)),
    ]
    result = purge_embargo_split(train, TEST_START, TEST_END)
    assert result.kept_keys == ["clean"]
    assert result.drop_reason_counts == {PURGE_OVERLAP: 1}


def test_touching_endpoint_counts_as_overlap():
    train = [_sample("touch", date(2025, 2, 1), TEST_START)]  # ends exactly at test_start
    result = purge_embargo_split(train, TEST_START, TEST_END)
    assert result.kept_keys == []
    assert result.drop_reason_counts == {PURGE_OVERLAP: 1}


def test_embargo_drops_samples_starting_just_after_test():
    train = [
        _sample("after_in_embargo", date(2025, 4, 3), date(2025, 4, 10)),
        _sample("after_outside_embargo", date(2025, 4, 30), date(2025, 5, 5)),
    ]
    result = purge_embargo_split(train, TEST_START, TEST_END, embargo_days=5)
    # 2025-04-03 is within 5 days of 2025-03-31; 2025-04-30 is not
    assert result.kept_keys == ["after_outside_embargo"]
    assert result.drop_reason_counts == {EMBARGO_AFTER: 1}


def test_zero_embargo_keeps_post_test_samples():
    train = [_sample("after", date(2025, 4, 1), date(2025, 4, 10))]
    result = purge_embargo_split(train, TEST_START, TEST_END, embargo_days=0)
    assert result.kept_keys == ["after"]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        LabeledSample("bad", date(2025, 3, 10), date(2025, 3, 1))  # end before start
    with pytest.raises(ValueError):
        purge_embargo_split([], TEST_END, TEST_START)  # test_end before test_start
    with pytest.raises(ValueError):
        purge_embargo_split([], TEST_START, TEST_END, embargo_days=-1)

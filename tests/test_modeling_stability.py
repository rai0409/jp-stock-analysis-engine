"""Tests for model stability diagnostics. Deterministic, offline."""

from __future__ import annotations

from jp_stock_analysis.modeling.stability import (
    STATUS_ALL_MISSING,
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_SINGLE_FOLD,
    build_stability_report,
    summarize_metric,
    write_stability_outputs,
)


def test_fold_metric_mean_std_min_max():
    s = summarize_metric([0.1, 0.2, 0.3], "rank_ic")
    assert round(s.mean, 6) == 0.2
    assert s.std is not None and s.std > 0
    assert s.min == 0.1 and s.max == 0.3
    assert s.status == STATUS_OK


def test_positive_period_rate():
    s = summarize_metric([0.1, -0.2, 0.3, -0.4], "spread")
    assert s.positive_period_rate == 0.5


def test_one_fold_degenerate():
    s = summarize_metric([0.5], "rank_ic")
    assert s.status == STATUS_SINGLE_FOLD
    assert s.std is None and s.cv is None
    assert s.mean == 0.5


def test_all_none_metrics():
    s = summarize_metric([None, None], "rank_ic")
    assert s.status == STATUS_ALL_MISSING
    assert s.mean is None


def test_empty_metrics():
    s = summarize_metric([], "rank_ic")
    assert s.status == STATUS_EMPTY


def test_worst_and_best_fold_indices():
    s = summarize_metric([0.3, 0.1, 0.5], "rank_ic")
    assert s.worst_fold == {"index": 1, "value": 0.1}
    assert s.best_fold == {"index": 2, "value": 0.5}


def test_cv_only_when_mean_nonzero():
    assert summarize_metric([1.0, 1.0, 1.0], "x").cv == 0.0  # zero variance -> cv 0
    # mean zero -> cv undefined (None)
    assert summarize_metric([-1.0, 1.0], "x").cv is None


def test_seed_stability_deterministic_and_report(tmp_path):
    fold_metrics = {
        "rank_ic": [0.1, 0.2, 0.15],
        "long_short_spread": [0.5, -0.2, 0.3],
    }
    seed_ic = [0.1, 0.12, 0.11, 0.13]
    r1 = build_stability_report(fold_metrics, horizon=20, is_synthetic=True, seed_ic=seed_ic)
    r2 = build_stability_report(fold_metrics, horizon=20, is_synthetic=True, seed_ic=seed_ic)
    assert r1.to_dict() == r2.to_dict()  # deterministic
    assert r1.seed_stability is not None
    assert r1.n_folds == 3
    paths = write_stability_outputs(r1, tmp_path / "out")
    assert paths["json_path"].exists()
    assert paths["csv_path"].exists()
    assert "SYNTHETIC" in paths["markdown_path"].read_text(encoding="utf-8")


def test_deterministic_model_note_when_no_seed_variance():
    report = build_stability_report(
        {"rank_ic": [0.1, 0.2]},
        horizon=20,
        seed_ic=[0.1, 0.1, 0.1],
        deterministic_model=True,
    )
    assert report.seed_stability.std == 0.0
    assert any("seed variance is 0" in w for w in report.warnings)

"""Tests for cross-sectional ranking validation metrics."""

from __future__ import annotations

from datetime import date

from jp_stock_analysis.modeling.ranking_metrics import (
    ScoredObservation,
    evaluate_horizon,
    evaluate_ranking,
    spearman,
    write_ranking_outputs,
)


def test_spearman_perfect_and_reversed_and_constant():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    assert spearman([1, 2, 3], [5, 5, 5]) is None  # zero variance
    assert spearman([1], [1]) is None  # too few points


def _obs(d: date, ticker: str, score: float, label: float, sector: str = "a"):
    return ScoredObservation(
        decision_date=d,
        ticker=ticker,
        score=score,
        sector=sector,
        labels={"forward_return_h5": label},
    )


def test_rank_ic_positive_when_score_tracks_return():
    d = date(2025, 1, 1)
    obs = [_obs(d, f"t{i}", float(i), float(i)) for i in range(6)]
    metrics = evaluate_horizon(obs, 5, n_quantiles=3)
    assert metrics.ic_mean == 1.0
    assert metrics.coverage_count == 6
    assert metrics.missing_label_count == 0


def test_quantile_spread_positive_when_top_outperforms():
    d = date(2025, 1, 1)
    # scores 0..5 with returns matching -> top quantile mean > bottom quantile mean
    obs = [_obs(d, f"t{i}", float(i), float(i)) for i in range(6)]
    metrics = evaluate_horizon(obs, 5, n_quantiles=2)
    assert metrics.quantile_spread_mean is not None
    assert metrics.quantile_spread_mean > 0
    assert metrics.hit_rate_top_positive == 1.0  # top bucket mean (4,5) > 0


def test_hit_rate_above_median_uses_universe_median():
    d = date(2025, 1, 1)
    obs = [_obs(d, f"t{i}", float(i), float(i)) for i in range(4)]
    metrics = evaluate_horizon(obs, 5, n_quantiles=2)
    # top bucket = {2,3} mean 2.5 > universe median (1.5)
    assert metrics.hit_rate_top_above_median == 1.0


def test_missing_labels_counted_not_fabricated():
    d = date(2025, 1, 1)
    obs = [
        ScoredObservation(d, "t1", 1.0, "a", {"forward_return_h5": None}),
        ScoredObservation(d, "t2", 2.0, "a", {"forward_return_h5": 5.0}),
    ]
    metrics = evaluate_horizon(obs, 5)
    assert metrics.missing_label_count == 1
    assert metrics.coverage_count == 1


def test_sector_neutral_ic_computed_separately():
    d = date(2025, 1, 1)
    obs = [
        _obs(d, "t1", 1.0, 1.0, sector="a"),
        _obs(d, "t2", 2.0, 2.0, sector="a"),
        _obs(d, "t3", 1.0, 1.0, sector="b"),
        _obs(d, "t4", 2.0, 2.0, sector="b"),
    ]
    metrics = evaluate_horizon(obs, 5, n_quantiles=2)
    assert metrics.sector_neutral_ic_mean is not None


def test_icir_safe_with_single_date():
    d = date(2025, 1, 1)
    obs = [_obs(d, f"t{i}", float(i), float(i)) for i in range(4)]
    metrics = evaluate_horizon(obs, 5)
    # one date -> ic_std is None -> ICIR None, never a crash
    assert metrics.ic_std is None
    assert metrics.icir is None


def test_evaluate_ranking_report_and_outputs(tmp_path):
    d1, d2 = date(2025, 1, 1), date(2025, 2, 1)
    obs = [_obs(d1, f"t{i}", float(i), float(i)) for i in range(5)]
    obs += [_obs(d2, f"t{i}", float(i), float(i)) for i in range(5)]
    report = evaluate_ranking(obs, [5], is_synthetic=True, n_quantiles=3)
    payload = report.to_dict()
    assert payload["is_synthetic"] is True
    assert "not real market evidence" in payload["synthetic_warning"]
    assert "not personalized financial advice" in payload["disclaimer"]

    paths = write_ranking_outputs(report, tmp_path / "out")
    assert paths["json_path"].exists()
    assert paths["csv_path"].exists()
    assert paths["markdown_path"].exists()
    assert "SYNTHETIC" in paths["markdown_path"].read_text(encoding="utf-8")

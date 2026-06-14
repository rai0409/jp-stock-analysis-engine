"""Tests for Numerai-style neutralization metrics. Deterministic, offline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from jp_stock_analysis.modeling.neutralization import (
    STATUS_CONSTANT_PREDICTION,
    STATUS_OK,
    ExposureObservation,
    MMCStyleObservation,
    exposure_diagnostics,
    mmc_style_contribution,
    neutralize,
    neutralized_rank_ic,
    sector_dummy_columns,
    write_neutralized_outputs,
)

DATES = [date(2025, 1, 1), date(2025, 2, 1)]


def test_neutralizing_reduces_correlation_with_exposure():
    # prediction is exposure + a small orthogonal-ish wiggle
    exposure = [float(i) for i in range(10)]
    predictions = [e + (1.0 if i % 2 else -1.0) for i, e in enumerate(exposure)]
    result = neutralize(predictions, {"f": exposure}, ["f"], proportion=1.0)
    assert result.status == STATUS_OK
    diag = exposure_diagnostics(predictions, result.neutralized, {"f": exposure}, ["f"])
    assert abs(diag.post_neutralization_exposure_corr["f"]) < abs(
        diag.pre_neutralization_exposure_corr["f"]
    )
    assert abs(diag.post_neutralization_exposure_corr["f"]) < 1e-6  # fully removed


def test_missing_exposure_column_is_reported_not_ignored():
    result = neutralize([1.0, 2.0, 3.0], {"present": [1.0, 2.0, 3.0]}, ["present", "absent"])
    assert "absent" in result.skipped_exposures
    assert result.skipped_exposures["absent"] == "missing_column"
    assert result.exposure_columns_used == ["present"]


def test_constant_exposure_is_skipped():
    result = neutralize([1.0, 2.0, 3.0], {"c": [5.0, 5.0, 5.0]}, ["c"])
    assert result.skipped_exposures.get("c") == "constant"


def test_constant_prediction_returns_status():
    result = neutralize([2.0, 2.0, 2.0], {"f": [1.0, 2.0, 3.0]}, ["f"])
    assert result.status == STATUS_CONSTANT_PREDICTION
    assert result.neutralized == [2.0, 2.0, 2.0]  # unchanged, not fabricated


def _exposure_obs():
    obs = []
    for d in DATES:
        for i in range(6):
            obs.append(
                ExposureObservation(
                    decision_date=d,
                    ticker=f"t{i}",
                    prediction=float(i),
                    forward_return=float(i) * (1.0 if d == DATES[0] else 1.5),
                    exposures={"mom": float(5 - i), "lev": float(i % 3)},
                    sector="a" if i < 3 else "b",
                )
            )
    return obs


def test_neutralized_rank_ic_per_decision_date():
    report = neutralized_rank_ic(_exposure_obs(), horizon=5, exposure_columns=["mom", "lev"])
    assert report.status == STATUS_OK
    assert set(report.rank_ic_by_date) == {d.isoformat() for d in DATES}
    assert report.ic_mean is not None
    assert report.raw_ic_mean is not None


def test_sector_dummies_expand_deterministically():
    dummies = sector_dummy_columns(["a", "b", "a", None])
    assert dummies == {"sector::a": [1.0, 0.0, 1.0, 0.0], "sector::b": [0.0, 1.0, 0.0, 0.0]}


def test_mmc_style_delta_for_base_and_candidate():
    # candidate tracks the target; base is unrelated -> candidate adds rank info
    obs = []
    for d in DATES:
        for i in range(6):
            obs.append(
                MMCStyleObservation(
                    decision_date=d,
                    ticker=f"t{i}",
                    base_prediction=float(i % 2),  # weakly informative
                    candidate_prediction=float(i),  # tracks return
                    forward_return=float(i),
                )
            )
    report = mmc_style_contribution(obs, horizon=5)
    assert report.status == STATUS_OK
    assert report.contribution_delta is not None
    assert report.candidate_ic_mean is not None
    assert report.delta_vs_base is not None
    assert "NOT official Numerai" in report.caveat


def test_outputs_written_and_synthetic_labelled(tmp_path):
    report = neutralized_rank_ic(
        _exposure_obs(), horizon=5, exposure_columns=["mom", "lev"], is_synthetic=True
    )
    paths = write_neutralized_outputs(report, tmp_path / "out")
    assert paths["json_path"].exists()
    assert paths["csv_path"].exists()
    assert "SYNTHETIC" in paths["markdown_path"].read_text(encoding="utf-8")


def test_no_external_or_network_dependency():
    """No Numerai package / network import (the word 'Numerai' in prose is fine)."""
    source = Path(
        "src/jp_stock_analysis/modeling/neutralization.py"
    ).read_text(encoding="utf-8")
    import_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    blob = "\n".join(import_lines).lower()
    for forbidden in ("numerai", "numerapi", "requests", "httpx", "urllib", "socket"):
        assert forbidden not in blob

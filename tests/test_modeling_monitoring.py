"""Tests for drift / monitoring. Deterministic, offline."""

from __future__ import annotations

from jp_stock_analysis.modeling.monitoring import (
    STATUS_ALL_MISSING,
    STATUS_OK,
    STATUS_TOO_FEW_PERIODS,
    build_monitoring_report,
    monitor_metric,
    write_monitoring_outputs,
)

PERIODS = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05"]


def test_drift_band_and_stats_computed():
    m = monitor_metric(PERIODS, [0.1, 0.12, 0.11, 0.13, 0.10], "rank_ic")
    assert m.status == STATUS_OK
    assert m.mean is not None and m.std is not None
    assert m.stability_band["lower"] is not None and m.stability_band["upper"] is not None
    assert m.worst_period["value"] == 0.1


def test_threshold_flagging():
    # a clear spike after a stable trailing window should be flagged
    m = monitor_metric(PERIODS, [0.1, 0.1, 0.1, 0.9, 0.1], "rank_ic", window=3, z_threshold=2.0)
    assert any(row["flagged"] for row in m.per_period)
    assert len(m.flagged_periods) >= 1


def test_zero_trailing_std_handled():
    m = monitor_metric(PERIODS, [0.5, 0.5, 0.5, 0.5, 0.5], "x", window=3)
    # constant series -> zero trailing std -> zscore None, nothing flagged, no div0
    assert all(row.get("zscore") is None for row in m.per_period if row.get("value") is not None)
    assert m.flagged_periods == []


def test_too_few_periods_status():
    m = monitor_metric(["p1"], [0.1], "rank_ic")
    assert m.status == STATUS_TOO_FEW_PERIODS


def test_all_missing_status():
    m = monitor_metric(PERIODS, [None, None, None, None, None], "rank_ic")
    assert m.status == STATUS_ALL_MISSING
    assert m.mean is None


def test_missing_metric_periods_handled():
    m = monitor_metric(PERIODS, [0.1, None, 0.2, None, 0.3], "rank_ic")
    assert m.n_periods == 3  # only present values counted
    missing_rows = [r for r in m.per_period if r.get("status") == "missing"]
    assert len(missing_rows) == 2


def test_report_warns_on_few_periods_and_writes(tmp_path):
    report = build_monitoring_report(["p1"], {"m": [0.1]}, is_synthetic=True)
    assert any("period" in w for w in report.warnings)
    paths = write_monitoring_outputs(report, tmp_path / "out")
    assert paths["json_path"].exists()
    assert paths["csv_path"].exists()
    assert "SYNTHETIC" in paths["markdown_path"].read_text(encoding="utf-8")


def test_report_research_only():
    payload = build_monitoring_report(
        PERIODS, {"rank_ic": [0.1, 0.2, 0.1, 0.2, 0.1]}, is_synthetic=True
    ).to_dict()
    assert payload["research_only"] is True
    assert "not real market evidence" in payload["synthetic_warning"]

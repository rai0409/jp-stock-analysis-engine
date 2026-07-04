from __future__ import annotations

import json

import pandas as pd

from jp_stock_analysis.analysis.jquants_extraction_audit import (
    OUTPUT_FILES,
    build_extraction_audit,
    build_required_fields_matrix,
    main,
    resolve_field,
)


def test_required_field_resolver_finds_alias():
    result = resolve_field(["Code", "DiscDate"], ("ticker", "Code"), "required")
    assert result["resolved_column"] == "Code"
    assert result["status"] == "alias_found"


def test_required_field_matrix_marks_missing_warn():
    matrix = build_required_fields_matrix({"financial_summary_raw": ["ticker", "DiscDate", "DocType"]})
    row = matrix[
        (matrix["dataset"] == "financial_summary_raw")
        & (matrix["field_purpose"] == "disclosure_time")
    ].iloc[0]
    assert row["status"] == "missing_should_warn"


def _write_sample_inputs(base):
    financial = base / "financial_summary_raw.csv"
    prices = base / "prices_daily_bars.csv"
    analysis = base / "analysis"
    output = base / "out"
    analysis.mkdir()

    pd.DataFrame(
        [
            {"Code": "72030", "DiscDate": "2024-01-02", "DiscTime": "15:00:00", "DocType": "A", "Sales": 100},
            {"Code": "72030", "DiscDate": "2024-01-02", "DiscTime": "15:00:00", "DocType": "A", "Sales": 100},
            {"Code": None, "DiscDate": "not-a-date", "DiscTime": None, "DocType": "B", "Sales": None},
        ]
    ).to_csv(financial, index=False)
    pd.DataFrame(
        [
            {"ticker": "7203", "date": "2024-01-02", "adj_close": 100, "open": 99, "high": 101, "low": 98, "close": 100, "volume": 10, "turnover_value": 1000},
            {"ticker": "7203", "date": "2024-01-03", "adj_close": 101, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 11, "turnover_value": 1111},
            {"ticker": "7203", "date": "2024-01-04", "adj_close": 102, "open": 101, "high": 103, "low": 100, "close": 102, "volume": 12, "turnover_value": 1224},
            {"ticker": "6758", "date": "2024-01-02", "adj_close": 0, "open": 10, "high": 8, "low": 9, "close": 10, "volume": -1, "turnover_value": 100},
            {"ticker": "6758", "date": "2024-01-02", "adj_close": 0, "open": 10, "high": 8, "low": 9, "close": 10, "volume": -1, "turnover_value": 100},
        ]
    ).to_csv(prices, index=False)
    pd.DataFrame(
        [
            {"ticker": "7203", "DiscDate": "2024-01-02", "liquidity_bucket": "high", "abnormal_ret_1d": 0.1}
        ]
    ).to_csv(analysis / "commercial_quality_positive_reacted_ranking_enriched.csv", index=False)
    (analysis / "commercial_daily_dashboard_audit.json").write_text(
        json.dumps({"warnings": ["sample warning"]}), encoding="utf-8"
    )
    (analysis / "commercial_quality_rankings_enriched_summary.json").write_text(
        json.dumps({"warnings": []}), encoding="utf-8"
    )
    return financial, prices, analysis, output


def test_sample_input_generates_reports_and_detects_quality_issues(tmp_path):
    financial, prices, analysis, output = _write_sample_inputs(tmp_path)

    report = build_extraction_audit(financial, prices, analysis, output)

    assert report["joinability"]["events"] >= 1
    assert report["commercial_readiness_score"]["extraction_readiness_score"] >= 0
    assert (output / OUTPUT_FILES["report_json"]).exists()
    assert (output / OUTPUT_FILES["report_md"]).exists()
    for key in [
        "missing_columns",
        "missing_values",
        "duplicate_keys",
        "coverage_by_date",
        "coverage_by_ticker",
        "unusable_rows",
        "required_fields_matrix",
    ]:
        assert (output / OUTPUT_FILES[key]).exists()

    duplicates = pd.read_csv(output / OUTPUT_FILES["duplicate_keys"])
    assert not duplicates.empty
    unusable = pd.read_csv(output / OUTPUT_FILES["unusable_rows"])
    assert "high_less_than_low" in set(unusable["issue_type"])
    assert "non_positive_price" in set(unusable["issue_type"])
    assert "missing_or_invalid_disc_date" in set(unusable["issue_type"])

    payload = json.loads((output / OUTPUT_FILES["report_json"]).read_text())
    json.dumps(payload, allow_nan=False)


def test_empty_csv_inputs_do_not_crash_and_write_report(tmp_path):
    financial = tmp_path / "financial_summary_raw.csv"
    prices = tmp_path / "prices_daily_bars.csv"
    analysis = tmp_path / "analysis"
    output = tmp_path / "out"
    analysis.mkdir()
    pd.DataFrame().to_csv(financial, index=False)
    pd.DataFrame().to_csv(prices, index=False)

    report = build_extraction_audit(financial, prices, analysis, output)

    assert report["financial_summary_raw"]["rows"] == 0
    assert report["prices_daily_bars"]["rows"] == 0
    assert (output / OUTPUT_FILES["report_json"]).exists()
    assert pd.read_csv(output / OUTPUT_FILES["required_fields_matrix"]).shape[0] > 0


def test_cli_help_works(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "financial-summary" in capsys.readouterr().out

"""No-network tests for TOPIX1000 universe coverage filtering."""

from __future__ import annotations

import csv
import json

from jp_stock_analysis.cli import main
from jp_stock_analysis.validation.universe_coverage import (
    OUTPUT_COLUMNS,
    filter_topix_universe_by_coverage,
    filter_usable_tickers,
    load_universe_coverage,
    summarize_universe_coverage,
    write_excluded_tickers_report,
)


def _write_coverage(path) -> None:
    path.write_text(
        "ticker,name_universe,universe_date,new_index_category,master_matched,"
        "price_ever_available,price_latest_available,first_price_date,last_price_date,"
        "price_rows,coverage_status,reason,company_name,sector,market,raw_code,"
        "sector_17,sector_33,error\n"
        "7203,トヨタ自動車,20260430,TOPIX Core30,True,True,True,"
        "2024-03-28,2024-11-15,158,usable_full_window,"
        "price_available_from_store_start_to_latest_or_near_latest,"
        "トヨタ自動車,輸送用機器,プライム,72030,自動車・輸送機,輸送用機器,\n"
        "9023,東京地下鉄,20260430,TOPIX Mid400,True,True,True,"
        "2024-10-23,2024-11-15,17,usable_partial_history,"
        "price_history_starts_after_store_start,東京地下鉄,陸運業,プライム,"
        "90230,運輸・物流,陸運業,\n"
        "167A,リョーサン菱洋ホールディングス,20260430,TOPIX Small 1,"
        "False,False,False,,, ,master_missing_and_no_price,"
        "2026_universe_member_not_available_in_current_price_window_or_master,"
        ",,,,,,missing_metadata\n"
        "5016,JX金属,20260430,TOPIX Small 1,True,False,False,,, ,"
        "master_matched_but_no_price,"
        "master_matched_but_no_price_in_current_store,JX金属,非鉄金属,プライム,"
        "50160,素材・化学,非鉄金属,\n",
        encoding="utf-8",
    )


def test_default_excludes_partial_missing_and_no_price(tmp_path):
    coverage_file = tmp_path / "coverage.csv"
    _write_coverage(coverage_file)

    coverage = load_universe_coverage(coverage_file)
    included = filter_usable_tickers(coverage)

    assert [row["ticker"] for row in included] == ["7203"]
    excluded_tickers = {row["ticker"] for row in coverage if row not in included}
    assert "9023" in excluded_tickers
    assert "167A" in excluded_tickers
    assert "5016" in excluded_tickers


def test_include_partial_history_includes_9023(tmp_path):
    coverage_file = tmp_path / "coverage.csv"
    _write_coverage(coverage_file)

    coverage = load_universe_coverage(coverage_file)
    included = filter_usable_tickers(coverage, include_partial_history=True)

    assert [row["ticker"] for row in included] == ["7203", "9023"]


def test_summary_counts_by_coverage_status(tmp_path):
    coverage_file = tmp_path / "coverage.csv"
    _write_coverage(coverage_file)

    summary = summarize_universe_coverage(load_universe_coverage(coverage_file))

    assert summary == {
        "universe_count": 4,
        "coverage_statuses": {
            "master_matched_but_no_price": 1,
            "master_missing_and_no_price": 1,
            "usable_full_window": 1,
            "usable_partial_history": 1,
        },
    }


def test_output_schema_stable_and_excluded_report_counts(tmp_path):
    coverage_file = tmp_path / "coverage.csv"
    output_file = tmp_path / "usable.csv"
    report_file = tmp_path / "excluded.json"
    _write_coverage(coverage_file)

    result = filter_topix_universe_by_coverage(
        coverage_file=coverage_file,
        output_file=output_file,
        excluded_report_file=report_file,
    )

    assert result.included_count == 1
    assert result.excluded_count == 3
    with output_file.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == OUTPUT_COLUMNS
    assert rows[0] == {
        "ticker": "7203",
        "name_universe": "トヨタ自動車",
        "universe_date": "20260430",
        "new_index_category": "TOPIX Core30",
        "coverage_status": "usable_full_window",
        "first_price_date": "2024-03-28",
        "last_price_date": "2024-11-15",
        "sector": "輸送用機器",
        "market": "プライム",
    }

    payload = json.loads(report_file.read_text(encoding="utf-8"))
    assert payload["coverage_file"] == str(coverage_file)
    assert payload["include_partial_history"] is False
    assert payload["included_count"] == 1
    assert payload["excluded_count"] == 3
    assert payload["included_statuses"] == {"usable_full_window": 1}
    assert payload["excluded_statuses"] == {
        "master_matched_but_no_price": 1,
        "master_missing_and_no_price": 1,
        "usable_partial_history": 1,
    }
    assert [row["ticker"] for row in payload["excluded_tickers"]] == [
        "9023",
        "167A",
        "5016",
    ]
    assert payload["excluded_tickers"][0] == {
        "ticker": "9023",
        "name_universe": "東京地下鉄",
        "new_index_category": "TOPIX Mid400",
        "coverage_status": "usable_partial_history",
        "reason": "price_history_starts_after_store_start",
    }


def test_excluded_report_counts_with_partial_history_included(tmp_path):
    coverage_file = tmp_path / "coverage.csv"
    report_file = tmp_path / "excluded.json"
    _write_coverage(coverage_file)

    payload = write_excluded_tickers_report(
        coverage=load_universe_coverage(coverage_file),
        coverage_file=coverage_file,
        output_file=report_file,
        include_partial_history=True,
        generated_at="2026-06-24T00:00:00+00:00",
    )

    assert payload["included_count"] == 2
    assert payload["excluded_count"] == 2
    assert payload["included_statuses"] == {
        "usable_full_window": 1,
        "usable_partial_history": 1,
    }
    assert payload["excluded_statuses"] == {
        "master_matched_but_no_price": 1,
        "master_missing_and_no_price": 1,
    }
    assert payload["generated_at"] == "2026-06-24T00:00:00+00:00"


def test_filter_topix_universe_by_coverage_cli(tmp_path, capsys):
    coverage_file = tmp_path / "coverage.csv"
    output_file = tmp_path / "usable.csv"
    report_file = tmp_path / "excluded.json"
    _write_coverage(coverage_file)

    rc = main(
        [
            "filter-topix-universe-by-coverage",
            "--coverage-file",
            str(coverage_file),
            "--output-file",
            str(output_file),
            "--excluded-report-file",
            str(report_file),
            "--include-partial-history",
        ]
    )

    assert rc == 0
    assert "included=2, excluded=2" in capsys.readouterr().out
    with output_file.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["ticker"] for row in rows] == ["7203", "9023"]

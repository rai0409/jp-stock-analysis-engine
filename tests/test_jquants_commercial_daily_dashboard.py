from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from jp_stock_analysis.analysis.jquants_commercial_daily_dashboard import (
    INPUT_FILES,
    OUTPUT_FILES,
    build_daily_dashboard,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ranking_row(ticker: str, disc_date: str, label: str, abnormal: float) -> dict[str, object]:
    return {
        "rank": 1,
        "ranking_label": label,
        "ranking_reason": f"{label}_reason",
        "DiscDate": disc_date,
        "ticker": ticker,
        "company_name": f"Company {ticker}",
        "first_time": "15:00:00",
        "sector33_name": "電気機器",
        "market_segment": "プライム",
        "topix_size_category": "TOPIX Mid400",
        "ranking_score": 100,
        "abnormal_ret_1d": abnormal,
        "ret_1d": abnormal + 0.01,
        "liquidity_bucket": "high",
        "candidate_bucket": "A",
        "avg_turnover_20d": 1_000_000_000,
        "turnover_spike_ratio": 2.0,
        "reaction_base_date": "2024-12-02",
    }


def _write_inputs(base: Path) -> None:
    data = {
        "positive_reacted": [_ranking_row("7203", "2024-12-01", "positive_reacted", 0.02)],
        "positive_unreacted": [_ranking_row("6758", "2024-12-01", "positive_unreacted", -0.01)],
        "negative_reacted": [_ranking_row("8306", "2024-12-02", "negative_reacted", -0.03)],
        "negative_reversal": [_ranking_row("9984", "2024-12-02", "negative_reversal", 0.04)],
        "excluded": [
            {
                "DiscDate": "2024-12-02",
                "ticker": "1111",
                "company_name": "Excluded",
                "sector33_name": "銀行業",
                "market_segment": "プライム",
                "topix_size_category": "TOPIX Large70",
                "liquidity_bucket": "low",
                "abnormal_ret_1d": None,
                "ret_1d": None,
                "exclusion_reasons": "low_liquidity",
            }
        ],
        "unclassified": [
            {
                "DiscDate": "2024-12-02",
                "ticker": "2222",
                "company_name": "Unclassified",
                "sector33_name": "小売業",
                "market_segment": "スタンダード",
                "topix_size_category": "TOPIX Small 1",
                "liquidity_bucket": "mid",
                "abnormal_ret_1d": 0,
                "ret_1d": 0.01,
                "unclassified_reason": "signal_not_positive_or_negative",
            }
        ],
    }
    for key, rows in data.items():
        pd.DataFrame(rows).to_csv(base / INPUT_FILES[key], index=False)


def _write_raw(price_store: Path, financial_summary: Path) -> None:
    pd.DataFrame(
        [
            {"ticker": "7203", "date": "2024-12-01", "adj_close": 100},
            {"ticker": "7203", "date": "2024-12-03", "adj_close": 101},
        ]
    ).to_csv(price_store, index=False)
    pd.DataFrame(
        [
            {"DiscDate": "2024-12-01", "ticker": "7203"},
            {"DiscDate": "2024-12-04", "ticker": "6758"},
        ]
    ).to_csv(financial_summary, index=False)


def test_daily_dashboard_outputs_counts_snapshot_and_group_summaries(tmp_path):
    analysis = tmp_path / "analysis"
    output = tmp_path / "out"
    analysis.mkdir()
    price_store = tmp_path / "prices_daily_bars.csv"
    financial_summary = tmp_path / "financial_summary_raw.csv"
    _write_inputs(analysis)
    _write_raw(price_store, financial_summary)
    before_price = _sha(price_store)
    before_fin = _sha(financial_summary)

    audit = build_daily_dashboard(analysis, price_store, financial_summary, output, latest_top_n=5)

    assert _sha(price_store) == before_price
    assert _sha(financial_summary) == before_fin
    summary = pd.read_csv(output / OUTPUT_FILES["dashboard_summary_csv"])
    assert len(summary) == 2
    assert int(summary["all_dashboard_rows"].sum()) == 6
    assert int(summary["total_classified_rows"].sum()) == 4
    assert int(summary["excluded_rows"].sum()) == 1
    assert int(summary["unclassified_rows"].sum()) == 1
    assert audit["total_dashboard_rows"] == 6

    snapshot = json.loads((output / OUTPUT_FILES["latest_snapshot_json"]).read_text())
    audit_json = json.loads((output / OUTPUT_FILES["audit_json"]).read_text())
    assert snapshot["latest_disc_date"] == "2024-12-02"
    assert snapshot["latest_price_date"] == "2024-12-03"
    assert snapshot["latest_financial_summary_disc_date"] == "2024-12-04"
    assert snapshot["total_dashboard_rows"] == audit_json["total_dashboard_rows"] == 6
    assert snapshot["total_unique_tickers"] == audit_json["total_unique_tickers"] == 6
    assert snapshot["latest_date_total_rows"] == 4
    assert snapshot["latest_date_unique_tickers"] == 4
    assert snapshot["latest_date_counts"]["negative_reacted"] == 1

    assert len(pd.read_csv(output / OUTPUT_FILES["sector_summary_csv"])) >= 1
    assert len(pd.read_csv(output / OUTPUT_FILES["market_segment_summary_csv"])) >= 1
    assert len(pd.read_csv(output / OUTPUT_FILES["topix_size_summary_csv"])) >= 1


def test_daily_dashboard_warns_on_missing_columns_and_handles_empty_inputs(tmp_path):
    analysis = tmp_path / "analysis"
    output = tmp_path / "out"
    analysis.mkdir()
    price_store = tmp_path / "prices_daily_bars.csv"
    financial_summary = tmp_path / "financial_summary_raw.csv"
    _write_raw(price_store, financial_summary)
    pd.DataFrame([{"DiscDate": "2024-12-01", "ticker": "7203"}]).to_csv(
        analysis / INPUT_FILES["positive_reacted"], index=False
    )
    for key, filename in INPUT_FILES.items():
        if key != "positive_reacted":
            pd.DataFrame(columns=["DiscDate", "ticker"]).to_csv(analysis / filename, index=False)

    audit = build_daily_dashboard(analysis, price_store, financial_summary, output)

    assert audit["total_dashboard_rows"] == 1
    snapshot = json.loads((output / OUTPUT_FILES["latest_snapshot_json"]).read_text())
    audit_json = json.loads((output / OUTPUT_FILES["audit_json"]).read_text())
    assert snapshot["total_dashboard_rows"] == audit_json["total_dashboard_rows"] == 1
    assert snapshot["total_unique_tickers"] == audit_json["total_unique_tickers"] == 1
    assert "latest_date_counts" in snapshot
    assert audit["missing_columns"]
    assert audit["warnings"]
    assert (output / OUTPUT_FILES["dashboard_summary_json"]).exists()

    empty_analysis = tmp_path / "empty_analysis"
    empty_output = tmp_path / "empty_out"
    empty_analysis.mkdir()
    empty_audit = build_daily_dashboard(empty_analysis, price_store, financial_summary, empty_output)
    empty_summary = pd.read_csv(empty_output / OUTPUT_FILES["dashboard_summary_csv"])
    assert empty_summary.empty
    assert empty_audit["total_dashboard_rows"] == 0
    assert len(empty_audit["missing_input_files"]) == len(INPUT_FILES)

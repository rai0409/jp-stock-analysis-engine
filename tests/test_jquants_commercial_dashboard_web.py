from __future__ import annotations

import json
import asyncio
from urllib.parse import urlsplit

import pandas as pd
import pytest
from fastapi import FastAPI

from jp_stock_analysis.web.commercial_dashboard_app import create_app


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture()
def dashboard_analysis_dir(tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    (analysis_dir / "commercial_daily_latest_snapshot.json").write_text(
        json.dumps(
            {
                "generated_at_jst": "2026-07-03T08:00:00+09:00",
                "latest_disc_date": "2026-03-26",
                "latest_price_date": "2026-03-26",
                "latest_financial_summary_disc_date": "2026-03-26",
                "total_dashboard_rows": 7,
                "total_unique_tickers": 5,
                "latest_date_total_rows": 2,
                "latest_date_unique_tickers": 2,
                "latest_date_counts": {"positive_reacted": 1, "unclassified": 1},
                "total_counts_by_classification": {
                    "positive_reacted": 2,
                    "positive_unreacted": 1,
                    "negative_reacted": 1,
                    "negative_reversal": 1,
                    "excluded": 1,
                    "unclassified": 1,
                },
                "warnings": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "commercial_daily_dashboard_audit.json").write_text(
        json.dumps(
            {
                "input_row_counts": {"positive_reacted": 2},
                "output_row_counts": {"summary": 2},
                "missing_input_files": [],
                "missing_columns": {},
                "warnings": [],
                "latest_price_date": "2026-03-26",
                "latest_financial_summary_disc_date": "2026-03-26",
                "latest_disc_date": "2026-03-26",
                "total_dashboard_rows": 7,
                "total_unique_tickers": 5,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _write_csv(
        analysis_dir / "commercial_daily_dashboard_summary.csv",
        [
            {
                "DiscDate": "2026-03-25",
                "total_classified_rows": 4,
                "positive_reacted_rows": 1,
                "all_dashboard_rows": 5,
                "unique_tickers": 4,
                "avg_abnormal_ret_1d": 0.01,
            },
            {
                "DiscDate": "2026-03-26",
                "total_classified_rows": 2,
                "positive_reacted_rows": 1,
                "all_dashboard_rows": 2,
                "unique_tickers": 2,
                "avg_abnormal_ret_1d": 0.02,
            },
        ],
    )
    _write_csv(
        analysis_dir / "commercial_daily_sector_summary.csv",
        [
            {
                "DiscDate": "2026-03-26",
                "dashboard_classification": "positive_reacted",
                "sector33_name": "Transportation Equipment",
                "rows": 2,
                "unique_tickers": 2,
                "avg_abnormal_ret_1d": 0.03,
            }
        ],
    )
    _write_csv(
        analysis_dir / "commercial_daily_market_segment_summary.csv",
        [
            {
                "DiscDate": "2026-03-26",
                "dashboard_classification": "positive_reacted",
                "market_segment": "Prime",
                "rows": 2,
                "unique_tickers": 2,
                "avg_abnormal_ret_1d": 0.03,
            }
        ],
    )
    _write_csv(
        analysis_dir / "commercial_daily_topix_size_summary.csv",
        [
            {
                "DiscDate": "2026-03-26",
                "dashboard_classification": "positive_reacted",
                "topix_size_category": "TOPIX Large70",
                "rows": 2,
                "unique_tickers": 2,
                "avg_abnormal_ret_1d": 0.03,
            }
        ],
    )

    base_ranking_row = {
        "rank": 1,
        "ranking_label": "positive_reacted",
        "ticker": "7203",
        "company_name": "Toyota Motor",
        "DiscDate": "2026-03-26",
        "first_time": "08:30",
        "sector33_name": "Transportation Equipment",
        "market_segment": "Prime",
        "topix_size_category": "TOPIX Large70",
        "candidate_bucket": "quality_screening",
        "ranking_score": 0.9,
        "abnormal_ret_1d": 0.04,
        "ret_1d": 0.05,
        "liquidity_bucket": "high",
        "is_liquid_enough": True,
        "price_reaction_status_v2": "full_20d_available",
        "commercial_signal": "upward_revision",
    }
    _write_csv(
        analysis_dir / "commercial_quality_positive_reacted_ranking_enriched.csv",
        [
            base_ranking_row,
            {**base_ranking_row, "rank": 2, "ticker": "6758", "company_name": "Sony Group", "abnormal_ret_1d": None},
        ],
    )
    for classification, filename in {
        "positive_unreacted": "commercial_quality_positive_unreacted_ranking_enriched.csv",
        "negative_reacted": "commercial_quality_negative_reacted_ranking_enriched.csv",
        "negative_reversal": "commercial_quality_negative_reversal_ranking_enriched.csv",
    }.items():
        row = {**base_ranking_row, "ranking_label": classification, "ticker": "8306", "company_name": "MUFG"}
        row.pop("topix_size_category")
        _write_csv(analysis_dir / filename, [row])

    _write_csv(
        analysis_dir / "commercial_quality_rankings_excluded_audit_enriched.csv",
        [{"ticker": "1111", "company_name": "Excluded Co", "DiscDate": "2026-03-25", "liquidity_bucket": "low"}],
    )
    _write_csv(
        analysis_dir / "commercial_quality_rankings_unclassified_audit_enriched.csv",
        [{"ticker": "2222", "company_name": "Unclassified Co", "DiscDate": "2026-03-26", "liquidity_bucket": "mid"}],
    )
    (analysis_dir / "commercial_daily_dashboard_summary.json").write_text("{}", encoding="utf-8")
    (analysis_dir / "commercial_quality_rankings_enriched_summary.json").write_text("{}", encoding="utf-8")
    (analysis_dir / "commercial_quality_improvement_final_summary.json").write_text("{}", encoding="utf-8")

    return analysis_dir


@pytest.fixture()
def dashboard_app(dashboard_analysis_dir):
    return create_app(dashboard_analysis_dir)


def _request(app: FastAPI, target: str) -> tuple[int, bytes, dict[str, str]]:
    async def run_request():
        parsed = urlsplit(target)
        messages = []
        request_sent = False

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": parsed.path,
            "raw_path": parsed.path.encode("utf-8"),
            "query_string": parsed.query.encode("utf-8"),
            "headers": [(b"host", b"testserver")],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "root_path": "",
        }

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.sleep(3600)
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        await asyncio.wait_for(app(scope, receive, send), timeout=5)
        return messages

    messages = asyncio.run(run_request())
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in start.get("headers", [])}
    return int(start["status"]), body, headers


def _json(app: FastAPI, path: str):
    status, body, _ = _request(app, path)
    assert status == 200
    return json.loads(body.decode("utf-8"))


def _text(app: FastAPI, path: str):
    status, body, _ = _request(app, path)
    assert status == 200
    return body.decode("utf-8")


def test_create_app_returns_fastapi(dashboard_analysis_dir):
    assert isinstance(create_app(dashboard_analysis_dir), FastAPI)


def test_health_returns_snapshot_status(dashboard_app):
    payload = _json(dashboard_app, "/health")
    assert payload["status"] == "ok"
    assert payload["total_dashboard_rows"] == 7


def test_snapshot_api_returns_total_dashboard_rows(dashboard_app):
    assert _json(dashboard_app, "/api/snapshot")["total_dashboard_rows"] == 7


def test_rankings_api_returns_rows_and_filters_without_optional_columns(dashboard_app):
    payload = _json(dashboard_app, "/api/rankings/positive_reacted?q=toyota&limit=5")
    assert payload["classification"] == "positive_reacted"
    assert payload["total"] == 1
    assert payload["rows"][0]["ticker"] == "7203"

    missing_optional = _json(
        dashboard_app, "/api/rankings/positive_unreacted?topix_size_category=TOPIX%20Large70"
    )
    assert missing_optional["rows"] == []


def test_invalid_ranking_classification_returns_404(dashboard_app):
    status, _, _ = _request(dashboard_app, "/api/rankings/invalid")
    assert status == 404


def test_daily_summary_returns_rows_and_total(dashboard_app):
    payload = _json(dashboard_app, "/api/daily-summary?limit=1")
    assert payload["total"] == 2
    assert len(payload["rows"]) == 1


def test_group_summary_endpoints_return_rows_and_total(dashboard_app):
    for path in ["/api/sector-summary", "/api/market-segment-summary", "/api/topix-size-summary"]:
        payload = _json(dashboard_app, path)
        assert payload["total"] == 1
        assert len(payload["rows"]) == 1


def test_download_allowlist_and_path_traversal(dashboard_app):
    status, _, _ = _request(dashboard_app, "/download/commercial_daily_dashboard_summary.csv")
    assert status == 200

    status, _, _ = _request(dashboard_app, "/download/../secret")
    assert status in {400, 404}


def test_json_responses_do_not_emit_nan(dashboard_app):
    payload = _json(dashboard_app, "/api/rankings/positive_reacted")
    assert payload["rows"][1]["abnormal_ret_1d"] is None
    json.dumps(payload, allow_nan=False)


def test_html_dashboard_renders_disclaimer_and_avoids_advice_terms(dashboard_app):
    text = _text(dashboard_app, "/")
    assert "Screening support only" in text
    assert "Not investment advice" in text

    lower = text.lower()
    for forbidden in ["buy", "sell", "recommend", "target price", "推奨", "買い", "売り", "目標株価"]:
        assert forbidden not in lower

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "schemas"

EXPECTED_REQUIRED = {
    "disclosure_event.schema.json": {
        "schema_version",
        "source",
        "disclosure_id",
        "local_code",
        "edinet_code",
        "issuer_name",
        "document_type",
        "event_type",
        "event_direction",
        "confidence",
        "disclosed_at",
        "available_from_timestamp",
        "raw_uri",
        "content_hash",
        "revision_of",
        "created_at",
    },
    "price_bar.schema.json": {
        "schema_version",
        "source",
        "local_code",
        "price_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover_value",
        "adjustment_factor",
        "is_adjusted",
        "available_from_timestamp",
        "created_at",
    },
    "security_master.schema.json": {
        "schema_version",
        "local_code",
        "edinet_code",
        "company_name",
        "market_segment",
        "sector33_code",
        "sector33_name",
        "topix_size_category",
        "listed_date",
        "delisted_date",
        "effective_from",
        "effective_to",
        "created_at",
    },
    "point_in_time_feature.schema.json": {
        "schema_version",
        "local_code",
        "feature_date",
        "asof_timestamp",
        "available_from_timestamp",
        "feature_cutoff_timestamp",
        "source_event_ids",
        "feature_name",
        "feature_value",
        "feature_type",
        "feature_group",
        "leakage_safe_flag",
        "leakage_status",
        "created_at",
    },
    "signal_output.schema.json": {
        "schema_version",
        "local_code",
        "signal_date",
        "asof_timestamp",
        "feature_cutoff_timestamp",
        "label_start_timestamp",
        "label_end_timestamp",
        "signal_name",
        "signal_score",
        "signal_direction",
        "ranking_bucket",
        "validation_status",
        "leakage_safe_flag",
        "cost_adjusted_available",
        "capacity_jpy",
        "research_only_flag",
        "created_at",
    },
}

SAMPLE_PAYLOADS = {
    "disclosure_event.schema.json": {
        "schema_version": "1.0.0",
        "source": "topix1000_disclosure_platform.gold",
        "disclosure_id": "S100TEST1",
        "local_code": "7203",
        "edinet_code": "E00001",
        "issuer_name": "Sample Company",
        "document_type": "annual_securities_report",
        "event_type": "earnings_revision",
        "event_direction": "positive",
        "confidence": 0.83,
        "disclosed_at": "2026-06-04T06:00:00Z",
        "available_from_timestamp": "2026-06-04T06:05:00Z",
        "raw_uri": "gold/disclosures/S100TEST1.json",
        "content_hash": "sha256:example",
        "revision_of": None,
        "created_at": "2026-06-04T06:10:00Z",
    },
    "price_bar.schema.json": {
        "schema_version": "1.0.0",
        "source": "topix1000_disclosure_platform.gold",
        "local_code": "7203",
        "price_date": "2026-06-04",
        "open": 3000.0,
        "high": 3050.0,
        "low": 2980.0,
        "close": 3025.0,
        "volume": 1000000,
        "turnover_value": 3025000000.0,
        "adjustment_factor": 1.0,
        "is_adjusted": True,
        "available_from_timestamp": "2026-06-04T06:30:00Z",
        "created_at": "2026-06-04T06:35:00Z",
    },
    "security_master.schema.json": {
        "schema_version": "1.0.0",
        "local_code": "7203",
        "edinet_code": "E00001",
        "company_name": "Sample Company",
        "market_segment": "Prime",
        "sector33_code": "3700",
        "sector33_name": "Transportation Equipment",
        "topix_size_category": "TOPIX Core30",
        "listed_date": "1949-05-16",
        "delisted_date": None,
        "effective_from": "2026-06-01T00:00:00Z",
        "effective_to": None,
        "created_at": "2026-06-01T00:05:00Z",
    },
    "point_in_time_feature.schema.json": {
        "schema_version": "1.0.0",
        "local_code": "7203",
        "feature_date": "2026-06-05",
        "asof_timestamp": "2026-06-05T00:00:00Z",
        "available_from_timestamp": "2026-06-05T00:00:00Z",
        "feature_cutoff_timestamp": "2026-06-04T23:59:59Z",
        "source_event_ids": ["S100TEST1"],
        "feature_name": "disclosure_sentiment_score",
        "feature_value": 0.42,
        "feature_type": "numeric",
        "feature_group": "disclosure",
        "leakage_safe_flag": True,
        "leakage_status": "pass",
        "created_at": "2026-06-05T00:01:00Z",
    },
    "signal_output.schema.json": {
        "schema_version": "1.0.0",
        "local_code": "7203",
        "signal_date": "2026-06-05",
        "asof_timestamp": "2026-06-05T00:00:00Z",
        "feature_cutoff_timestamp": "2026-06-04T23:59:59Z",
        "label_start_timestamp": "2026-06-05T00:00:00Z",
        "label_end_timestamp": "2026-06-12T00:00:00Z",
        "signal_name": "research_quality_rank",
        "signal_score": 0.67,
        "signal_direction": "positive",
        "ranking_bucket": 1,
        "validation_status": "analyst_review_required",
        "leakage_safe_flag": True,
        "cost_adjusted_available": False,
        "capacity_jpy": None,
        "research_only_flag": True,
        "created_at": "2026-06-05T00:02:00Z",
    },
}


def _load_schema(schema_name: str) -> dict:
    with (SCHEMA_DIR / schema_name).open(encoding="utf-8") as file:
        return json.load(file)


def _assert_required_fields_present(schema: dict, payload: dict) -> None:
    missing = set(schema["required"]) - set(payload)
    assert not missing


def test_contract_schemas_are_valid_json_and_use_draft_2020_12():
    for schema_name in EXPECTED_REQUIRED:
        schema = _load_schema(schema_name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert "properties" in schema


def test_contract_schemas_include_expected_required_fields():
    for schema_name, expected_required in EXPECTED_REQUIRED.items():
        schema = _load_schema(schema_name)
        assert set(schema["required"]) == expected_required
        assert expected_required <= set(schema["properties"])


def test_sample_payloads_satisfy_required_fields():
    for schema_name, payload in SAMPLE_PAYLOADS.items():
        schema = _load_schema(schema_name)
        _assert_required_fields_present(schema, payload)


def test_point_in_time_feature_requires_leakage_controls():
    schema = _load_schema("point_in_time_feature.schema.json")
    required = set(schema["required"])

    assert {
        "asof_timestamp",
        "available_from_timestamp",
        "feature_cutoff_timestamp",
        "leakage_safe_flag",
        "leakage_status",
    } <= required


def test_signal_output_requires_research_only_point_in_time_controls():
    schema = _load_schema("signal_output.schema.json")
    required = set(schema["required"])

    assert {
        "asof_timestamp",
        "feature_cutoff_timestamp",
        "label_start_timestamp",
        "label_end_timestamp",
        "validation_status",
        "leakage_safe_flag",
        "research_only_flag",
    } <= required
    assert schema["properties"]["research_only_flag"]["const"] is True

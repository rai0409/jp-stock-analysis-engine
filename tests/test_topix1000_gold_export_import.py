from __future__ import annotations

import json

import pytest

from jp_stock_analysis.errors import ProviderError
from jp_stock_analysis.providers.topix1000_gold_export import (
    DATASET_SPECS,
    Topix1000GoldExportProvider,
    _load_required_fields,
)


def test_loads_gold_export_fixture_dataframes(fixtures_dir):
    provider = Topix1000GoldExportProvider(fixtures_dir / "topix1000_gold_export")

    disclosures = provider.load_disclosure_events()
    prices = provider.load_price_bars()
    security_master = provider.load_security_master()
    features = provider.load_point_in_time_features()
    signals = provider.load_dataset("signal_outputs", filename="signal_outputs.json")

    assert list(disclosures["local_code"]) == ["7203", "6758"]
    assert set(prices["local_code"]) == {"7203", "6758"}
    assert set(security_master["local_code"]) == {"7203", "6758"}
    assert list(features["source_event_ids"])[0] == ["S100GOLD1"]
    assert signals.loc[0, "research_only_flag"] is True


def test_required_fields_and_provenance_are_preserved(fixtures_dir):
    provider = Topix1000GoldExportProvider(fixtures_dir / "topix1000_gold_export")
    disclosures = provider.load_disclosure_events()
    features = provider.load_point_in_time_features()

    row = disclosures.set_index("local_code").loc["7203"]
    assert row["disclosure_id"] == "S100GOLD1"
    assert row["content_hash"] == "sha256:gold-disclosure-1"
    assert row["raw_uri"] == "gold/disclosures/S100GOLD1.json"
    assert row["revision_of"] is None

    feature = features.set_index("local_code").loc["7203"]
    assert feature["source_event_ids"] == ["S100GOLD1"]
    assert feature["feature_cutoff_timestamp"] == "2026-06-04T23:59:59Z"
    assert feature["leakage_safe_flag"] is True
    assert feature["leakage_status"] == "pass"


def test_missing_required_point_in_time_field_fails(tmp_path):
    export_dir = tmp_path / "gold"
    export_dir.mkdir()
    bad_record = {
        "schema_version": "1.0.0",
        "local_code": "7203",
        "feature_date": "2026-06-05",
        "asof_timestamp": "2026-06-05T00:00:00Z",
        "available_from_timestamp": "2026-06-04T06:05:00Z",
        "source_event_ids": ["S100GOLD1"],
        "feature_name": "disclosure_event_confidence",
        "feature_value": 0.82,
        "feature_type": "numeric",
        "feature_group": "disclosure",
        "leakage_safe_flag": True,
        "leakage_status": "pass",
        "created_at": "2026-06-05T00:01:00Z",
    }
    (export_dir / "point_in_time_features.jsonl").write_text(
        json.dumps(bad_record) + "\n", encoding="utf-8"
    )

    provider = Topix1000GoldExportProvider(export_dir)
    with pytest.raises(ProviderError, match="feature_cutoff_timestamp"):
        provider.load_point_in_time_features()


def test_unsafe_leakage_controls_fail_in_strict_mode(tmp_path):
    export_dir = tmp_path / "gold"
    export_dir.mkdir()
    unsafe_record = {
        "schema_version": "1.0.0",
        "local_code": "7203",
        "feature_date": "2026-06-05",
        "asof_timestamp": "2026-06-05T00:00:00Z",
        "available_from_timestamp": "2026-06-04T06:05:00Z",
        "feature_cutoff_timestamp": "2026-06-04T23:59:59Z",
        "source_event_ids": ["S100GOLD1"],
        "feature_name": "disclosure_event_confidence",
        "feature_value": 0.82,
        "feature_type": "numeric",
        "feature_group": "disclosure",
        "leakage_safe_flag": False,
        "leakage_status": "fail",
        "created_at": "2026-06-05T00:01:00Z",
    }
    (export_dir / "point_in_time_features.jsonl").write_text(
        json.dumps(unsafe_record) + "\n", encoding="utf-8"
    )

    provider = Topix1000GoldExportProvider(export_dir)
    with pytest.raises(ProviderError, match="leakage controls"):
        provider.load_point_in_time_features()


def test_disclosure_price_and_security_master_join_by_local_code(fixtures_dir):
    provider = Topix1000GoldExportProvider(fixtures_dir / "topix1000_gold_export")

    joined = (
        provider.load_disclosure_events()
        .merge(provider.load_price_bars(), on="local_code", suffixes=("_disclosure", "_price"))
        .merge(provider.load_security_master(), on="local_code")
    )

    assert set(joined["local_code"]) == {"7203", "6758"}
    assert {"disclosure_id", "price_date", "company_name"} <= set(joined.columns)


def test_adapter_required_fields_match_existing_schema_files():
    assert _load_required_fields(DATASET_SPECS["disclosure_events"].schema_filename) == {
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
    }
    assert {
        "asof_timestamp",
        "available_from_timestamp",
        "feature_cutoff_timestamp",
        "leakage_safe_flag",
        "leakage_status",
    } <= _load_required_fields(DATASET_SPECS["point_in_time_features"].schema_filename)

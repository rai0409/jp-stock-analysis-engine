"""Tests for the topix1000_disclosure_platform export provider.

All inputs are synthetic fixtures under tests/fixtures/topix1000_export/.
No network, no database, no real EDINET content.
"""

from __future__ import annotations

import json

import pytest

from jp_stock_analysis.analysis.disclosure_nlp import RuleBasedDisclosureAnalyzer
from jp_stock_analysis.cli import main
from jp_stock_analysis.errors import ProviderError
from jp_stock_analysis.providers.topix1000_export import Topix1000ExportProvider


@pytest.fixture
def export_dir(fixtures_dir):
    return fixtures_dir / "topix1000_export"


def _write_export(root, documents, index=None):
    """Build a minimal export directory under ``root``."""
    (root / "disclosures").mkdir(parents=True, exist_ok=True)
    for doc in documents:
        path = root / "disclosures" / f"{doc['doc_id']}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    if index is None:
        index = {"documents": [{"doc_id": doc["doc_id"]} for doc in documents]}
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def test_loads_fixture_documents_keyed_by_normalized_ticker(export_dir):
    provider = Topix1000ExportProvider(export_dir)
    documents = provider.load_documents()
    # 5-digit EDINET sec_code "72030"/"67580" normalize to listing codes
    assert set(documents) == {"7203", "6758"}
    assert provider.warnings == []


def test_maps_metadata_into_document_and_source_metadata(export_dir):
    document = Topix1000ExportProvider(export_dir).load_documents()["7203"]
    assert document.doc_id == "S100TEST1"
    assert document.edinet_code == "E00001"
    assert document.company_name == "サンプル自動車株式会社"
    assert document.document_type == "annual_securities_report"
    assert document.fiscal_year == 2025
    assert document.source.endswith("S100TEST1.json")
    meta = document.source_metadata
    assert meta["doc_id"] == "S100TEST1"
    assert meta["target_date"] == "2026-06-04"
    assert meta["filing_type_key"] == "annual_securities_report"
    assert meta["form_code"] == "030000"
    assert meta["doc_type_code"] == "120"
    assert meta["source_path.original_zip"] == "raw/S100TEST1/original.zip"
    assert meta["source_path.document_pdf"] == "raw/S100TEST1/document.pdf"
    assert meta["source_path.csv_zip"] == "raw/S100TEST1/csv.zip"
    assert meta["xbrl.facts_count"] == "1234"
    # unknown optional fields ("future_unknown_field", "key_facts_sample") tolerated


def test_text_present_flows_into_disclosure_nlp(export_dir):
    document = Topix1000ExportProvider(export_dir).load_documents()["7203"]
    assert "増収" in document.text
    assert document.warnings == []
    result = RuleBasedDisclosureAnalyzer().analyze(document)
    assert result.findings
    assert result.confidence_score > 0
    assert result.document_type == "annual_securities_report"
    assert result.fiscal_year == 2025
    assert result.source_metadata["doc_id"] == "S100TEST1"
    assert result.source_metadata["source_path.document_pdf"] == "raw/S100TEST1/document.pdf"


def test_null_text_becomes_metadata_only_with_low_confidence(export_dir):
    document = Topix1000ExportProvider(export_dir).load_documents()["6758"]
    assert document.text == ""  # never fabricated
    assert any("no extracted text" in warning for warning in document.warnings)
    assert document.doc_id == "S100TEST2"
    result = RuleBasedDisclosureAnalyzer().analyze(document)
    assert result.findings == []
    assert result.confidence_score == 0.0
    assert any("no extracted text" in warning for warning in result.warnings)
    assert any("empty disclosure text" in warning for warning in result.warnings)
    assert result.source_metadata["doc_id"] == "S100TEST2"


def test_get_disclosures_satisfies_provider_protocol(export_dir):
    provider = Topix1000ExportProvider(export_dir)
    documents = provider.get_disclosures("7203")
    assert [d.doc_id for d in documents] == ["S100TEST1"]
    assert provider.get_disclosures("9999") == []


def test_missing_export_dir_raises_clear_error(tmp_path):
    with pytest.raises(ProviderError, match="export directory not found"):
        Topix1000ExportProvider(tmp_path / "nope")


def test_missing_index_raises_clear_error(tmp_path):
    with pytest.raises(ProviderError, match="index.json"):
        Topix1000ExportProvider(tmp_path)


def test_invalid_index_json_raises_clear_error(tmp_path):
    (tmp_path / "index.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ProviderError, match="invalid JSON"):
        Topix1000ExportProvider(tmp_path)


def test_index_without_document_list_raises_clear_error(tmp_path):
    (tmp_path / "index.json").write_text('{"documents": "oops"}', encoding="utf-8")
    with pytest.raises(ProviderError, match="'documents' list"):
        Topix1000ExportProvider(tmp_path)


def test_missing_per_document_file_raises_clear_error(tmp_path):
    (tmp_path / "index.json").write_text(
        '{"documents": [{"doc_id": "S100GONE"}]}', encoding="utf-8"
    )
    provider = Topix1000ExportProvider(tmp_path)
    with pytest.raises(ProviderError, match="S100GONE"):
        provider.load_documents()


def test_invalid_per_document_json_raises_clear_error(tmp_path):
    _write_export(tmp_path, [])
    (tmp_path / "index.json").write_text(
        '{"documents": [{"doc_id": "S100BAD"}]}', encoding="utf-8"
    )
    (tmp_path / "disclosures" / "S100BAD.json").write_text("[broken", encoding="utf-8")
    with pytest.raises(ProviderError, match="invalid JSON"):
        Topix1000ExportProvider(tmp_path).load_documents()


def test_document_without_any_code_raises_clear_error(tmp_path):
    _write_export(tmp_path, [{"doc_id": "S100NOCODE", "text": "増収"}])
    with pytest.raises(ProviderError, match="no ticker, sec_code, or edinet_code"):
        Topix1000ExportProvider(tmp_path).load_documents()


def test_empty_export_warns_and_returns_no_documents(tmp_path):
    (tmp_path / "index.json").write_text('{"documents": []}', encoding="utf-8")
    provider = Topix1000ExportProvider(tmp_path)
    assert provider.load_documents() == {}
    assert any("no documents" in warning for warning in provider.warnings)


def test_ticker_field_wins_and_edinet_code_is_fallback(tmp_path):
    _write_export(
        tmp_path,
        [
            {"doc_id": "S100A", "ticker": "7203", "sec_code": "99990", "text": "増収"},
            {"doc_id": "S100B", "edinet_code": "E99999", "text": None},
        ],
    )
    documents = Topix1000ExportProvider(tmp_path).load_documents()
    assert set(documents) == {"7203", "E99999"}
    assert any("keyed by EDINET code" in w for w in documents["E99999"].warnings)


def test_duplicate_ticker_selects_latest_target_date_deterministically(tmp_path):
    _write_export(
        tmp_path,
        [
            {"doc_id": "S100OLD", "ticker": "7203", "target_date": "2025-06-04", "text": "減収"},
            {"doc_id": "S100NEW", "ticker": "7203", "target_date": "2026-06-04", "text": "増収"},
        ],
    )
    provider = Topix1000ExportProvider(tmp_path)
    documents = provider.load_documents()
    assert documents["7203"].doc_id == "S100NEW"
    assert any("S100OLD" in warning for warning in provider.warnings)
    assert len(provider.get_disclosures("7203")) == 2


def test_cli_smoke_with_topix1000_export(fixtures_dir, export_dir, tmp_path):
    argv = [
        "analyze",
        "--prices", str(fixtures_dir / "prices_sample.csv"),
        "--fundamentals", str(fixtures_dir / "fundamentals_sample.csv"),
        "--metadata", str(fixtures_dir / "company_metadata_sample.csv"),
        "--disclosure-provider", "topix1000-export",
        "--topix1000-export-dir", str(export_dir),
        "--output-dir", str(tmp_path),
    ]
    assert main(argv) == 0
    payload = json.loads((tmp_path / "screening.json").read_text(encoding="utf-8"))
    assert payload["signal_mode"] == "analysis_only"
    by_ticker = {entry["ticker"]: entry for entry in payload["results"]}
    # 7203 has export text -> findings with export provenance
    disclosure = by_ticker["7203"]["disclosure"]
    assert disclosure["document_type"] == "annual_securities_report"
    assert disclosure["fiscal_year"] == 2025
    assert disclosure["findings"]
    assert disclosure["source_metadata"]["doc_id"] == "S100TEST1"
    # 6758 export text is null -> metadata-only, low confidence, no fabricated findings
    metadata_only = by_ticker["6758"]["disclosure"]
    assert metadata_only["findings"] == []
    assert metadata_only["confidence_score"] == 0.0
    assert metadata_only["source_metadata"]["doc_id"] == "S100TEST2"
    # 9984 has no export document at all
    assert by_ticker["9984"]["disclosure"] is None


def test_cli_topix1000_export_requires_export_dir(tmp_path):
    argv = [
        "analyze",
        "--prices", "unused.csv",
        "--disclosure-provider", "topix1000-export",
        "--output-dir", str(tmp_path),
    ]
    with pytest.raises(SystemExit):
        main(argv)


def test_cli_rejects_mixing_disclosures_dir_with_export_provider(fixtures_dir, tmp_path):
    argv = [
        "analyze",
        "--prices", str(fixtures_dir / "prices_sample.csv"),
        "--disclosures", str(fixtures_dir / "disclosures"),
        "--disclosure-provider", "topix1000-export",
        "--topix1000-export-dir", str(fixtures_dir / "topix1000_export"),
        "--output-dir", str(tmp_path),
    ]
    with pytest.raises(SystemExit):
        main(argv)


def test_cli_missing_export_dir_is_reported_not_traceback(fixtures_dir, tmp_path, capsys):
    argv = [
        "analyze",
        "--prices", str(fixtures_dir / "prices_sample.csv"),
        "--disclosure-provider", "topix1000-export",
        "--topix1000-export-dir", str(tmp_path / "missing"),
        "--output-dir", str(tmp_path / "out"),
    ]
    assert main(argv) == 1
    assert "export directory not found" in capsys.readouterr().err

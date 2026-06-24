"""No-network tests for the J-Quants listed master export."""

from __future__ import annotations

import csv
import json

from jp_stock_analysis.cli import main
from jp_stock_analysis.schemas import CompanyMetadata
from jp_stock_analysis.validation.jquants_listed_master import (
    OUTPUT_COLUMNS,
    export_jquants_listed_master_csv,
)


class FakeMetadataProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_metadata(self, ticker: str) -> CompanyMetadata | None:
        self.calls.append(ticker)
        if ticker == "167A":
            return CompanyMetadata(
                ticker=ticker,
                company_name="アルファ株式会社",
                sector="情報・通信業",
                market="グロース",
                source_metadata={
                    "source": "jquants",
                    "raw_code": "167A",
                    "company_name_en": "Alpha Inc.",
                    "sector_17": "情報通信・サービスその他",
                    "sector_33": "情報・通信業",
                    "market": "グロース",
                },
            )
        return None


def test_listed_master_export_schema_alpha_tickers_matching_report_and_secrets(
    tmp_path,
    monkeypatch,
):
    secret = "test-secret-value"
    monkeypatch.setenv("JQUANTS_API_KEY", secret)
    universe = tmp_path / "universe.csv"
    universe.write_text(
        "ticker,name,date,new_index_category\n"
        "167A,アルファ,2024-10-31,new\n"
        "7203,トヨタ,2024-10-31,existing\n",
        encoding="utf-8",
    )
    output = tmp_path / "listed.csv"
    report = tmp_path / "report.json"

    result = export_jquants_listed_master_csv(
        FakeMetadataProvider(),
        universe_file=universe,
        output_file=output,
        report_file=report,
        sleep_seconds=0,
        allow_network=False,
        endpoint_url_for_listed_info="https://api.jquants.com/v2/equities/master",
    )

    assert result.universe_count == 2
    assert result.matched_count == 1
    assert result.missing_count == 1
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0].keys() == set(OUTPUT_COLUMNS)
    assert list(rows[0].keys()) == OUTPUT_COLUMNS
    assert rows[0]["ticker"] == "167A"
    assert rows[0]["matched"] == "true"
    assert rows[0]["company_name"] == "アルファ株式会社"
    assert rows[0]["raw_code"] == "167A"
    assert rows[0]["company_name_en"] == "Alpha Inc."
    assert rows[0]["sector_17"] == "情報通信・サービスその他"
    assert rows[0]["sector_33"] == "情報・通信業"
    assert json.loads(rows[0]["source_metadata_json"])["raw_code"] == "167A"

    assert rows[1]["ticker"] == "7203"
    assert rows[1]["matched"] == "false"
    assert rows[1]["error"] == "missing_metadata"
    assert json.loads(rows[1]["source_metadata_json"]) == {}

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["universe_count"] == 2
    assert payload["matched_count"] == 1
    assert payload["missing_count"] == 1
    assert payload["missing_tickers"] == ["7203"]
    assert payload["matched_by_new_index_category"] == {"new": 1}
    assert payload["missing_by_new_index_category"] == {"existing": 1}
    assert payload["endpoint_url_for_listed_info"] == (
        "https://api.jquants.com/v2/equities/master"
    )
    assert payload["api_key_status"] == "PRESENT"
    assert payload["secret_included"] is False

    combined = output.read_text(encoding="utf-8") + report.read_text(encoding="utf-8")
    assert secret not in combined


def test_fetch_listed_master_cli_requires_allow_network(tmp_path, capsys):
    universe = tmp_path / "universe.csv"
    universe.write_text(
        "ticker,name,date,new_index_category\n167A,アルファ,2024-10-31,new\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "fetch-jquants-listed-master",
            "--universe-file",
            str(universe),
            "--output-file",
            str(tmp_path / "listed.csv"),
            "--report-file",
            str(tmp_path / "report.json"),
        ]
    )

    assert rc == 1
    assert "requires --allow-network" in capsys.readouterr().err

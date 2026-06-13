"""CLI smoke tests against the static fixtures. No network access."""

from __future__ import annotations

import csv
import json

from jp_stock_analysis.cli import main
from jp_stock_analysis.config import DEFAULT_DISCLAIMER

TICKERS = ("6758", "7203", "9984")


def _run(fixtures_dir, output_dir, mode=None) -> int:
    argv = [
        "analyze",
        "--prices", str(fixtures_dir / "prices_sample.csv"),
        "--fundamentals", str(fixtures_dir / "fundamentals_sample.csv"),
        "--metadata", str(fixtures_dir / "company_metadata_sample.csv"),
        "--disclosures", str(fixtures_dir / "disclosures"),
        "--output-dir", str(output_dir),
    ]
    if mode is not None:
        argv += ["--signal-mode", mode]
    return main(argv)


def test_analysis_only_smoke(fixtures_dir, tmp_path, capsys):
    assert _run(fixtures_dir, tmp_path) == 0
    assert str(tmp_path) in capsys.readouterr().out

    assert (tmp_path / "screening.csv").exists()
    assert (tmp_path / "screening.json").exists()
    for ticker in TICKERS:
        assert (tmp_path / f"{ticker}.md").exists()

    payload = json.loads((tmp_path / "screening.json").read_text(encoding="utf-8"))
    assert payload["disclaimer"] == DEFAULT_DISCLAIMER
    assert payload["signal_mode"] == "analysis_only"
    for entry in payload["results"] + payload["screening"]:
        assert "signal" not in entry
        assert "screening_label" not in entry

    report = (tmp_path / "7203.md").read_text(encoding="utf-8")
    assert DEFAULT_DISCLAIMER in report
    assert "## Research Signal" not in report
    assert "## Screening" not in report


def test_screening_mode_assigns_labels(fixtures_dir, tmp_path):
    assert _run(fixtures_dir, tmp_path, mode="screening") == 0
    with (tmp_path / "screening.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(row["screening_label"] for row in rows)
    assert "trade_signal" not in rows[0]
    # downtrending loss-maker must rank last
    assert rows[-1]["ticker"] == "9984"
    ranks = [int(row["rank"]) for row in rows]
    assert ranks == sorted(ranks)

    report = (tmp_path / "7203.md").read_text(encoding="utf-8")
    assert "## Screening" in report
    assert "## Research Signal" not in report


def test_trade_signal_mode_emits_signals(fixtures_dir, tmp_path):
    assert _run(fixtures_dir, tmp_path, mode="trade_signal") == 0
    payload = json.loads((tmp_path / "screening.json").read_text(encoding="utf-8"))
    assert payload["signal_mode"] == "trade_signal"
    signals = {entry["ticker"]: entry["signal"] for entry in payload["results"]}
    assert set(signals) == set(TICKERS)
    for signal in signals.values():
        assert signal["label"] in {
            "buy_signal", "hold_signal", "sell_signal",
            "watch_signal", "avoid_signal", "insufficient_data",
        }
        assert signal["disclaimer"] == DEFAULT_DISCLAIMER
        assert signal["thresholds_used"]
        assert signal["rationale"]
    # synthetic 9984 is a deteriorating loss-maker with critical disclosure language
    assert signals["9984"]["label"] in {"sell_signal", "avoid_signal"}

    report = (tmp_path / "7203.md").read_text(encoding="utf-8")
    assert "## Research Signal" in report
    assert "## Screening" in report


def test_rag_export_stable_paths_in_json(fixtures_dir, tmp_path):
    """Future RAG ingestion pins to these JSON paths; keep them stable."""
    assert _run(fixtures_dir, tmp_path, mode="trade_signal") == 0
    payload = json.loads((tmp_path / "screening.json").read_text(encoding="utf-8"))
    for entry in payload["results"]:
        assert entry["ticker"]
        assert "company_name" in entry
        assert entry["fundamentals"]["fiscal_year"] == 2024
        disclosure = entry["disclosure"]
        assert disclosure["document_type"] == "local_text"
        assert "fiscal_year" in disclosure  # null for local text files, but path is stable
        for finding in disclosure["findings"]:
            assert finding["evidence_text"]
            assert finding["summary"]
        positive = [f for f in disclosure["findings"] if f["category"] == "positive_factor"]
        assert isinstance(positive, list)  # positive_factors derivable by category filter
        assert entry["risks"]["flags"] is not None
        assert entry["score"]["reasons"]  # analysis summary equivalent
        assert "signal" in entry  # only present because trade_signal mode is enabled


def test_output_is_deterministic(fixtures_dir, tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    assert _run(fixtures_dir, first_dir) == 0
    assert _run(fixtures_dir, second_dir) == 0
    first = (first_dir / "screening.json").read_text(encoding="utf-8")
    second = (second_dir / "screening.json").read_text(encoding="utf-8")
    assert first == second


def test_fundamentals_only_ticker_gets_deterministic_date(fixtures_dir, tmp_path):
    extra = tmp_path / "fundamentals.csv"
    extra.write_text("ticker,fiscal_year,revenue\n1111,2024,1000\n", encoding="utf-8")
    argv = [
        "analyze",
        "--prices", str(fixtures_dir / "prices_sample.csv"),
        "--fundamentals", str(extra),
        "--output-dir", str(tmp_path / "out"),
    ]
    assert main(argv) == 0
    payload = json.loads((tmp_path / "out" / "screening.json").read_text(encoding="utf-8"))
    entry = next(e for e in payload["results"] if e["ticker"] == "1111")
    assert entry["analysis_date"] == "2024-12-31"  # derived from fiscal year, not today()
    assert entry["momentum"] is None


def test_prices_only_run_still_works(fixtures_dir, tmp_path):
    argv = [
        "analyze",
        "--prices", str(fixtures_dir / "prices_sample.csv"),
        "--output-dir", str(tmp_path),
    ]
    assert main(argv) == 0
    payload = json.loads((tmp_path / "screening.json").read_text(encoding="utf-8"))
    assert payload["result_count"] == 3
    for entry in payload["results"]:
        assert entry["fundamentals"] is None
        assert entry["warnings"]

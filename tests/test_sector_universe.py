"""Sector-relative validation against the larger synthetic J-Quants universe.

Universe: tests/fixtures/jquants_universe/ — 12 synthetic codes, 80 daily bars
each, two fiscal years of statements. Sectors: 輸送用機器 (5 companies,
strong→weak gradient 7001→7005), 電気機器 (4, 6501→6504), 情報・通信業
(2, 9001/9002), plus 9101 with no listed_info (missing-sector case).
All offline and deterministic; no network, no API key.
"""

from __future__ import annotations

import csv
import json

import pytest
from conftest import FIXTURES_DIR

from jp_stock_analysis.cli import analyze_data, main
from jp_stock_analysis.providers.jquants import ENV_API_KEY, JQuantsProvider

UNIVERSE_DIR = FIXTURES_DIR / "jquants_universe"

TRANSPORT = ["7001", "7002", "7003", "7004", "7005"]
ELECTRONICS = ["6501", "6502", "6503", "6504"]
INFOCOMM = ["9001", "9002"]
NO_METADATA = ["9101"]
ALL_CODES = TRANSPORT + ELECTRONICS + INFOCOMM + NO_METADATA

VALID_SIGNALS = {
    "buy_signal", "hold_signal", "sell_signal",
    "watch_signal", "avoid_signal", "insufficient_data",
}


def _run_universe(out_dir, mode=None, codes=ALL_CODES) -> dict:
    argv = ["analyze", "--provider", "jquants-cache",
            "--jquants-cache-dir", str(UNIVERSE_DIR)]
    for code in codes:
        argv += ["--jquants-code", code]
    argv += ["--output-dir", str(out_dir)]
    if mode is not None:
        argv += ["--signal-mode", mode]
    assert main(argv) == 0
    return json.loads((out_dir / "screening.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def universe(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("universe_out")
    payload = _run_universe(out_dir)
    return out_dir, payload


def test_universe_analyzes_offline(universe, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    _, payload = universe
    assert payload["result_count"] == 12
    assert {entry["ticker"] for entry in payload["results"]} == set(ALL_CODES)
    assert payload["signal_mode"] == "analysis_only"  # default unchanged


def test_peer_counts_match_sector_sizes(universe):
    _, payload = universe
    by_ticker = {entry["ticker"]: entry for entry in payload["results"]}
    for code in TRANSPORT:
        relative = by_ticker[code]["sector_relative"]
        assert relative["sector"] == "輸送用機器"
        assert relative["peer_count"] == 5
    for code in ELECTRONICS:
        assert by_ticker[code]["sector_relative"]["peer_count"] == 4
    for code in INFOCOMM:
        relative = by_ticker[code]["sector_relative"]
        assert relative["peer_count"] == 2
        assert any("small sector peer group" in w for w in relative["warnings"])
    for code in NO_METADATA:
        assert by_ticker[code]["sector_relative"] is None


def test_same_sector_ranking_follows_designed_gradient(universe):
    _, payload = universe
    by_ticker = {entry["ticker"]: entry for entry in payload["results"]}
    scores = {
        code: by_ticker[code]["sector_relative"]["sector_relative_score"]
        for code in TRANSPORT
    }
    assert all(score is not None for score in scores.values())
    assert max(scores, key=scores.get) == "7001"  # strongest by design
    assert min(scores, key=scores.get) == "7005"  # loss-maker by design
    assert scores["7001"] > scores["7003"] > scores["7005"]

    best = by_ticker["7001"]["sector_relative"]
    assert best["revenue_growth_percentile"] == 100.0
    assert best["roe_percentile"] == 100.0
    # 7005 has negative EPS: PER percentile must be None (missing), not faked
    worst = by_ticker["7005"]["sector_relative"]
    assert worst["per_percentile"] is None
    assert any("per_percentile" in w for w in worst["warnings"])

    electronics = {
        code: by_ticker[code]["sector_relative"]["sector_relative_score"]
        for code in ELECTRONICS
    }
    assert max(electronics, key=electronics.get) == "6501"
    assert min(electronics, key=electronics.get) == "6504"


def test_sector_relative_score_is_deterministic(universe, tmp_path):
    out_dir, _ = universe
    second = tmp_path / "rerun"
    _run_universe(second)
    assert (
        (out_dir / "screening.json").read_text(encoding="utf-8")
        == (second / "screening.json").read_text(encoding="utf-8")
    )


def test_final_score_unchanged_by_sector_attachment(tmp_path):
    provider = JQuantsProvider(cache_dir=UNIVERSE_DIR)
    prices = {code: provider.get_prices(code) for code in TRANSPORT}
    fundamentals = {code: provider.get_statements(code) for code in TRANSPORT}
    metadata = {
        code: company
        for code in TRANSPORT
        if (company := provider.get_metadata(code)) is not None
    }
    with_sectors = analyze_data(prices, fundamentals, metadata, {}, tmp_path / "a")
    without_sectors = analyze_data(prices, fundamentals, {}, {}, tmp_path / "b")

    finals_with = {r.ticker: r.score.final_score for r in with_sectors["results"]}
    finals_without = {r.ticker: r.score.final_score for r in without_sectors["results"]}
    assert finals_with == finals_without  # attachment never moves final_score
    assert all(r.sector_relative is not None for r in with_sectors["results"])
    assert all(r.sector_relative is None for r in without_sectors["results"])


def test_trade_signal_sector_factor_is_evidence_only(tmp_path):
    payload = _run_universe(tmp_path, mode="trade_signal")
    by_ticker = {entry["ticker"]: entry for entry in payload["results"]}

    def sector_factors(code: str) -> list[str]:
        return [
            factor
            for factor in by_ticker[code]["signal"]["supporting_factors"]
            if factor.startswith("sector_relative_score=")
        ]

    for entry in payload["results"]:
        signal = entry["signal"]
        assert signal["label"] in VALID_SIGNALS
        assert signal["thresholds_used"]
        assert signal["disclaimer"]
        # sector-relative must never become a threshold (no label influence)
        assert all("sector" not in key for key in signal["thresholds_used"])
        # any buy must rest on >=2 CORE (non-valuation, non-sector) factors
        if signal["label"] == "buy_signal":
            core = [
                factor
                for factor in signal["supporting_factors"]
                if not factor.startswith("sector_relative_score=")
            ]
            assert len(core) >= 2

    # 7001: eligible (score 92.9, 5 peers) -> factor present, marked evidence-only
    factors_7001 = sector_factors("7001")
    assert len(factors_7001) == 1
    assert "supporting evidence only" in factors_7001[0]
    assert "5 peers" in factors_7001[0]
    # 9001: high score but only 2 peers -> not eligible
    assert by_ticker["9001"]["sector_relative"]["sector_relative_score"] >= 70
    assert sector_factors("9001") == []
    # 7005: loss-maker is not rescued by sector data
    assert by_ticker["7005"]["signal"]["label"] in {
        "sell_signal", "avoid_signal", "insufficient_data"
    }
    assert sector_factors("7005") == []


def test_csv_blanks_sector_score_only_where_unavailable(universe):
    out_dir, _ = universe
    with (out_dir / "screening.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    by_ticker = {row["ticker"]: row for row in rows}
    assert by_ticker["9101"]["sector_relative_score"] == ""
    for code in TRANSPORT + ELECTRONICS + INFOCOMM:
        assert by_ticker[code]["sector_relative_score"] != ""
        float(by_ticker[code]["sector_relative_score"])  # parseable number


def test_markdown_section_only_for_sector_members(universe):
    out_dir, _ = universe
    assert "## Sector Relative" in (out_dir / "7001.md").read_text(encoding="utf-8")
    assert "## Sector Relative" not in (out_dir / "9101.md").read_text(encoding="utf-8")

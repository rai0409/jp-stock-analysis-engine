"""Tests for sector-relative scoring. Deterministic, offline."""

from __future__ import annotations

import csv
import json
from datetime import date

import pytest
from conftest import make_result

from jp_stock_analysis.analysis.sector_relative import (
    attach_sector_relative,
    compute_sector_relative,
)
from jp_stock_analysis.cli import main
from jp_stock_analysis.schemas import (
    CompanyMetadata,
    FundamentalMetrics,
    MomentumMetrics,
    RiskMetrics,
    StockAnalysisResult,
    ValuationMetrics,
)


def _full_result(
    ticker: str,
    per: float | None = 15.0,
    pbr: float | None = 1.5,
    revenue_growth: float | None = 5.0,
    operating_margin: float | None = 10.0,
    roe: float | None = 8.0,
    return_3m: float | None = 5.0,
    return_6m: float | None = None,
    risk_score: float = 20.0,
) -> StockAnalysisResult:
    return StockAnalysisResult(
        ticker=ticker,
        analysis_date=date(2025, 6, 30),
        fundamentals=FundamentalMetrics(
            ticker=ticker,
            revenue_growth_yoy=revenue_growth,
            operating_margin=operating_margin,
            roe=roe,
            confidence_score=80.0,
        ),
        valuation=ValuationMetrics(ticker=ticker, per=per, pbr=pbr, confidence_score=80.0),
        momentum=MomentumMetrics(
            ticker=ticker,
            observations=120,
            return_3m=return_3m,
            return_6m=return_6m,
            confidence_score=50.0,
        ),
        risks=RiskMetrics(ticker=ticker, risk_score=risk_score, confidence_score=90.0),
    )


def _metadata(sectors: dict[str, str | None]) -> dict[str, CompanyMetadata]:
    return {
        ticker: CompanyMetadata(ticker=ticker, sector=sector)
        for ticker, sector in sectors.items()
    }


THREE_AUTO = _metadata({"A": "自動車", "B": "自動車", "C": "自動車"})


def test_percentiles_rank_within_sector():
    results = [
        _full_result("A", per=10.0, revenue_growth=15.0, risk_score=10.0),
        _full_result("B", per=15.0, revenue_growth=10.0, risk_score=10.0),
        _full_result("C", per=20.0, revenue_growth=5.0, risk_score=50.0),
    ]
    computed = compute_sector_relative(results, THREE_AUTO)
    # PER is lower-better: cheapest gets 100
    assert computed["A"].per_percentile == 100.0
    assert computed["B"].per_percentile == 50.0
    assert computed["C"].per_percentile == 0.0
    # revenue growth is higher-better
    assert computed["A"].revenue_growth_percentile == 100.0
    assert computed["C"].revenue_growth_percentile == 0.0
    # risk is lower-better with a tie at the top: A and B tie, each beats C
    assert computed["A"].risk_percentile == 75.0
    assert computed["B"].risk_percentile == 75.0
    assert computed["C"].risk_percentile == 0.0
    assert all(c.peer_count == 3 and c.sector == "自動車" for c in computed.values())


def test_sector_relative_score_is_mean_of_available_percentiles():
    results = [
        _full_result("A", per=10.0),
        _full_result("B", per=20.0),
    ]
    computed = compute_sector_relative(results, _metadata({"A": "電機", "B": "電機"}))
    a = computed["A"]
    available = [
        a.per_percentile,
        a.pbr_percentile,
        a.revenue_growth_percentile,
        a.operating_margin_percentile,
        a.roe_percentile,
        a.momentum_percentile,
        a.risk_percentile,
    ]
    values = [v for v in available if v is not None]
    assert a.sector_relative_score == pytest.approx(sum(values) / len(values), abs=0.05)


def test_missing_metric_yields_none_percentile_and_warning():
    results = [
        _full_result("A", revenue_growth=None),
        _full_result("B", revenue_growth=10.0),
        _full_result("C", revenue_growth=5.0),
    ]
    computed = compute_sector_relative(results, THREE_AUTO)
    assert computed["A"].revenue_growth_percentile is None
    assert any("revenue_growth_percentile" in w for w in computed["A"].warnings)
    assert computed["A"].sector_relative_score is not None  # from remaining metrics
    # B still ranks against C even though A is missing
    assert computed["B"].revenue_growth_percentile == 100.0


def test_momentum_falls_back_to_6m_return():
    results = [
        _full_result("A", return_3m=None, return_6m=20.0),
        _full_result("B", return_3m=5.0),
    ]
    computed = compute_sector_relative(results, _metadata({"A": "電機", "B": "電機"}))
    assert computed["A"].momentum_percentile == 100.0  # 20 (6m fallback) beats 5


def test_single_company_sector_gets_no_metrics():
    results = [_full_result("A"), _full_result("B")]
    metadata = _metadata({"A": "自動車", "B": "電機"})
    computed = compute_sector_relative(results, metadata)
    assert computed == {}
    attach_sector_relative(results, metadata)
    assert all(result.sector_relative is None for result in results)


def test_missing_sector_metadata_gets_no_metrics():
    results = [_full_result("A"), _full_result("B")]
    computed = compute_sector_relative(results, _metadata({"A": None, "B": None}))
    assert computed == {}
    assert compute_sector_relative(results, {}) == {}


def test_small_peer_group_warns_and_lowers_confidence():
    results = [_full_result("A"), _full_result("B")]
    computed = compute_sector_relative(results, _metadata({"A": "電機", "B": "電機"}))
    assert any("small sector peer group" in w for w in computed["A"].warnings)
    assert computed["A"].confidence_score <= 50.0  # 2/4 peer factor


def test_deterministic_and_final_score_untouched():
    results = [
        _full_result("A", per=10.0),
        _full_result("B", per=20.0),
    ]
    metadata = _metadata({"A": "電機", "B": "電機"})
    first = compute_sector_relative(results, metadata)
    second = compute_sector_relative(results, metadata)
    assert first == second
    # attaching must not change scores or labels
    plain = make_result("X")
    final_before = plain.score.final_score
    attach_sector_relative([plain], {})
    assert plain.score.final_score == final_before
    assert plain.screening_label is None and plain.signal is None


def test_cli_shared_sector_produces_sector_relative_output(fixtures_dir, tmp_path):
    metadata_csv = tmp_path / "metadata.csv"
    metadata_csv.write_text(
        "ticker,company_name,sector,market\n"
        "7203,サンプル自動車株式会社,輸送用機器,プライム\n"
        "6758,サンプル電機株式会社,輸送用機器,プライム\n"
        "9984,サンプルホールディングス株式会社,情報・通信業,プライム\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    argv = [
        "analyze",
        "--prices", str(fixtures_dir / "prices_sample.csv"),
        "--fundamentals", str(fixtures_dir / "fundamentals_sample.csv"),
        "--metadata", str(metadata_csv),
        "--disclosures", str(fixtures_dir / "disclosures"),
        "--output-dir", str(out_dir),
    ]
    assert main(argv) == 0

    payload = json.loads((out_dir / "screening.json").read_text(encoding="utf-8"))
    by_ticker = {entry["ticker"]: entry for entry in payload["results"]}
    for ticker in ("7203", "6758"):
        sector_relative = by_ticker[ticker]["sector_relative"]
        assert sector_relative is not None
        assert sector_relative["sector"] == "輸送用機器"
        assert sector_relative["peer_count"] == 2
        assert sector_relative["sector_relative_score"] is not None
    assert by_ticker["9984"]["sector_relative"] is None  # lone sector

    # 7203 (cheap, growing) must beat 6758 (fair, slow) within the sector
    assert (
        by_ticker["7203"]["sector_relative"]["sector_relative_score"]
        > by_ticker["6758"]["sector_relative"]["sector_relative_score"]
    )

    report_7203 = (out_dir / "7203.md").read_text(encoding="utf-8")
    assert "## Sector Relative" in report_7203
    assert (out_dir / "9984.md").read_text(encoding="utf-8").count("## Sector Relative") == 0

    with (out_dir / "screening.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "sector_relative_score" in rows[0]
    blank = next(row for row in rows if row["ticker"] == "9984")
    assert blank["sector_relative_score"] == ""


def test_cli_default_fixtures_unchanged_without_shared_sectors(fixtures_dir, tmp_path):
    """Standard fixtures have three distinct sectors: behavior stays as before."""
    argv = [
        "analyze",
        "--prices", str(fixtures_dir / "prices_sample.csv"),
        "--fundamentals", str(fixtures_dir / "fundamentals_sample.csv"),
        "--metadata", str(fixtures_dir / "company_metadata_sample.csv"),
        "--disclosures", str(fixtures_dir / "disclosures"),
        "--output-dir", str(tmp_path),
    ]
    assert main(argv) == 0
    payload = json.loads((tmp_path / "screening.json").read_text(encoding="utf-8"))
    assert all(entry["sector_relative"] is None for entry in payload["results"])
    assert "## Sector Relative" not in (tmp_path / "7203.md").read_text(encoding="utf-8")
    header = (tmp_path / "screening.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "sector_relative_score" not in header

"""Command-line interface.

Example:

    python -m jp_stock_analysis.cli analyze \\
        --prices tests/fixtures/prices_sample.csv \\
        --fundamentals tests/fixtures/fundamentals_sample.csv \\
        --metadata tests/fixtures/company_metadata_sample.csv \\
        --disclosures tests/fixtures/disclosures \\
        --output-dir /tmp/jp_stock_analysis_out \\
        --signal-mode analysis_only

The default mode is ``analysis_only``; ``trade_signal`` is explicit opt-in.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from jp_stock_analysis.analysis.disclosure_nlp import RuleBasedDisclosureAnalyzer
from jp_stock_analysis.analysis.fundamentals import analyze_fundamentals
from jp_stock_analysis.analysis.momentum import analyze_momentum
from jp_stock_analysis.analysis.risk import analyze_risks
from jp_stock_analysis.analysis.scoring import score_stock
from jp_stock_analysis.analysis.screening import screen_stocks
from jp_stock_analysis.analysis.signal_engine import generate_signals
from jp_stock_analysis.analysis.valuation import analyze_valuation
from jp_stock_analysis.config import AnalysisConfig
from jp_stock_analysis.providers.local_csv import (
    load_company_metadata_csv,
    load_disclosure_texts,
    load_fundamentals_csv,
    load_prices_csv,
)
from jp_stock_analysis.reports.csv_report import write_screening_csv
from jp_stock_analysis.reports.json_report import write_json_report
from jp_stock_analysis.reports.markdown_report import write_markdown_report
from jp_stock_analysis.schemas import (
    CompanyMetadata,
    DisclosureDocument,
    FinancialStatement,
    PriceBar,
    SignalMode,
    StockAnalysisResult,
)


def _analyze_ticker(
    ticker: str,
    bars: list[PriceBar],
    statements: list[FinancialStatement],
    metadata: CompanyMetadata | None,
    document: DisclosureDocument | None,
    config: AnalysisConfig,
) -> StockAnalysisResult:
    warnings: list[str] = []

    momentum = analyze_momentum(bars) if bars else None
    if momentum is None:
        warnings.append("no price data for ticker")

    fundamentals = None
    if statements:
        ordered = sorted(statements, key=lambda s: (s.fiscal_year is None, s.fiscal_year or 0))
        previous = ordered[-2] if len(ordered) > 1 else None
        fundamentals = analyze_fundamentals(ordered[-1], previous)
    else:
        warnings.append("no financial statements for ticker")

    valuation = None
    if statements and bars:
        latest_bar = max(bars, key=lambda bar: bar.date)
        eps_growth = fundamentals.eps_growth_yoy if fundamentals else None
        valuation = analyze_valuation(
            sorted(statements, key=lambda s: (s.fiscal_year is None, s.fiscal_year or 0))[-1],
            latest_bar.close,
            eps_growth,
        )
    else:
        warnings.append("valuation skipped: needs both a statement and a market price")

    disclosure = RuleBasedDisclosureAnalyzer().analyze(document) if document else None
    if disclosure is None:
        warnings.append("no disclosure text for ticker")

    risks = analyze_risks(fundamentals, valuation, momentum, disclosure)
    score = score_stock(fundamentals, valuation, momentum, disclosure, risks, config)

    if bars:
        analysis_date = max(bars, key=lambda bar: bar.date).date
    else:
        # keep output deterministic when only fundamentals are available
        fiscal_years = [s.fiscal_year for s in statements if s.fiscal_year is not None]
        if fiscal_years:
            analysis_date = date(max(fiscal_years), 12, 31)
            warnings.append("analysis date approximated from fiscal year (no price data)")
        else:
            analysis_date = date.today()
            warnings.append("analysis date set to today (no dated inputs)")
    return StockAnalysisResult(
        ticker=ticker,
        company_name=metadata.company_name if metadata else None,
        analysis_date=analysis_date,
        signal_mode=config.signal_mode,
        fundamentals=fundamentals,
        valuation=valuation,
        momentum=momentum,
        disclosure=disclosure,
        risks=risks,
        score=score,
        warnings=warnings,
        confidence_score=score.confidence_score,
    )


def run_analysis(
    prices_path: str | Path,
    output_dir: str | Path,
    fundamentals_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    disclosures_dir: str | Path | None = None,
    signal_mode: SignalMode = "analysis_only",
) -> dict[str, object]:
    """Run the full pipeline and write reports. Returns output paths/results."""
    config = AnalysisConfig(signal_mode=signal_mode)

    prices = load_prices_csv(prices_path)
    fundamentals = load_fundamentals_csv(fundamentals_path) if fundamentals_path else {}
    metadata = load_company_metadata_csv(metadata_path) if metadata_path else {}
    disclosures = load_disclosure_texts(disclosures_dir) if disclosures_dir else {}

    tickers = sorted(set(prices) | set(fundamentals))
    results = [
        _analyze_ticker(
            ticker,
            prices.get(ticker, []),
            fundamentals.get(ticker, []),
            metadata.get(ticker),
            disclosures.get(ticker),
            config,
        )
        for ticker in tickers
    ]

    screening = screen_stocks(results, config)
    if config.signal_mode in ("screening", "trade_signal"):
        labels = {entry.ticker: entry.screening_label for entry in screening}
        for result in results:
            result.screening_label = labels.get(result.ticker)
    if config.signal_mode == "trade_signal":
        generate_signals(results, config)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_screening_csv(results, screening, out_dir / "screening.csv")
    json_path = write_json_report(results, screening, out_dir / "screening.json", config)
    markdown_paths = [write_markdown_report(result, out_dir, config) for result in results]

    return {
        "output_dir": out_dir,
        "csv_path": csv_path,
        "json_path": json_path,
        "markdown_paths": markdown_paths,
        "results": results,
        "screening": screening,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jp_stock_analysis",
        description="Japanese stock analysis engine (self-directed research tool)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="run the analysis pipeline on local files")
    analyze.add_argument("--prices", required=True, help="path to prices CSV")
    analyze.add_argument("--fundamentals", default=None, help="path to fundamentals CSV")
    analyze.add_argument("--metadata", default=None, help="path to company metadata CSV")
    analyze.add_argument("--disclosures", default=None, help="directory of <ticker>.txt files")
    analyze.add_argument("--output-dir", required=True, help="directory for generated reports")
    analyze.add_argument(
        "--signal-mode",
        default="analysis_only",
        choices=["analysis_only", "screening", "trade_signal"],
        help="analysis_only (default), screening, or trade_signal (explicit opt-in)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        outputs = run_analysis(
            prices_path=args.prices,
            output_dir=args.output_dir,
            fundamentals_path=args.fundamentals,
            metadata_path=args.metadata,
            disclosures_dir=args.disclosures,
            signal_mode=args.signal_mode,
        )
        print(f"Reports written to: {outputs['output_dir']}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

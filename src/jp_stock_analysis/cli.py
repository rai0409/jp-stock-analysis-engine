"""Command-line interface.

Local-file example (default provider):

    python -m jp_stock_analysis.cli analyze \\
        --prices tests/fixtures/prices_sample.csv \\
        --fundamentals tests/fixtures/fundamentals_sample.csv \\
        --metadata tests/fixtures/company_metadata_sample.csv \\
        --disclosures tests/fixtures/disclosures \\
        --output-dir /tmp/jp_stock_analysis_out \\
        --signal-mode analysis_only

J-Quants cache example (offline, no API key needed):

    python -m jp_stock_analysis.cli analyze \\
        --provider jquants-cache --jquants-cache-dir .cache/jquants \\
        --jquants-code 7203 --output-dir /tmp/out

``--provider jquants-live`` additionally fetches missing data from the
J-Quants API (requires the JQUANTS_API_KEY environment variable) and writes
it to the cache. The default provider remains ``local`` and the default mode
remains ``analysis_only``; ``trade_signal`` is explicit opt-in.

Disclosures can alternatively come from a topix1000_disclosure_platform
export (file-based, offline; see docs/topix1000_export_provider.md):

    python -m jp_stock_analysis.cli analyze \\
        --prices tests/fixtures/prices_sample.csv \\
        --disclosure-provider topix1000-export \\
        --topix1000-export-dir tests/fixtures/topix1000_export \\
        --output-dir /tmp/out
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from jp_stock_analysis.analysis.disclosure_nlp import RuleBasedDisclosureAnalyzer
from jp_stock_analysis.analysis.fundamentals import analyze_fundamentals
from jp_stock_analysis.analysis.momentum import analyze_momentum
from jp_stock_analysis.analysis.risk import analyze_risks
from jp_stock_analysis.analysis.scoring import score_stock
from jp_stock_analysis.analysis.screening import screen_stocks
from jp_stock_analysis.analysis.sector_relative import attach_sector_relative
from jp_stock_analysis.analysis.signal_engine import generate_signals
from jp_stock_analysis.analysis.valuation import analyze_valuation
from jp_stock_analysis.config import AnalysisConfig
from jp_stock_analysis.errors import JPStockAnalysisError, ProviderError
from jp_stock_analysis.providers.jquants import JQuantsProvider
from jp_stock_analysis.providers.local_csv import (
    load_company_metadata_csv,
    load_disclosure_texts,
    load_fundamentals_csv,
    load_prices_csv,
)
from jp_stock_analysis.providers.topix1000_export import Topix1000ExportProvider
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
from jp_stock_analysis.validation.forward_returns import (
    load_forward_return_report,
    write_forward_return_outputs,
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


def analyze_data(
    prices: dict[str, list[PriceBar]],
    fundamentals: dict[str, list[FinancialStatement]],
    metadata: dict[str, CompanyMetadata],
    disclosures: dict[str, DisclosureDocument],
    output_dir: str | Path,
    signal_mode: SignalMode = "analysis_only",
) -> dict[str, object]:
    """Run the analysis pipeline on already-loaded data and write reports."""
    config = AnalysisConfig(signal_mode=signal_mode)
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
    attach_sector_relative(results, metadata)

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


def run_analysis(
    prices_path: str | Path,
    output_dir: str | Path,
    fundamentals_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    disclosures_dir: str | Path | None = None,
    signal_mode: SignalMode = "analysis_only",
) -> dict[str, object]:
    """Run the full pipeline from local files. Returns output paths/results."""
    prices = load_prices_csv(prices_path)
    fundamentals = load_fundamentals_csv(fundamentals_path) if fundamentals_path else {}
    metadata = load_company_metadata_csv(metadata_path) if metadata_path else {}
    disclosures = load_disclosure_texts(disclosures_dir) if disclosures_dir else {}
    return analyze_data(prices, fundamentals, metadata, disclosures, output_dir, signal_mode)


def _load_jquants_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, list[PriceBar]],
    dict[str, list[FinancialStatement]],
    dict[str, CompanyMetadata],
]:
    """Load prices/fundamentals/metadata via the J-Quants provider.

    Prices are mandatory per code; statements and metadata degrade to warnings
    when their caches are absent (mirroring optional local inputs).
    """
    provider = JQuantsProvider(
        cache_dir=args.jquants_cache_dir,
        live=args.provider == "jquants-live",
    )
    prices: dict[str, list[PriceBar]] = {}
    fundamentals: dict[str, list[FinancialStatement]] = {}
    metadata: dict[str, CompanyMetadata] = {}
    for code in args.jquants_code:
        prices[code] = provider.get_prices(code, from_date=args.from_date, to_date=args.to_date)
        try:
            fundamentals[code] = provider.get_statements(code)
        except ProviderError as exc:
            print(f"warning: {exc}", file=sys.stderr)
        try:
            company = provider.get_metadata(code)
            if company is not None:
                metadata[code] = company
        except ProviderError as exc:
            print(f"warning: {exc}", file=sys.stderr)
    return prices, fundamentals, metadata


def _load_disclosures(args: argparse.Namespace) -> dict[str, DisclosureDocument]:
    """Load disclosures from the selected disclosure provider."""
    if args.disclosure_provider == "topix1000-export":
        provider = Topix1000ExportProvider(args.topix1000_export_dir)
        documents = provider.load_documents()
        for warning in provider.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        return documents
    return load_disclosure_texts(args.disclosures) if args.disclosures else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jp_stock_analysis",
        description="Japanese stock analysis engine (self-directed research tool)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="run the analysis pipeline")
    analyze.add_argument(
        "--provider",
        default="local",
        choices=["local", "jquants-cache", "jquants-live"],
        help="data source: local files (default), J-Quants cache (offline), "
        "or J-Quants live fetch (requires JQUANTS_API_KEY)",
    )
    analyze.add_argument("--prices", default=None, help="path to prices CSV (local provider)")
    analyze.add_argument("--fundamentals", default=None, help="path to fundamentals CSV")
    analyze.add_argument("--metadata", default=None, help="path to company metadata CSV")
    analyze.add_argument("--disclosures", default=None, help="directory of <ticker>.txt files")
    analyze.add_argument(
        "--disclosure-provider",
        default="local",
        choices=["local", "topix1000-export"],
        help="disclosure source: local <ticker>.txt files (default) or a "
        "topix1000_disclosure_platform export directory",
    )
    analyze.add_argument(
        "--topix1000-export-dir",
        default=None,
        help="topix1000 export directory containing index.json "
        "(required with --disclosure-provider topix1000-export)",
    )
    analyze.add_argument("--output-dir", required=True, help="directory for generated reports")
    analyze.add_argument(
        "--signal-mode",
        default="analysis_only",
        choices=["analysis_only", "screening", "trade_signal"],
        help="analysis_only (default), screening, or trade_signal (explicit opt-in)",
    )
    analyze.add_argument(
        "--jquants-cache-dir",
        default=".cache/jquants",
        help="J-Quants cache directory (default: .cache/jquants)",
    )
    analyze.add_argument(
        "--jquants-code",
        action="append",
        default=None,
        help="stock code for J-Quants providers; repeatable",
    )
    analyze.add_argument("--from-date", default=None, help="YYYY-MM-DD price range start")
    analyze.add_argument("--to-date", default=None, help="YYYY-MM-DD price range end")

    validate = subparsers.add_parser(
        "validate-forward-returns",
        help="measure realized forward returns from a screening.json and a later "
        "prices CSV (research-only; no trading signals)",
    )
    validate.add_argument(
        "--screening-json",
        required=True,
        help="path to a screening.json produced by the analyze command",
    )
    validate.add_argument(
        "--prices",
        required=True,
        help="path to a local prices CSV with ticker,date,close columns",
    )
    validate.add_argument(
        "--output-dir", required=True, help="directory for forward-return outputs"
    )
    validate.add_argument(
        "--horizons",
        default="5,20,60",
        help="comma-separated trading-row horizons (default: 5,20,60)",
    )
    validate.add_argument(
        "--analysis-date",
        default=None,
        help="YYYY-MM-DD fallback analysis date for tickers whose screening.json "
        "result has no analysis_date",
    )
    validate.add_argument(
        "--no-markdown",
        action="store_true",
        help="skip writing forward_returns.md",
    )
    return parser


def _parse_horizons(raw: str) -> list[int]:
    horizons: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid horizon {token!r}: must be an integer") from exc
        if value < 1:
            raise ValueError(f"invalid horizon {value}: must be a positive integer")
        horizons.append(value)
    if not horizons:
        raise ValueError("--horizons must contain at least one positive integer")
    return horizons


def _run_validate_forward_returns(args: argparse.Namespace) -> int:
    try:
        horizons = _parse_horizons(args.horizons)
        analysis_date_override = None
        if args.analysis_date:
            analysis_date_override = date.fromisoformat(args.analysis_date)
        report = load_forward_return_report(
            args.screening_json,
            args.prices,
            horizons,
            analysis_date_override,
        )
        paths = write_forward_return_outputs(
            report, args.output_dir, write_markdown=not args.no_markdown
        )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"Forward-return validation written to: {paths['json_path'].parent}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-forward-returns":
        return _run_validate_forward_returns(args)
    if args.command != "analyze":
        return 2
    if args.provider == "local" and not args.prices:
        parser.error("--provider local requires --prices")
    if args.provider != "local" and not args.jquants_code:
        parser.error(f"--provider {args.provider} requires at least one --jquants-code")
    if args.disclosure_provider == "topix1000-export" and not args.topix1000_export_dir:
        parser.error("--disclosure-provider topix1000-export requires --topix1000-export-dir")
    if args.disclosure_provider == "topix1000-export" and args.disclosures:
        parser.error("--disclosures applies to --disclosure-provider local only")

    try:
        disclosures = _load_disclosures(args)
        if args.provider == "local":
            prices = load_prices_csv(args.prices)
            fundamentals = load_fundamentals_csv(args.fundamentals) if args.fundamentals else {}
            metadata = load_company_metadata_csv(args.metadata) if args.metadata else {}
        else:
            prices, fundamentals, metadata = _load_jquants_inputs(args)
        outputs = analyze_data(
            prices, fundamentals, metadata, disclosures, args.output_dir, args.signal_mode
        )
    except JPStockAnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Reports written to: {outputs['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

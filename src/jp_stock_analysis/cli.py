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
from jp_stock_analysis.modeling.baseline_ranker import score_baseline, scored_observations
from jp_stock_analysis.modeling.dataset import build_modeling_dataset, write_dataset_outputs
from jp_stock_analysis.modeling.factors import ALL_FACTORS
from jp_stock_analysis.modeling.feature_importance import (
    coefficient_importance,
    permutation_importance,
)
from jp_stock_analysis.modeling.fixtures import build_synthetic_bundle
from jp_stock_analysis.modeling.linear_models import ElasticNetRanker, RidgeRanker
from jp_stock_analysis.modeling.ml_models import MODEL_TYPES, train_ranking_model
from jp_stock_analysis.modeling.neutralization import (
    ExposureObservation,
    neutralized_rank_ic,
    write_neutralized_outputs,
)
from jp_stock_analysis.modeling.portfolio_metrics import (
    evaluate_portfolio,
    observations_from_scored,
    write_portfolio_outputs,
)
from jp_stock_analysis.modeling.ranking_metrics import evaluate_ranking, write_ranking_outputs
from jp_stock_analysis.modeling.report import (
    build_modeling_report,
    write_modeling_report_outputs,
)
from jp_stock_analysis.modeling.stability import (
    build_stability_report,
    compute_fold_metrics,
    synthetic_seed_ic,
    write_stability_outputs,
)
from jp_stock_analysis.modeling.walk_forward import (
    MODE_EXPANDING,
    MODE_ROLLING,
    build_walk_forward_plan,
    write_walk_forward_outputs,
)
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
from jp_stock_analysis.validation.jquants_prices import export_jquants_prices_csv
from jp_stock_analysis.validation.no_lookahead import (
    load_bundle_disclosure_date,
    load_readiness_report,
    write_readiness_outputs,
)
from jp_stock_analysis.validation.price_prep import _parse_tickers, prepare_price_csv


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

    prepare = subparsers.add_parser(
        "prepare-price-csv",
        help="validate and normalize a local raw price CSV into the "
        "ticker,date,close shape forward-return validation consumes "
        "(offline; no fetching, no fabrication)",
    )
    prepare.add_argument("--input", required=True, help="path to a local raw price CSV")
    prepare.add_argument(
        "--output", required=True, help="path to write the normalized ticker,date,close CSV"
    )
    prepare.add_argument(
        "--tickers",
        required=True,
        help="comma-separated tickers to keep (e.g. 3928,4107,4264)",
    )
    prepare.add_argument(
        "--from-date",
        required=True,
        help="YYYY-MM-DD; rows on or after this date count toward --min-rows-after",
    )
    prepare.add_argument(
        "--min-rows-after",
        default=None,
        type=int,
        help="require at least this many rows on or after --from-date per ticker",
    )

    fetch = subparsers.add_parser(
        "fetch-jquants-prices",
        help="export a local ticker,date,close CSV from the J-Quants provider "
        "(cache-only by default; --allow-network permits a live fetch needing "
        "JQUANTS_API_KEY). Research-only; no trading signals",
    )
    fetch.add_argument(
        "--tickers",
        required=True,
        help="comma-separated tickers to fetch (e.g. 3928,4107,4264)",
    )
    fetch.add_argument(
        "--out",
        required=True,
        help="path to write the raw ticker,date,close CSV",
    )
    fetch.add_argument("--from-date", default=None, help="YYYY-MM-DD price range start (optional)")
    fetch.add_argument("--to-date", default=None, help="YYYY-MM-DD price range end (optional)")
    fetch.add_argument(
        "--price-field",
        default="close",
        choices=["close", "adjusted_close"],
        help="which value fills the output 'close' column: raw close (default) "
        "or adjusted_close (J-Quants AdjC; column stays named 'close'). With "
        "adjusted_close the export fails if any row lacks an adjusted close",
    )
    fetch.add_argument(
        "--cache-dir",
        default=".cache/jquants",
        help="J-Quants cache directory (default: .cache/jquants)",
    )
    fetch.add_argument(
        "--allow-network",
        action="store_true",
        help="permit a live J-Quants fetch when the cache is missing "
        "(requires JQUANTS_API_KEY); default is cache-only / offline",
    )

    readiness = subparsers.add_parser(
        "check-forward-readiness",
        help="check strict no-look-ahead readiness for forward-return validation "
        "(disclosure-axis): can a decision date on/after the bundle disclosure "
        "date have enough later price rows per horizon? Offline; research-only",
    )
    readiness.add_argument(
        "--fundamentals",
        required=True,
        help="path to the bundle fundamentals CSV (ticker universe)",
    )
    readiness.add_argument(
        "--prices",
        default=None,
        help="path to a local prices CSV (ticker,date,close); omit to treat all "
        "price data as missing",
    )
    readiness.add_argument(
        "--disclosure-index",
        default=None,
        help="path to the topix1000 export index.json (reads target_date as the "
        "bundle disclosure date)",
    )
    readiness.add_argument(
        "--disclosure-date",
        default=None,
        help="YYYY-MM-DD bundle disclosure date override (wins over "
        "--disclosure-index)",
    )
    readiness.add_argument(
        "--output-dir", required=True, help="directory for readiness outputs"
    )
    readiness.add_argument(
        "--horizons",
        default="5,20,60",
        help="comma-separated trading-row horizons (default: 5,20,60)",
    )
    readiness.add_argument(
        "--no-markdown", action="store_true", help="skip writing forward_readiness.md"
    )

    # ----- modeling subcommands (offline; research-only; no trading signals) ----
    build_ds = subparsers.add_parser(
        "build-modeling-dataset",
        help="build an offline modeling dataset (factors + forward-return labels) "
        "with no-look-ahead guardrails; research-only",
    )
    _add_modeling_input_args(build_ds)

    rank = subparsers.add_parser(
        "evaluate-factor-ranking",
        help="evaluate the baseline factor ranker with Rank IC / quantile metrics "
        "(research-only; no trading signals)",
    )
    _add_modeling_input_args(rank)

    wf = subparsers.add_parser(
        "run-walk-forward-ranking",
        help="generate domain-aware walk-forward folds over decision dates "
        "(research-only)",
    )
    _add_modeling_input_args(wf)
    wf.add_argument(
        "--mode",
        default=MODE_EXPANDING,
        choices=[MODE_EXPANDING, MODE_ROLLING],
        help="expanding (default) or rolling training window",
    )
    wf.add_argument("--min-train-periods", default=1, type=int)
    wf.add_argument("--test-periods", default=1, type=int)

    train = subparsers.add_parser(
        "train-ranking-model",
        help="train a ranking model (baseline, or optional LightGBM/CatBoost; "
        "missing optional deps are skipped, not failed); research-only",
    )
    _add_modeling_input_args(train)
    train.add_argument(
        "--model-type", default=MODEL_TYPES[0], choices=list(MODEL_TYPES)
    )
    train.add_argument("--horizon", default=20, type=int, help="training label horizon")

    report = subparsers.add_parser(
        "modeling-report",
        help="produce the full offline modeling report (coverage, ranking, "
        "walk-forward, model comparison, no-look-ahead status); research-only",
    )
    _add_modeling_input_args(report)
    report.add_argument(
        "--mode",
        default=MODE_EXPANDING,
        choices=[MODE_EXPANDING, MODE_ROLLING],
    )
    report.add_argument("--min-train-periods", default=1, type=int)
    report.add_argument("--test-periods", default=1, type=int)
    _add_portfolio_args(report)
    _add_neutralize_args(report)

    portfolio = subparsers.add_parser(
        "evaluate-portfolio-ranking",
        help="JPX-style long-short spread evaluation (Sharpe-like, turnover, "
        "drawdown, optional transaction cost); research-only, no trading signals",
    )
    _add_modeling_input_args(portfolio)
    _add_portfolio_args(portfolio)
    portfolio.add_argument(
        "--horizon", default=20, type=int, help="forward-return horizon to evaluate"
    )
    portfolio.add_argument(
        "--periods-per-year",
        default=None,
        type=int,
        help="if given, also report an annualized Sharpe-like score",
    )

    neutralized = subparsers.add_parser(
        "evaluate-neutralized-ranking",
        help="Numerai-style neutralized Rank IC + exposure diagnostics "
        "(research-only; not official Numerai scoring)",
    )
    _add_modeling_input_args(neutralized)
    _add_neutralize_args(neutralized)
    neutralized.add_argument(
        "--horizon", default=20, type=int, help="forward-return horizon to evaluate"
    )

    linear = subparsers.add_parser(
        "train-linear-ranking-model",
        help="train a deterministic Ridge or real coordinate-descent Elastic Net "
        "ranker on the modeling dataset; research-only, no trading signals",
    )
    _add_modeling_input_args(linear)
    linear.add_argument(
        "--linear-model-type", default="ridge", choices=["ridge", "elastic_net"]
    )
    linear.add_argument("--horizon", default=20, type=int, help="training label horizon")
    linear.add_argument("--alpha", default=1.0, type=float)
    linear.add_argument(
        "--l1-ratio", default=0.5, type=float, help="Elastic Net L1 fraction in [0,1]"
    )
    linear.add_argument("--max-iter", default=1000, type=int)
    linear.add_argument("--tol", default=1e-6, type=float)
    linear.add_argument(
        "--feature-importance",
        action="store_true",
        help="also write coefficient + permutation feature importance",
    )

    stability = subparsers.add_parser(
        "evaluate-model-stability",
        help="walk-forward + seed stability of the baseline ranker "
        "(research diagnostics only)",
    )
    _add_modeling_input_args(stability)
    stability.add_argument("--horizon", default=20, type=int)
    stability.add_argument(
        "--seed-count", default=4, type=int, help="synthetic seed-noise probe count"
    )
    return parser


def _add_portfolio_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--portfolio-mode",
        default="quantile",
        choices=["quantile", "count"],
        help="select legs by quantile (default) or fixed count",
    )
    sub.add_argument("--portfolio-top-n", default=1, type=int)
    sub.add_argument("--portfolio-bottom-n", default=1, type=int)
    sub.add_argument("--portfolio-top-quantile", default=0.2, type=float)
    sub.add_argument("--portfolio-bottom-quantile", default=0.2, type=float)
    sub.add_argument(
        "--portfolio-rank-weighted",
        action="store_true",
        help="rank-weight each leg (top long / bottom short get larger weights)",
    )
    sub.add_argument(
        "--transaction-cost-bps",
        default=0.0,
        type=float,
        help="simplified turnover-based cost in bps (default 0; research approximation)",
    )


def _add_neutralize_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--neutralize-exposures",
        default="momentum_60d,leverage",
        help="comma-separated factor exposure columns to neutralize against "
        "(sector dummies are always added). Default: momentum_60d,leverage",
    )
    sub.add_argument(
        "--neutralize-proportion",
        default=1.0,
        type=float,
        help="neutralization strength 0..1 (default 1.0 = full)",
    )


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


def _run_prepare_price_csv(args: argparse.Namespace) -> int:
    try:
        tickers = _parse_tickers(args.tickers)
        from_date = date.fromisoformat(args.from_date)
        if args.min_rows_after is not None and args.min_rows_after < 0:
            raise ValueError("--min-rows-after must be a non-negative integer")
        result = prepare_price_csv(
            args.input,
            args.output,
            tickers,
            from_date,
            min_rows_after=args.min_rows_after,
        )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(
        f"Prepared {result.total_rows_written} rows for "
        f"{len(result.tickers)} ticker(s) -> {result.output_path}"
    )
    return 0


def _run_fetch_jquants_prices(args: argparse.Namespace) -> int:
    try:
        tickers = _parse_tickers(args.tickers)
        from_date = date.fromisoformat(args.from_date) if args.from_date else None
        to_date = date.fromisoformat(args.to_date) if args.to_date else None
        provider = JQuantsProvider(cache_dir=args.cache_dir, live=args.allow_network)
        result = export_jquants_prices_csv(
            provider,
            tickers,
            args.out,
            from_date=from_date,
            to_date=to_date,
            price_field=args.price_field,
        )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(
        f"Exported {result.total_rows_written} rows ({result.price_field}) for "
        f"{len(result.tickers)} ticker(s) -> {result.output_path}"
    )
    return 0


def _run_check_forward_readiness(args: argparse.Namespace) -> int:
    try:
        horizons = _parse_horizons(args.horizons)
        disclosure_override = (
            date.fromisoformat(args.disclosure_date) if args.disclosure_date else None
        )
        if disclosure_override is None and args.disclosure_index is None:
            raise ValueError(
                "provide --disclosure-index or --disclosure-date for the bundle "
                "disclosure date"
            )
        report = load_readiness_report(
            args.fundamentals,
            args.prices,
            horizons,
            index_json_path=args.disclosure_index,
            disclosure_date_override=disclosure_override,
        )
        paths = write_readiness_outputs(
            report, args.output_dir, write_markdown=not args.no_markdown
        )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = report.to_dict()
    print(
        f"Strict no-look-ahead readiness: {report.overall_status.upper()} "
        f"({summary['eligible_ticker_count']}/{summary['ticker_count']} tickers "
        f"eligible). Disclosure date: {summary['bundle_disclosure_date']}. "
        f"Written to: {paths['json_path'].parent}"
    )
    return 0


# --------------------------------------------------------------------------- #
# Modeling subcommands (offline; research-only; no trading signals)
# --------------------------------------------------------------------------- #
def _add_modeling_input_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--synthetic",
        action="store_true",
        help="use the built-in deterministic synthetic fixture bundle (offline; "
        "SYNTHETIC ONLY — results are not market evidence)",
    )
    sub.add_argument("--prices", default=None, help="prices CSV (ticker,date,close)")
    sub.add_argument("--fundamentals", default=None, help="fundamentals CSV")
    sub.add_argument("--metadata", default=None, help="company metadata CSV")
    sub.add_argument(
        "--decision-dates",
        default=None,
        help="comma-separated YYYY-MM-DD decision dates (file inputs)",
    )
    sub.add_argument(
        "--disclosure-date", default=None, help="YYYY-MM-DD bundle disclosure date"
    )
    sub.add_argument(
        "--disclosure-index",
        default=None,
        help="topix1000 export index.json (reads target_date as the disclosure date)",
    )
    sub.add_argument(
        "--horizons", default="5,20,60", help="comma-separated horizons (default 5,20,60)"
    )
    sub.add_argument(
        "--include-non-consolidated",
        action="store_true",
        help="include non_consolidated rows (excluded by default; never pooled silently)",
    )
    sub.add_argument(
        "--n-quantiles", default=5, type=int, help="quantile buckets (default 5)"
    )
    sub.add_argument("--output-dir", required=True, help="directory for outputs")


def _load_modeling_dataset(args: argparse.Namespace):
    """Build a ModelingDataset + prices from --synthetic or file inputs."""
    horizons = _parse_horizons(args.horizons)
    if args.synthetic:
        bundle = build_synthetic_bundle()
        dataset = build_modeling_dataset(
            bundle.fundamentals,
            bundle.prices,
            bundle.metadata,
            bundle.narratives,
            decision_dates=bundle.decision_dates,
            horizons=horizons,
            bundle_disclosure_date=bundle.bundle_disclosure_date,
            include_non_consolidated=args.include_non_consolidated,
            is_synthetic=True,
        )
        return dataset, bundle.prices, bundle.bundle_disclosure_date

    if not args.prices or not args.fundamentals:
        raise ValueError("file inputs require --prices and --fundamentals (or use --synthetic)")
    if not args.decision_dates:
        raise ValueError("file inputs require --decision-dates")
    decision_dates = [
        date.fromisoformat(d.strip()) for d in args.decision_dates.split(",") if d.strip()
    ]
    if not decision_dates:
        raise ValueError("--decision-dates must contain at least one YYYY-MM-DD")
    disclosure_date = None
    if args.disclosure_date:
        disclosure_date = date.fromisoformat(args.disclosure_date)
    elif args.disclosure_index:
        disclosure_date = load_bundle_disclosure_date(args.disclosure_index)

    prices = load_prices_csv(args.prices)
    fundamentals = load_fundamentals_csv(args.fundamentals)
    metadata = load_company_metadata_csv(args.metadata) if args.metadata else {}
    dataset = build_modeling_dataset(
        fundamentals,
        prices,
        metadata,
        decision_dates=decision_dates,
        horizons=horizons,
        bundle_disclosure_date=disclosure_date,
        include_non_consolidated=args.include_non_consolidated,
        is_synthetic=False,
    )
    return dataset, prices, disclosure_date


def _run_build_modeling_dataset(args: argparse.Namespace) -> int:
    try:
        dataset, _prices, _disclosure = _load_modeling_dataset(args)
        paths = write_dataset_outputs(dataset, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Modeling dataset: {len(dataset.included())}/{len(dataset.observations)} "
        f"eligible observations. Written to: {paths['csv_path'].parent}"
    )
    return 0


def _run_evaluate_factor_ranking(args: argparse.Namespace) -> int:
    try:
        dataset, _prices, _disclosure = _load_modeling_dataset(args)
        scores = score_baseline(dataset)
        scored = [s for s in scored_observations(dataset, scores) if s.score is not None]
        report = evaluate_ranking(
            scored,
            dataset.horizons,
            model_label="baseline_factor_ranker",
            is_synthetic=dataset.is_synthetic,
            n_quantiles=args.n_quantiles,
        )
        paths = write_ranking_outputs(report, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Factor ranking validation written to: {paths['json_path'].parent}")
    return 0


def _run_walk_forward_ranking(args: argparse.Namespace) -> int:
    try:
        dataset, _prices, _disclosure = _load_modeling_dataset(args)
        plan = build_walk_forward_plan(
            dataset.decision_dates,
            horizons=dataset.horizons,
            mode=args.mode,
            min_train_periods=args.min_train_periods,
            test_periods=args.test_periods,
        )
        paths = write_walk_forward_outputs(plan, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Walk-forward plan: {len(plan.folds)} folds ({plan.mode}). "
        f"Written to: {paths['json_path'].parent}"
    )
    return 0


def _run_train_ranking_model(args: argparse.Namespace) -> int:
    try:
        dataset, _prices, _disclosure = _load_modeling_dataset(args)
        result = train_ranking_model(
            dataset, args.model_type, horizon=args.horizon, n_quantiles=args.n_quantiles
        )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        import json as _json

        (out_dir / "model_result.json").write_text(
            _json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if result.is_trained and result.scored:
            ranking = evaluate_ranking(
                result.scored,
                dataset.horizons,
                model_label=args.model_type,
                is_synthetic=dataset.is_synthetic,
                n_quantiles=args.n_quantiles,
            )
            write_ranking_outputs(ranking, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Model `{args.model_type}`: status={result.status}. "
        f"Written to: {args.output_dir}"
    )
    return 0


def _run_modeling_report(args: argparse.Namespace) -> int:
    try:
        dataset, prices, disclosure_date = _load_modeling_dataset(args)
        report = build_modeling_report(
            dataset,
            prices,
            bundle_disclosure_date=disclosure_date,
            n_quantiles=args.n_quantiles,
            walk_forward_mode=args.mode,
            min_train_periods=args.min_train_periods,
            test_periods=args.test_periods,
            portfolio_top_quantile=args.portfolio_top_quantile,
            portfolio_bottom_quantile=args.portfolio_bottom_quantile,
            portfolio_rank_weighted=args.portfolio_rank_weighted,
            transaction_cost_bps=args.transaction_cost_bps,
            neutralize_factors=_parse_exposures(args.neutralize_exposures),
            neutralize_proportion=args.neutralize_proportion,
        )
        paths = write_modeling_report_outputs(report, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Modeling report written to: {paths['json_path'].parent}")
    return 0


def _parse_exposures(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _run_evaluate_portfolio_ranking(args: argparse.Namespace) -> int:
    try:
        dataset, _prices, _disclosure = _load_modeling_dataset(args)
        scores = score_baseline(dataset)
        scored = [s for s in scored_observations(dataset, scores) if s.score is not None]
        report = evaluate_portfolio(
            observations_from_scored(scored, args.horizon),
            horizon=args.horizon,
            model_label="baseline_factor_ranker",
            is_synthetic=dataset.is_synthetic,
            mode=args.portfolio_mode,
            top_n=args.portfolio_top_n,
            bottom_n=args.portfolio_bottom_n,
            top_quantile=args.portfolio_top_quantile,
            bottom_quantile=args.portfolio_bottom_quantile,
            rank_weighted=args.portfolio_rank_weighted,
            transaction_cost_bps=args.transaction_cost_bps,
            periods_per_year=args.periods_per_year,
        )
        paths = write_portfolio_outputs(report, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Long-short evaluation status={report.status}, "
        f"Sharpe-like={report.series.sharpe_like}. Written to: {paths['json_path'].parent}"
    )
    return 0


def _run_evaluate_neutralized_ranking(args: argparse.Namespace) -> int:
    try:
        dataset, _prices, _disclosure = _load_modeling_dataset(args)
        scores = score_baseline(dataset)
        scored = [s for s in scored_observations(dataset, scores) if s.score is not None]
        exposure_obs, exposure_columns = _build_exposure_observations(
            dataset, scored, args.horizon, _parse_exposures(args.neutralize_exposures)
        )
        report = neutralized_rank_ic(
            exposure_obs,
            horizon=args.horizon,
            exposure_columns=exposure_columns,
            proportion=args.neutralize_proportion,
            is_synthetic=dataset.is_synthetic,
        )
        paths = write_neutralized_outputs(report, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Neutralized rank IC status={report.status}, "
        f"mean={report.ic_mean}. Written to: {paths['json_path'].parent}"
    )
    return 0


def _run_train_linear_ranking_model(args: argparse.Namespace) -> int:
    try:
        dataset, _prices, _disclosure = _load_modeling_dataset(args)
        label_key = f"forward_return_h{args.horizon}"
        labelled = sorted(
            (o for o in dataset.included() if o.labels.get(label_key) is not None),
            key=lambda o: (o.decision_date, o.ticker),
        )
        if not labelled:
            raise ValueError(f"no labelled observations at horizon {args.horizon}")
        matrix = [[o.features.get(f) for f in ALL_FACTORS] for o in labelled]
        target = [float(o.labels[label_key]) for o in labelled]
        if args.linear_model_type == "ridge":
            model = RidgeRanker(alpha=args.alpha)
        else:
            model = ElasticNetRanker(
                alpha=args.alpha, l1_ratio=args.l1_ratio, max_iter=args.max_iter, tol=args.tol
            )
        predictions = model.fit_predict(matrix, target, list(ALL_FACTORS))

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        import json as _json

        (out_dir / "model_metadata.json").write_text(
            _json.dumps(model.model_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (out_dir / "coefficients.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = _csv.writer(handle, lineterminator="\n")
            writer.writerow(["feature", "coefficient", "scaled_coefficient"])
            scaled = model.model_metadata["scaled_coefficients"]
            for feature, coef in model.coefficients.items():
                writer.writerow([feature, f"{coef:.8f}", f"{scaled.get(feature, 0.0):.8f}"])
        with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = _csv.writer(handle, lineterminator="\n")
            writer.writerow(["ticker", "decision_date", "prediction", "forward_return"])
            for obs, pred in zip(labelled, predictions, strict=True):
                writer.writerow(
                    [
                        obs.ticker,
                        obs.decision_date.isoformat(),
                        f"{pred:.8f}",
                        obs.labels[label_key],
                    ]
                )

        if args.feature_importance:
            coef_imp = coefficient_importance(
                model.model_metadata["scaled_coefficients"], is_synthetic=dataset.is_synthetic
            )
            perm = permutation_importance(
                model,
                matrix,
                list(ALL_FACTORS),
                [o.decision_date for o in labelled],
                [o.labels[label_key] for o in labelled],
                seed=0,
                is_synthetic=dataset.is_synthetic,
            )
            (out_dir / "feature_importance.json").write_text(
                _json.dumps(
                    {"coefficient": coef_imp.to_dict(), "permutation": perm.to_dict()},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Trained {args.linear_model_type}: status={model.status}. "
        f"Written to: {args.output_dir}"
    )
    return 0


def _run_evaluate_model_stability(args: argparse.Namespace) -> int:
    try:
        dataset, _prices, _disclosure = _load_modeling_dataset(args)
        scores = score_baseline(dataset)
        scored = [s for s in scored_observations(dataset, scores) if s.score is not None]
        plan = build_walk_forward_plan(dataset.decision_dates, horizons=[args.horizon])
        fold_metrics = compute_fold_metrics(scored, plan.folds, horizon=args.horizon)
        seed_ic = synthetic_seed_ic(
            scored, horizon=args.horizon, seeds=list(range(max(1, args.seed_count)))
        )
        report = build_stability_report(
            fold_metrics,
            horizon=args.horizon,
            is_synthetic=dataset.is_synthetic,
            seed_ic=seed_ic,
        )
        paths = write_stability_outputs(report, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Model stability written to: {paths['json_path'].parent}")
    return 0


def _build_exposure_observations(dataset, scored, horizon, factor_columns):
    """CLI helper: build neutralization inputs (factor columns + sector dummies)."""
    features_by_key = {(o.ticker, o.decision_date): o.features for o in dataset.included()}
    sectors = sorted({o.sector for o in scored if o.sector})
    label_key = f"forward_return_h{horizon}"
    obs = []
    for s in scored:
        features = features_by_key.get((s.ticker, s.decision_date), {})
        exposures = {col: features.get(col) for col in factor_columns}
        for sector in sectors:
            exposures[f"sector::{sector}"] = 1.0 if s.sector == sector else 0.0
        obs.append(
            ExposureObservation(
                decision_date=s.decision_date,
                ticker=s.ticker,
                prediction=s.score,
                forward_return=s.labels.get(label_key),
                exposures=exposures,
                sector=s.sector,
            )
        )
    return obs, [*factor_columns, *(f"sector::{s}" for s in sectors)]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-forward-returns":
        return _run_validate_forward_returns(args)
    if args.command == "prepare-price-csv":
        return _run_prepare_price_csv(args)
    if args.command == "fetch-jquants-prices":
        return _run_fetch_jquants_prices(args)
    if args.command == "check-forward-readiness":
        return _run_check_forward_readiness(args)
    if args.command == "build-modeling-dataset":
        return _run_build_modeling_dataset(args)
    if args.command == "evaluate-factor-ranking":
        return _run_evaluate_factor_ranking(args)
    if args.command == "run-walk-forward-ranking":
        return _run_walk_forward_ranking(args)
    if args.command == "train-ranking-model":
        return _run_train_ranking_model(args)
    if args.command == "modeling-report":
        return _run_modeling_report(args)
    if args.command == "evaluate-portfolio-ranking":
        return _run_evaluate_portfolio_ranking(args)
    if args.command == "evaluate-neutralized-ranking":
        return _run_evaluate_neutralized_ranking(args)
    if args.command == "train-linear-ranking-model":
        return _run_train_linear_ranking_model(args)
    if args.command == "evaluate-model-stability":
        return _run_evaluate_model_stability(args)
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

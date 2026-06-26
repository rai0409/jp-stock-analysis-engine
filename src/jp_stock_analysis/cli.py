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
from jp_stock_analysis.data.incremental_prices import (
    update_incremental_price_store,
    verify_price_store,
    write_reports,
)
from jp_stock_analysis.errors import JPStockAnalysisError, ProviderError
from jp_stock_analysis.modeling.audit import (
    build_audit_manifest,
    current_git_commit,
    fingerprint_file,
    project_version,
    write_audit_manifest_outputs,
)
from jp_stock_analysis.modeling.audit_bundle import (
    export_audit_bundle,
    verify_audit_bundle,
    write_audit_bundle_verification_outputs,
)
from jp_stock_analysis.modeling.baseline_history import (
    summarize_baseline_history,
    verify_baseline_history,
    write_history_outputs,
    write_verification_outputs,
)
from jp_stock_analysis.modeling.baseline_ranker import score_baseline, scored_observations
from jp_stock_analysis.modeling.constraints import (
    ConstraintConfig,
    PositionBook,
    apply_constraints,
)
from jp_stock_analysis.modeling.dataset import build_modeling_dataset, write_dataset_outputs
from jp_stock_analysis.modeling.determinism import (
    compare_artifact_trees,
    write_determinism_report,
)
from jp_stock_analysis.modeling.factors import ALL_FACTORS
from jp_stock_analysis.modeling.feature_importance import (
    coefficient_importance,
    permutation_importance,
)
from jp_stock_analysis.modeling.fixtures import build_synthetic_bundle
from jp_stock_analysis.modeling.linear_models import ElasticNetRanker, RidgeRanker
from jp_stock_analysis.modeling.ml_models import MODEL_TYPES, train_ranking_model
from jp_stock_analysis.modeling.monitoring import (
    build_monitoring_report,
    write_monitoring_outputs,
)
from jp_stock_analysis.modeling.neutralization import (
    ExposureObservation,
    neutralized_rank_ic,
    write_neutralized_outputs,
)
from jp_stock_analysis.modeling.pipeline import PipelineConfig, run_pipeline
from jp_stock_analysis.modeling.portfolio_metrics import (
    evaluate_portfolio,
    observations_from_scored,
    write_portfolio_outputs,
)
from jp_stock_analysis.modeling.ranking_metrics import evaluate_ranking, write_ranking_outputs
from jp_stock_analysis.modeling.regression_baseline import (
    GOLDEN_RUN_ID,
    GOLDEN_TIMESTAMP,
    capture_baseline,
    compare_to_baseline,
    load_baseline,
    run_golden_synthetic_pipeline,
    write_baseline,
    write_regression_report,
)
from jp_stock_analysis.modeling.report import (
    build_modeling_report,
    write_modeling_report_outputs,
)
from jp_stock_analysis.modeling.run_compare import (
    compare_runs,
    promote_pipeline_baseline,
    write_promotion_record_outputs,
    write_run_comparison_outputs,
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
from jp_stock_analysis.validation.jquants_daily_bars import (
    DEFAULT_ADJUSTED_CLOSE_FILE,
    DEFAULT_FEATURE_FILE,
    FIELD_COVERAGE_REPORT,
    QUALITY_REPORT,
    build_daily_bars_analysis_features,
    fetch_jquants_daily_bars_incremental,
    write_daily_bars_quality_reports,
)
from jp_stock_analysis.validation.jquants_listed_master import (
    export_jquants_listed_master_csv,
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

    incremental = subparsers.add_parser(
        "fetch-jquants-prices-incremental",
        help="incrementally build a local J-Quants price store by date "
        "(offline/cache-safe unless --allow-network is explicit); research-only",
    )
    incremental.add_argument("--universe-file", required=True)
    incremental.add_argument("--store-dir", required=True)
    incremental.add_argument("--start-date", required=True)
    incremental.add_argument("--end-date", default=None)
    incremental.add_argument("--mode", default="date", choices=["date"])
    incremental.add_argument(
        "--price-field",
        required=True,
        choices=["adjusted_close", "close"],
        help="adjusted_close is recommended; no raw-close fallback is ever applied",
    )
    incremental.add_argument("--cache-dir", default=".cache/jquants")
    incremental.add_argument("--allow-network", action="store_true")
    incremental.add_argument("--sleep-seconds", default=13.0, type=float)
    incremental.add_argument("--max-retries", default=8, type=int)
    incremental.add_argument("--backoff-multiplier", default=2.0, type=float)
    incremental.add_argument("--continue-on-date-error", action="store_true")
    incremental.add_argument("--universe-name", default=None)

    daily_bars = subparsers.add_parser(
        "fetch-jquants-daily-bars-incremental",
        help="incrementally fetch production daily bars by date into prices_daily_bars.csv "
        "(requires explicit --allow-network)",
    )
    daily_bars.add_argument("--universe-file", required=True)
    daily_bars.add_argument("--store-dir", required=True)
    daily_bars.add_argument("--start-date", required=True)
    daily_bars.add_argument("--end-date", required=True)
    daily_bars.add_argument("--sleep-seconds", default=90.0, type=float)
    daily_bars.add_argument("--max-retries", default=2, type=int)
    daily_bars.add_argument("--cache-dir", default=".cache/jquants")
    daily_bars.add_argument("--allow-network", action="store_true")

    listed_master = subparsers.add_parser(
        "fetch-jquants-listed-master",
        help="export J-Quants listed master metadata for a universe "
        "(requires explicit --allow-network for live API use)",
    )
    listed_master.add_argument("--universe-file", required=True)
    listed_master.add_argument("--output-file", required=True)
    listed_master.add_argument("--report-file", required=True)
    listed_master.add_argument("--cache-dir", default=".cache/jquants")
    listed_master.add_argument("--sleep-seconds", default=0.5, type=float)
    listed_master.add_argument(
        "--allow-network",
        action="store_true",
        help="permit live J-Quants calls through the provider (requires JQUANTS_API_KEY)",
    )

    verify_store = subparsers.add_parser(
        "verify-price-store",
        help="verify local incremental price store coverage and h5/h20/h60 "
        "decision-date eligibility; research-only",
    )
    verify_store.add_argument("--store-dir", required=True)
    verify_store.add_argument("--universe-file", default=None)

    verify_daily_bars = subparsers.add_parser(
        "verify-jquants-daily-bars",
        help="write daily-bars quality and field coverage reports",
    )
    verify_daily_bars.add_argument("--store-dir", required=True)
    verify_daily_bars.add_argument("--universe-file", required=True)
    verify_daily_bars.add_argument(
        "--adjusted-close-file",
        default=DEFAULT_ADJUSTED_CLOSE_FILE,
        help="adjusted-close store to compare against",
    )
    verify_daily_bars.add_argument("--output-report", default=None)
    verify_daily_bars.add_argument("--field-coverage-report", default=None)

    daily_features = subparsers.add_parser(
        "build-daily-bars-analysis-features",
        help="build modeling-oriented daily-bars features from adjusted OHLC and liquidity fields",
    )
    daily_features.add_argument("--daily-bars-file", required=True)
    daily_features.add_argument("--coverage-file", default=None)
    daily_features.add_argument("--output-file", default=DEFAULT_FEATURE_FILE)
    daily_features.add_argument("--lookback-days", default=20, type=int)
    daily_features.add_argument("--min-average-turnover", default=None, type=float)
    daily_features.add_argument("--include-partial-history", action="store_true")

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

    constraints = subparsers.add_parser(
        "evaluate-portfolio-constraints",
        help="apply position/liquidity/sector/turnover constraints to the baseline "
        "long-short book (research feasibility approximation; NOT a recommended "
        "portfolio, NOT order execution)",
    )
    _add_modeling_input_args(constraints)
    constraints.add_argument("--horizon", default=20, type=int)
    constraints.add_argument("--portfolio-top-quantile", default=0.2, type=float)
    constraints.add_argument("--portfolio-bottom-quantile", default=0.2, type=float)
    constraints.add_argument("--max-weight-per-name", default=None, type=float)
    constraints.add_argument("--max-sector-weight", default=None, type=float)
    constraints.add_argument(
        "--max-participation-rate",
        default=None,
        type=float,
        help="requires an ADV column; synthetic fixtures have none (never fabricated)",
    )
    constraints.add_argument("--min-adv", default=None, type=float)
    constraints.add_argument("--max-total-turnover", default=None, type=float)
    constraints.add_argument("--transaction-cost-bps", default=0.0, type=float)

    audit = subparsers.add_parser(
        "build-audit-manifest",
        help="build a deterministic reproducibility manifest (input fingerprints, "
        "model versions, synthetic-vs-real); research-only",
    )
    audit.add_argument("--synthetic", action="store_true")
    audit.add_argument(
        "--input", action="append", default=None, help="input file to fingerprint; repeatable"
    )
    audit.add_argument("--output-dir", required=True)
    audit.add_argument(
        "--fixed-timestamp",
        default=None,
        help="fix created_at_utc for deterministic runs (e.g. tests)",
    )
    audit.add_argument(
        "--run-id", default=None, help="fix run_id for deterministic runs (else derived)"
    )

    monitoring = subparsers.add_parser(
        "evaluate-model-monitoring",
        help="drift/stability monitoring over decision dates from a metrics CSV or "
        "the synthetic baseline (research diagnostics only)",
    )
    _add_modeling_input_args(monitoring)
    monitoring.add_argument("--horizon", default=20, type=int)
    monitoring.add_argument(
        "--metrics-csv",
        default=None,
        help="CSV with a period column + metric columns (overrides --synthetic series)",
    )
    monitoring.add_argument("--period-column", default="decision_date")
    monitoring.add_argument("--window", default=3, type=int)
    monitoring.add_argument("--z-threshold", default=2.0, type=float)

    pipeline = subparsers.add_parser(
        "run-modeling-pipeline",
        help="run the full offline modeling pipeline into a stamped run directory "
        "(dataset -> ... -> audit + artifact manifest); research-only",
    )
    _add_modeling_input_args(pipeline)
    pipeline.add_argument("--run-id", default="run")
    pipeline.add_argument("--fixed-timestamp", default=None)
    pipeline.add_argument("--adv", default=None, help="optional ADV CSV (ticker,adv)")
    _add_pipeline_config_args(pipeline)

    verify = subparsers.add_parser(
        "verify-pipeline-determinism",
        help="run the pipeline twice and compare artifact trees (canonicalizing "
        "only declared volatile fields); research-only",
    )
    _add_modeling_input_args(verify)
    verify.add_argument("--run-id-prefix", default="det")
    verify.add_argument("--fixed-timestamp", default="1970-01-01T00:00:00Z")
    verify.add_argument("--adv", default=None, help="optional ADV CSV (ticker,adv)")
    verify.add_argument(
        "--fail-on-difference",
        action="store_true",
        help="exit nonzero if the two runs differ (default: report only)",
    )
    _add_pipeline_config_args(verify)

    regression = subparsers.add_parser(
        "check-pipeline-regression",
        help="run the pipeline and compare it against a committed golden baseline "
        "(unexpected change detection); research-only",
    )
    _add_modeling_input_args(regression)
    regression.add_argument(
        "--baseline-path",
        default="tests/fixtures/pipeline_baseline/golden_pipeline_baseline.json",
        help="path to the golden baseline JSON",
    )
    regression.add_argument("--run-id", default=GOLDEN_RUN_ID)
    regression.add_argument("--fixed-timestamp", default=GOLDEN_TIMESTAMP)
    regression.add_argument("--adv", default=None, help="optional ADV CSV (ticker,adv)")
    regression.add_argument(
        "--update-baseline",
        action="store_true",
        help="INTENTIONALLY regenerate the golden baseline from this run (reviewed)",
    )
    regression.add_argument("--fail-on-regression", action="store_true")
    regression.add_argument(
        "--strict-new-artifacts",
        action="store_true",
        help="treat an unexpected new artifact as a regression",
    )

    compare = subparsers.add_parser(
        "compare-pipeline-runs",
        help="diff two pipeline run directories (A vs B) into a neutral artifact / "
        "metric-delta report (descriptive only, never better/worse); research-only",
    )
    compare.add_argument("--run-a", required=True, help="run directory A")
    compare.add_argument("--run-b", required=True, help="run directory B")
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--run-id-a", default=None)
    compare.add_argument("--run-id-b", default=None)
    compare.add_argument("--fixed-timestamp", default=None)
    compare.add_argument(
        "--strict-new-artifacts",
        action="store_true",
        help="record only_in_b artifacts (informational; never a performance claim)",
    )
    compare.add_argument(
        "--canonicalize",
        action="store_true",
        default=True,
        help="canonicalize declared volatile fields before comparing (default on)",
    )

    promote = subparsers.add_parser(
        "promote-pipeline-baseline",
        help="promote a run to the approved golden baseline AFTER explicit review "
        "(writes an auditable provenance record); research-only",
    )
    promote.add_argument("--from-run", required=True, help="run directory to promote")
    promote.add_argument("--baseline-path", required=True)
    promote.add_argument("--output-dir", required=True)
    promote.add_argument("--reviewer-note", default="")
    promote.add_argument(
        "--approve", action="store_true", help="explicit approval to update the baseline"
    )
    promote.add_argument(
        "--require-approval",
        action="store_true",
        help="block promotion unless --approve is also given",
    )
    promote.add_argument("--run-id", default=GOLDEN_RUN_ID)
    promote.add_argument("--fixed-timestamp", default=GOLDEN_TIMESTAMP)
    promote.add_argument(
        "--previous-baseline-path",
        default=None,
        help="prior baseline for the metric-delta comparison (defaults to --baseline-path)",
    )
    promote.add_argument(
        "--ledger-path",
        default=None,
        help="append a hash-chained promotion entry to this append-only ledger "
        "(only on an approved promotion; a broken chain blocks the promotion)",
    )

    history = subparsers.add_parser(
        "show-baseline-history",
        help="print the baseline promotion lineage from the hash-chained ledger "
        "(research-only; audit trail, not a performance claim)",
    )
    history.add_argument(
        "--ledger-path",
        default="tests/fixtures/pipeline_baseline/baseline_history.jsonl",
    )
    history.add_argument("--output-dir", default=None)

    lineage = subparsers.add_parser(
        "verify-baseline-lineage",
        help="verify the baseline-history hash chain is intact (detects silent "
        "edits / broken parents / out-of-order entries); research-only",
    )
    lineage.add_argument(
        "--ledger-path",
        default="tests/fixtures/pipeline_baseline/baseline_history.jsonl",
    )
    lineage.add_argument("--output-dir", default=None)
    lineage.add_argument("--fail-on-invalid", action="store_true")

    export_bundle = subparsers.add_parser(
        "export-audit-bundle",
        help="package the current baseline, ledger, and optional reports into a "
        "self-contained reproducibility audit bundle; research-only",
    )
    export_bundle.add_argument("--output-dir", required=True)
    export_bundle.add_argument("--bundle-id", default=None)
    export_bundle.add_argument("--fixed-timestamp", default=None)
    export_bundle.add_argument(
        "--baseline-path",
        default="tests/fixtures/pipeline_baseline/golden_pipeline_baseline.json",
    )
    export_bundle.add_argument(
        "--ledger-path",
        default="tests/fixtures/pipeline_baseline/baseline_history.jsonl",
    )
    export_bundle.add_argument("--promotion-record", default=None)
    export_bundle.add_argument("--promotion-record-dir", default=None)
    export_bundle.add_argument("--pipeline-run-dir", default=None)
    export_bundle.add_argument("--determinism-report", default=None)
    export_bundle.add_argument("--regression-report", default=None)
    export_bundle.add_argument("--synthetic", action="store_true")
    export_bundle.add_argument("--include-fresh-checks", action="store_true")

    verify_bundle = subparsers.add_parser(
        "verify-audit-bundle",
        help="verify audit bundle manifest fingerprints, ledger lineage, and "
        "baseline/ledger consistency; research-only",
    )
    verify_bundle.add_argument("--bundle-dir", required=True)
    verify_bundle.add_argument("--output-dir", default=None)
    verify_bundle.add_argument("--fail-on-invalid", action="store_true")
    return parser


def _add_pipeline_config_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--linear-models", default="ridge,elastic_net")
    sub.add_argument("--alpha", default=0.05, type=float)
    sub.add_argument("--l1-ratio", default=0.5, type=float)
    sub.add_argument("--portfolio-top-quantile", default=0.2, type=float)
    sub.add_argument("--portfolio-bottom-quantile", default=0.2, type=float)
    sub.add_argument("--portfolio-rank-weighted", action="store_true")
    sub.add_argument("--transaction-cost-bps", default=0.0, type=float)
    sub.add_argument("--max-weight-per-name", default=None, type=float)
    sub.add_argument("--max-sector-weight", default=None, type=float)
    sub.add_argument("--max-participation-rate", default=None, type=float)
    sub.add_argument("--min-adv", default=None, type=float)
    sub.add_argument("--monitoring-window", default=3, type=int)
    sub.add_argument("--monitoring-threshold", default=2.0, type=float)


def _pipeline_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    linear_models = tuple(
        m.strip() for m in args.linear_models.split(",") if m.strip()
    )
    return PipelineConfig(
        linear_models=linear_models,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        portfolio_top_quantile=args.portfolio_top_quantile,
        portfolio_bottom_quantile=args.portfolio_bottom_quantile,
        portfolio_rank_weighted=args.portfolio_rank_weighted,
        transaction_cost_bps=args.transaction_cost_bps,
        max_weight_per_name=args.max_weight_per_name,
        max_sector_weight=args.max_sector_weight,
        max_participation_rate=args.max_participation_rate,
        min_adv=args.min_adv,
        monitoring_window=args.monitoring_window,
        monitoring_threshold=args.monitoring_threshold,
    )


def _load_adv_csv(path: str | None) -> dict[str, float] | None:
    if not path:
        return None
    import csv as _csv

    adv: dict[str, float] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in _csv.DictReader(handle):
            ticker = (row.get("ticker") or row.get("code") or "").strip()
            raw = (row.get("adv") or "").strip()
            if ticker and raw:
                adv[ticker] = float(raw)
    return adv or None


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


def _run_fetch_jquants_prices_incremental(args: argparse.Namespace) -> int:
    try:
        from_date = date.fromisoformat(args.start_date)
        to_date = date.fromisoformat(args.end_date) if args.end_date else None
        provider = JQuantsProvider(cache_dir=args.cache_dir, live=args.allow_network)
        result = update_incremental_price_store(
            provider,
            universe_file=args.universe_file,
            store_dir=args.store_dir,
            start_date=from_date,
            end_date=to_date,
            price_field=args.price_field,
            allow_network=args.allow_network,
            mode=args.mode,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
            backoff_multiplier=args.backoff_multiplier,
            continue_on_date_error=args.continue_on_date_error,
            universe_name=args.universe_name,
        )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(
        f"Incremental J-Quants store updated: rows={result.row_count}, "
        f"tickers={result.ticker_count}, added={result.rows_added}, "
        f"failed_dates={len(result.failed_dates)} -> {result.price_file}"
    )
    return 0 if not result.failed_dates else 2


def _run_fetch_jquants_daily_bars_incremental(args: argparse.Namespace) -> int:
    if not args.allow_network:
        print(
            "error: fetch-jquants-daily-bars-incremental requires --allow-network for live "
            "J-Quants daily bars ingestion",
            file=sys.stderr,
        )
        return 1
    try:
        provider = JQuantsProvider(cache_dir=args.cache_dir, live=args.allow_network)
        result = fetch_jquants_daily_bars_incremental(
            provider,
            universe_file=args.universe_file,
            store_dir=args.store_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            allow_network=args.allow_network,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        "Daily bars store updated: "
        f"rows={result.rows}, tickers={result.tickers}, added={result.added}, "
        f"date_min={result.date_min}, date_max={result.date_max}, "
        f"failed_dates={len(result.failed_dates)} -> {result.output_file}"
    )
    if result.empty_dates:
        print(f"Empty/non-trading dates recorded: {len(result.empty_dates)}")
    return 0 if not result.failed_dates else 2


def _run_fetch_jquants_listed_master(args: argparse.Namespace) -> int:
    if not args.allow_network:
        print(
            "error: fetch-jquants-listed-master requires --allow-network for live "
            "J-Quants metadata export",
            file=sys.stderr,
        )
        return 1
    try:
        provider = JQuantsProvider(cache_dir=args.cache_dir, live=args.allow_network)
        result = export_jquants_listed_master_csv(
            provider,
            universe_file=args.universe_file,
            output_file=args.output_file,
            report_file=args.report_file,
            sleep_seconds=args.sleep_seconds,
            allow_network=args.allow_network,
            endpoint_url_for_listed_info=provider.endpoint_url("listed_info"),
        )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(
        f"J-Quants listed master exported: universe={result.universe_count}, "
        f"matched={result.matched_count}, missing={result.missing_count} -> "
        f"{result.output_path}"
    )
    print(f"Report written to: {result.report_path}")
    return 0


def _run_verify_price_store(args: argparse.Namespace) -> int:
    try:
        report = verify_price_store(args.store_dir, universe_file=args.universe_file)
        write_reports(args.store_dir, universe_file=args.universe_file)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    latest = report["latest_eligible_decision_dates"]
    print(
        f"Price store: rows={report['rows']}, tickers={report['ticker_count']}, "
        f"dates={report['date_min']}..{report['date_max']}, "
        f"duplicates={report['duplicate_ticker_date_rows']}, latest={latest}"
    )
    return 0


def _run_verify_jquants_daily_bars(args: argparse.Namespace) -> int:
    try:
        output_report = args.output_report or str(Path(args.store_dir) / QUALITY_REPORT)
        field_report = args.field_coverage_report or str(
            Path(args.store_dir) / FIELD_COVERAGE_REPORT
        )
        report, _coverage = write_daily_bars_quality_reports(
            store_dir=args.store_dir,
            universe_file=args.universe_file,
            adjusted_close_file=args.adjusted_close_file,
            output_report=output_report,
            field_coverage_report=field_report,
        )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Daily bars quality: rows={report['rows']}, tickers={report['tickers']}, "
        f"dates={report['date_min']}..{report['date_max']}, "
        f"duplicates={report['duplicate_ticker_date_rows']}, "
        f"status={report['overall_status']} -> {output_report}"
    )
    print(f"Field coverage written to: {field_report}")
    return 0


def _run_build_daily_bars_analysis_features(args: argparse.Namespace) -> int:
    try:
        features = build_daily_bars_analysis_features(
            daily_bars_file=args.daily_bars_file,
            coverage_file=args.coverage_file,
            output_file=args.output_file,
            lookback_days=args.lookback_days,
            min_average_turnover=args.min_average_turnover,
            include_partial_history=args.include_partial_history,
        )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Daily bars features written: rows={len(features)} -> {args.output_file}")
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


def _run_evaluate_portfolio_constraints(args: argparse.Namespace) -> int:
    try:
        dataset, _prices, _disclosure = _load_modeling_dataset(args)
        scores = score_baseline(dataset)
        scored = [s for s in scored_observations(dataset, scores) if s.score is not None]
        portfolio = evaluate_portfolio(
            observations_from_scored(scored, args.horizon),
            horizon=args.horizon,
            is_synthetic=dataset.is_synthetic,
            top_quantile=args.portfolio_top_quantile,
            bottom_quantile=args.portfolio_bottom_quantile,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        ok_dates = [s for s in portfolio.per_date if s.status == "ok"]
        if not ok_dates:
            raise ValueError("no valid long-short date to constrain")
        latest = ok_dates[-1]
        sector_of = {
            o.ticker: o.sector
            for o in dataset.included()
            if o.decision_date == latest.decision_date
        }
        book = PositionBook(
            long_weights=dict(latest.long_weights),
            short_weights=dict(latest.short_weights),
            sector_of=sector_of,
            adv_of=None,  # synthetic fixtures carry no ADV; never fabricated
        )
        config = ConstraintConfig(
            max_weight_per_name=args.max_weight_per_name,
            max_sector_weight=args.max_sector_weight,
            max_participation_rate=args.max_participation_rate,
            min_adv=args.min_adv,
            max_total_turnover=args.max_total_turnover,
        )
        result = apply_constraints(book, config)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        import json as _json

        payload = {
            "decision_date": latest.decision_date.isoformat(),
            "is_synthetic": dataset.is_synthetic,
            "synthetic_warning": (
                "SYNTHETIC FIXTURE RESULTS — not real market evidence."
                if dataset.is_synthetic
                else None
            ),
            "constraints": result.to_dict(),
            "portfolio_commercial": portfolio.commercial,
        }
        (out_dir / "constrained_portfolio.json").write_text(
            _json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (out_dir / "constrained_portfolio.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = _csv.writer(handle, lineterminator="\n")
            writer.writerow(["leg", "ticker", "unconstrained_weight", "constrained_weight"])
            for leg, leg_result in (("long", result.long), ("short", result.short)):
                names = sorted(set(leg_result.unconstrained) | set(leg_result.constrained))
                for ticker in names:
                    writer.writerow(
                        [
                            leg,
                            ticker,
                            f"{leg_result.unconstrained.get(ticker, 0.0):.8f}",
                            f"{leg_result.constrained.get(ticker, 0.0):.8f}",
                        ]
                    )
        (out_dir / "constrained_portfolio.md").write_text(
            "# Constrained Portfolio (research feasibility, NOT a recommendation)\n\n"
            f"{result.disclaimer}\n\n- status: {result.status}\n"
            f"- applied: {result.applied_constraints}\n",
            encoding="utf-8",
        )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Constraints status={result.status}. Written to: {args.output_dir}")
    return 0


def _run_build_audit_manifest(args: argparse.Namespace) -> int:
    try:
        fingerprints = [fingerprint_file(p) for p in (args.input or [])]
        manifest = build_audit_manifest(
            command={"command": "build-audit-manifest", "inputs": args.input or []},
            input_fingerprints=fingerprints,
            is_synthetic=args.synthetic,
            git_commit=current_git_commit("."),
            version=project_version(),
            run_id=args.run_id,
            created_at_utc=args.fixed_timestamp,
            stable=True,
        )
        paths = write_audit_manifest_outputs(manifest, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Audit manifest run_id={manifest['run_id']} written to: {paths['json_path'].parent}")
    return 0


def _run_evaluate_model_monitoring(args: argparse.Namespace) -> int:
    try:
        if args.metrics_csv:
            periods, columns = _load_metrics_csv(args.metrics_csv, args.period_column)
            is_synthetic = False
        else:
            dataset, _prices, _disclosure = _load_modeling_dataset(args)
            scores = score_baseline(dataset)
            scored = [s for s in scored_observations(dataset, scores) if s.score is not None]
            portfolio = evaluate_portfolio(
                observations_from_scored(scored, args.horizon),
                horizon=args.horizon,
                is_synthetic=dataset.is_synthetic,
            )
            periods = [s.decision_date.isoformat() for s in portfolio.per_date]
            columns = {"long_short_spread": [s.spread_return for s in portfolio.per_date]}
            is_synthetic = dataset.is_synthetic
        report = build_monitoring_report(
            periods, columns, window=args.window, z_threshold=args.z_threshold,
            is_synthetic=is_synthetic,
        )
        paths = write_monitoring_outputs(report, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Monitoring written to: {paths['json_path'].parent}")
    return 0


def _pipeline_inputs(args: argparse.Namespace):
    """Build (dataset, prices, disclosure_date, input_files, adv) for the pipeline."""
    dataset, prices, disclosure_date = _load_modeling_dataset(args)
    input_files = [
        p for p in (getattr(args, "fundamentals", None), getattr(args, "prices", None),
                    getattr(args, "metadata", None)) if p
    ]
    adv = _load_adv_csv(args.adv)
    return dataset, prices, disclosure_date, input_files, adv


def _run_run_modeling_pipeline(args: argparse.Namespace) -> int:
    try:
        dataset, prices, disclosure_date, input_files, adv = _pipeline_inputs(args)
        summary = run_pipeline(
            dataset,
            prices,
            output_dir=args.output_dir,
            run_id=args.run_id,
            fixed_timestamp=args.fixed_timestamp,
            disclosure_date=disclosure_date,
            config=_pipeline_config_from_args(args),
            input_files=input_files,
            adv=adv,
            git_commit=current_git_commit("."),
            version=project_version(),
        )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Pipeline run '{args.run_id}' ({summary['step_count']} steps, "
        f"no-look-ahead={summary['no_look_ahead_status']}). "
        f"Run dir: {summary['run_directory']}"
    )
    return 0


def _run_verify_pipeline_determinism(args: argparse.Namespace) -> int:
    try:
        config = _pipeline_config_from_args(args)
        run_id = f"{args.run_id_prefix}_run"
        base = Path(args.output_dir)
        run_dirs = []
        for suffix in ("a", "b"):
            dataset, prices, disclosure_date, input_files, adv = _pipeline_inputs(args)
            parent = base / f"{args.run_id_prefix}_{suffix}"
            run_pipeline(
                dataset,
                prices,
                output_dir=parent,
                run_id=run_id,
                fixed_timestamp=args.fixed_timestamp,
                disclosure_date=disclosure_date,
                config=config,
                input_files=input_files,
                adv=adv,
                git_commit=current_git_commit("."),
                version=project_version(),
            )
            run_dirs.append(parent / run_id)
        # canonicalize only the two differing parent absolute paths (declared volatile)
        comparison = compare_artifact_trees(
            run_dirs[0], run_dirs[1], volatile_values=[str(d) for d in run_dirs]
        )
        paths = write_determinism_report(comparison, base)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Determinism: {comparison['overall'].upper()} "
        f"({comparison['counts']}). Report: {paths['json_path']}"
    )
    if args.fail_on_difference and comparison["overall"] != "identical":
        return 2
    return 0


def _run_check_pipeline_regression(args: argparse.Namespace) -> int:
    try:
        if args.synthetic:
            run_dir = run_golden_synthetic_pipeline(
                args.output_dir, run_id=args.run_id, fixed_timestamp=args.fixed_timestamp
            )
        else:
            dataset, prices, disclosure_date, input_files, adv = _pipeline_inputs(args)
            run_pipeline(
                dataset, prices, output_dir=args.output_dir, run_id=args.run_id,
                fixed_timestamp=args.fixed_timestamp, disclosure_date=disclosure_date,
                input_files=input_files, adv=adv, git_commit=current_git_commit("."),
                version=project_version(),
            )
            run_dir = Path(args.output_dir) / args.run_id

        if args.update_baseline:
            baseline = capture_baseline(
                run_dir, run_id=args.run_id, fixed_timestamp=args.fixed_timestamp,
                is_synthetic=args.synthetic,
            )
            write_baseline(baseline, args.baseline_path)
            print(
                f"warning: INTENTIONAL baseline update written to {args.baseline_path} "
                "(review before committing)",
                file=sys.stderr,
            )
        else:
            baseline = load_baseline(args.baseline_path)
        report = compare_to_baseline(
            run_dir, baseline, run_id=args.run_id, fixed_timestamp=args.fixed_timestamp,
            strict_new_artifacts=args.strict_new_artifacts,
        )
        paths = write_regression_report(report, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Regression check: detected={report['regression_detected']} "
        f"({report['counts']}). Report: {paths['json_path']}"
    )
    if args.fail_on_regression and report["regression_detected"]:
        return 2
    return 0


def _run_compare_pipeline_runs(args: argparse.Namespace) -> int:
    try:
        report = compare_runs(
            args.run_a, args.run_b,
            run_id_a=args.run_id_a, run_id_b=args.run_id_b,
            fixed_timestamp=args.fixed_timestamp,
            strict_new_artifacts=args.strict_new_artifacts,
        )
        paths = write_run_comparison_outputs(report, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Run comparison: {report['comparison_status']} ({report['counts']}). "
        f"Report: {paths['json_path']}"
    )
    return 0


def _run_promote_pipeline_baseline(args: argparse.Namespace) -> int:
    try:
        record, updated = promote_pipeline_baseline(
            args.from_run, args.baseline_path,
            reviewer_note=args.reviewer_note,
            require_approval=args.require_approval,
            approved=args.approve,
            previous_baseline_path=args.previous_baseline_path,
            run_id=args.run_id,
            fixed_timestamp=args.fixed_timestamp,
            ledger_path=args.ledger_path,
        )
        paths = write_promotion_record_outputs(record, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    ledger_note = (
        f" Ledger: {record['ledger_append_status']}"
        if record.get("ledger_append_status")
        else ""
    )
    if updated:
        print(
            f"warning: baseline INTENTIONALLY updated at {args.baseline_path} "
            f"(reviewer note recorded). Record: {paths['json_path']}",
            file=sys.stderr,
        )
        print(f"Baseline promoted.{ledger_note} Record: {paths['json_path']}")
        return 0
    print(
        f"Promotion BLOCKED ({record['status']}); baseline NOT updated. "
        f"Record: {paths['json_path']}",
        file=sys.stderr,
    )
    return 2


def _run_show_baseline_history(args: argparse.Namespace) -> int:
    try:
        summary = summarize_baseline_history(args.ledger_path)
        if args.output_dir:
            write_history_outputs(summary, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Baseline history: {summary['entry_count']} entries, chain "
        f"{summary['chain_status'].upper()}."
    )
    for row in summary["entries"]:
        print(
            f"  #{row['entry_index']} {row['entry_hash_short']} "
            f"(parent {row['parent_hash_short']}) — {row['reviewer_note'] or '—'}"
        )
    return 0


def _run_verify_baseline_lineage(args: argparse.Namespace) -> int:
    try:
        report = verify_baseline_history(args.ledger_path)
        if args.output_dir:
            write_verification_outputs(report, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Baseline lineage: {report['status'].upper()} "
        f"({report['entry_count']} entries, {len(report['issues'])} issue(s))."
    )
    for issue in report["issues"]:
        print(f"  - {issue}", file=sys.stderr)
    if args.fail_on_invalid and report["status"] != "valid":
        return 2
    return 0


def _run_export_audit_bundle(args: argparse.Namespace) -> int:
    try:
        manifest = export_audit_bundle(
            args.output_dir,
            baseline_path=args.baseline_path,
            ledger_path=args.ledger_path,
            promotion_record_path=args.promotion_record,
            promotion_record_dir=args.promotion_record_dir,
            pipeline_run_dir=args.pipeline_run_dir,
            determinism_report_path=args.determinism_report,
            regression_report_path=args.regression_report,
            synthetic=args.synthetic,
            fixed_timestamp=args.fixed_timestamp,
            bundle_id=args.bundle_id,
            include_fresh_checks=args.include_fresh_checks,
        )
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Audit bundle exported: {args.output_dir} "
        f"({len(manifest['bundle_contents'])} files, "
        f"fingerprint={manifest['overall_bundle_fingerprint'][:12]}...)."
    )
    return 0


def _run_verify_audit_bundle(args: argparse.Namespace) -> int:
    try:
        report = verify_audit_bundle(
            args.bundle_dir, fail_on_invalid=args.fail_on_invalid
        )
        if args.output_dir:
            write_audit_bundle_verification_outputs(report, args.output_dir)
    except (ValueError, JPStockAnalysisError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Audit bundle: {report['status'].upper()} "
        f"({report['file_count']} files, {len(report['issues'])} issue(s))."
    )
    for issue in report["issues"]:
        print(f"  - {issue}", file=sys.stderr)
    if args.fail_on_invalid and report["status"] != "valid":
        return 2
    return 0


def _load_metrics_csv(path: str, period_column: str):
    import csv as _csv

    rows = list(_csv.DictReader(Path(path).open(encoding="utf-8-sig")))
    if not rows or period_column not in rows[0]:
        raise ValueError(f"metrics CSV must have a '{period_column}' column")
    periods = [row[period_column] for row in rows]
    metric_names = [c for c in rows[0] if c != period_column]
    columns: dict[str, list[float | None]] = {}
    for name in metric_names:
        values: list[float | None] = []
        for row in rows:
            raw = (row.get(name) or "").strip()
            values.append(float(raw) if raw else None)
        columns[name] = values
    return periods, columns


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
    if args.command == "fetch-jquants-prices-incremental":
        return _run_fetch_jquants_prices_incremental(args)
    if args.command == "fetch-jquants-daily-bars-incremental":
        return _run_fetch_jquants_daily_bars_incremental(args)
    if args.command == "fetch-jquants-listed-master":
        return _run_fetch_jquants_listed_master(args)
    if args.command == "verify-price-store":
        return _run_verify_price_store(args)
    if args.command == "verify-jquants-daily-bars":
        return _run_verify_jquants_daily_bars(args)
    if args.command == "build-daily-bars-analysis-features":
        return _run_build_daily_bars_analysis_features(args)
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
    if args.command == "evaluate-portfolio-constraints":
        return _run_evaluate_portfolio_constraints(args)
    if args.command == "build-audit-manifest":
        return _run_build_audit_manifest(args)
    if args.command == "evaluate-model-monitoring":
        return _run_evaluate_model_monitoring(args)
    if args.command == "run-modeling-pipeline":
        return _run_run_modeling_pipeline(args)
    if args.command == "verify-pipeline-determinism":
        return _run_verify_pipeline_determinism(args)
    if args.command == "check-pipeline-regression":
        return _run_check_pipeline_regression(args)
    if args.command == "compare-pipeline-runs":
        return _run_compare_pipeline_runs(args)
    if args.command == "promote-pipeline-baseline":
        return _run_promote_pipeline_baseline(args)
    if args.command == "show-baseline-history":
        return _run_show_baseline_history(args)
    if args.command == "verify-baseline-lineage":
        return _run_verify_baseline_lineage(args)
    if args.command == "export-audit-bundle":
        return _run_export_audit_bundle(args)
    if args.command == "verify-audit-bundle":
        return _run_verify_audit_bundle(args)
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

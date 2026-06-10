Implement Japanese Stock Analysis Engine v1 in this repository.

Important:
Do not stop at planning.
Implement the code, fixtures, tests, CLI, and docs.
Run tests and ruff before final response.
Fix failures until they pass or report the exact blocker.

Repository state:
- This is a fresh scaffold.
- Package path: `src/jp_stock_analysis/`
- Tests path: `tests/`
- Fixtures path: `tests/fixtures/`
- Docs path: `docs/`
- `pyproject.toml` already provides pandas, numpy, pydantic, pytest, and ruff.
- Use Python 3.12-compatible code.
- Do not implement RAG integration.
- Do not implement broker execution, auto-trading, position sizing, portfolio allocation, leverage, margin, or derivatives.
- Do not require network access.
- Do not require paid APIs.
- Use deterministic local fixtures only.

Project meaning:
“Commercial-grade” means public-release-quality, maintainable, tested, reproducible, and useful for my own stock research/trading analysis.
It does not mean SaaS, third-party investment advisory, or paid financial advice.

Core goal:
Create a working v1 vertical slice:

local CSV/TXT inputs
→ fundamentals analysis
→ valuation analysis
→ momentum analysis
→ Japanese disclosure/NLP analysis
→ risk analysis
→ integrated scoring
→ screening
→ optional trade_signal mode
→ JSON/CSV/Markdown reports
→ CLI
→ pytest and ruff pass

Required modes:
1. `analysis_only`
   - default mode
   - outputs metrics, risks, score breakdown, warnings, evidence, and reports
   - must not output screening labels
   - must not output trade signals

2. `screening`
   - outputs screening labels:
     - `strong_candidate`
     - `candidate`
     - `watchlist`
     - `avoid_candidate`
     - `insufficient_data`
   - must not output buy/sell/hold/watch/avoid trade signals

3. `trade_signal`
   - explicit opt-in only
   - outputs self-directed research signals:
     - `buy_signal`
     - `hold_signal`
     - `sell_signal`
     - `watch_signal`
     - `avoid_signal`
     - `insufficient_data`
   - every signal must include:
     - label
     - confidence
     - rationale
     - evidence
     - blocking_risks
     - supporting_factors
     - thresholds_used
     - disclaimer

Required disclaimer:
Every report and every trade signal must include:
“This output is for analytical and self-directed research purposes. It is not personalized financial advice.”

Important signal rules:
- Default must be `analysis_only`.
- `trade_signal` must require explicit CLI flag/config.
- Valuation alone must never create a buy_signal or sell_signal.
- buy_signal requires multiple confirming dimensions:
  - high final_score
  - sufficient confidence
  - acceptable risk_score
  - no critical risk flags
  - at least one positive fundamental/growth/momentum/disclosure factor
- sell_signal requires weak final_score or severe risk deterioration.
- insufficient_data must be used when confidence is too low.
- No position sizing.
- No broker execution.
- No portfolio optimization.

Implementation strategy:
Build a reliable v1, not a huge unfinished framework.
Keep code small, typed, and testable.
Use dataclasses or Pydantic v2 models.
Prefer clear deterministic logic over complex ML.
Do not add new heavy dependencies.
Do not add network clients as hard dependencies.

Do not manually create massive CSV fixtures.
For tests requiring long price history, generate deterministic price data inside test helper functions.
Static fixtures can be small but must support CLI smoke tests.
The CLI tests may create temporary CSV files programmatically when needed.

Files to implement:

src/jp_stock_analysis/
- `__init__.py`
- `config.py`
- `schemas.py`
- `errors.py`
- `cli.py`

src/jp_stock_analysis/providers/
- `__init__.py`
- `base.py`
- `local_csv.py`
- `local_json.py`
- `jquants_stub.py`
- `edinet_stub.py`
- `tdnet_stub.py`
- `news_stub.py`

src/jp_stock_analysis/analysis/
- `__init__.py`
- `fundamentals.py`
- `valuation.py`
- `momentum.py`
- `disclosure_nlp.py`
- `risk.py`
- `scoring.py`
- `screening.py`
- `signal_engine.py`

src/jp_stock_analysis/reports/
- `__init__.py`
- `json_report.py`
- `csv_report.py`
- `markdown_report.py`

tests/
- `conftest.py` if useful
- `test_fundamentals.py`
- `test_valuation.py`
- `test_momentum.py`
- `test_disclosure_nlp.py`
- `test_risk.py`
- `test_scoring.py`
- `test_screening.py`
- `test_signal_engine.py`
- `test_reports.py`
- `test_cli.py`

fixtures:
- `tests/fixtures/prices_sample.csv`
- `tests/fixtures/fundamentals_sample.csv`
- `tests/fixtures/company_metadata_sample.csv`
- `tests/fixtures/disclosures/7203.txt`
- `tests/fixtures/disclosures/6758.txt`
- `tests/fixtures/disclosures/9984.txt`

docs:
- `docs/jp_stock_analysis_engine.md`
- `docs/future_rag_integration_separate_project.md`

1. Schemas

Implement in `schemas.py`.

Create these models:
- PriceBar
- FinancialStatement
- CompanyMetadata
- DisclosureDocument
- FundamentalMetrics
- ValuationMetrics
- MomentumMetrics
- DisclosureFinding
- DisclosureAnalysisResult
- RiskFlag
- RiskMetrics
- ScoreBreakdown
- ScreeningResult
- SignalResult
- StockAnalysisResult

Use simple types that serialize easily to JSON.

Each result model should support:
- ticker
- company_name where applicable
- analysis_date
- warnings
- confidence_score
- source_metadata

Use Literal or Enum for:
- SignalMode: `analysis_only`, `screening`, `trade_signal`
- ScreeningLabel
- TradeSignalLabel
- RiskSeverity
- FindingCategory

Do not make schemas too strict for partial data.

2. Config

Implement in `config.py`.

Create:
- DEFAULT_DISCLAIMER
- ScoreWeights
- SignalThresholds
- AnalysisConfig

Defaults:
- signal_mode = `analysis_only`

Score weights:
- quality_score = 0.25
- growth_score = 0.20
- valuation_score = 0.20
- momentum_score = 0.15
- disclosure_score = 0.10
- risk_adjustment = 0.10

Thresholds:
- strong_candidate_threshold = 80
- candidate_threshold = 65
- watchlist_threshold = 50
- avoid_threshold = 35
- buy_signal_threshold = 78
- sell_signal_threshold = 35
- min_confidence_for_signal = 55
- max_risk_score_for_buy_signal = 45

Validate:
- weights non-negative
- thresholds between 0 and 100
- signal_mode is valid

3. Errors

Implement in `errors.py`:
- JPStockAnalysisError
- DataValidationError
- ProviderError
- InsufficientDataError

Use warnings in result objects for normal missing financial data.

4. Providers

Implement local providers.

`providers/base.py`:
Define protocols or abstract classes:
- PriceDataProvider
- FundamentalsProvider
- MetadataProvider
- DisclosureProvider

`providers/local_csv.py`:
Implement:
- load_prices_csv(path)
- load_fundamentals_csv(path)
- load_company_metadata_csv(path)
- load_disclosure_texts(directory)

Requirements:
- read local CSV/text only
- normalize common column name variations
- support multiple tickers
- return schema objects or simple grouped dicts
- handle optional missing columns predictably

`providers/local_json.py`:
- basic local JSON helper
- no network

Stub providers:
- JQuantsProvider
- EDINETProvider
- TDnetProvider
- NewsProvider

Each stub:
- import-safe
- no network
- raises ProviderError with clear message if used
- documents intended future fields

5. Fundamental analysis

Implement in `analysis/fundamentals.py`.

Functions:
- safe_divide(numerator, denominator)
- pct_change(current, previous)
- analyze_fundamentals(current, previous=None)
- analyze_fundamentals_by_ticker(statements)

Metrics:
- revenue_growth_yoy
- operating_income_growth_yoy
- net_income_growth_yoy
- eps_growth_yoy
- operating_margin
- net_margin
- roe
- roa
- equity_ratio
- fcf_margin
- dividend_payout_ratio

Rules:
- zero division returns None and warning
- missing previous-year data returns growth metrics None and warning
- negative EPS allowed but warned
- fiscal period mismatch warned
- missing values reduce confidence
- do not fabricate values

6. Valuation analysis

Implement in `analysis/valuation.py`.

Functions:
- analyze_valuation(statement, market_price, eps_growth_yoy=None)
- classify_valuation(metrics)

Metrics:
- per
- pbr
- psr
- dividend_yield
- peg
- market_cap
- valuation_classification

Rules:
- PER = market_price / EPS
- PBR = market_price / BPS
- market_cap = market_price * shares_outstanding when possible
- PSR = market_cap / revenue
- dividend_yield = dividends_per_share / market_price
- PEG = PER / EPS growth rate when EPS growth is positive
- negative EPS makes PER unavailable with warning
- classification:
  - cheap
  - fair
  - expensive
  - unavailable

7. Momentum analysis

Implement in `analysis/momentum.py`.

Functions:
- analyze_momentum(price_bars)
- calculate_returns
- moving_average
- volatility
- max_drawdown

Metrics:
- return_1m
- return_3m
- return_6m
- return_12m
- moving_average_20d
- moving_average_60d
- moving_average_120d
- moving_average_200d
- volatility_annualized
- max_drawdown
- volume_trend

Rules:
- use adjusted_close if available, otherwise close
- sort by date
- insufficient history returns None metrics and warnings
- confidence drops when history is short
- volume trend compares recent average to prior average when enough data exists

8. Disclosure/NLP analysis

Implement in `analysis/disclosure_nlp.py`.

Create:
- DisclosureNLPProvider protocol/interface
- RuleBasedDisclosureAnalyzer
- NoOpLLMDisclosureAnalyzer

Default:
- RuleBasedDisclosureAnalyzer

The rule-based analyzer must detect Japanese phrases for:
- positive_factor
- negative_factor
- risk_factor
- growth_driver
- management_outlook
- business_environment
- guidance_revision
- one_time_factor
- uncertainty

Each DisclosureFinding:
- category
- summary
- evidence_text
- severity
- confidence
- rule_id

Use simple deterministic keyword/pattern matching.
Do not require LLM.
Do not call network.
Do not infer unsupported claims.

Keyword examples:

Positive:
- 増収
- 増益
- 需要が堅調
- 受注が増加
- 価格改定が寄与
- 利益率が改善

Negative:
- 減収
- 減益
- 需要が減少
- コスト上昇
- 原材料価格の高騰
- 為替の影響

Risk:
- 事業等のリスク
- 重要なリスク
- 継続企業の前提
- 競争激化
- 規制変更
- サプライチェーン
- 金利上昇
- 為替変動

Growth:
- 成長投資
- 新規事業
- 海外展開
- DX
- 研究開発
- 設備投資

Outlook:
- 見通し
- 予想
- 計画
- 中期経営計画
- 通期

Guidance revision:
- 上方修正
- 下方修正
- 業績予想を修正
- 配当予想を修正

One-time:
- 特別利益
- 特別損失
- 一過性
- 減損損失
- 固定資産売却益

Uncertainty:
- 不透明
- 未確定
- 可能性があります
- 懸念
- 変動する可能性

9. Risk analysis

Implement in `analysis/risk.py`.

Function:
- analyze_risks(fundamentals, valuation, momentum, disclosure)

Risk flags:
- negative_eps
- declining_revenue
- declining_operating_income
- high_valuation_weak_growth
- low_equity_ratio
- high_volatility
- large_drawdown
- negative_disclosure_tone
- many_uncertainty_mentions
- insufficient_data

Each RiskFlag:
- risk_id
- severity: low, medium, high, critical
- explanation
- evidence
- confidence

Risk score:
- 0 = low risk
- 100 = high risk

10. Integrated scoring

Implement in `analysis/scoring.py`.

Function:
- score_stock(fundamentals, valuation, momentum, disclosure, risks, config)

Scores:
- quality_score
- growth_score
- valuation_score
- momentum_score
- disclosure_score
- risk_score
- confidence_score
- final_score

Rules:
- 0 to 100 scale
- conservative
- reproducible
- reasons for each sub-score
- missing data lowers confidence
- risk_score adjusts final_score downward
- do not fabricate scores from unavailable metrics

11. Screening

Implement in `analysis/screening.py`.

Functions:
- screen_stocks(results, config)
- assign_screening_label(score_breakdown, config)

Behavior:
- accepts multiple StockAnalysisResult objects
- sorts by final_score descending
- analysis_only: no screening_label
- screening: add screening_label
- trade_signal: keep screening structure compatible; signal is generated separately

Labels:
- strong_candidate
- candidate
- watchlist
- avoid_candidate
- insufficient_data

12. Signal engine

Implement in `analysis/signal_engine.py`.

Functions:
- generate_signal(result, config)
- generate_signals(results, config)

Behavior:
- only generate trade signals when signal_mode is `trade_signal`
- analysis_only and screening must not create buy/sell labels
- include thresholds_used
- include disclaimer

Labels:
- buy_signal
- hold_signal
- sell_signal
- watch_signal
- avoid_signal
- insufficient_data

Rules:
- buy_signal requires:
  - final_score >= buy_signal_threshold
  - confidence_score >= min_confidence_for_signal
  - risk_score <= max_risk_score_for_buy_signal
  - no critical risk flag
  - multiple supporting factors
- sell_signal:
  - final_score <= sell_signal_threshold or critical/severe risk deterioration
- watch_signal:
  - promising but insufficient confirmation
- hold_signal:
  - mixed or neutral
- avoid_signal:
  - high-risk or structurally weak
- insufficient_data:
  - confidence too low or major data unavailable

13. Reports

Implement:
- `reports/json_report.py`
- `reports/csv_report.py`
- `reports/markdown_report.py`

JSON:
- write screening/results as JSON
- include signal only when enabled
- include disclaimer

CSV:
- write screening.csv with:
  - rank
  - ticker
  - company_name
  - final_score
  - quality_score
  - growth_score
  - valuation_score
  - momentum_score
  - disclosure_score
  - risk_score
  - confidence_score
  - screening_label if available
  - trade_signal if available
  - warnings_count

Markdown per ticker:
Sections:
- Executive summary
- Data coverage
- Fundamental metrics
- Valuation metrics
- Momentum metrics
- Disclosure analysis
- Risk flags
- Integrated score
- Screening label, only when mode is screening
- Research signal, only when mode is trade_signal
- Evidence and warnings
- Limitations
- Disclaimer

14. CLI

Implement in `cli.py` using argparse.

Command:

python -m jp_stock_analysis.cli analyze \
  --prices tests/fixtures/prices_sample.csv \
  --fundamentals tests/fixtures/fundamentals_sample.csv \
  --metadata tests/fixtures/company_metadata_sample.csv \
  --disclosures tests/fixtures/disclosures \
  --output-dir /tmp/jp_stock_analysis_out

Optional:
--signal-mode analysis_only|screening|trade_signal

CLI behavior:
1. load inputs
2. group data by ticker
3. use latest price as market price
4. run fundamentals
5. run valuation
6. run momentum
7. run disclosure analysis
8. run risk analysis
9. run scoring
10. run screening
11. run signal engine only when trade_signal
12. write:
    - screening.csv
    - screening.json
    - one markdown report per ticker
13. print output directory

15. Fixtures

Create deterministic small static fixtures:
- `prices_sample.csv`
- `fundamentals_sample.csv`
- `company_metadata_sample.csv`
- disclosure text files:
  - `7203.txt`
  - `6758.txt`
  - `9984.txt`

Use synthetic but realistic data.
Do not claim real company facts.
Tickers may be real-looking but data must be synthetic.

Static CSVs can be compact.
For long-history momentum tests, generate data in test code instead of writing huge fixture files.

16. Tests

Implement deterministic tests.

Required:
- fundamentals:
  - margins, ROE, growth
  - missing previous data
  - negative EPS warning

- valuation:
  - PER/PBR/PSR/dividend_yield/PEG
  - negative EPS handling
  - classification

- momentum:
  - returns, moving averages, volatility, drawdown
  - insufficient history

- disclosure_nlp:
  - positive/negative/risk/uncertainty/guidance detection
  - evidence_text present
  - no LLM/network

- risk:
  - flags negative EPS, declining revenue, high valuation weak growth, large drawdown
  - risk score 0-100

- scoring:
  - reproducible final score
  - risk lowers score
  - missing data lowers confidence

- screening:
  - analysis_only has no screening label
  - screening mode assigns labels
  - descending rank

- signal_engine:
  - analysis_only produces no trade signal
  - screening produces no trade signal
  - trade_signal produces explicit signal with thresholds/evidence/disclaimer
  - valuation alone does not create buy_signal

- reports:
  - JSON report created
  - CSV report created
  - Markdown contains disclaimer and required sections

- CLI:
  - smoke test writes expected outputs to tmp directory
  - test at least analysis_only and trade_signal modes
  - screening mode if runtime remains reasonable

17. Docs

Implement `docs/jp_stock_analysis_engine.md`:
- overview
- supported inputs
- provider strategy
- analysis methods
- scoring logic
- screening modes
- trade_signal mode
- CLI usage
- output files
- limitations
- future J-Quants/EDINET/TDnet/news provider plan

Implement `docs/future_rag_integration_separate_project.md`:
- RAG is out of scope here
- future separate project may ingest JSON/Markdown outputs
- recommended name: `jp_stock_rag_service`
- required export fields:
  - ticker
  - company_name
  - fiscal_year
  - document_type
  - evidence_text
  - analysis_summary
  - risks
  - positive_factors
  - score_breakdown
  - signal if enabled
- do not implement vector DB, embeddings, retrieval API, or chatbot here

18. Validation

Run:
python -m pytest
ruff check .

If failures occur, fix them.
Do not fake success.

Final response must include:
1. Summary
2. Files added/modified
3. How to run
4. Test results
5. Signal mode behavior
6. Generated outputs
7. Known limitations
8. Next recommended implementation step

Begin implementation now.

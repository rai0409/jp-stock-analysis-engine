# Japanese Stock Analysis Engine

A modular, reproducible Python engine for analyzing Japanese stocks from local
data files. Built for self-directed trading research and screening — not for
investment advisory, broker execution, or automated trading.

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## Overview

The engine runs a deterministic pipeline per ticker:

```
local CSV/TXT inputs
  → fundamentals analysis
  → valuation analysis
  → momentum analysis
  → Japanese disclosure (rule-based NLP) analysis
  → risk analysis
  → integrated scoring
  → screening
  → optional trade_signal mode
  → JSON / CSV / Markdown reports
```

Key principles:

- **Never fabricate data.** Missing inputs produce `None` metrics, warnings,
  and reduced confidence — never invented values.
- **Strict separation** of raw inputs, derived metrics, scores, screening
  labels, and trade signals.
- **Valuation alone never creates a buy/sell signal.** Signals require
  multiple confirming dimensions.
- **Reproducible.** Same inputs always produce identical outputs; tests use
  deterministic local fixtures with no network access.

## Supported Inputs

| Input | Format | Required columns / convention |
|---|---|---|
| Prices | CSV | `ticker`, `date`, `close` (optional: open/high/low, `adjusted_close`, `volume`) |
| Fundamentals | CSV | `ticker` plus optional figures (revenue, operating_income, net_income, eps, bps, dividends_per_share, shares_outstanding, total_assets, equity, operating_cash_flow, capital_expenditure) |
| Company metadata | CSV | `ticker` plus optional `company_name`, `sector`, `market` |
| Disclosures | directory of `<ticker>.txt` | UTF-8 Japanese disclosure text |

Common column aliases are normalized automatically (`code`/`symbol` → ticker,
`adj_close` → adjusted_close, `売上高` → revenue, etc.).

## Provider Strategy

`providers/base.py` defines `PriceDataProvider`, `FundamentalsProvider`,
`MetadataProvider`, and `DisclosureProvider` protocols. v1 ships:

- `local_csv.py` / `local_json.py` — the default providers; local files only.
- `jquants.py` — optional cache-first J-Quants V2 provider; offline against
  `.cache/jquants/`, live fetch is explicit opt-in via `JQUANTS_API_KEY`.
  See `docs/jquants_provider.md`.
- `edinet_stub.py`, `tdnet_stub.py`, `news_stub.py` — import-safe placeholders
  that raise `ProviderError` if used. They document the intended future fields
  and keep network clients out of the dependency tree.

## Analysis Methods

- **Fundamentals** (`analysis/fundamentals.py`): YoY growth (revenue,
  operating income, net income, EPS), margins, ROE/ROA, equity ratio, FCF
  margin, dividend payout. Zero-division and missing data return `None` with
  warnings; negative EPS is allowed but warned.
- **Valuation** (`analysis/valuation.py`): PER, PBR, PSR, dividend yield, PEG
  (positive EPS growth only), market cap, and a cheap/fair/expensive/
  unavailable classification.
- **Momentum** (`analysis/momentum.py`): 1/3/6/12-month returns, 20/60/120/
  200-day moving averages, annualized volatility, max drawdown (negative
  percent), volume trend. Uses adjusted close when available; short history
  reduces confidence.
- **Disclosure NLP** (`analysis/disclosure_nlp.py`): deterministic Japanese
  keyword rules across nine categories (positive/negative factors, risk
  factors, growth drivers, outlook, business environment, guidance revisions,
  one-time factors, uncertainty). Every finding carries `evidence_text`,
  `category`, `severity`, `confidence`, and `rule_id`. A `NoOpLLM` analyzer
  exists as a future opt-in extension point; no LLM or network is required.
- **Risk** (`analysis/risk.py`): evidence-backed flags (negative EPS,
  declining revenue/operating income, expensive valuation with weak growth,
  low equity ratio, high volatility, large drawdown, negative disclosure tone,
  repeated uncertainty, insufficient data) aggregated into a 0-100 risk score
  (0 = low risk).

## Scoring Logic

`analysis/scoring.py` maps metrics onto 0-100 sub-scores (quality, growth,
valuation, momentum, disclosure). A sub-score whose inputs are all missing is
`None`, never a fabricated neutral. The final score is the weighted average of
available sub-scores (default weights 0.25/0.20/0.20/0.15/0.10) minus a risk
penalty (`risk_score × 0.10`). Every sub-score includes a human-readable
reason, and the confidence score reflects component coverage and data quality.

## Modes

1. **`analysis_only` (default)** — metrics, risks, score breakdown, evidence,
   warnings, reports. No screening labels, no trade signals.
2. **`screening`** — adds ranked labels: `strong_candidate` (≥80),
   `candidate` (≥65), `watchlist` (≥50, and the conservative 35–50 borderline
   zone), `avoid_candidate` (≤35), `insufficient_data` (confidence below 55
   or no final score). No trade signals.
3. **`trade_signal` (explicit opt-in)** — research signals with thresholds,
   evidence, confidence, rationale, blocking risks, and supporting factors:
   - `buy_signal`: final ≥78 AND confidence ≥55 AND risk ≤45 AND no critical
     flag AND ≥2 non-valuation supporting factors.
   - `sell_signal`: final ≤35, or a critical risk flag with a sub-candidate score.
   - `avoid_signal`: critical flag or risk score ≥70 (≥60 with a weak score).
   - `watch_signal`: promising (≥65) but unconfirmed.
   - `hold_signal`: mixed/neutral.
   - `insufficient_data`: confidence below the minimum.

No position sizing, portfolio allocation, leverage, derivatives, or broker
execution — in any mode.

## CLI Usage

```bash
python -m jp_stock_analysis.cli analyze \
  --prices tests/fixtures/prices_sample.csv \
  --fundamentals tests/fixtures/fundamentals_sample.csv \
  --metadata tests/fixtures/company_metadata_sample.csv \
  --disclosures tests/fixtures/disclosures \
  --output-dir /tmp/jp_stock_analysis_out \
  --signal-mode analysis_only   # or screening / trade_signal
```

Only `--prices` and `--output-dir` are required; missing inputs degrade
gracefully with warnings.

## Output Files

| File | Content |
|---|---|
| `screening.csv` | One ranked row per ticker: scores, confidence, warnings count; label/signal columns only when the mode produced them |
| `screening.json` | Full results + screening payload with disclaimer and mode |
| `<ticker>.md` | Per-ticker report: executive summary, data coverage, all metric tables, disclosure findings with evidence, risk flags, score breakdown with reasons, mode-dependent screening/signal sections, warnings, limitations, disclaimer |

## Limitations

- Rule-based disclosure analysis matches keywords; it does not understand
  context, negation, or nuance.
- Scores are research heuristics calibrated on common-sense bands, not
  backtested predictors.
- No survivorship-bias handling, corporate-action adjustment, or sector
  normalization in v1.
- Synthetic test fixtures use real-looking tickers but contain no real
  company facts.

## Future Provider Plan

- **J-Quants**: daily quotes and statements behind `PriceDataProvider` /
  `FundamentalsProvider`, opt-in via API token; cached locally to keep
  analysis reproducible.
- **EDINET**: securities-report sections (business risks, MD&A) as
  `DisclosureDocument`s for the rule-based analyzer.
- **TDnet**: timely disclosures (earnings/dividend revisions) to drive the
  guidance-revision rules with fresh text.
- **News**: optional headline provider, kept strictly separate from the
  deterministic core.

All future providers stay optional dependencies; tests will continue to run
fully offline. RAG integration is out of scope for this repository — see
`future_rag_integration_separate_project.md`.

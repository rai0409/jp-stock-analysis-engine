# Factor Ranking Validation (research-only)

**Scope:** `modeling/factors.py`, `modeling/baseline_ranker.py`,
`modeling/ranking_metrics.py`.

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice. These metrics measure cross-sectional ranking
> association only and make no predictive or trading-profitability claim.

## Factor groups

| group | factors | needs |
| --- | --- | --- |
| value | earnings_yield, book_to_market, sales_to_price | price + shares |
| quality | roe, roa, operating_margin, equity_ratio | latest statement |
| growth | revenue_growth_yoy, net_income_growth_yoy | prior-year statement |
| momentum | momentum_20d/60d/120d | adjusted-close history |
| risk | volatility, max_drawdown, leverage | price history / equity |
| disclosure | narrative_available, risk_keyword_count, sentiment_placeholder | narrative (placeholder; no LLM) |

A missing or zero denominator yields `None` (never a fabricated value, never a
crash). Each factor has an explicit direction (`FACTOR_DIRECTION`); lower-is-better
factors (volatility, leverage, risk-keyword count) are inverted before ranking.

### Normalisation

Cross-sectional, per decision date: winsorize (clip to a quantile band) →
z-score, or **sector-relative** z-score. `None` stays `None` (a missing-value
indicator); it is never imputed. Zero / degenerate variance maps present values
to 0.0 deterministically.

## Baseline factor ranker

Equal weight within each group (mean of available factor z-scores), then
configurable group weights (conservative equal weights by default). Outputs per
observation: `factor_score`, `factor_rank` (1 = best, dense, per date),
`sector_neutral_factor_score`, `missing_feature_count`, `model_version`. No
buy/sell label — `analysis_only`.

## Ranking metrics (per horizon)

- **Rank IC / Spearman** per decision date (Pearson on average ranks; no scipy).
- **IC mean**, **IC std**, **ICIR** = mean(IC)/std(IC) with safe zero-std (→ `None`).
- **Sector-neutral Rank IC** (score and label demeaned within sector).
- **Top-vs-bottom quantile spread** (mean) and the **quantile return table**.
- **Hit rate**: share of dates whose top-quantile mean return is `> 0`, and `>`
  the universe median that date.
- **Coverage count** and **missing-label count** (never fabricated).
- **Reliability-grade dispersion** when grades are supplied.

Reports are written as JSON, CSV, and Markdown, always carrying the research-only
disclaimer and — on synthetic data — a `SYNTHETIC FIXTURE RESULTS` banner.

## Determinism

All statistics are pure-Python and deterministic. Spearman of perfectly ordered
inputs is `1.0`; reversed is `-1.0`; constant inputs return `None` (no NaN).

## Caveat

On the synthetic bundle, any positive IC is a fixture artefact (a faint
quality→drift link is baked in to keep metrics non-degenerate). It is **not** a
finding and **not** market evidence.

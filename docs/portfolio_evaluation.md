# JPX-Style Long-Short Spread Evaluation (research-only)

**Module:** `src/jp_stock_analysis/modeling/portfolio_metrics.py`
**CLI:** `evaluate-portfolio-ranking` (+ integrated into `modeling-report`)

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## What this is — and is not

A **research metric** inspired by long-short spread competition scoring (e.g. the
JPX Tokyo Stock Exchange Prediction Sharpe-of-spread). It is **not** a trading
system and makes no claim of exchange/execution realism:

- no buy/sell signals, no order routing, no position sizing, no automation;
- no predictive-performance claim;
- it consumes **forward-return labels already produced upstream** (dataset /
  validation) and **never fetches prices** — real inputs must be
  **adjusted-close-derived and point-in-time** upstream;
- synthetic-fixture results are clearly labelled and are **not market evidence**.

## How it works

Per decision date, over the cross-section with both a prediction score and a
realised forward return:

1. **Rank** by prediction score (desc).
2. **Select legs** — top-N long / bottom-N short, or top/bottom **quantile**.
   Legs never overlap; if they would, the date returns `insufficient_names`.
3. **Weight** — `equal` or `rank_weighted` (linearly decreasing weights so the
   most-favourable long / most-unfavourable short get the largest weight); an
   optional per-observation `weight` multiplies the base weight. Weights are
   normalised per leg.
4. **Spread** = `long_leg_return − short_leg_return`.

### Series summary
`mean_spread`, `std_spread`, **Sharpe-like = mean/std** (the JPX metric *type*;
`None` if <2 periods or zero variance), optional **annualized** Sharpe-like when
`periods_per_year` is given, `hit_rate` (share of spreads > 0), a **cumulative
equity curve** (compounding `1 + spread/100`), **max drawdown**, and
worst/best period.

### Turnover
One-sided per leg between consecutive decision dates
(`0.5 · Σ|w_t − w_{t−1}|`); `total_turnover = long + short` (range 0..2).
Initial-build turnover on the first date is excluded. Reports `average_turnover`
and `max_turnover`.

### Transaction cost (optional, simplified)
`net_spread = gross_spread − turnover · cost_bps / 100` (returns in percent;
equivalently, with fractional returns, `gross − turnover·cost_bps/10000`).
**Default 0 bps.** This is a **simplified research approximation, not execution
simulation**.

### Degenerate handling (status, never exceptions)
`insufficient_names` (too few names / leg overlap), `constant_predictions`
(no rank signal), `no_valid_dates`, `degenerate_series` (Sharpe undefined).
Missing returns drop a name from the usable universe deterministically.

## CLI

```
python -m jp_stock_analysis.cli evaluate-portfolio-ranking \
    --synthetic --horizon 20 --portfolio-top-quantile 0.2 \
    --portfolio-bottom-quantile 0.2 --portfolio-rank-weighted \
    --transaction-cost-bps 10 --output-dir out/
```

Outputs `portfolio_metrics.{json,csv,md}`. Use `--prices/--fundamentals/...`
instead of `--synthetic` for file inputs. The `modeling-report` command also
emits a per-horizon long-short summary table.

## When real data exists
Build the dataset with point-in-time disclosure dates and adjusted-close prices,
confirm `check-forward-readiness` is `ELIGIBLE`, then evaluate. Only then is a
Sharpe-of-spread number meaningful — and it remains a research statistic, not a
performance guarantee or a recommendation.

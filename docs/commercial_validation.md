# Commercial Validation & Audit (research-only)

**Modules:** `constraints.py`, `portfolio_metrics.py` (extended), `audit.py`,
`monitoring.py`
**CLI:** `evaluate-portfolio-constraints`, `build-audit-manifest`,
`evaluate-model-monitoring` (+ integrated into `modeling-report`)

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## What this is — and is not

P4 is **commercial-validation infrastructure, NOT commercial-ready proof**. It is
offline, deterministic, synthetic-tested research diagnostics:

- no predictive-performance claim, no buy/sell signal, no trading automation;
- the constrained book is a **feasibility approximation, not a recommended
  portfolio** and not order execution;
- **liquidity/ADV constraints require real liquidity data** to be meaningful;
  synthetic fixtures carry no ADV and ADV is **never fabricated**;
- the transaction-cost model is **simplified**, not execution simulation;
- benchmark-relative and sector-relative returns are research diagnostics;
- audit manifests improve **reproducibility** but do **not** prove model validity;
- synthetic results are clearly labelled and are **not market evidence**.
- Real-data results remain gated on strict no-look-ahead **P0**.

## Constraints (`constraints.py`)

`apply_constraints(book, config, prior_book=None)` applies, deterministically, to
a one-date long/short weight book:

- **Position**: `max_weight_per_name`, `max/min_long_names`, `max/min_short_names`,
  `max_sector_weight` (when sectors exist).
- **Liquidity/ADV**: `min_adv`, `max_participation_rate`,
  `max_notional_fraction_of_adv`. If ADV data is missing the result is
  `liquidity_data_missing` (ADV is never fabricated) unless the caller explicitly
  allows proceeding without it.
- **Turnover**: `max_total_turnover` / `max_leg_turnover` — the book is scaled
  toward prior holdings to meet the limit (deterministic).
- **Infeasible** cases (too few names, all illiquid, impossible sector cap,
  missing columns) return a clear `status` + `infeasible_reason`, never an
  exception.

Output: constrained vs unconstrained weights, gross exposure before/after,
applied constraints, infeasible reason, and turnover before/after when prior
holdings are supplied. Absolute exposure is reduced or preserved, never
increased.

## Portfolio commercial metrics (`portfolio_metrics.py`, extended)

`evaluate_portfolio(...).commercial` (also `compute_commercial_validation`) adds:

- **Benchmark-relative**: per-date universe mean return and sector mean return,
  long/short excess over universe, long excess over its sector benchmark, and a
  benchmark-relative Sharpe-like of the long-leg excess. `universe_excess_returns`
  gives per-name excess that sums to ~0 cross-sectionally.
- **Cost decomposition**: gross spread, turnover cost, liquidity cost (omitted
  unless real liquidity data exists), net spread, and cumulative gross/net equity
  curves.
- **Exposure decomposition**: long/short/net sector exposure (when sectors exist).
- **Concentration**: Herfindahl index, effective number of names, top weight,
  per leg.

Existing portfolio metrics are unchanged; `commercial_validation` is an additive
key.

## Audit manifest (`audit.py`)

`build_audit_manifest(...)` produces a deterministic run manifest: input
fingerprints (content SHA-256, basename, CSV row/column counts — absolute temp
paths excluded by default), model versions, feature/target columns, horizons,
no-look-ahead status, a synthetic-vs-real flag, output files, and warnings. In
`stable` mode an unset `run_id` is derived from a hash of the inputs/config and
the timestamp defaults to the epoch, so **identical inputs yield an identical
manifest and changed inputs change the fingerprint**. **Secrets are scrubbed**
(any key containing key/token/secret/password/credential/api is redacted) and
never serialized.

## Monitoring / drift (`monitoring.py`)

`monitor_metric` / `build_monitoring_report` compute, per metric over ordered
periods: rolling mean/std, a trailing-window z-score, threshold flagging, a
stability band, and worst/best period. Zero trailing std yields no z-score (never
a divide-by-zero); too-few-periods / all-missing / missing-metric cases return a
clear status. Suitable for Rank IC, neutralized Rank IC, long-short spread,
Sharpe-like, turnover, exposure, feature coverage, and prediction-distribution
drift.

## CLI

```
python -m jp_stock_analysis.cli evaluate-portfolio-constraints --synthetic \
    --horizon 20 --max-weight-per-name 0.34 --max-sector-weight 0.6 --output-dir out/
# -> constrained_portfolio.{json,csv,md}

python -m jp_stock_analysis.cli build-audit-manifest --synthetic \
    --input fundamentals.csv --input prices.csv --output-dir out/
# -> audit_manifest.{json,md}   (--run-id / --fixed-timestamp for deterministic runs)

python -m jp_stock_analysis.cli evaluate-model-monitoring --synthetic \
    --horizon 20 --window 3 --z-threshold 2.0 --output-dir out/
# -> monitoring.{json,csv,md}   (or --metrics-csv path with a period column)
```

`modeling-report` includes a "Commercial validation (research diagnostics)"
section built from all four modules.

## Real-data run order

1. `check-forward-readiness` (must be ELIGIBLE).
2. `build-modeling-dataset`.
3. train baseline / ridge / Elastic Net / optional GBDT models.
4. evaluate Rank IC / portfolio spread / neutralization.
5. run walk-forward.
6. evaluate stability.
7. apply constraints and cost/liquidity checks (`evaluate-portfolio-constraints`;
   real ADV required for liquidity to be meaningful).
8. generate audit manifest (`build-audit-manifest`).
9. generate `modeling-report`.

Nothing here constitutes a predictive or trading claim until strict real
no-look-ahead validation succeeds.

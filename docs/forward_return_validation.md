# Forward-Return Validation Harness

## Purpose

The engine produces a point-in-time `screening.json` containing, per ticker,
`final_score`, `confidence_score`, `data_coverage_score`, `screening_score`,
`screening_eligible`, and `reliability_grade`. Until now there was no offline
way to check whether the reliability-aware fields (`screening_score`,
`reliability_grade`, `screening_eligible`) are *more informative about realized
forward returns* than the raw `final_score`.

`validate-forward-returns` closes that gap. It joins a past `screening.json`
with a **later** local prices CSV and reports the realized forward returns each
ticker would have shown over one or more horizons, grouped by the screening
fields. This is descriptive evidence for research — nothing more.

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice. The harness emits **no trading signals, no
> portfolio construction, and no position sizing**. It does not tell you what to
> buy, sell, or hold. Realized historical returns do not predict future returns.

It is fully offline and deterministic: no network, no paid API, no randomness.

## Inputs

### `--screening-json`
A `screening.json` produced by `python -m jp_stock_analysis.cli analyze`. The
harness reads:

- `screening[]` — one entry per ticker, providing the grouping fields
  (`final_score`, `screening_score`, `confidence_score`, `data_coverage_score`,
  `screening_eligible`, `reliability_grade`).
- `results[]` — used only to recover each ticker's `analysis_date`
  (the point-in-time decision date).

### `--prices`
A local CSV with at least `ticker,date,close`. The same column-alias rules as
the `analyze` command apply (`code`/`symbol` for ticker, `終値` for close, etc.).
Rows are grouped by ticker and sorted ascending by date. **This CSV must contain
prices dated after the screening's `analysis_date`** — that is the future window
being measured.

### `--horizons`
Comma-separated positive integers (default `5,20,60`). Each is a number of
**trading rows** (not calendar days) after the base price.

### `--analysis-date` (optional)
`YYYY-MM-DD` fallback used only for tickers whose `results` entry has no
`analysis_date`. If a ticker has no `analysis_date` and no override is supplied,
its forward returns are left uncomputed and a warning is emitted.

## Forward-return definition

For each ticker:

1. `analysis_date` comes from the matching `results` entry, else from
   `--analysis-date`.
2. Price rows are sorted ascending. The **base** row is the first row *strictly
   after* `analysis_date`. A row dated on or before `analysis_date` can never be
   the base or a target (no look-ahead onto the decision date).
3. For horizon `N`, the **target** row is `N` positions after the base row in
   the strictly-after sequence (index `N`, 0-based). The forward return is:

   ```
   (target_close / base_close - 1) * 100      # percent
   ```

4. Prices are never interpolated. If fewer than `N + 1` rows exist strictly
   after `analysis_date`, the horizon is marked `insufficient_history`.

Per-cell `status` values: `ok`, `no_price_data`, `no_base_price`,
`insufficient_history`, `invalid_base_price`.

## CLI usage

```bash
python -m jp_stock_analysis.cli validate-forward-returns \
    --screening-json /path/to/screening.json \
    --prices /path/to/prices.csv \
    --output-dir /tmp/forward_validation_out \
    --horizons 5,20,60
```

Optional flags: `--analysis-date 2024-03-01`, `--no-markdown`.

The `analyze` command is unchanged.

## Output schema

Three files are written to `--output-dir`:

### `forward_returns.json`

```jsonc
{
  "disclaimer": "...research-only... no trading signals...",
  "horizons": [5, 20, 60],
  "ticker_count": 3,
  "warnings": [],
  "per_ticker_forward_returns": [
    {
      "ticker": "1001",
      "analysis_date": "2024-03-01",
      "analysis_date_source": "result",     // "result" | "override" | "missing"
      "final_score": 82.0,
      "screening_score": 78.0,
      "confidence_score": 85.0,
      "data_coverage_score": 100.0,
      "screening_eligible": true,
      "reliability_grade": "high",
      "returns": [
        {
          "horizon": 5,
          "status": "ok",
          "available": true,
          "forward_return": 4.0984,        // percent
          "base_date": "2024-03-04",
          "base_price": 122.0,
          "target_date": "2024-03-11",
          "target_price": 127.0
        }
      ],
      "warnings": []
    }
  ],
  "grouped_summary": [
    {
      "dimension": "screening_eligible",   // also: reliability_grade,
                                           // screening_score_bucket,
                                           // final_score_bucket
      "group": "true",
      "horizon": 5,
      "count": 2,                          // tickers in the group
      "available_horizon_count": 2,
      "missing_horizon_count": 0,
      "mean_forward_return": -0.5824,
      "median_forward_return": -0.5824,
      "hit_rate_positive": 0.5,
      "min_forward_return": -5.2632,
      "max_forward_return": 4.0984
    }
  ]
}
```

Score buckets are fixed-width (10 points): `0-10`, `10-20`, … `90-100`, plus
`none` when the score is absent. Fixed edges (rather than sample deciles) keep
group keys stable regardless of how many tickers are present.

### `forward_returns.csv`
One row per `(ticker, horizon)` with the per-cell return fields plus the carried
screening fields — convenient for spreadsheet pivots.

### `forward_returns.md`
A human-readable summary: the grouped-summary table and a per-ticker table, with
the research-only disclaimer at the top.

## Interpretation

The question this answers: *do the reliability-aware fields separate winners
from losers better than `final_score` alone?* Compare, at each horizon:

- `screening_eligible = true` vs `false`,
- `reliability_grade` high vs medium vs low,
- top vs bottom `screening_score_bucket`,
- top vs bottom `final_score_bucket`.

If (for example) high `reliability_grade` shows a higher `mean_forward_return`
and `hit_rate_positive` than the top `final_score_bucket`, that is evidence the
reliability adjustment adds information. A single run on a handful of tickers is
anecdotal; meaningful conclusions need a broad universe and several
non-overlapping `analysis_date` snapshots.

## Limitations

- Realized returns are **historical and descriptive**; they do not predict
  future returns and are not a backtest of a trading strategy.
- Horizons count **trading rows present in the CSV**, not calendar days. Gaps,
  holidays, or a sparse CSV shift what "20 rows" means.
- Uses raw `close` (not `adjusted_close`); splits/dividends in the price feed
  affect returns. Supply an adjusted series if that matters for your study.
- No transaction costs, slippage, liquidity, or survivorship handling — none of
  which are in scope for a descriptive measurement.
- Small samples and many groups make per-group statistics noisy.

## Not financial advice / out of scope

This harness deliberately does **not**:

- output buy / sell / hold or any trade signal,
- construct a portfolio or allocate capital,
- size positions or apply leverage / margin / derivatives,
- provide personalized financial advice.

It measures realized forward returns and reports descriptive statistics. Keep
`analysis_only` as the engine's default mode; `trade_signal` remains an explicit
opt-in elsewhere and is unaffected by this harness.

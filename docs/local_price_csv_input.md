# Local price CSV input for forward-return validation

## Why this exists

The forward-return validation harness (`validate-forward-returns`, see
`docs/forward_return_validation.md`) measures realized returns *after* an
analysis date. It needs a local `ticker,date,close` price history that the
engine does **not** produce itself: prices are user-supplied. The
`prepare-price-csv` subcommand accepts a local raw price CSV in several common
schemas, validates and normalizes it, and writes the exact shape the harness
consumes.

This step is **offline only**. It never fetches data (no network, no J-Quants,
no EDINET), never fabricates or interpolates prices, and emits no trading
signals. It only reshapes and checks a CSV you already have locally. Sample and
synthetic prices (e.g. `tests/fixtures/prices_sample.csv`, matplotlib's
`Stocks.csv`, or hand-made `/tmp` smoke files) are **not** real data and must
not be treated as predictive validation.

## Required output schema

Exactly three columns, sorted by `(ticker, date)`:

```
ticker,date,close
```

- `ticker` — string listing code, `.T` suffix removed (e.g. `7203.T` → `7203`);
  alphanumeric codes such as `286A` are preserved.
- `date` — `YYYY-MM-DD`.
- `close` — numeric raw close price.

## Accepted source schemas

Header matching is case-insensitive; extra columns are ignored.

- `ticker,date,close`
- `ticker,date,open,high,low,close,volume` (the `close` column is taken; the
  rest of OHLCV is ignored)
- `code,date,close`
- `Code,Date,Close`
- `LocalCode,Date,Close`

Dates may be `YYYY-MM-DD`, `YYYY/MM/DD`, or `YYYYMMDD`. `close` may contain
thousands separators (e.g. `1,234`). Rows whose ticker is not requested are
dropped. Any structural failure (missing column, unparseable date, non-numeric
close, an absent requested ticker, or insufficient coverage) exits non-zero
with a clear message and writes no output file.

## Required coverage for the current topix1000 validation

The three tickers with real EDINET-derived fundamentals are **3928, 4107,
4264** (analysis date `2026-03-27`). To validate forward returns at horizons
`5,20,60`, each ticker needs at least **60 trading rows on or after
2026-03-28** (the day after the analysis date). Supply that coverage in your
local raw CSV; the preparation step enforces it via `--min-rows-after 60`.

## Commands

### 1. Prepare the price CSV

```bash
python -m jp_stock_analysis.cli prepare-price-csv \
  --input /path/to/local_raw_prices.csv \
  --output /tmp/topix1000_forward_prices.csv \
  --tickers 3928,4107,4264 \
  --from-date 2026-03-28 \
  --min-rows-after 60
```

### 2. Run analyze (produces `screening.json`) with the topix1000 bundle

```bash
python -m jp_stock_analysis.cli analyze \
  --prices /tmp/topix1000_forward_prices.csv \
  --metadata /tmp/topix1000_engine_bundle/metadata.csv \
  --fundamentals /tmp/topix1000_engine_bundle/fundamentals.csv \
  --disclosure-provider topix1000-export \
  --topix1000-export-dir /tmp/topix1000_annual_report_export_linked \
  --output-dir /tmp/jstocks_topix1000_forward_input \
  --signal-mode analysis_only
```

### 3. Run forward-return validation

```bash
python -m jp_stock_analysis.cli validate-forward-returns \
  --screening-json /tmp/jstocks_topix1000_forward_input/screening.json \
  --prices /tmp/topix1000_forward_prices.csv \
  --output-dir /tmp/jstocks_forward_validation_topix1000 \
  --horizons 5,20,60
```

(Invoke with the repo venv and `PYTHONPATH=src`, e.g.
`PYTHONPATH=src .venv/bin/python -m jp_stock_analysis.cli …`.)

## Caveats

- **Raw close, not adjusted close.** The prepared CSV uses raw close prices.
  Corporate actions (splits, dividends) are not accounted for; a `.T`-suffix
  source does not imply adjusted data. Forward returns computed from raw closes
  can be distorted around such events.
- **No predictive conclusion until real prices are supplied and the harness is
  run.** As of this writing no real local price history for 3928/4107/4264
  after 2026-03-27 has been found on this machine, so no real forward-return
  validation has been completed. The numbers only become meaningful once a
  genuine local CSV is prepared with the command above and the harness is run.
- **Small sample.** Even with real prices, only 3 tickers have fundamentals;
  any result is descriptive, not statistically significant.
- **No financial advice, no trading automation.** This pipeline is a
  self-directed research tool. It produces no buy/sell/hold signals, no
  portfolio construction, and no position sizing, and is not personalized
  financial advice.

# J-Quants Daily Bars Pipeline

This pipeline builds a production-usable local daily bars foundation for the
TOPIX1000 usable universe. It complements the existing adjusted-close store
without modifying it.

## Confirmed Live Fields

The live J-Quants V2 endpoint is:

```text
/v2/equities/bars/daily?date=YYYY-MM-DD
```

Rows are under the `data` key. Confirmed fields include:

```text
Date, Code, O, H, L, C, UL, LL, Vo, Va, AdjFactor, AdjO, AdjH, AdjL, AdjC, AdjVo
```

The CLI uses the existing `JQuantsProvider` request path. Live requests use the
`x-api-key` header. It does not use an Authorization Bearer token.

## Why These Fields Matter

`adj_close` is the anchor for returns, labels, and consistency with the existing
adjusted-close store. `turnover_value` is essential for commercial-grade
liquidity analysis because share volume alone is not comparable across price
levels. `volume` supports activity and spike detection. Adjusted open, high,
and low support gap, range, intraday return, and volatility features.
`adjustment_factor` supports split and adjustment checks. Raw OHLC is retained
for audit, but adjusted OHLC should be preferred for modeling. Upper and lower
limit flags help identify stop-limit events and outlier regimes.

## Relation To Existing Adjusted Close Store

The existing file:

```text
/tmp/jquants_topix1000_price_store/prices_adjusted_close.csv
```

has a `close` column that represents adjusted close when it was collected with
`--price-field adjusted_close`. The daily bars verifier compares:

```text
prices_daily_bars.csv adj_close
prices_adjusted_close.csv close
```

Tiny floating point differences are tolerated. The daily bars pipeline never
overwrites the adjusted-close store.

## Outputs

Daily bars CSV:

```text
{store_dir}/prices_daily_bars.csv
```

Columns:

```text
ticker,date,adj_close,turnover_value,volume,adj_open,adj_high,adj_low,
adjustment_factor,open,high,low,close,adj_volume,upper_limit_flag,
lower_limit_flag,source_fields_json
```

Fetch state:

```text
{store_dir}/daily_bars_fetch_state.json
```

Quality report:

```text
{store_dir}/daily_bars_quality_report.json
```

Field coverage report:

```text
{store_dir}/daily_bars_field_coverage_report.json
```

Feature output:

```text
/tmp/jquants_topix1000_price_store/daily_bars_analysis_features.csv
```

## Recommended First Run

Use a three-day window first:

```bash
cd PROJECT_ROOT

set -a
[ -f .env ] && . ./.env
set +a

export JQUANTS_API_VERSION=v2
export JQUANTS_DAILY_QUOTES_PATH=/equities/bars/daily

.venv/bin/python -m jp_stock_analysis.cli fetch-jquants-daily-bars-incremental \
  --universe-file /tmp/jquants_topix1000_price_store/topix1000_usable_tickers.csv \
  --store-dir /tmp/jquants_topix1000_price_store \
  --start-date 2024-12-11 \
  --end-date 2024-12-13 \
  --sleep-seconds 90 \
  --max-retries 2 \
  --allow-network
```

Then verify:

```bash
.venv/bin/python -m jp_stock_analysis.cli verify-jquants-daily-bars \
  --store-dir /tmp/jquants_topix1000_price_store \
  --universe-file /tmp/jquants_topix1000_price_store/topix1000_usable_tickers.csv \
  --adjusted-close-file /tmp/jquants_topix1000_price_store/prices_adjusted_close.csv
```

Then build features:

```bash
.venv/bin/python -m jp_stock_analysis.cli build-daily-bars-analysis-features \
  --daily-bars-file /tmp/jquants_topix1000_price_store/prices_daily_bars.csv \
  --coverage-file /tmp/jquants_topix1000_price_store/topix1000_universe_price_master_coverage.csv \
  --output-file /tmp/jquants_topix1000_price_store/daily_bars_analysis_features.csv
```

## API Limit Guidance

The ingestion command fetches by date and defaults to `--sleep-seconds 90` and
`--max-retries 2`. Keep the first runs small, inspect state and quality reports,
then extend the date range incrementally. Failures are recorded by date and do
not delete existing rows. Non-trading days or empty responses are recorded as
empty dates and are non-fatal.

## No-Look-Ahead Caution

The feature builder computes rolling values using only current and prior rows
within each ticker. Downstream modeling still needs decision-date alignment:
features for a date should only be used where that date's market data would have
been known before the prediction or ranking decision.

## Raw Versus Adjusted OHLC

Raw OHLC (`open`, `high`, `low`, `close`) reflects the exchange-reported price
scale at the time. Adjusted OHLC (`adj_open`, `adj_high`, `adj_low`,
`adj_close`) is adjusted for corporate actions and should be used for return,
gap, range, volatility, and modeling-oriented features. Raw OHLC remains in the
CSV for audit and source-data inspection.

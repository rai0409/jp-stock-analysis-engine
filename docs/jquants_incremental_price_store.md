# Incremental J-Quants Price Store (research-only)

This collector builds a local `ticker,date,close` price store from J-Quants V2
daily bars by date. It is research-data infrastructure only. It does not produce
buy/sell recommendations, trading automation, or predictive-performance claims.

## Why Incremental

J-Quants Free-plan coverage is delayed/rolling. A row that is available today may
leave the live API window later. The local store preserves already-fetched
adjusted-close rows so historical readiness checks can keep using the same
locally retained data without fabricating missing prices.

## Store Layout

`fetch-jquants-prices-incremental` writes:

- `prices_adjusted_close.csv`
- `fetch_state.json`
- `coverage_report.json`
- `eligibility_report.json`
- `logs/fetch_<timestamp>.json`

The CSV schema is always:

```text
ticker,date,close
```

When `--price-field adjusted_close` is used, the `close` column contains J-Quants
`AdjC` / `PriceBar.adjusted_close` for downstream compatibility. If any returned
row lacks `AdjC`, the date fails clearly. There is no raw-close fallback.

## Usage

```bash
.venv/bin/python -m jp_stock_analysis.cli fetch-jquants-prices-incremental \
  --universe-file /tmp/topix1000_tickers.csv \
  --store-dir /tmp/jquants_topix1000_price_store \
  --start-date 2024-03-25 \
  --price-field adjusted_close \
  --allow-network
```

Defaults are conservative for the Free plan:

- `--sleep-seconds 13`
- `--max-retries 8`
- `--backoff-multiplier 2.0`

Without `--allow-network`, the command only uses local date caches and is safe
for offline tests or dry runs.

## Resume Behavior

On each run the collector reads `prices_adjusted_close.csv` and
`fetch_state.json`, computes weekdays from `--start-date` through `--end-date`
or today, and fetches only dates whose stored rows do not cover the full
universe. Rows are deduplicated by `(ticker, date)` and sorted by ticker/date.

If one date fails after retries:

- default: stop safely after recording the failed date in `fetch_state.json`;
- with `--continue-on-date-error`: record the failed date and continue.

The collector refuses to mix `price_field` values in an existing store.

## 429 Handling

HTTP 429 / rate-limit errors are retried with exponential backoff:

`sleep_seconds`, then `sleep_seconds * backoff_multiplier`, and so on up to
`max_retries`. Retry events are logged without secrets.

## Verification

```bash
.venv/bin/python -m jp_stock_analysis.cli verify-price-store \
  --store-dir /tmp/jquants_topix1000_price_store \
  --universe-file /tmp/topix1000_tickers.csv
```

The verification reports:

- row count, ticker count, date range;
- duplicate ticker/date row count;
- missing universe tickers;
- per-ticker row-count summary;
- latest h5/h20/h60 eligible decision dates from actual rows;
- excluded tickers and reasons.

## Limitations

The TOPIX1000 universe file is a fixed selected universe unless you supply a
point-in-time membership file. That creates survivor-bias risk for historical
research. The collector solves local price retention and adjusted-close
integrity; it does not by itself make a backtest point-in-time-valid.

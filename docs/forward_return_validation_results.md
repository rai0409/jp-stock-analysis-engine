# Forward-return validation results

**Status: BLOCKED — no real run completed.**

As of 2026-06-13 no real forward-return validation has been performed, because
no real local price history exists on this machine for the three tickers with
real EDINET-derived fundamentals (3928, 4107, 4264) covering dates after the
analysis date 2026-03-27.

## What was searched (offline, no network)

- `tests/fixtures/prices_sample.csv` — sample tickers 6758/7203/9984, **not**
  the topix1000 tickers; fixture data, not real.
- `/tmp/*.csv` — only tiny synthetic smoke files (2 dates, round numbers) for
  3928/4107/4264; explicitly not real prices.
- `LOCAL_DATA_DIR/...` — EDINET raw/derived data and feature marts; **no price
  CSV**.
- No J-Quants cache present (`.cache/jquants` absent).

The forward-return harness itself is implemented, tested (no-look-ahead
verified), and deterministic; the only missing input is real prices.

## How to unblock (exact commands)

1. Obtain a real local raw price CSV for 3928/4107/4264 with at least 60
   trading rows on or after 2026-03-28 (offline; do not fabricate). Accepted
   source schemas are listed in `docs/local_price_csv_input.md`.

2. Normalize it:

   ```bash
   python -m jp_stock_analysis.cli prepare-price-csv \
     --input /path/to/local_raw_prices.csv \
     --output /tmp/topix1000_forward_prices.csv \
     --tickers 3928,4107,4264 \
     --from-date 2026-03-28 \
     --min-rows-after 60
   ```

3. Run analyze with the topix1000 bundle:

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

4. Run validation:

   ```bash
   python -m jp_stock_analysis.cli validate-forward-returns \
     --screening-json /tmp/jstocks_topix1000_forward_input/screening.json \
     --prices /tmp/topix1000_forward_prices.csv \
     --output-dir /tmp/jstocks_forward_validation_topix1000 \
     --horizons 5,20,60
   ```

Then replace this file with the real results, documenting the source path, row
counts per ticker, and horizons. Note the caveats: raw (not adjusted) close,
and a 3-ticker sample is descriptive only, not statistically significant. This
is a research tool: no trading signals, no portfolio construction, no financial
advice.

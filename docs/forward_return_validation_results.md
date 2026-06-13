# Forward-return validation results

**Status: BLOCKED — no real run completed.**

As of 2026-06-13 no real forward-return validation has been performed, because
no real local price history exists on this machine for the three tickers with
real EDINET-derived fundamentals (3928, 4107, 4264) covering dates after the
analysis date 2026-03-27, and no J-Quants credentials are available to fetch
it offline-safely.

## What was searched (offline, no network)

- Local J-Quants cache: only the synthetic test fixture
  `tests/fixtures/jquants_cache/daily_quotes/7203.json` (ticker 7203, sample
  data) exists — **not** the target tickers, and not real.
- `tests/fixtures/prices_sample.csv` — sample tickers 6758/7203/9984; fixture
  data, not real.
- `/tmp/*.csv` — only tiny synthetic smoke files (2 dates, round numbers) for
  3928/4107/4264; explicitly not real prices.
- `LOCAL_DATA_DIR/...` — EDINET raw/derived data and feature marts; **no price
  CSV**, no rows matching the target tickers.

## Environment check (names only; no values printed)

- `JQUANTS_API_KEY`: **MISSING**
- `JQUANTS_API_BASE_URL`: MISSING (optional override)
- `JQUANTS_API_VERSION`: MISSING (optional override)
- `JQUANTS_DAILY_QUOTES_PATH`: MISSING (optional override)

The engine's J-Quants provider authenticates with a single `JQUANTS_API_KEY`
sent as the `x-api-key` header (see `docs/jquants_provider.md`); it does **not**
use email/password/refresh-token. A live fetch therefore requires
`JQUANTS_API_KEY` to be exported in the environment. Note also that the
provider's default `/v2/...` endpoint paths are **UNVERIFIED** (a live probe
returned HTTP 403); if the live fetch fails on endpoint resolution, override
`JQUANTS_API_BASE_URL` / `JQUANTS_API_VERSION` / `JQUANTS_DAILY_QUOTES_PATH`
per the official spec at https://jpx-jquants.com/spec/.

## How to unblock (exact commands)

The acquisition path now exists: `fetch-jquants-prices` (cache-only by default;
`--allow-network` permits a live fetch when the cache is missing).

1. Export credentials (value never shown):

   ```bash
   export JQUANTS_API_KEY=...   # required for a live fetch
   ```

2. Fetch real raw prices (network only with `--allow-network`):

   ```bash
   PYTHONPATH=src python -m jp_stock_analysis.cli fetch-jquants-prices \
     --tickers 3928,4107,4264 \
     --from-date 2026-03-28 \
     --out /tmp/topix1000_forward_prices_raw.csv \
     --allow-network
   ```

   (Omit `--allow-network` to reuse an existing local cache offline.)

3. Normalize / validate coverage:

   ```bash
   PYTHONPATH=src python -m jp_stock_analysis.cli prepare-price-csv \
     --input /tmp/topix1000_forward_prices_raw.csv \
     --output /tmp/topix1000_forward_prices.csv \
     --tickers 3928,4107,4264 \
     --from-date 2026-03-28 \
     --min-rows-after 60
   ```

4. Analyze with the topix1000 bundle:

   ```bash
   PYTHONPATH=src python -m jp_stock_analysis.cli analyze \
     --prices /tmp/topix1000_forward_prices.csv \
     --metadata /tmp/topix1000_engine_bundle/metadata.csv \
     --fundamentals /tmp/topix1000_engine_bundle/fundamentals.csv \
     --disclosure-provider topix1000-export \
     --topix1000-export-dir /tmp/topix1000_annual_report_export_linked \
     --output-dir /tmp/jstocks_topix1000_forward_input \
     --signal-mode analysis_only
   ```

5. Validate forward returns:

   ```bash
   PYTHONPATH=src python -m jp_stock_analysis.cli validate-forward-returns \
     --screening-json /tmp/jstocks_topix1000_forward_input/screening.json \
     --prices /tmp/topix1000_forward_prices.csv \
     --output-dir /tmp/jstocks_forward_validation_topix1000 \
     --horizons 5,20,60
   ```

Then replace this file with the real results: source, per-ticker row counts,
analysis date, horizons, output paths, grouped-summary interpretation, and
whether `screening_score`/`reliability_grade` ordered forward returns better
than raw `final_score`.

## Caveats (apply even once unblocked)

- **No predictive conclusion can be drawn** until real prices are supplied and
  the harness is run; nothing here is validated.
- **Raw close, not adjusted close** — corporate actions are not accounted for.
- **Small sample**: only 3 tickers have fundamentals; any result is descriptive,
  not statistically significant.
- **No-look-ahead** is enforced by the harness (base price is the first row
  strictly after the analysis date).
- This is a self-directed research tool: no trading signals, no portfolio
  construction, no position sizing, and not personalized financial advice.

**Do not tag** a release on this state: no real validation has been completed.

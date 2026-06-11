# J-Quants Provider (Optional, Cache-First)

The `JQuantsProvider` (`src/jp_stock_analysis/providers/jquants.py`) adds
optional J-Quants V2 API support behind the existing provider protocols
(`PriceDataProvider`, `FundamentalsProvider`, `MetadataProvider`). It does not
change analysis behavior: data from J-Quants flows into the same `PriceBar` /
`FinancialStatement` / `CompanyMetadata` schemas as local CSV files.

## Setting the API key

Live fetching requires a J-Quants API key in an environment variable:

```bash
export JQUANTS_API_KEY="your-api-key"
```

- **Never commit the key.** Do not put it in files in this repository;
  `.env`-style files are gitignored and are not read by this engine.
- The engine only reads `JQUANTS_API_KEY` from the process environment, and
  only when a live fetch actually happens.

## Cache-first design

```
get_prices("7203")
  └─ .cache/jquants/daily_quotes/7203.json exists?
       ├─ yes → read it (offline, no key needed)
       └─ no  → live mode enabled?
                 ├─ no  → ProviderError (cache mode never fetches)
                 └─ yes → key set? → fetch → write cache → return
```

- Cache layout is deterministic: `<cache_dir>/<dataset>/<code>.json` with
  datasets `daily_quotes`, `statements`, `listed_info` (JSON list of raw rows).
- Default cache directory: `.cache/jquants/` (gitignored).
- Cache reads work fully offline; cache writes happen only on explicit live
  fetch.

## CLI usage

```bash
# offline, against an existing cache (no API key required)
python -m jp_stock_analysis.cli analyze \
  --provider jquants-cache \
  --jquants-cache-dir .cache/jquants \
  --jquants-code 7203 --jquants-code 6758 \
  --output-dir outputs/jq

# live opt-in: fetch missing data, write cache, then analyze
JQUANTS_API_KEY=... python -m jp_stock_analysis.cli analyze \
  --provider jquants-live \
  --jquants-code 7203 \
  --from-date 2025-01-01 --to-date 2025-06-30 \
  --output-dir outputs/jq
```

- The default provider remains `local` (CSV/TXT files); existing commands are
  unchanged.
- `--disclosures` still works with J-Quants providers (local text files).
- Prices are mandatory per code; missing statement/metadata caches degrade to
  stderr warnings, mirroring optional local inputs.

## Field-mapping assumptions

Endpoint paths, the `x-api-key` auth header, and response field names are
adapter-level assumptions isolated in `_DATASETS` and the `_map_*` helpers of
`jquants.py` — re-check the official J-Quants V2 documentation before first
live use. Notable mappings: `AdjustmentClose` → `adjusted_close`,
`NetSales` → `revenue`, `Profit` → `net_income`, fiscal year = calendar year
the reporting period ends in. J-Quants statements have no capital-expenditure
column, so `capital_expenditure` stays `None` (never fabricated).

## Tests are offline

All J-Quants tests run against synthetic cache fixtures under
`tests/fixtures/jquants_cache/` plus an injected fake transport for the fetch
logic. No test makes a network call or needs `JQUANTS_API_KEY`.

## Plan and licensing notes

- Free-plan J-Quants subscriptions deliver data with a delay (recent weeks may
  be missing); paid plans reduce the lag. Expect `to-date`-recent gaps
  depending on your plan.
- J-Quants raw data must not be redistributed. Cache files are for your local
  research only; this repository ships only synthetic fixtures.

## Out of scope

RAG, EDINET, TDnet, broker execution, auto-trading, position sizing, leverage,
margin, derivatives, and portfolio allocation remain out of scope for this
repository.

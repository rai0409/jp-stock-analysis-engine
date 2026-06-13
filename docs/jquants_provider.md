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

## Endpoint configuration (V2, verified 2026-06-13)

Live probes on 2026-06-13 confirmed J-Quants **V1 is retired** (HTTP 410,
`J-QuantsはV2に移行しました。`) and that V2 **restructured the routes**. The old
`/v2/prices/daily_quotes` guess returns HTTP 403 "endpoint does not exist"; the
correct V2 routes were verified at HTTP 200 with the `x-api-key` header. The API
key must **not** be sent as a Bearer token (the gateway rejects an
`Authorization` header). The provider therefore:

- sends the dashboard-issued API key as the `x-api-key` header only;
- defaults to the verified V2 routes and reads rows from the top-level `data`
  key; field names are mapped V2-first with V1 fallbacks (older caches still
  load);
- keeps every endpoint component overridable (explicit constructor arguments
  win over the environment):

| Environment variable | Default (V2, verified) | V1 (retired) |
|---|---|---|
| `JQUANTS_API_BASE_URL` | `https://api.jquants.com` | — |
| `JQUANTS_API_VERSION` | `v2` | `v1` |
| `JQUANTS_DAILY_QUOTES_PATH` | `/equities/bars/daily` | `/prices/daily_quotes` |
| `JQUANTS_STATEMENTS_PATH` | `/fins/summary` | `/fins/statements` |
| `JQUANTS_LISTED_INFO_PATH` | `/equities/master` | `/listed/info` |

Resolved URL: `<base>/<version><path>`. Error messages distinguish auth
failure, endpoint-not-found, V1-gone/migrated, and plan/date-coverage limits;
they never contain the API key. Cache reads ignore endpoint configuration
entirely. Migration guide: https://jpx-jquants.com/ja/spec/migration-v1-v2.

> A stale local `.env` pinning `JQUANTS_API_VERSION='v1'` /
> `JQUANTS_DAILY_QUOTES_PATH='/prices/daily_quotes'` overrides the correct V2
> defaults and reintroduces the HTTP 410. Remove or update those two lines.

## Field-mapping assumptions

Response field names are isolated in `_DATASETS` and the `_map_*` helpers of
`jquants.py`. V2 abbreviated names are mapped first, with V1 names as fallback:
`C`/`Close` → close, `AdjC`/`AdjustmentClose` → `adjusted_close`, `Vo`/`Volume`
→ volume; `Sales`/`NetSales` → `revenue`, `OP`/`OperatingProfit` →
`operating_income`, `NP`/`Profit` → `net_income`, `Eq`/`Equity` → equity,
`TA`/`TotalAssets` → total assets, `CFO` → operating cash flow. Fiscal year =
calendar year the reporting period ends in. J-Quants statements have no
capital-expenditure column, so `capital_expenditure` stays `None` (never
fabricated). The V2 feed also exposes adjusted OHLC (`AdjC`/`AdjO`/…), mapped to
`PriceBar.adjusted_close`. The price exporter (`fetch-jquants-prices`) writes
raw close by default and adjusted close on request via
`--price-field adjusted_close` (output column stays named `close`; the export
fails clearly if any requested row lacks an adjusted close — no silent
fallback). See `docs/local_price_csv_input.md`.

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

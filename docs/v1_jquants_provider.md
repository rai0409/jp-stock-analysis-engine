# v1 J-Quants Provider — Validation Note

**Date:** 2026-06-11
**Scope:** Optional J-Quants V2 provider with local cache, behind existing
provider protocols. Analysis pipeline behavior unchanged. No RAG, EDINET,
TDnet, broker execution, auto-trading, position sizing, leverage, margin,
derivatives, or portfolio allocation.

## Files changed

- `src/jp_stock_analysis/providers/jquants.py` — **new**: cache-first provider,
  V2 adapter assumptions isolated in `_DATASETS` / `_map_*` helpers, pagination
  support, injectable transport for offline testing.
- `src/jp_stock_analysis/providers/jquants_stub.py` — **removed** (superseded).
- `src/jp_stock_analysis/providers/__init__.py` — import updated.
- `src/jp_stock_analysis/cli.py` — split `run_analysis` into
  `analyze_data(prices, fundamentals, metadata, disclosures, ...)` +
  local-file `run_analysis` wrapper (same signature as before); added
  `--provider local|jquants-cache|jquants-live`, `--jquants-cache-dir`,
  `--jquants-code` (repeatable), `--from-date`, `--to-date`; engine errors now
  exit 1 with a clean stderr message.
- `tests/test_jquants_provider.py` — **new**: 12 offline tests.
- `tests/test_providers.py` — stub assertions for J-Quants moved/replaced.
- `tests/fixtures/jquants_cache/` — synthetic cache fixtures
  (`daily_quotes/7203.json` 80 bars, `statements/7203.json` 2 fiscal years,
  `listed_info/7203.json`).
- `.gitignore` — added `.cache/`.
- `docs/jquants_provider.md` — **new** usage/design doc.

## Provider modes

| Mode | Network | API key | Behavior |
|---|---|---|---|
| `local` (default) | no | no | CSV/TXT files, unchanged |
| `jquants-cache` | no | no | reads `.cache/jquants/`; `ProviderError` on cache miss |
| `jquants-live` | opt-in | `JQUANTS_API_KEY` | cache-first; fetches+writes cache only on miss |

## Cache format

Deterministic paths `<cache_dir>/<dataset>/<code>.json`, datasets
`daily_quotes` / `statements` / `listed_info`, each a JSON list of raw J-Quants
rows. Default `<cache_dir>` is `.cache/jquants/` (gitignored, never committed).

## Environment variables

- `JQUANTS_API_KEY` — read from the process environment only, required only at
  the moment a live fetch happens. Never read from files; never hardcoded.

## Commands run

```
python -m pytest          → 99 passed
ruff check .              → All checks passed
CLI local fixture smoke   → exit 0, reports written (default behavior unchanged)
CLI jquants-cache smoke   → exit 0 offline against synthetic fixtures, no key set
CLI jquants-live, no key  → exit 1, clean error naming JQUANTS_API_KEY, no output written
```

## Test results

99 passed (87 pre-existing + 12 new), all deterministic and offline. The fetch
path (auth header, pagination, cache write-back) is tested with an injected
fake transport — no real network anywhere. Existing analysis_only / screening /
trade_signal behavior, default mode, and valuation-alone buy protection are
unchanged and still covered by the pre-existing tests.

## Limitations

- Endpoint paths, the `x-api-key` header, and response field names are
  documented assumptions; verify against official J-Quants V2 docs before
  first live use (all mapping is isolated in `jquants.py`).
- No capital-expenditure column in J-Quants statements → `fcf_margin` is
  unavailable for J-Quants-sourced fundamentals (warned, not fabricated).
- Free-plan data delay applies to live fetches depending on the subscription.
- Cache invalidation is manual (delete the per-code JSON file to refetch).
- J-Quants raw data must not be redistributed; only synthetic fixtures ship.

## Next recommended step

Run a real `jquants-live` fetch for one code with your own key, eyeball the
written cache against the documented field assumptions, and adjust the
`_map_*` candidate names if the official V2 schema differs. After that, the
EDINET disclosure provider is the next functional gap (real disclosure text
instead of local files).

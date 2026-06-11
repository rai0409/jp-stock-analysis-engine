# v1 J-Quants Endpoint Diagnosis

**Date:** 2026-06-11
**Scope:** Diagnose and harden the J-Quants live-fetch endpoint/auth
assumptions based on real probe results. No live API calls were made during
this step; all new tests use a fake transport. Cache mode and the analysis
pipeline are unaffected.

## Probe results (from `tools/jquants_probe.py`)

| Probe | Result |
|---|---|
| `GET /v2/prices/daily_quotes` with `x-api-key` | HTTP 403 — `"The requested endpoint does not exist. Please check the URL, HTTP method, and API version:https://jpx-jquants.com/spec/"` |
| `GET /v2/prices/daily_quotes` with `Authorization: Bearer <api-key>` | HTTP 403 — Authorization header malformed |
| v1 legacy token flow | skipped (`JQUANTS_EMAIL`/`JQUANTS_PASSWORD` not set) |

## Root cause classification

**Adapter-level endpoint/version mismatch — not an auth-key problem and not a
pipeline problem.** The service explicitly says the `/v2/...` path does not
exist for this method/version, so the provider's assumed V2 paths are wrong
(or the v2 surface differs from the assumed layout). Separately, the probe
confirms the API key is not usable as a Bearer token, validating the
provider's existing `x-api-key` header choice. `jquants-cache` mode and all
analysis behavior are unaffected.

## What changed (`src/jp_stock_analysis/providers/jquants.py`)

1. **Endpoint components are now configurable** — base URL, API version, and
   per-dataset paths are resolved at construction time as
   *explicit argument > environment variable > default*:
   `JQUANTS_API_BASE_URL` (default `https://api.jquants.com`),
   `JQUANTS_API_VERSION` (default `v2`), `JQUANTS_DAILY_QUOTES_PATH`,
   `JQUANTS_STATEMENTS_PATH`, `JQUANTS_LISTED_INFO_PATH`. The resolved URL is
   `<base>/<version><path>`; defaults reproduce the previous URLs exactly.
   A public `endpoint_url(dataset)` exposes the resolution for diagnostics.
2. **HTTP error diagnostics** — `urllib.error.HTTPError` bodies are read
   (truncated to 300 chars) and classified:
   - body contains "endpoint does not exist" → `ProviderError` explaining the
     likely version/path mismatch and naming the exact override variables and
     the official spec URL;
   - body indicates a malformed/Bearer Authorization problem →
     `ProviderError` stating the API key is not a Bearer token and that this
     provider sends `x-api-key`;
   - anything else → `ProviderError` with status code and truncated body.
   **Error messages never contain the API key** (asserted by tests).
3. Documentation: module docstring and `docs/jquants_provider.md` now record
   the probe findings and the override table.

Nothing changed in cache behavior, schema mapping, the CLI, or any analysis
module.

## Tests added (all offline, fake transport)

- "endpoint does not exist" body → helpful `ProviderError` naming
  `JQUANTS_API_VERSION`/path variables and the spec URL; key never leaked.
- malformed-Bearer body → "not a Bearer token" / "x-api-key" guidance.
- other HTTP errors → status + body reported (HTTP 429 case).
- env overrides change the requested URL
  (`https://alt.example/v9/markets/daily?...`); constructor args win over env.
- cache-first unchanged: with a cache hit, a poisoned endpoint config and a
  transport that fails the test if called are never touched.
- default endpoint URLs pinned (`https://api.jquants.com/v2/...`).
- an autouse fixture clears the five endpoint env vars so developer shells
  cannot affect the suite.

## Commands run

```
python -m pytest          → 131 passed
ruff check .              → All checks passed (probe tool lint auto-fixed)
CLI jquants-cache smoke   → exit 0 offline, even with JQUANTS_API_VERSION=v999 set
CLI local fixture smoke   → exit 0
```

## Manual live command to try next

Consult https://jpx-jquants.com/spec/ for the correct paths, then re-probe
without code changes (key comes from your shell env; it is never printed):

```bash
# example: try the v1 surface (note: v1 may require the legacy
# refresh-token flow, which this provider intentionally does not implement)
JQUANTS_API_VERSION=v1 python tools/jquants_probe.py

# once the spec confirms the right base/version/path, run a single live fetch:
JQUANTS_API_BASE_URL=https://api.jquants.com \
JQUANTS_API_VERSION=<version-from-spec> \
JQUANTS_DAILY_QUOTES_PATH=<path-from-spec> \
python -m jp_stock_analysis.cli analyze --provider jquants-live \
  --jquants-code 7203 --from-date 2025-01-01 --to-date 2025-03-31 \
  --output-dir outputs/jq_live_check
```

If the live fetch succeeds, inspect `.cache/jquants/daily_quotes/7203.json`
against the documented field mappings and update `_map_*` candidates if the
real schema differs. If the correct surface turns out to require the v1
idToken flow, implementing that auth flow becomes the next scoped task.

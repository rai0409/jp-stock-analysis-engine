# Forward-return validation results

**Status: real prices acquired and a real forward-return run completed within
the data-coverage window — but NO predictive conclusion can be drawn (n=3, and
one ticker's raw series contains an unadjusted stock split).**

Updated 2026-06-13.

## J-Quants V2 endpoint resolution (fixed)

The previous blocker was the J-Quants endpoint, not the key. Confirmed by live
probes on 2026-06-13 (secret-safe; the API key is sent only in the `x-api-key`
header and was never printed):

- `JQUANTS_API_KEY`: **PRESENT**.
- **V1 is retired.** `GET /v1/prices/daily_quotes` → HTTP 410
  `J-QuantsはV2に移行しました。` (migration guide:
  https://jpx-jquants.com/ja/spec/migration-v1-v2).
- The old guess `/v2/prices/daily_quotes` → HTTP 403
  `The requested endpoint does not exist.` The V2 routes were **restructured**.
- **Verified V2 routes** (HTTP 200 with `x-api-key`):

  | dataset        | V1 (retired)              | V2 (current)              |
  | -------------- | ------------------------- | ------------------------- |
  | daily OHLC     | `/v1/prices/daily_quotes` | `/v2/equities/bars/daily` |
  | financials     | `/v1/fins/statements`     | `/v2/fins/summary`        |
  | listed master  | `/v1/listed/info`         | `/v2/equities/master`     |

- **Auth (unchanged model, simpler):** V2 uses a dashboard-issued API key in the
  `x-api-key` header. The V1 ID-token / refresh-token flow is gone. An
  `Authorization` header is rejected by the API gateway.
- **Response shape changed:** rows are under the top-level `data` key (was
  `daily_quotes` / `statements` / `info`), and field names are abbreviated
  (`O`,`H`,`L`,`C`,`Vo`,`AdjC`,`AdjFactor`,`Date`,`Code`; financials use
  `Sales`,`OP`,`NP`,`EPS`,`BPS`,`Eq`,`TA`,`CFO`,…).

The provider now defaults to the verified V2 paths and the `data` rows key, and
maps the V2 field names (with V1 fallbacks for older caches). Endpoint
overrides (`JQUANTS_API_BASE_URL`, `JQUANTS_API_VERSION`,
`JQUANTS_DAILY_QUOTES_PATH`, …) still work. Error messages now distinguish
auth failure, endpoint-not-found, V1-gone/migrated, and plan/coverage limits.

> Note: a local `.env` that still pins `JQUANTS_API_VERSION='v1'` and
> `JQUANTS_DAILY_QUOTES_PATH='/prices/daily_quotes'` will override the correct
> V2 defaults and reintroduce the 410. Remove those two overrides (or set them
> to `v2` and `/equities/bars/daily`). The API key line is untouched.

## Real price acquisition

```bash
PYTHONPATH=src python -m jp_stock_analysis.cli fetch-jquants-prices \
  --tickers 3928,4107,4264 --out /tmp/topix1000_forward_prices_raw.csv \
  --cache-dir /tmp/jq_cache_live --allow-network
```

- **Source:** J-Quants V2 `/v2/equities/bars/daily`, raw close (`PriceBar.close`).
- **Rows fetched:** 488 per ticker.

| ticker | rows | covered range            |
| ------ | ---- | ------------------------ |
| 3928   | 488  | 2024-03-21 → 2026-03-19  |
| 4107   | 488  | 2024-03-21 → 2026-03-19  |
| 4264   | 488  | 2024-03-21 → 2026-03-19  |

## The 2026-03-28 target is outside plan coverage (blocked)

The requested window (forward returns from **2026-03-28** onward) cannot be
satisfied. The subscription covers **2024-03-21 ~ 2026-03-21**; a fetch from
2026-03-28 returns:

```
HTTP 400 :: "Your subscription covers the following dates: 2024-03-21 ~ 2026-03-21.
            If you want more data, please check other plans:..."
```

`prepare-price-csv --from-date 2026-03-28 --min-rows-after 60` therefore fails
the coverage check (0 rows on/after 2026-03-28 for every ticker). **No real
prices exist on/after 2026-03-28 on this plan**; none were fabricated.

## Real run performed within the covered window (deviation, clearly labelled)

To produce a genuine forward-return run with real prices, the decision/analysis
date was moved inside the covered window: **2025-11-28** (the latest covered
date that still leaves ≥60 forward trading rows, through 2026-03-19). The
`analyze` step was fed real closes up to 2025-11-28 (so `analysis_date =
2025-11-28`); `validate-forward-returns` was fed the full real series so it
could look forward. No-look-ahead is enforced (base = first row strictly after
2025-11-28 = 2025-12-01).

Outputs: `/tmp/jstocks_forward_validation_topix1000/forward_returns.{json,csv,md}`

| ticker | final_score | screening_score | grade  | h5     | h20      | h60      |
| ------ | ----------- | --------------- | ------ | ------ | -------- | -------- |
| 3928   | 41.8        | 12.3            | medium | −3.12% | +17.58%  | +16.41%  |
| 4107   | 84.6        | 24.9            | medium | −2.50% | −86.15%† | −82.36%† |
| 4264   | 33.0        | 9.7             | medium | −0.07% | +1.77%   | +25.94%  |

† **Data artifact, not an economic return.** Ticker 4107's raw close drops
43050 → 4985 on 2025-12-29 (−88% in one day): an **unadjusted stock split**.
Its h20/h60 figures are corrupted by the split, not real losses. The V2 feed
*does* provide adjusted close (`AdjC`), but this export uses raw close by
design (see caveat below).

## Interpretation

**No predictive conclusion can be drawn.** Two independent reasons:

1. **n = 3** — descriptive only, never statistically significant.
2. **Split contamination** — 4107 (the single highest `final_score`) is
   dominated by an unadjusted corporate action, so any apparent
   "high final_score → large negative return" relationship here is a raw-close
   artifact, not signal.

On the two clean tickers (3928, 4264) both `screening_score` and `final_score`
order them the same way (3928 above 4264), and 3928's longer-horizon returns
were higher — but with two clean points this is anecdote, not evidence. The
question "does `screening_score`/`reliability_grade` beat `final_score`?"
remains **unanswered** and requires (a) adjusted close and (b) a broad universe
over multiple non-overlapping decision dates.

## Caveats

- **Raw close, not adjusted close.** Materialized here: 4107's split corrupts
  its forward returns. A clean run needs adjusted close (`AdjC` is available in
  the V2 feed) — see the next-step prompt.
- **Plan/coverage window.** Data ends 2026-03-21; the literal 2026-03-28 target
  is unreachable on this subscription.
- **Decision-date deviation.** The run uses analysis date 2025-11-28 (inside
  coverage), not 2026-03-28; results are not the originally requested window.
- **Small sample.** Only 3 tickers have real fundamentals.
- **No-look-ahead** is enforced (base = first row strictly after the analysis
  date).
- **No financial advice, no trading automation.** Self-directed research only:
  no buy/sell/hold signals, no portfolio construction, no position sizing.

**Do not tag** a release as a validated predictive result: the run is real but
inconclusive (n=3 and a split artifact).

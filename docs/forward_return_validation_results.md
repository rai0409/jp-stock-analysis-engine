# Forward-return validation results

**Status: real prices acquired and a real forward-return run completed on
adjusted close within the data-coverage window. The 4107 raw-close split
artifact is removed. Still NO predictive conclusion (n=3, no statistical
significance; `screening_score` and `final_score` rank identically here).**

Updated 2026-06-13 (adjusted-close run added).

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

### Adjusted-close run (primary)

- **Price field:** `adjusted_close` via J-Quants V2 `AdjC` (fetched with
  `fetch-jquants-prices --price-field adjusted_close`; the CSV `close` column
  holds back-adjusted values).
- **Rows:** 488 per ticker (3928, 4107, 4264); coverage 2024-03-21 → 2026-03-19.
- **Decision date:** 2025-11-28; horizons 5/20/60.
- **Outputs:**
  `/tmp/jstocks_forward_validation_topix1000_adjusted/forward_returns.{json,csv,md}`

| ticker | final_score | screening_score | grade  | h5     | h20     | h60     |
| ------ | ----------- | --------------- | ------ | ------ | ------- | ------- |
| 3928   | 41.8        | 12.3            | medium | −3.12% | +17.58% | +16.41% |
| 4107   | 84.6        | 24.9            | medium | −2.50% | +38.47% | +76.39% |
| 4264   | 33.0        | 9.7             | medium | −0.07% | +1.77%  | +25.94% |

Group means (all three are `screening_eligible=true`, grade `medium`):
h5 −1.90%, h20 +19.27%, h60 +39.58% (hit-rate-positive 0.00 / 1.00 / 1.00).

### The 4107 split artifact is removed

Raw vs adjusted for 4107 (the only ticker with a corporate action — a 1:10
split; raw close collapses 43050 → 4985 on 2025-12-29):

| horizon | raw close (artifact) | adjusted close (fixed) |
| ------- | -------------------- | ---------------------- |
| h5      | −2.50%               | −2.50%                 |
| h20     | **−86.15%**          | **+38.47%**            |
| h60     | **−82.36%**          | **+76.39%**            |

3928 and 4264 are byte-identical between the raw and adjusted runs (no corporate
action), confirming the change is isolated to the adjusted ticker.

## Interpretation

**Still no predictive conclusion can be drawn**, but for a cleaner reason than
before — the data artifact is gone, leaving only the sample-size limit:

1. **n = 3** — descriptive only, never statistically significant.
2. **`screening_score` and `final_score` rank the three identically** here
   (4107 > 3928 > 4264), and `reliability_grade` is constant (`medium` for all),
   so this run **cannot distinguish** whether the reliability-aware fields beat
   raw `final_score`. Neither outperformed the other.

Descriptively, on adjusted close the highest-scored name (4107) did post the
strongest 20- and 60-day forward returns, and the top-scored ordering matched
forward returns at h20 (4107 > 3928 > 4264); at h60 the lowest-scored 4264
(+25.94%) edged the mid-scored 3928 (+16.41%). With three points and one
decision date this is anecdote, not evidence. Answering "does
`screening_score`/`reliability_grade` beat `final_score`?" requires a broad
universe with score dispersion across grades, over multiple non-overlapping
decision dates.

## Caveats

- **Adjusted close used (primary run); raw close still the export default.**
  Adjusted close removes split/dividend distortion (it fixed 4107). It is
  selected explicitly with `--price-field adjusted_close`; raw close remains the
  default. A raw run for the same window is preserved at
  `/tmp/jstocks_forward_validation_topix1000/` for comparison.
- **Plan/coverage window.** Data ends 2026-03-21; the literal 2026-03-28 target
  is unreachable on this subscription.
- **Decision-date deviation.** The run uses analysis date 2025-11-28 (inside
  coverage), not 2026-03-28; results are not the originally requested window.
- **Small sample, no statistical significance.** Only 3 tickers have real
  fundamentals, all graded `medium`; any ordering is descriptive only.
- **No-look-ahead** is enforced (base = first row strictly after the analysis
  date).
- **No financial advice, no trading automation.** Self-directed research only:
  no buy/sell/hold signals, no portfolio construction, no position sizing.

**Do not tag** a release as a validated predictive result: the adjusted run is
real and the artifact is fixed, but n=3 with identical score rankings is not
strong enough to conclude anything predictive.

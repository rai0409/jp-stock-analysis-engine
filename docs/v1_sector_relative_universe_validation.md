# v1 Sector-Relative Universe Validation

**Date:** 2026-06-11
**Scope:** Larger deterministic offline universe for sector-relative scoring
validation, using synthetic jquants-cache fixtures only. No J-Quants live
calls (still deferred due to HTTP 403). No RAG/EDINET/TDnet/broker/trading
logic. `final_score` and trade-signal behavior unchanged.

## Fixture universe structure

`tests/fixtures/jquants_universe/` — a self-contained J-Quants cache directory
(`daily_quotes/`, `statements/`, `listed_info/`), 12 synthetic codes, fully
deterministic (closed-form prices: drift + sine wiggle over 80 weekdays from
2025-01-06; two fiscal years of statements with designed gradients). All data
is synthetic; no real J-Quants rows are committed.

| Sector | Codes | Peer count | Design |
|---|---|---|---|
| 輸送用機器 | 7001–7005 | 5 | gradient: 7001 strong (cheap, growing, +12% margin) → 7005 loss-maker (negative EPS, −11% revenue, equity ratio 18%, downtrend) |
| 電気機器 | 6501–6504 | 4 | same gradient pattern, 6501 best → 6504 weakest |
| 情報・通信業 | 9001, 9002 | 2 | small-peer-group case (warns, reduced confidence) |
| (none) | 9101 | — | prices + statements but **no `listed_info`** → missing-sector degradation |

## Validation results

From `tests/test_sector_universe.py` (8 tests) and the CLI smoke:

- **Offline analysis works at scale:** 12 codes analyzed via
  `--provider jquants-cache` with no API key set; exit 0; the missing
  `listed_info` for 9101 degrades to a stderr warning, not a failure.
- **Peer counts correct:** 5 / 4 / 2 per sector; 9101 → `sector_relative: null`.
- **Rankings follow the designed gradients:** 7001 highest sector-relative
  score in transport (92.9), 7005 lowest (16.7); 6501 highest in electronics,
  6504 lowest. 7001 scores 100.0 revenue-growth and ROE percentiles; 7005's
  PER percentile is `None` (negative EPS) with a warning — never faked.
- **Deterministic:** two CLI runs produce byte-identical `screening.json`.
- **`final_score` untouched:** identical final scores with and without sector
  metadata attached (direct `analyze_data` comparison).
- **Trade signals unchanged and not wired to sector data:** all signal labels
  valid, thresholds/disclaimer present, no "sector" text in supporting
  factors/evidence/thresholds; the designed loss-maker 7005 never buys.
- **CSV/Markdown gating:** `sector_relative_score` numeric for all sector
  members, blank for 9101; `## Sector Relative` section present for 7001,
  absent for 9101.

CLI smoke (12 codes, analysis_only): absolute final-score ranking and
sector-relative scores agree with the designed gradients
(7001 75.8/92.9 at rank 1 … 7005 11.7/16.7 at rank 12).

## Commands run

```
python -m pytest                          → 117 passed
ruff check .                              → All checks passed
CLI jquants-cache smoke (12 codes)        → exit 0, no API key, warning for 9101 only
CLI local fixture smoke                   → exit 0, default behavior unchanged
```

## Limitations

- The universe is synthetic with intentionally clean gradients; real sectors
  have noisier cross-metric patterns and mid-rank ordering is not guaranteed.
- 80 trading days per code: 6m/12m returns and MA120/200 unavailable
  (warned); momentum percentiles rest on 3m returns.
- No disclosure texts in the universe: disclosure scores are absent and every
  code carries the `insufficient_data` risk flag (risk floor 15), which
  compresses risk percentile spread at the top.
- Sector grouping still requires exact sector-string matches.
- J-Quants live validation remains deferred; field mappings stay assumptions.

## Next recommended step

Wire `sector_relative_score` into the signal engine as supporting evidence
only — appended to `supporting_factors` when ≥ 70 with ≥ 4 sector peers, never
counting toward the ≥2-factor buy requirement on its own and never touching
`final_score` — using this universe as the regression bed.

# v1 Sector-Relative Scoring — Validation Note

**Date:** 2026-06-11
**Scope:** Additive sector-relative analysis layer using local / jquants-cache
data only. No J-Quants live calls, no RAG/EDINET/TDnet, no broker or trading
logic. Existing score, mode, and signal behavior unchanged.

## What it does

For every analyzed ticker whose `CompanyMetadata.sector` is shared by at least
one other ticker **in the same analysis run**, the engine computes percentile
ranks against those same-sector peers for seven metrics:

| Metric | Direction |
|---|---|
| PER | lower is better (inverted) |
| PBR | lower is better (inverted) |
| Revenue growth YoY | higher is better |
| Operating margin | higher is better |
| ROE | higher is better |
| Momentum (3m return, falling back to 6m) | higher is better |
| Risk score | lower is better (inverted) |

Percentile = share of same-sector peers beaten (ties count half), 0–100, so
**100 always means most favorable**. `sector_relative_score` is the mean of
available percentiles.

## Design decisions

- **Separate from `final_score`.** The new score is reported alongside, never
  blended in. Absolute scoring, screening labels, and trade signals are
  byte-for-byte unchanged when no sectors overlap (verified by tests against
  the standard fixtures, which use three distinct sectors).
- **Nothing fabricated.** No sector metadata, a lone company in its sector, or
  a missing underlying metric ⇒ `None` plus warnings — never a neutral filler.
- **Conservative confidence.** Coverage (available percentiles / 7) scaled by
  a peer-count factor (`min(1, peers/4)`); small peer groups warn explicitly.
- **Universe-relative, not market-relative.** Peers are the tickers in the
  current run, not the whole TSE sector. This is documented in the report
  section header and is the honest v1 interpretation for a screening tool.

## Schema added

`SectorRelativeMetrics` (in `schemas.py`): `ticker`, `sector`, `peer_count`,
seven `*_percentile` fields, `sector_relative_score`, plus the standard
`warnings`/`confidence_score`/`source_metadata`. Attached to
`StockAnalysisResult.sector_relative` (optional; `null` in JSON when absent).

## Output changes (all conditional on data availability)

- **JSON:** `results[].sector_relative` — object when computed, `null` otherwise.
- **Markdown:** `## Sector Relative` section, only when computed.
- **CSV:** `sector_relative_score` column, only when at least one row has it
  (same dynamic-column pattern as `screening_label`/`trade_signal`).

## Files changed

- `src/jp_stock_analysis/schemas.py` — `SectorRelativeMetrics`, result field
- `src/jp_stock_analysis/analysis/sector_relative.py` — **new** (compute + attach)
- `src/jp_stock_analysis/analysis/__init__.py` — exports
- `src/jp_stock_analysis/cli.py` — `attach_sector_relative` after per-ticker analysis
- `src/jp_stock_analysis/reports/markdown_report.py` — conditional section
- `src/jp_stock_analysis/reports/csv_report.py` — conditional column
- `tests/test_sector_relative.py` — **new**, 10 tests
- `docs/jp_stock_analysis_engine.md` — analysis-methods note

## Commands run

```
python -m pytest                 → 109 passed
ruff check .                     → All checks passed
CLI local smoke (default meta)   → exit 0; no sector column/section (3 distinct sectors)
CLI local smoke (shared sector)  → exit 0; sector column + "## Sector Relative" present
CLI jquants-cache smoke          → exit 0 offline, no API key
```

## Test results

109 passed (99 pre-existing + 10 new), deterministic and offline. New coverage:
direction-aware percentiles with ties, mean-of-available scoring, missing-metric
warnings, 6m momentum fallback, lone-sector / missing-metadata absence, small
peer-group confidence, determinism, final-score immutability, CLI behavior with
and without shared sectors (JSON/Markdown/CSV).

## Larger synthetic universe (validation)

A 12-code synthetic jquants-cache universe now lives under
`tests/fixtures/jquants_universe/` for offline validation at a realistic
scale (see `docs/v1_sector_relative_universe_validation.md`):

- 輸送用機器 — 5 companies (`7001`–`7005`, designed strong→weak gradient)
- 電気機器 — 4 companies (`6501`–`6504`, same gradient pattern)
- 情報・通信業 — 2 companies (`9001`, `9002`; small-peer-group case)
- `9101` — prices/statements but no `listed_info` (missing-sector case)

`tests/test_sector_universe.py` proves peer counts, gradient-consistent
rankings, determinism, safe degradation, and that `final_score` and
trade-signal behavior are untouched by sector-relative attachment.

## Peer-count requirements

- Minimum 2 same-sector companies in the analyzed universe, or no
  sector-relative metrics are produced at all.
- Fewer than 4 peers ⇒ explicit "small sector peer group" warning and a
  proportionally reduced confidence score; 2-company sectors can only produce
  0/50/100 percentiles and should be read as coarse direction, not ranking.

## Limitations

- Peer group = current analysis universe, which may be tiny; percentiles with
  2–3 peers are coarse (warned, confidence-scaled).
- Sector strings must match exactly; no sector-code normalization in v1
  (J-Quants `Sector33CodeName` vs local CSV labels must agree to group).
- **`sector_relative_score` is still NOT used in `final_score` or screening
  labels.** In `trade_signal` mode it now appears as *supporting evidence
  only*, gated by `SignalThresholds.sector_support_*` (defaults: score ≥ 70,
  ≥ 4 peers, confidence ≥ 50): the factor string is appended after the signal
  label is decided and never counts toward the two-core-factor buy
  requirement — see `docs/v1_sector_relative_signal_support.md` and
  `docs/v1_sector_signal_threshold_config.md`.
- **J-Quants live validation remains deferred** (HTTP 403); all validation
  uses synthetic cache fixtures.

## Next recommended step

**Done:** the sector-support eligibility constants are now `SignalThresholds`
config fields with today's values as defaults
(`docs/v1_sector_signal_threshold_config.md`). Next functional gap: a real
disclosure provider (EDINET, cache-first) or resolving the deferred J-Quants
live validation.

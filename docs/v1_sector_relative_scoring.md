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

## Limitations

- Peer group = current analysis universe, which may be tiny; percentiles with
  2–3 peers are coarse (warned).
- Sector strings must match exactly; no sector-code normalization in v1
  (J-Quants `Sector33CodeName` vs local CSV labels must agree to group).
- `sector_relative_score` does not feed `final_score`, screening labels, or
  trade signals yet — by design until validated against a larger universe.

## Next recommended step

Wire `sector_relative_score` into the signal engine as *supporting evidence
only* (e.g. listing it in `supporting_factors` when ≥ 70 with ≥ 4 peers) once
a realistically sized universe (10+ tickers per sector from jquants-cache) has
been exercised — keeping it out of `final_score` and never sufficient for a
buy on its own.

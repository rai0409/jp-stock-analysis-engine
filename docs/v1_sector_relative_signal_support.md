# v1 Sector-Relative Signal Support — Validation Note

**Date:** 2026-06-11
**Scope:** `sector_relative_score` is now surfaced in `trade_signal` mode as
supporting evidence only. No change to `final_score`, `score_stock`, screening
labels, thresholds, or CSV/Markdown sector reporting. No J-Quants live calls
(still deferred); local and jquants-cache modes only.

## What changed

`src/jp_stock_analysis/analysis/signal_engine.py` only:

- New `_sector_relative_factor(result)` helper producing an evidence string
  like `sector_relative_score=92.9 (>= 70, 5 peers; supporting evidence only)`.
- `generate_signal` appends that string to `supporting_factors` **after** the
  signal label has already been decided from core factors alone.
- `_supporting_factors` (the buy-gate input) is unchanged in behavior and its
  docstring now states that valuation AND sector-relative are excluded.

## Exact eligibility rule

The factor is appended if and only if all of:

1. `result.sector_relative` exists with a non-`None` `sector_relative_score`
2. `sector_relative_score >= 70`
3. `peer_count >= 4`
4. `SectorRelativeMetrics.confidence_score >= 50`

…and it is appended regardless of label, purely as displayed evidence.

## Why sector_relative_score does not drive final_score

It is universe-relative: the same company scores differently depending on
which peers happen to be in the run, so folding it into the absolute
`final_score` would make scores non-comparable across runs and silently
double-count the same underlying metrics (PER, growth, ROE…) that already
feed the absolute sub-scores.

## Why it does not satisfy the buy-factor requirement

A stock can be "best in sector" in a structurally weak or uniformly expensive
sector — relative strength is not absolute quality. The buy gate therefore
still requires **>= 2 core factors from quality/growth/momentum/disclosure**;
valuation and sector-relative both stay outside that count. The factor is
computed after the label decision, so it cannot influence any branch.

## Commands run

```
python -m pytest                                    → 121 passed
ruff check .                                        → All checks passed
CLI jquants-cache smoke (12-code universe,
  trade_signal)                                     → exit 0:
    7001/6501 watch_signal + sector factor (5/4 peers, scores 92.9/83.3)
    9001 watch_signal WITHOUT factor (score 85.7 but only 2 peers)
    7004/6504 sell_signal, 7005 insufficient_data — not rescued
CLI local fixture smoke (analysis_only default)     → exit 0, unchanged
```

## Test results

121 passed (117 before; one universe test rewritten for the new contract,
5 added). New coverage: eligible factor appended on a buy signal that already
has >= 2 core factors; valuation + sector-relative with zero core factors must
not buy (factor shown as labelled evidence only); ineligibility for low peers /
low score / low confidence / missing score; analysis_only and screening remain
signal-free even with sector data attached; universe-level checks that no
"sector" key enters `thresholds_used`, every buy rests on >= 2 core factors,
and 7005 is never rescued. Pre-existing valuation-alone protection tests are
untouched and green.

## Limitations

- ~~Eligibility constants (70 / 4 peers / 50 confidence) are engine constants,
  not yet `SignalThresholds` config fields.~~ **Resolved:** now configurable
  via `SignalThresholds.sector_support_*` with identical defaults — see
  `docs/v1_sector_signal_threshold_config.md`.
- Peer groups come from the analyzed universe, so the factor's meaning depends
  on which codes were included in the run.
- The synthetic universe has no disclosure texts, so signal confidences sit
  near the minimum; real disclosure inputs will shift labels.
- J-Quants live validation remains deferred (HTTP 403).

## Next recommended step

**Done:** the eligibility constants were promoted into `SignalThresholds`
(`sector_support_score_threshold` / `sector_support_min_peers` /
`sector_support_min_confidence`) with identical defaults and validation; see
`docs/v1_sector_signal_threshold_config.md`.

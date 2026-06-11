# v1 Sector-Support Threshold Config — Validation Note

**Date:** 2026-06-11
**Scope:** Configuration-only refactor. The sector-relative supporting-evidence
gates moved from `signal_engine.py` module constants into `SignalThresholds`.
No change to `final_score`, `score_stock`, screening labels, decision
thresholds, default mode, or any safety rule. No J-Quants live calls; local
and jquants-cache modes only.

## What changed

- `src/jp_stock_analysis/config.py` — three new `SignalThresholds` fields
  (below) with validation; class docstring explains the evidence-only role.
- `src/jp_stock_analysis/analysis/signal_engine.py` — the
  `_SECTOR_SUPPORT_*` module constants were removed;
  `_sector_relative_factor(result, thresholds)` now reads
  `config.thresholds`. The factor string embeds the configured score
  threshold (e.g. `(>= 70, 5 peers; supporting evidence only)`).

## New config fields and defaults

| Field | Default | Validation |
|---|---|---|
| `sector_support_score_threshold` | `70.0` | 0–100 |
| `sector_support_min_peers` | `4` | positive integer (≥ 1) |
| `sector_support_min_confidence` | `50.0` | 0–100 |

Validation uses the existing `SignalThresholds` style: the `"*"` range
validator still covers every 0–100 field, with `sector_support_min_peers`
exempted from the range check and validated separately as a positive integer.

## Why these values are NOT in `thresholds_used`

`thresholds_used` documents the values that can influence label selection.
The sector-support gates are evidence-only and evaluated **after** the label
is decided, so including them would misrepresent them as decision inputs —
and the universe safety test explicitly asserts no sector key ever appears in
`thresholds_used`. They are config-driven but deliberately excluded; this is
documented in the `SignalThresholds` docstring and the signal-engine module
docstring.

## Compatibility and safety confirmations

- **Defaults reproduce previous behavior exactly.** Asserted by
  `test_sector_support_defaults_match_previous_constants` and re-confirmed by
  the 12-code universe smoke: 7001/6501 carry the factor, 9001 (2 peers) and
  7005 do not — identical to the pre-config run, including the factor string.
- **Custom thresholds change evidence eligibility only.** Tightening any of
  the three fields removes the factor without moving the label or
  `thresholds_used`; loosening adds it, again without label movement.
- **Sector-relative remains evidence-only**: it never counts toward the
  two-core-factor buy requirement, valuation-alone protection stays green,
  `analysis_only` stays signal-free, `screening` stays trade-signal-free, and
  `final_score` is untouched (covered by existing regression tests).

## Commands run

```
python -m pytest                                  → 125 passed
ruff check .                                      → All checks passed
CLI jquants-cache smoke (12-code universe,
  trade_signal, defaults)                         → exit 0, factor distribution
                                                    identical to pre-config run
CLI local fixture smoke (analysis_only default)   → exit 0, unchanged
```

## Test results

125 passed (121 before + 4 new): config defaults, per-field custom-threshold
eligibility (tighten ×3 / loosen ×1, labels and thresholds_used invariant),
thresholds_used exclusion, and validation errors (score 150, confidence −1,
peers 0, plus an existing-field guard at 101).

## Next recommended step

The sector-relative feature line is now complete and configurable. The next
functional gap is disclosure data sourcing: an EDINET provider (cache-first,
stub-replacing, same pattern as J-Quants) would feed real disclosure text into
the existing rule-based analyzer — currently the only pipeline input that has
no non-local provider path. Alternatively, resolve the deferred J-Quants live
validation (HTTP 403) to confirm the V2 field-mapping assumptions.

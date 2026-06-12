# Confidence-Aware Screening Guard

**Date:** 2026-06-12
**Scope:** Reliability-aware screening fields and ranking. No change to
`final_score`, sub-scores, screening labels, trade-signal gates, or modes.

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## Why `final_score` and `screening_score` differ

`final_score` is the weighted average of the *available* sub-scores
(`analysis/scoring.py`). That is correct and transparent, but it has a sharp
edge: when only one sub-score is computable, that single value *is* the final
score. The motivating real case (linked topix1000 export, synthetic ticker
9991): no fundamentals, no valuation, no disclosure text, only **2 daily price
bars** — yet `max_drawdown = 0` makes the momentum sub-score 100 and the final
score **98.5**, while `confidence_score` is only **12.2**. Ranking by
`final_score` alone put this least-known ticker at **#1**.

The guard (`src/jp_stock_analysis/analysis/reliability.py`) keeps
`final_score` untouched for transparency and adds deterministic reliability
fields that screening uses for ranking and presentation.

## Fields (in `screening.json` `screening[]`, `screening.csv`, and Markdown)

| Field | Meaning |
|---|---|
| `final_score` | Unchanged raw weighted score (0–100). |
| `confidence_score` | Unchanged: mean of the five component confidences. |
| `data_coverage_score` | 20 points per covered component out of fundamentals, valuation, momentum, disclosure, risk. A component is covered only if its analysis exists **and** reported non-zero confidence — a metadata-only disclosure (no text) does not count. |
| `screening_score` | Reliability-adjusted ranking score: `final_score × (confidence/100) × (coverage/100)`, rounded to 1 decimal. |
| `screening_eligible` | Deterministic gate, see below. |
| `reliability_grade` | `low` whenever ineligible; `high` when confidence ≥ 70 **and** coverage ≥ 80; otherwise `medium`. |
| `warnings` | Screening entries carry explicit `screening reliability: …` warnings explaining every failed gate. |

## Eligibility rules (configurable via `SignalThresholds` in `config.py`)

A result is `screening_eligible = false` when **any** of:

- `confidence_score < screening_min_confidence` (default **30**)
- `data_coverage_score < screening_min_coverage` (default **40**)
- fewer than `screening_min_subscores` (default **2**) of the five major
  sub-scores (quality, growth, valuation, momentum, disclosure) are available
- `final_score` is unavailable

Boundaries are inclusive: exactly 30 confidence / 40 coverage passes.

## Ranking

`screen_stocks` (`analysis/screening.py`) sorts by:

1. `screening_eligible` (eligible first)
2. `screening_score` descending (missing last)
3. `final_score` descending (missing last)
4. ticker (deterministic tie-break)

`final_score` alone never determines rank when reliability information
disagrees: an eligible 65-point result outranks an ineligible 98.5-point one.
`screening_label` semantics are unchanged — labels already require
`confidence ≥ min_confidence_for_signal` (55), which is stricter than the
eligibility confidence gate, so a low-confidence result still labels
`insufficient_data` in screening mode.

## Interpretation examples

| Case | final | confidence | coverage | screening_score | eligible | grade |
|---|---|---|---|---|---|---|
| 9991: 2 price bars only | 98.5 | 12.2 | 40 | ≈ 4.8 | false | low |
| Full data, solid stock | 70.0 | 80.0 | 100 | 56.0 | true | high |
| Full data, marginal confidence | 70.0 | 30.0 | 100 | 21.0 | true | medium |
| No computable score | n/a | 0 | any | n/a | false | low |

Reading rule: **a high `final_score` with a `low` reliability grade means
"too little data to know", never "strong candidate".** The Markdown executive
summary prints this warning explicitly, and the screening warnings list names
each failed gate.

## Limitations and boundaries

- This is a data-sufficiency guard, not a forecast of returns; scores remain
  research heuristics. Not personalized financial advice.
- Coverage counts component presence, not depth (300 bars and 60 bars both
  count momentum as covered; depth is reflected in `confidence_score`).
- No broker execution, auto-trading, position sizing, or portfolio
  allocation logic exists in this engine; `analysis_only` remains the default
  mode and `trade_signal` remains explicit opt-in. The valuation-alone
  buy-signal protection in `analysis/signal_engine.py` is unchanged.

## Tests

`tests/test_reliability.py` (10 tests): the 9991 two-bar case end-to-end
through the real analyzers, eligibility boundaries, ranking demotion,
CSV/JSON/Markdown field presence, the low-reliability Markdown warning,
unchanged `analysis_only`/disclaimer behavior, and `insufficient_data`
trade-signal output for the low-confidence case.

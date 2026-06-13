# Commercial Readiness Gap

**Scope:** what is built (P1–P3 modeling infrastructure) vs what remains blocked
by real data (P0 strict broad validation).

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## Why P0 is skipped here

P0 = strict broad no-look-ahead forward-return validation over a real universe.
It is **blocked by data coverage**, not by missing code:

- the topix1000 annual bundle's disclosure/target date is **2026-03-27**;
- adjusted-close price coverage currently **ends 2026-03-19**;
- a strict study needs a decision date **on or after** the disclosure date with
  enough *later* adjusted-close rows per horizon.

So there are zero eligible price rows after the disclosure date today. The
`check-forward-readiness` command reports this deterministically and we do **not**
weaken it. This task therefore builds P1–P3 so the moment valid point-in-time
fundamentals/prices arrive, validation runs end-to-end.

## What is implemented (offline, synthetic-tested)

- **P1** accounting-basis-aware fundamentals (export column + dataset filtering;
  consolidated/non_consolidated never pooled silently).
- **P1** disclosure narrative extraction contract (schema + placeholder features;
  no LLM/NLP yet).
- **P2** modeling dataset builder, factor feature engineering, cross-sectional
  ranking validation, baseline factor ranker.
- **P3** walk-forward validation, purged/embargo splitting, optional
  LightGBM/CatBoost adapters, full offline modeling report.
- Deterministic synthetic fixtures so all of the above is testable now.

## What remains blocked by real data

| capability | blocker |
| --- | --- |
| strict broad no-look-ahead validation (P0) | price coverage ends before disclosure date |
| real Rank IC / ICIR as evidence | needs real point-in-time fundamentals + later adjusted closes |
| real walk-forward results | needs multiple real decision dates with labels |
| real narrative features | needs an actual narrative extractor (contract ready, extractor not built) |
| real ML model comparison | optional backends + real labelled data |

## Hard guarantees preserved

- No predictive-performance claim, no buy/sell signal, no trade automation.
- `analysis_only` default unchanged.
- Strict no-look-ahead readiness check unchanged.
- Synthetic results are explicitly flagged as non-evidence.
- LightGBM/CatBoost remain optional; minimal install and tests never require them.

## Exit criteria for a commercial-grade real run

1. Point-in-time fundamentals with correct `accounting_basis`.
2. Adjusted-close prices extending well past the disclosure date (≥ `max(horizon)+1`
   rows after the decision date per ticker).
3. `check-forward-readiness` returns `ELIGIBLE` for a meaningful ticker count.
4. Then run the pipeline order in `docs/modeling_pipeline.md`.

# Pipeline Regression Baseline & Change Detection (research-only)

**Module:** `regression_baseline.py`
**CLI:** `check-pipeline-regression`
**Golden fixture:** `tests/fixtures/pipeline_baseline/golden_pipeline_baseline.json`

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## What this is — and is not

The regression baseline detects **unexpected pipeline output changes** against a
blessed reference. It is complementary to the determinism gate:

- **Determinism gate** (`verify-pipeline-determinism`) checks *repeated-run
  reproducibility* — two runs now should be byte-identical.
- **Regression baseline** (`check-pipeline-regression`) checks *future changes*
  against a committed golden reference — did a code change alter a metric/artifact?

It **does not** prove model validity or market performance. The golden baseline is
**synthetic-only and not market evidence**. Baseline updates must be **explicit and
reviewed**; the official fixture requires an approved promotion and ledger. Secrets are never
captured; provenance manifests (`audit_manifest.*`, `artifact_manifest.*`) and the
run-narrative `pipeline_summary.*` are excluded by default because they are
intentionally run/commit-specific.

## Golden baseline

A small (~17 KB) **canonicalized fingerprint set** of a fixed synthetic pipeline
run (run id `golden`, fixed timestamp epoch, canonical synthetic config), committed
under version control. Per tracked artifact it stores: relative path, type, a
**canonical SHA** (with declared volatile fields — `run_id`, timestamps, absolute
paths — canonicalized out), a **raw SHA**, and safe semantic metadata (CSV
rows/columns, JSON top-level keys, line counts, a few headline metrics). It
contains **no absolute paths, no timestamps, no secrets, no run-specific values**.

## Classification

Each tracked artifact is classified by comparing a fresh run to the baseline:

| class | meaning | regression? |
| --- | --- | --- |
| `unchanged` | canonical + raw fingerprints match | no |
| `volatile_only` | canonical matches, only a declared volatile field differs | no |
| `changed` | canonical fingerprint differs (a real content/metric change) | **yes** |
| `missing` | a baseline artifact is absent from the fresh run | **yes** |
| `new` | a fresh artifact not in the baseline | only with `--strict-new-artifacts` |
| `unreadable` | the artifact could not be parsed | reported |

`changed` diffs report changed JSON top-level keys, CSV row/column changes, text
line-count changes, and **headline-metric** before/after values (e.g.
`ranking_metrics.json` `horizons.0.ic_mean`). Numeric differences and missing
artifacts are **never** ignored.

## CLI

```
# check a fresh synthetic run against the committed golden baseline (CI-style)
python -m jp_stock_analysis.cli check-pipeline-regression --synthetic \
    --fail-on-regression --output-dir out/
# -> pipeline_regression_report.{json,md}; exits nonzero on a regression

# Generate a research-only candidate; do not write the official fixture here.
python -m jp_stock_analysis.cli check-pipeline-regression --synthetic \
    --update-baseline --baseline-path /tmp/candidate-baseline.json --output-dir /tmp/candidate-run
```

Options: `--baseline-path`, `--run-id` (default `golden`), `--fixed-timestamp`
(default epoch), `--strict-new-artifacts`, `--fail-on-regression`. File inputs
(`--prices/--fundamentals/...`) are supported for non-synthetic runs against a
user-supplied baseline.

## Optional pipeline hook

`run_pipeline(..., baseline=<loaded baseline>)` embeds a compact
`regression_summary` (counts + `regression_detected`) into the returned summary
and `pipeline_summary.json`. It is optional — normal `run-modeling-pipeline` does
no baseline checking — and `pipeline_summary` is excluded from baseline tracking,
so the hook never causes a self-referential regression.

## Updating the baseline intentionally

The official fixture requires an explicitly approved promotion with a reviewer
note and its matching ledger. Direct `--update-baseline` writes to it are refused.
Inspect the candidate and the semantic change before approval; never update it
merely to silence an unexplained regression. Temporary research baselines remain
available through `--update-baseline`.

## Comparing arbitrary runs & promoting a new baseline

`check-pipeline-regression` compares a fresh run to the *committed* golden
baseline. To diff two arbitrary runs (A vs B), or to bless a new run as the
approved reference, see `docs/run_comparison.md`:

- `compare-pipeline-runs --run-a … --run-b …` — neutral artifact/metric-delta
  report (descriptive only, never better/worse).
- `promote-pipeline-baseline --from-run … --reviewer-note … --require-approval
  --approve` — explicit, auditable baseline update with a provenance record.
  Promotion is an *approved reference*, not a model improvement. Add
  `--ledger-path …` to append a tamper-evident entry to the **hash-chained
  baseline history ledger** (`show-baseline-history` / `verify-baseline-lineage`;
  see `docs/baseline_history.md`).

## Real-data note

This is synthetic-only change detection. Real-data results still require
`check-forward-readiness=ELIGIBLE`, and **P0 strict no-look-ahead remains required
before any predictive claim**. The regression baseline guards reproducibility of
the *pipeline*, not the validity of any model.

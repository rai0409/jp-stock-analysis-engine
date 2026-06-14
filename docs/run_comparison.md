# Pipeline Run Comparison & Baseline Promotion (research-only)

**Module:** `run_compare.py`
**CLI:** `compare-pipeline-runs`, `promote-pipeline-baseline`

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## What this is — and is not

- `compare-pipeline-runs` is for **diagnostic artifact comparison** of two pipeline
  run directories (A vs B). **Metric deltas are descriptive only** — directions are
  `increased` / `decreased` / `changed` / `unchanged` and are **never** labelled
  better/worse and imply **no performance claim**.
- `promote-pipeline-baseline` updates the **approved reference baseline only after
  explicit review** and writes an auditable provenance record. **Promotion does not
  mean model improvement** — only that a run was approved as the new reference.
- Synthetic baselines/results are **not market evidence**. Real-data interpretation
  requires `check-forward-readiness=ELIGIBLE`, and **P0 strict no-look-ahead remains
  required before any predictive claim**. Promotion records improve **auditability,
  not model validity**.

It reuses the regression-baseline canonicalization and fingerprinting (no
duplicate hashing logic); secrets are scrubbed and absolute temp paths are kept
out of stable records.

## `compare-pipeline-runs`

Per artifact (tracked set = the deterministic metric/content outputs; provenance
manifests and `pipeline_summary` are excluded), each is classified:

| class | meaning |
| --- | --- |
| `unchanged` | canonical + raw fingerprints match |
| `volatile_only` | canonical matches; only a declared volatile field differs |
| `changed` | canonical fingerprint differs (a real content/metric change) |
| `only_in_a` | present in A, absent in B |
| `only_in_b` | present in B, absent in A |
| `unreadable` | could not be parsed |

`comparison_status` is one of `identical` / `changed` / `missing_artifacts` /
`new_artifacts` / `unreadable_artifacts`. For `changed` artifacts the report gives
neutral deltas: changed JSON top-level keys, CSV row/column changes and per-column
numeric-mean deltas (best-effort), text line-count changes, and a **headline metric
table** with `A` / `B` / `direction` (neutral). Numeric differences and missing
artifacts are **never** ignored.

```
python -m jp_stock_analysis.cli compare-pipeline-runs \
    --run-a out/run_a/golden --run-b out/run_b/golden --output-dir out/cmp
# -> run_comparison.{json,md}
```

## `promote-pipeline-baseline`

Promotion is explicit and auditable. It **never** promotes silently and **never**
overwrites the baseline without approval:

- `--require-approval` set and `--approve` missing → **blocked** (`baseline NOT
  updated`), exits nonzero.
- a non-empty `--reviewer-note` is required.
- when approved, it captures a new canonical baseline from the run, writes it to
  `--baseline-path`, and writes `baseline_promotion_record.{json,md}`.

The provenance record contains: schema version, (fixed) timestamp, source run id +
fingerprint, baseline basename (no absolute temp paths), previous/new baseline
fingerprints, the reviewer note, the approval flags, artifact classification counts
and **headline metric deltas vs the previous baseline** (descriptive), warnings, a
synthetic flag, and a secret-scrubbed config.

```
# blocked unless approved:
python -m jp_stock_analysis.cli promote-pipeline-baseline \
    --from-run out/run/golden --baseline-path tests/fixtures/pipeline_baseline/golden_pipeline_baseline.json \
    --reviewer-note "approved: intentional factor weight change, reviewed by <name>" \
    --require-approval --approve --output-dir out/promo
```

After promotion, review the written baseline diff and the promotion record in code
review and commit them deliberately.

## How this fits

- **Determinism gate** (`verify-pipeline-determinism`): two runs *now* are byte-identical.
- **Regression baseline** (`check-pipeline-regression`): a fresh run vs the committed golden baseline.
- **Run comparison** (`compare-pipeline-runs`): any two runs A vs B, neutral deltas.
- **Promotion** (`promote-pipeline-baseline`): explicitly bless a run as the new reference, with provenance.
- **History ledger** (`show-baseline-history` / `verify-baseline-lineage`): an
  append-only, hash-chained record of every promotion (see `docs/baseline_history.md`).
  Pass `promote-pipeline-baseline --ledger-path …` to append on an approved promotion.

All are reproducibility / change-detection / auditability tools — none is a
model-validity or performance check.

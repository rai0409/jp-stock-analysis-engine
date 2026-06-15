# End-to-End Pipeline Runner & Determinism Gate (research-only)

**Modules:** `pipeline.py`, `determinism.py`, `audit.py` (artifact manifest)
**CLI:** `run-modeling-pipeline`, `verify-pipeline-determinism`
**Audit bundle:** see `docs/audit_bundle.md`

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## What this is — and is not

`run-modeling-pipeline` executes the P1–P4 modeling steps in a fixed order into a
stamped run directory and records a per-step summary, a real audit manifest, and
an artifact-manifest index. It is an **offline research pipeline runner, NOT a
trading system**:

- no buy/sell signal, no trading automation, no predictive-performance claim;
- **synthetic pipeline outputs are not market evidence**;
- **no-look-ahead BLOCKED is surfaced, not bypassed** (the readiness step records
  `blocked` and the summary carries the status);
- liquidity constraints require **real ADV/liquidity data** to be meaningful
  (synthetic fixtures carry none; ADV is never fabricated);
- the **determinism gate checks reproducibility, not model validity**;
- audit manifests improve **traceability, not performance**.

## Pipeline steps (in order)

1. build modeling dataset → 2. readiness (BLOCKED surfaced) → 3. ranking metrics →
4. JPX-style portfolio metrics → 5. neutralized ranking → 6. linear models (ridge
+ real Elastic Net) → 7. ensemble/blend → 8. stability → 9. feature importance →
10. constraints → 11. monitoring/drift → 12. consolidated modeling report →
13. **audit manifest** (fingerprints the actual produced files + real inputs) →
14. **artifact manifest index** → 15. pipeline summary.

Each step records `status`, inputs, outputs (relative paths), warnings, and a
skipped reason. Ordinary blocked/ineligible/degenerate cases do not crash the
pipeline.

## Run directory

`output_dir/<run_id>/` holds every artifact: `pipeline_summary.{json,md}`,
`artifact_manifest.{json,md}`, `audit_manifest.{json,md}`,
`modeling_report.{json,md}`, and per-step subdirectories (`dataset/`,
`readiness/`, `ranking/`, `portfolio/`, `neutralization/`, `linear/<model>/`,
`ensemble/`, `stability/`, `feature_importance/`, `constraints/`, `monitoring/`).
Paths in manifests are **relative** (no absolute temp paths), so runs are
location-independent.

## Artifact manifest index

Every produced artifact is fingerprinted: relative path, type, producing step,
size, SHA-256, CSV row/column counts, JSON top-level keys, research-only and
synthetic flags. Deterministic ordering by relative path. The index excludes
itself and the summary (to avoid self-reference). The audit manifest separately
fingerprints the real output files and the real input files.

## Determinism gate

`verify-pipeline-determinism` runs the pipeline **twice** (same `run_id`, same
`fixed_timestamp`, into two parent directories) and compares the artifact trees.
With a fixed run id and timestamp and relative paths, all artifacts are
**byte-identical** — canonicalization is a safety net for declared volatile fields
only (`created_at_utc`, `run_id`, absolute paths, elapsed seconds). It **does not
ignore** numeric/metric differences or missing/extra artifacts: a changed metric
or a removed file is reported as a difference. With `--fail-on-difference` the
command exits nonzero on any difference. Outputs: `determinism_report.{json,md}`.

## CLI

```
# run the full pipeline (synthetic)
python -m jp_stock_analysis.cli run-modeling-pipeline --synthetic \
    --run-id run --fixed-timestamp 1970-01-01T00:00:00Z \
    --transaction-cost-bps 10 --max-weight-per-name 0.34 --output-dir out/

# verify two runs are identical
python -m jp_stock_analysis.cli verify-pipeline-determinism --synthetic \
    --run-id-prefix det --fail-on-difference --output-dir out/
```

File inputs instead of `--synthetic`: `--prices --fundamentals --metadata
--decision-dates --disclosure-date|--disclosure-index --horizons`, plus optional
`--adv` (a `ticker,adv` CSV) for liquidity constraints. Model / portfolio /
constraints / monitoring options mirror the standalone commands
(`--linear-models`, `--alpha`, `--l1-ratio`, `--portfolio-*`,
`--max-weight-per-name`, `--max-sector-weight`, `--max-participation-rate`,
`--min-adv`, `--monitoring-window`, `--monitoring-threshold`).

## Regression baseline (vs the determinism gate)

The determinism gate (`verify-pipeline-determinism`) checks that *repeated runs
now* are byte-identical. The **regression baseline** (`check-pipeline-regression`,
see `docs/pipeline_regression.md`) checks *future changes* against a committed
golden reference — catching a code change that alters a metric or artifact. Both
are reproducibility checks, not model-validity checks. To diff two arbitrary runs
(A vs B) or to explicitly promote a new approved baseline with a provenance
record, see `compare-pipeline-runs` / `promote-pipeline-baseline` in
`docs/run_comparison.md` — metric deltas there are descriptive only (never
better/worse) and a promotion is an approved reference, not an improvement. Every
approved promotion can append a tamper-evident entry to the hash-chained
**baseline history ledger** (`show-baseline-history` / `verify-baseline-lineage`;
see `docs/baseline_history.md`).

`export-audit-bundle` packages the current golden baseline, baseline history
ledger, optional promotion records, and optional determinism/regression reports
into a self-contained audit artifact. `verify-audit-bundle` checks the internal
fingerprints, ledger chain, baseline-vs-ledger-head match, and overall bundle
fingerprint. A bundle is a reproducibility artifact, not a performance claim;
synthetic bundles are not market evidence.

## Recommended real-data run order

1. regenerate / verify the real topix1000 bundle (point-in-time fundamentals +
   `accounting_basis`).
2. fetch adjusted-close prices extending past the disclosure date.
3. `check-forward-readiness` (must be ELIGIBLE; the pipeline surfaces BLOCKED).
4. `run-modeling-pipeline` (with `--adv` for real liquidity constraints).
5. `verify-pipeline-determinism`.
6. review the audit manifest.
7. run `check-pipeline-regression`.
8. export and verify an audit bundle.
9. review the modeling report.
10. only then consider further validation.

P0 strict no-look-ahead remains required before any predictive claim. Nothing
here constitutes a predictive or trading claim until that validation succeeds on
real data.

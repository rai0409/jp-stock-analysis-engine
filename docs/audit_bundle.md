# Offline Audit Bundle (research-only)

**Module:** `audit_bundle.py`
**CLI:** `export-audit-bundle`, `verify-audit-bundle`

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## What this is

An audit bundle is a self-contained reproducibility artifact. It packages the
approved golden pipeline baseline, the hash-chained baseline history ledger,
optional promotion records, optional determinism and regression reports, and
selected pipeline manifests into one directory with a top-level
`audit_bundle_manifest.json`.

The bundle is for auditability and reproducibility only. It does **not** prove
model validity, does **not** prove market performance, creates no buy/sell
recommendations, and is not trading automation. Synthetic bundles are clearly
labelled synthetic and are **not market evidence**.

## Manifest and fingerprints

`audit_bundle_manifest.json` records:

- schema version, bundle id, fixed timestamp, synthetic flag, and research-only
  flag;
- every included file by deterministic relative path;
- per-file size, raw SHA-256, canonical SHA-256, and safe JSON/CSV/text metadata;
- ledger verification status, head hash, and entry count;
- bundled baseline fingerprint and ledger-head baseline fingerprint;
- whether the baseline fingerprint matches the ledger head;
- determinism/regression status, or `unavailable` when reports were not supplied;
- warnings;
- an overall bundle fingerprint.

The overall bundle fingerprint is recomputed over canonical manifest content
excluding `overall_bundle_fingerprint` plus the included files' canonical
fingerprints. Manifests avoid absolute paths, raw environment-specific values,
secrets, and raw timestamps when `--fixed-timestamp` is supplied.

## Verification

`verify-audit-bundle` checks:

- `audit_bundle_manifest.json` exists and is readable;
- manifest fingerprints match actual bundle contents;
- required baseline and ledger files exist;
- no secret-like fields or absolute paths appear in stable/canonical fields;
- ledger chain integrity verifies;
- bundled baseline fingerprint matches the manifest;
- ledger head `new_baseline_fingerprint` / `baseline_fingerprint` matches the
  bundled baseline when available;
- promotion records reference ledger entries when possible;
- overall bundle fingerprint recomputes exactly;
- determinism and regression reports are readable when included.

Tampered files, missing files, broken ledger chains, baseline/ledger mismatches,
manifest/content mismatches, and invalid bundle fingerprints are reported as
`invalid` with explicit issues. They are detected, not hidden.

## Export

Synthetic bundle from committed fixtures:

```bash
python -m jp_stock_analysis.cli export-audit-bundle \
  --synthetic \
  --bundle-id synthetic-audit-19700101 \
  --fixed-timestamp 1970-01-01T00:00:00Z \
  --baseline-path tests/fixtures/pipeline_baseline/golden_pipeline_baseline.json \
  --ledger-path tests/fixtures/pipeline_baseline/baseline_history.jsonl \
  --output-dir out/audit_bundle
```

Including reports from prior offline runs:

```bash
python -m jp_stock_analysis.cli export-audit-bundle \
  --synthetic \
  --bundle-id synthetic-audit-with-reports \
  --fixed-timestamp 1970-01-01T00:00:00Z \
  --baseline-path tests/fixtures/pipeline_baseline/golden_pipeline_baseline.json \
  --ledger-path tests/fixtures/pipeline_baseline/baseline_history.jsonl \
  --determinism-report out/determinism/determinism_report.json \
  --regression-report out/regression/pipeline_regression_report.json \
  --pipeline-run-dir out/pipeline/golden \
  --output-dir out/audit_bundle
```

## Verify

```bash
python -m jp_stock_analysis.cli verify-audit-bundle \
  --bundle-dir out/audit_bundle \
  --output-dir out/audit_bundle_verification \
  --fail-on-invalid
```

Outputs are `audit_bundle_verification.json` and
`audit_bundle_verification.md`.

## CI use

For synthetic offline CI:

```bash
python -m jp_stock_analysis.cli check-pipeline-regression --synthetic --fail-on-regression --output-dir out/regression
python -m jp_stock_analysis.cli verify-baseline-lineage --fail-on-invalid
python -m jp_stock_analysis.cli export-audit-bundle --synthetic --fixed-timestamp 1970-01-01T00:00:00Z --output-dir out/audit_bundle
python -m jp_stock_analysis.cli verify-audit-bundle --bundle-dir out/audit_bundle --fail-on-invalid
```

This proves internal consistency and tamper evidence only.

## Real-data caveat

Real-data audit bundles can package real runs, but predictive interpretation
requires `check-forward-readiness=ELIGIBLE` first. P0 strict broad no-look-ahead
validation remains required before any predictive or performance claim.

Current blocker remains data coverage: disclosure date **2026-03-27** while
adjusted-close price coverage previously ended **2026-03-19**.

# Golden pipeline regression baseline (synthetic-only)

`golden_pipeline_baseline.json` is a **canonicalized fingerprint set** of a fixed
synthetic `run-modeling-pipeline` run (run id `golden`, fixed timestamp epoch,
canonical synthetic config). It is the blessed reference used by
`check-pipeline-regression` to detect unexpected pipeline output changes.

- **SYNTHETIC ONLY — not market evidence.** It proves nothing about model validity
  or market performance; it only records what the deterministic synthetic pipeline
  produces.
- Contains canonical + raw fingerprints and safe semantic metadata
  (CSV rows/columns, JSON top-level keys, line counts, a few headline metrics).
  It contains **no absolute paths, no timestamps, no secrets, no run-specific
  values** (run id / timestamp / absolute paths are canonicalized out of the
  canonical fingerprint).
- Provenance manifests (`audit_manifest.*`, `artifact_manifest.*`) are excluded
  because they are intentionally run/commit-specific.

Regenerate only intentionally, with review, via
`check-pipeline-regression --synthetic --update-baseline` (see
`docs/pipeline_regression.md`).

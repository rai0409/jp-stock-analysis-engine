# Baseline History & Hash-Chained Lineage Ledger (research-only)

**Module:** `baseline_history.py`
**CLI:** `show-baseline-history`, `verify-baseline-lineage` (+ `promote-pipeline-baseline --ledger-path`)
**Seed fixture:** `tests/fixtures/pipeline_baseline/baseline_history.jsonl`
**Audit bundle:** see `docs/audit_bundle.md`

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## What this is — and is not

The baseline history is an **append-only, hash-chained audit ledger** of baseline
promotions. Each entry's `entry_hash` is computed over its canonicalized content
(excluding the hash itself) and references the previous entry's hash
(`parent_hash`), so the promotion history is **tamper-evident**: a silent edit, a
broken parent link, an out-of-order index, a duplicate hash, or malformed JSONL is
**detected and reported, not hidden**.

It **does not** prove model validity or market performance. Ledger entries are
**synthetic-only** unless generated from real-data runs. A promotion means
"approved reference", **not** "better model". Secrets are scrubbed; committed
entries carry **no absolute paths, no secrets, and only a canonical fixed
timestamp**. Real-data baselines require `check-forward-readiness=ELIGIBLE` before
any predictive interpretation, and **P0 strict no-look-ahead remains required
before predictive claims**.

## Ledger entry

One JSON object per line. Each entry contains: `ledger_schema_version`,
`entry_index`, `parent_hash`, `entry_hash`, `promotion_record_fingerprint`,
`baseline_fingerprint` / `new_baseline_fingerprint` / `previous_baseline_fingerprint`,
`source_run_id`, `reviewer_note`, `approved`, `approval_required`, `synthetic`,
`research_only`, `created_at_utc` (canonical/fixed in tests), a compact
`headline_metric_delta_summary`, `artifact_classification_counts`, `warnings`, and
secret-scrubbed `metadata`. The first (index 0) entry has `parent_hash = GENESIS`.

## Verification

`verify-baseline-lineage` (and `verify_baseline_history`) checks: valid JSONL,
`entry_index` sequence, `parent_hash` chaining, recomputed `entry_hash` equality,
required fields present, no secret-like fields, no absolute paths, no duplicate
hashes, and a valid genesis parent. It returns `valid` / `invalid` with an
explicit `issues` list, entry count, head hash, and first/last entry metadata.

`verify-audit-bundle` repeats this ledger-chain verification inside a packaged
bundle and also requires the bundled golden baseline fingerprint to match the
ledger head. A valid bundle proves internal consistency and tamper evidence only;
it does not prove predictive validity or market performance.

## Promotion → ledger

`promote-pipeline-baseline --ledger-path <ledger>` appends exactly one entry
**only on a successful (approved) promotion**. A blocked promotion appends
nothing. To avoid a partial state (a promoted baseline with an un-appendable
ledger), the existing chain is **verified before any write**: a broken/unreadable
chain **blocks the whole promotion** (status `blocked_broken_ledger`) and the
baseline is **not** updated. On success the promotion record records
`ledger_append_status` and `appended_entry_hash`.

## CLI

```
# show the lineage (verifies the chain too)
python -m jp_stock_analysis.cli show-baseline-history \
    --ledger-path tests/fixtures/pipeline_baseline/baseline_history.jsonl --output-dir out/

# verify the chain (CI-style)
python -m jp_stock_analysis.cli verify-baseline-lineage \
    --ledger-path tests/fixtures/pipeline_baseline/baseline_history.jsonl --fail-on-invalid

# promote and append to the ledger (explicit approval + reviewer note)
python -m jp_stock_analysis.cli promote-pipeline-baseline \
    --from-run out/run/golden \
    --baseline-path tests/fixtures/pipeline_baseline/golden_pipeline_baseline.json \
    --reviewer-note "approved: intentional change X, reviewed by <name>" \
    --require-approval --approve \
    --ledger-path tests/fixtures/pipeline_baseline/baseline_history.jsonl \
    --output-dir out/promo

# package baseline + ledger into a self-contained audit bundle
python -m jp_stock_analysis.cli export-audit-bundle --synthetic \
    --fixed-timestamp 1970-01-01T00:00:00Z --output-dir out/audit_bundle

# verify bundle fingerprints, ledger chain, and baseline/ledger consistency
python -m jp_stock_analysis.cli verify-audit-bundle \
    --bundle-dir out/audit_bundle --fail-on-invalid
```

## Reviewing ledger changes in code review

Because the ledger is committed and hash-chained, a reviewer should: (1) confirm
only **new** lines were appended (no edits to existing lines); (2) run
`verify-baseline-lineage --fail-on-invalid`; (3) read the appended entry's
`reviewer_note` and metric-delta summary. An edit to any historical line will make
verification fail — that is the point.

## How this fits

- **Determinism gate**: two runs *now* are byte-identical.
- **Regression baseline**: a fresh run vs the committed golden baseline.
- **Run comparison**: any two runs A vs B (neutral deltas).
- **Promotion**: explicitly bless a run as the new reference.
- **History ledger** *(this)*: an append-only, tamper-evident record of every promotion.

All are reproducibility / auditability tools — none is a model-validity or
performance check.

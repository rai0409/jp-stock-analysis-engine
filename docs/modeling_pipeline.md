# Modeling Pipeline (offline, research-only)

**Scope:** P1–P3 commercial-grade modeling infrastructure under
`src/jp_stock_analysis/modeling/`. Offline, deterministic, and testable on
synthetic fixtures before any real fundamentals/prices exist.

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## What this is — and is not

This is **research infrastructure**, not a trading system:

- It produces **no** buy/sell signals and **no** trade automation.
- It makes **no** predictive-performance claim. The ranking metrics measure
  cross-sectional *association* only.
- `analysis_only` remains the default posture of the whole project.
- **Synthetic-fixture results are not market evidence.** They prove only that the
  pipeline runs deterministically.
- Consolidated and `non_consolidated` fundamentals are **never** pooled silently;
  `non_consolidated` rows are excluded by default.
- The strict no-look-ahead readiness check is **not** weakened.

P0 (strict broad no-look-ahead validation) is intentionally **skipped** here: the
current data coverage blocks it (the bundle disclosure date is after the adjusted
price coverage end). This task builds the P1–P3 machinery so that when valid
point-in-time fundamentals/prices arrive, validation can run immediately.

## Modules

| module | role |
| --- | --- |
| `modeling/factors.py` | explainable factor features + normalisation |
| `modeling/dataset.py` | model-ready observations + no-look-ahead guardrails |
| `modeling/ranking_metrics.py` | Rank IC / ICIR / quantile spread / hit rate |
| `modeling/walk_forward.py` | domain-aware walk-forward fold generation |
| `modeling/purged.py` | purged / embargo splitting for forward-return labels |
| `modeling/baseline_ranker.py` | transparent equal-weight factor ranker |
| `modeling/ml_models.py` | optional LightGBM / CatBoost adapters (skip if absent) |
| `modeling/portfolio_metrics.py` | JPX-style long-short spread / Sharpe-like / turnover / drawdown / optional cost (see `docs/portfolio_evaluation.md`) |
| `modeling/neutralization.py` | Numerai-style neutralization, neutralized Rank IC, MMC-style delta (see `docs/neutralization_metrics.md`) |
| `modeling/linear_models.py` | deterministic Ridge + real coordinate-descent Elastic Net (see `docs/model_diversity.md`) |
| `modeling/ensemble.py` | rank-average / weighted-blend ensembles + diversity diagnostics |
| `modeling/stability.py` | walk-forward / seed stability summaries |
| `modeling/feature_importance.py` | coefficient + permutation feature importance |
| `modeling/constraints.py` | position/liquidity/sector/turnover feasibility constraints (see `docs/commercial_validation.md`) |
| `modeling/audit.py` | deterministic reproducibility manifest + artifact-manifest index (secret-scrubbed) |
| `modeling/monitoring.py` | drift / stability monitoring across decision dates |
| `modeling/pipeline.py` | deterministic end-to-end pipeline runner (see `docs/pipeline_runner.md`) |
| `modeling/determinism.py` | canonicalization + artifact-tree comparison (determinism gate) |
| `modeling/report.py` | full offline modeling report |
| `modeling/fixtures.py` | deterministic synthetic bundle (SYNTHETIC ONLY) |

## Data contract

The dataset consumes the topix1000 export contract:

- fundamentals CSV now carries an `accounting_basis` column
  (`consolidated` / `non_consolidated` / `unknown`); see the platform's
  `docs/engine_bundle_schema.md`. The loader threads it onto
  `FinancialStatement.accounting_basis`.
- disclosures carry a narrative-extraction contract (`extraction_status`, …);
  see the platform's `docs/disclosure_narrative_contract.md`. Today extraction is
  `not_attempted`, so only presence/placeholder narrative features are derived
  (no LLM, no external NLP).

## CLI commands

All are offline. `--synthetic` uses the built-in deterministic fixture bundle.

```
python -m jp_stock_analysis.cli build-modeling-dataset   --synthetic --output-dir out/
python -m jp_stock_analysis.cli evaluate-factor-ranking  --synthetic --output-dir out/
python -m jp_stock_analysis.cli run-walk-forward-ranking --synthetic --output-dir out/
python -m jp_stock_analysis.cli train-ranking-model      --synthetic --model-type baseline_factor_ranker --output-dir out/
python -m jp_stock_analysis.cli evaluate-portfolio-ranking   --synthetic --horizon 20 --output-dir out/
python -m jp_stock_analysis.cli evaluate-neutralized-ranking --synthetic --horizon 20 --output-dir out/
python -m jp_stock_analysis.cli train-linear-ranking-model   --synthetic --linear-model-type elastic_net --horizon 20 --feature-importance --output-dir out/
python -m jp_stock_analysis.cli evaluate-model-stability      --synthetic --horizon 20 --output-dir out/
python -m jp_stock_analysis.cli evaluate-portfolio-constraints --synthetic --horizon 20 --max-weight-per-name 0.34 --output-dir out/
python -m jp_stock_analysis.cli build-audit-manifest         --synthetic --input fundamentals.csv --output-dir out/
python -m jp_stock_analysis.cli evaluate-model-monitoring     --synthetic --horizon 20 --output-dir out/
python -m jp_stock_analysis.cli modeling-report          --synthetic --output-dir out/
# or run everything in one deterministic pass (see docs/pipeline_runner.md):
python -m jp_stock_analysis.cli run-modeling-pipeline        --synthetic --run-id run --fixed-timestamp 1970-01-01T00:00:00Z --output-dir out/
python -m jp_stock_analysis.cli verify-pipeline-determinism  --synthetic --fail-on-difference --output-dir out/
```

The `modeling-report` long-short and neutralization sections accept
`--portfolio-top-quantile`, `--portfolio-bottom-quantile`,
`--portfolio-rank-weighted`, `--transaction-cost-bps`, `--neutralize-exposures`,
and `--neutralize-proportion`. See `docs/portfolio_evaluation.md` and
`docs/neutralization_metrics.md`.

File inputs instead of `--synthetic`:

```
python -m jp_stock_analysis.cli modeling-report \
    --prices prices.csv --fundamentals fundamentals.csv --metadata metadata.csv \
    --decision-dates 2026-03-27,2026-06-30 --disclosure-date 2026-03-27 \
    --horizons 5,20,60 --output-dir out/
```

`--disclosure-index path/to/index.json` reads the bundle `target_date` instead of
`--disclosure-date`.

## Run order when real data becomes available

1. **build modeling dataset** — assemble observations + labels with guardrails.
2. **compute factors** — already inside the dataset build (`modeling/factors.py`).
3. **run readiness check** — `check-forward-readiness` (strict no-look-ahead).
   Do not proceed past a `BLOCKED` verdict.
4. **run factor ranking validation** — `evaluate-factor-ranking`.
5. **run walk-forward validation** — `run-walk-forward-ranking`.
5b. **run long-short / neutralized evaluation** — `evaluate-portfolio-ranking`
   and `evaluate-neutralized-ranking` (research metrics; no trading signal).
5c. **train linear baselines** — `train-linear-ranking-model` (ridge / Elastic
   Net) and **evaluate stability** — `evaluate-model-stability`
   (research diagnostics; see `docs/model_diversity.md`).
5d. **apply constraints & cost/liquidity checks** —
   `evaluate-portfolio-constraints` (real ADV required for liquidity to be
   meaningful) and **monitor drift** — `evaluate-model-monitoring`
   (see `docs/commercial_validation.md`).
5e. **generate audit manifest** — `build-audit-manifest` (reproducibility).
6. **optionally train LightGBM/CatBoost** — `train-ranking-model` (install the
   `lightgbm` / `catboost` / `all-modeling` extras first).
7. **compare against baseline** — the report's model-comparison table.
8. **generate report** — `modeling-report`.

Or run all of the above deterministically in one pass with
`run-modeling-pipeline` and reproduce-check it with `verify-pipeline-determinism`
(see `docs/pipeline_runner.md`).

## Optional dependencies

LightGBM and CatBoost are **optional extras** (`lightgbm`, `catboost`,
`all-modeling`). When absent, the adapters return `optional_dependency_missing`
and the test-suite still passes — they are never required for a minimal install.

## Limitations

- No predictive-performance claim; ranking association only.
- Synthetic results are not market evidence.
- Real validation needs point-in-time disclosure dates and adjusted-close prices,
  and must pass strict no-look-ahead first.
- Narrative features are placeholders until real extraction exists.

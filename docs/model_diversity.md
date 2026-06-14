# Model Diversity, Stability & Explainability (research-only)

**Modules:** `linear_models.py`, `ensemble.py`, `stability.py`,
`feature_importance.py`
**CLI:** `train-linear-ranking-model`, `evaluate-model-stability` (+ integrated
into `modeling-report`)

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## What this is — and is not

Offline, deterministic **research diagnostics**: linear baselines, ensembling,
walk-forward/seed stability, and feature importance. They make **no** predictive
claim, emit **no** buy/sell signal, and synthetic results are clearly labelled
and are **not market evidence**. Real-data GBDT training stays gated on strict
no-look-ahead P0 (see `docs/commercial_readiness_gap.md`).

## Linear baselines (`linear_models.py`)

Both share a `fit` / `predict` / `fit_predict` API, `model_metadata`,
`coefficients`, `intercept`, `status`, `warnings`, `model_version`. Preprocessing:
training-set **median imputation** (values recorded and reused at predict time)
and optional standardization. Degenerate cases (too few rows, constant target,
all-missing/zero-variance feature, non-finite values) return a clear `status`
and `warnings` instead of raising; invalid `alpha`/`l1_ratio`/`max_iter`/`tol`
raise `ValueError`.

### Ridge
Deterministic closed-form ridge: `beta = (XᵀX + alpha·I)⁻¹ Xᵀy` on standardized,
centered data; the intercept is unpenalized. Singular Gram matrices are absorbed
by the ridge term (pseudo-inverse fallback with a warning).

### Elastic Net — real coordinate descent (NOT a placeholder, NOT sklearn)
Minimises

    f(beta) = 1/(2n)·||y − Xβ||² + alpha·l1_ratio·||β||₁
              + 0.5·alpha·(1 − l1_ratio)·||β||²

with **coordinate descent + soft-thresholding**:

- `soft_threshold(z, gamma) = sign(z)·max(|z| − gamma, 0)`;
- per-feature update `beta_j = soft_threshold(rho_j, alpha·l1_ratio) / (norm_j + alpha·(1 − l1_ratio))`,
  with residual updating and a deterministic feature order;
- the intercept is kept separate (unpenalized);
- **`l1_ratio = 0`** → L2-only (ridge-like, no exact zeros);
  **`l1_ratio = 1`** → L1-only (lasso-like sparsity);
  **`0 < l1_ratio < 1`** → true Elastic Net (sparsity + shrinkage);
- stored: feature means/scales, target mean, imputation values, scaled and
  unscaled coefficients, intercept on the original scale;
- convergence tracking: `n_iter`, `converged`, `objective_history`
  (non-increasing within tolerance), `max_coefficient_change`, `final_objective`;
- non-convergence returns `status="not_converged"` with the last deterministic
  coefficients and a warning — it does not raise.

Elastic Net can help with **sparse, correlated factor sets** (it selects and
shrinks), but **sparsity is not proof of alpha**.

## Ensemble / blending (`ensemble.py`)

- **Rank-average**: per decision date, rank each model and average the ranks
  (deterministic ties) → an `ensemble_score`.
- **Weighted blend**: normalise weights, blend per-decision-date **standardized**
  predictions (scale-robust); missing/degenerate weights return a status.
- **Diversity**: pairwise Spearman correlations and a `diversity_score`
  (`1 − mean pairwise corr`); near-identical models are flagged.

Outputs are `ScoredObservation` lists → they flow into Rank IC, portfolio
metrics, and neutralization unchanged.

## Stability (`stability.py`)

Per-metric summary across folds/seeds: mean, std, min, max, CV (when the mean is
non-zero), `positive_period_rate`, worst/best fold. `compute_fold_metrics`
derives per-fold Rank IC, long-short spread, Sharpe-like, and hit rate from
walk-forward folds. `synthetic_seed_ic` is a **seed-noise robustness probe**
(synthetic only); for a deterministic model the per-seed *retraining* variance is
reported as not-applicable. Degenerate cases (one fold, all-None, empty) return a
clear status.

## Feature importance (`feature_importance.py`)

- **Coefficient importance**: normalised |standardized coefficients| (sums to 1;
  all-zero handled).
- **Permutation importance** (synthetic/offline, deterministic seed): Rank IC
  degradation when one feature is shuffled; unavailable metrics return a status.

**Feature importance is explanatory research output only, not causal proof.**

## CLI

```
# train a linear ranker (ridge or real Elastic Net) + optional importance
python -m jp_stock_analysis.cli train-linear-ranking-model --synthetic \
    --linear-model-type elastic_net --horizon 20 --alpha 0.05 --l1-ratio 0.5 \
    --feature-importance --output-dir out/
# -> model_metadata.json, coefficients.csv, predictions.csv, feature_importance.json

# walk-forward + seed stability of the baseline ranker
python -m jp_stock_analysis.cli evaluate-model-stability --synthetic \
    --horizon 20 --seed-count 4 --output-dir out/
# -> model_stability.{json,csv,md}
```

`modeling-report` includes a "Model diversity, stability & explainability"
section built from all four modules. Use `--prices/--fundamentals/...` instead of
`--synthetic` for file inputs.

## Real-data run order

1. `check-forward-readiness` (must be ELIGIBLE; do not proceed on BLOCKED).
2. `build-modeling-dataset`.
3. train baseline factor / ridge / Elastic Net / optional GBDT models.
4. evaluate Rank IC / portfolio spread / neutralization.
5. run walk-forward.
6. evaluate stability.
7. generate `modeling-report`.

LightGBM / CatBoost remain **optional** throughout; absence is a clean skip.
Nothing here constitutes a predictive or trading claim until strict real
no-look-ahead validation succeeds.

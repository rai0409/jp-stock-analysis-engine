# Numerai-Style Neutralization & Neutralized Rank Metrics (research-only)

**Module:** `src/jp_stock_analysis/modeling/neutralization.py`
**CLI:** `evaluate-neutralized-ranking` (+ integrated into `modeling-report`)

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## What this is — and is not

Inspired by neutralized cross-sectional ranking concepts (regression-residual
feature/market neutralization and rank correlation). It is **NOT official Numerai
scoring** unless exactly matched, makes no predictive/trading claim, and uses no
Numerai package and no network. Synthetic results are labelled and are **not
market evidence**.

## Components

### 1. Regression-residual neutralization — `neutralize`
`neutralized = prediction − proportion · X · lstsq(X, prediction)` over rows
where the prediction is present, with an intercept column. `proportion` (default
1.0) controls strength (1.0 = fully remove the linear exposure). Deterministic
least squares (`numpy.linalg.lstsq`).

- Requested exposure columns that are **absent / all-missing / constant /
  length-mismatched** are recorded in `skipped_exposures` — **never silently
  ignored**.
- Missing values inside a *used* column are mean-imputed for the design matrix
  only (predictions are never fabricated).
- Degenerate cases return an explicit `status`
  (`constant_prediction`, `no_exposures_used`, `insufficient_points`).

### 2. Neutralized Rank IC — `neutralized_rank_ic`
Per decision date: neutralize the prediction against the exposures (factor
columns + one-hot **sector dummies**), then Spearman rank-correlate the residual
with the forward return. Reports per-date IC, **mean / std / ICIR**, the **raw**
(pre-neutralization) IC mean for contrast, and exposure diagnostics.

### 3. Exposure diagnostics — `exposure_diagnostics`
`pre_neutralization_exposure_corr` and `post_neutralization_exposure_corr` per
exposure, plus `max_abs_exposure_corr_before` / `…_after`. After full
neutralization the post-correlation collapses toward 0 (verified in tests).

### 4. MMC-style contribution delta — `mmc_style_contribution`
Given a **base** and a **candidate** prediction, neutralize the candidate against
the base and measure the residual's Rank IC vs the forward return.

> **MMC-STYLE only** — this is the candidate's mean neutralized-vs-base Rank IC.
> It is **NOT** official Numerai Meta Model Contribution unless exactly matched.

Requires **≥2 model predictions**. In the modeling report it compares the
baseline ranker against the first trained optional model (LightGBM/CatBoost);
when no optional backend is installed there is only the baseline, so the report
marks MMC-style as unavailable.

## CLI

```
python -m jp_stock_analysis.cli evaluate-neutralized-ranking \
    --synthetic --horizon 20 --neutralize-exposures momentum_60d,leverage \
    --neutralize-proportion 1.0 --output-dir out/
```

Sector dummies are always added to the requested factor exposures. Outputs
`neutralized_metrics.{json,csv,md}`.

## When real data exists
Neutralized Rank IC and exposure diagnostics become meaningful only on real
point-in-time data that has passed `check-forward-readiness`. Until then these
are infrastructure checks on synthetic fixtures, not evidence.

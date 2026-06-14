# Competition Benchmark Gap Analysis

**Date:** 2026-06-14
**Scope:** Compare the current offline stock-analysis modeling system against
public financial-ML / equity-prediction competitions and their top solutions,
then produce a hallucination-free gap analysis and improvement plan.

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice and contains no trading or investment
> recommendations.

## Evidence policy used in this document

- **VERIFIED** — confirmed from an official or reliably machine-readable source
  (official competition repo/docs, official GitHub, fetched page) during this
  analysis, with the URL cited.
- **NOT VERIFIED** — plausible/commonly-reported but not machine-confirmed here
  (e.g. Kaggle leaderboard/writeup pages are JavaScript-rendered and were not
  readable by the fetch tool). Treated as unconfirmed.
- **Synthetic fixture results are NOT market evidence.** No "we can beat the
  winners" claim is made anywhere, because no real strict validation exists.

---

## 1. Preflight (confirmed, not assumed)

| repo | HEAD | expected commit present | tests | ruff |
| --- | --- | --- | --- | --- |
| `topix1000_disclosure_platform` | `e9c7665` | yes (`e9c7665` "Add accounting-basis and narrative export contracts") | export suites `25 passed` | clean |
| `jp-stock-analysis-engine` | `f1bbfd8` | yes (`f1bbfd8` "Add offline factor ranking and walk-forward modeling infrastructure") | `290 passed, 1 skipped` | clean |

Commands run: `.venv/bin/python -m pytest` (engine, full); targeted export
suites in the platform repo (`test_export_engine_fundamentals.py`,
`test_export_disclosures.py`); `.venv/bin/ruff check .`. The one engine skip is
the LightGBM-installed training path (LightGBM/CatBoost intentionally absent).
`topix1000_disclosure_platform` has a pre-existing untracked
`apps/edinet_ingest/scripts/` directory (left untouched).

---

## 2. Current system inspection (Task 1)

### 2.1 `topix1000_disclosure_platform` — point-in-time metadata

Confirmed in `apps/edinet_ingest/src/edinet_ingest/export/`:

- **`accounting_basis`** export column: `consolidated` / `non_consolidated` /
  `unknown` (`engine_fundamentals.py`). Consolidated always wins; a row never
  blends bases. A parent-only fallback exists but is **OFF by default**; basis
  detection/labelling is always on and fabricates nothing. Summary reports
  `accounting_basis_counts`. (docs/engine_bundle_schema.md)
- **Narrative contract** (`disclosures.py`): `extraction_status` (always
  `not_attempted` today), `extraction_error`, `narrative_text`,
  `narrative_sections`, `source_doc_id`, `disclosure_date`. **No real narrative
  extractor exists — contract only**, no OCR/API/LLM. (docs/disclosure_narrative_contract.md)

**Point-in-time metadata sufficiency for competition-grade validation:**

| field | present | note |
| --- | --- | --- |
| `disclosure_date` | YES | narrative contract + bundle `target_date` |
| `target_date` | YES | bundle disclosure/target date |
| `ticker` | YES | 4-char listing code |
| `accounting_basis` | YES | consolidated/non_consolidated/unknown |
| `source_doc_id` | YES | EDINET doc id provenance |
| `extraction_status` | YES | `not_attempted` (no extractor yet) |

**Verdict:** the *schema* is point-in-time-capable. What is missing is **breadth
and price coverage**, not metadata fields: only 3 tickers have real adjusted-close
history and it ends **2026-03-19**, before the **2026-03-27** disclosure date
(see §2.3).

### 2.2 `jp-stock-analysis-engine` — modeling package

Confirmed modules in `src/jp_stock_analysis/modeling/`:

| module | implemented | notes |
| --- | --- | --- |
| `dataset.py` | YES | observation key (ticker, decision_date, disclosure_date, horizon); horizons **5/20/60** (`DEFAULT_HORIZONS`); excludes `disclosure_date > decision_date` and `non_consolidated` by default; labels `forward_return_h{5,20,60}` + sector-demeaned `excess_return_*` |
| `factors.py` | YES | value / quality / growth / momentum / risk + disclosure placeholders; winsorize, z-score, sector-relative z-score; divide-by-zero → None |
| `ranking_metrics.py` | YES | Rank IC/Spearman, IC mean/std, **ICIR**, sector-neutral IC, quantile spread (mean), decile/quintile table, hit rate |
| `walk_forward.py` | YES | expanding/rolling folds, min-train/test periods, fold table |
| `purged.py` | YES | purge on label-window overlap + embargo days |
| `baseline_ranker.py` | YES | equal-weight cross-sectional ranker, `factor_score`/`factor_rank`/sector-neutral, `model_version` |
| `ml_models.py` | YES | optional LightGBM/CatBoost ranker+regressor; clean `optional_dependency_missing` when absent |
| `report.py` | YES | coverage, basis distribution, ranking, walk-forward, model comparison, no-look-ahead status, limitations |
| `fixtures.py` | YES | deterministic synthetic bundle (non-evidence) |

CLI commands confirmed: `build-modeling-dataset`, `evaluate-factor-ranking`,
`run-walk-forward-ranking`, `train-ranking-model`, `modeling-report`,
`check-forward-readiness`.

**Confirmed gaps (grep-verified, not assumed):**

- **No portfolio long-short return series, no Sharpe of that series, no
  turnover, no transaction-cost / liquidity model, no strategy drawdown.** The
  only Sharpe-like quantity is `ICIR` (mean/std of the IC time series), and
  `quantile_spread_mean` is a *mean* spread, not a risk-adjusted (mean/std)
  portfolio return. → this is exactly the JPX-style evaluator gap (§6 P1).
- **No factor/market/size/value neutralization** beyond sector-relative
  z-score, and **no meta-model-contribution (MMC)-style** metric. → Numerai-style
  gap (§6 P2).
- **No linear/ridge/elastic-net** model, **no ensemble/blending/stacking**, no
  seed-stability harness, no feature-importance export.

### 2.3 Real-validation status (grounded in local docs)

`docs/forward_return_validation_results.md` and `docs/commercial_readiness_gap.md`
confirm: the only real-price run (`n=3`, tickers 3928/4107/4264, 488 rows each,
coverage ending **2026-03-19**) is a **PIPELINE PROOF ONLY** — its decision date
(2025-11-28) precedes the bundle disclosure date (2026-03-27), so it is
look-ahead with respect to disclosure availability. Strict broad no-look-ahead
validation is **BLOCKED** by data coverage, proven deterministically by
`check-forward-readiness`. **There is no real predictive evidence of any kind.**

---

## 3. Comparable competitions extracted (Task 2)

### A. JPX Tokyo Stock Exchange Prediction (Kaggle, 2022)
- **Source:** official J-Quants repo `https://github.com/J-Quants/JPXTokyoStockExchangePrediction`; competition `https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction`.
- **Domain / task:** Japanese equities (~2,000 stocks); cross-sectional **ranking** by expected forward return.
- **Target / horizon:** short-horizon daily forward return from adjusted close (next-day rate of change). *(metric type VERIFIED; exact day offset per official definition)*
- **Metric:** **Sharpe ratio of the daily spread return** = mean(daily spread return) / std(daily spread return), where the daily spread return is a **rank-weighted top-200 long minus bottom-200 short** portfolio. *(VERIFIED metric type — official repo + search; exact linear weight values NOT re-verified here.)*
- **Top private-LB scores: 1st = 0.381 … 10th = 0.280.** *(VERIFIED — official J-Quants repo lists 1st–10th models with these scores.)*
- **Models (strong/known solutions):** GBDT / **LightGBM** prevalent; at least one verified strong solution (Adam-Chellaoui, ranked 71/2033, top 4%) combined **LSTM + LightGBM** (`https://github.com/Adam-Chellaoui/Predicting-the-Japan-Stock-Market-Kaggle-Challenge`). Broad winning-method list NOT VERIFIED.
- **Validation design / leakage:** time-ordered API evaluation; known caveats are non-stationarity and small Sharpe differences between ranks (0.381 vs 0.280 is a narrow band → high variance / overfitting risk). 
- **Directly comparable to our project?** **YES.** **Confidence: HIGH.**

### B. Numerai Tournament / Numerai Signals
- **Source:** `https://docs.numer.ai/numerai-tournament/scoring/correlation-corr`.
- **Domain / task:** obfuscated cross-sectional global-equity prediction; ongoing tournament (no fixed leaderboard "score to beat").
- **Target / horizon:** stock-specific returns; **20-day main target (CORR20V2)** and **60-day auxiliary (CORJ60)**, plus CORT20. *(VERIFIED.)*
- **Metric:** **Numerai Corr** — rank-correlation of predictions vs target, predictions Gaussianized and raised to power 1.5; *"You are only evaluated on your prediction's ranks."* Plus **MMC (Meta Model Contribution)** and **feature neutralization**. *(CORR + targets VERIFIED; MMC/neutralization referenced on the docs but exact formulas NOT fully fetched → concept VERIFIED, formula NOT VERIFIED.)*
- **Models (top):** ensembles of GBDTs + NNs with heavy neutralization. NOT VERIFIED in specifics.
- **Directly comparable?** **YES, conceptually** (rank correlation + neutralization is exactly our factor/ranking domain). **Confidence: HIGH (concepts), LOW (specific scores).**

### C. Jane Street Real-Time Market Data Forecasting (Kaggle, 2024)
- **Source:** `https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting`.
- **Domain / task:** anonymized real market data; time-series regression of a responder.
- **Target / horizon:** `responder_6` over a fixed undisclosed horizon; sample weights provided.
- **Metric:** **sample-weighted zero-mean R²** = `1 − Σ wᵢ(yᵢ−ŷᵢ)² / Σ wᵢyᵢ²`. *(VERIFIED — consistent across sources.)*
- **Models (top):** NN/GBDT ensembles with online updating. NOT VERIFIED in specifics.
- **Directly comparable?** **PARTIALLY** — high-noise financial prediction, but **anonymized high-frequency-style market data**, not fundamental cross-section. **Confidence: MEDIUM.**

### D. Optiver — Trading at the Close (Kaggle, 2023)
- **Source:** `https://www.kaggle.com/competitions/optiver-trading-at-the-close`; solution repo `https://github.com/nimashahbazi/optiver-trading-close`.
- **Domain / task:** NASDAQ **closing-auction** microstructure; predict short-term price move (stock WAP move relative to a synthetic index).
- **Metric:** **MAE.** *(VERIFIED.)*
- **Top score:** 1st-place MAE ≈ **5.3070** *(reported by secondary sources — NOT VERIFIED via official page)*; a verified strong NN solution (LSTM/ConvNet on raw + imbalance features) scored ~5.34–5.35 public *(VERIFIED via the GitHub repo above)*.
- **Models (top):** **GBDT (LightGBM) + NN (LSTM/CNN) ensembles**, online retraining. *(LSTM/ConvNet themes VERIFIED via repo; full ensemble composition NOT VERIFIED.)*
- **Directly comparable?** **LOW** — order-book/auction microstructure, not fundamentals. **Confidence: LOW.**

---

## 4. Competition benchmark matrix (Task 3)

| Competition | Similarity | Universe / data | Target / horizon | Metric | Top-solution methods | Top score (verified?) | Key winning techniques | Our current equivalent | Our missing equivalent | Compare now? | Plausibly beat? | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **JPX Tokyo Stock Exchange Prediction** | HIGH | ~2,000 JP stocks, OHLC adjusted | next-day forward return, rank | Sharpe of daily top200−bottom200 spread return | GBDT/LightGBM (+ some LSTM) | 1st 0.381 … 10th 0.280 (VERIFIED, official repo) | rank-weighted long-short, target denoising, robust CV, ensembling | factor ranker + Rank IC/ICIR + quantile spread | **JPX Sharpe-of-spread evaluator, rank-weighted top-N portfolio, turnover** | **NO** (no real broad data) | **UNKNOWN** | HIGH (infra), NONE (results) |
| **Numerai** | HIGH (concept) | obfuscated global equities | 20d / 60d stock-specific return, rank | Numerai Corr (rank, Gaussianized^1.5) + MMC | GBDT+NN ensembles, neutralization | ongoing (no fixed target) | feature/market neutralization, era-wise CV, meta-model contribution | sector-relative z-score, sector-neutral Rank IC | **full feature/market/size/value neutralization, MMC-style metric** | **NO** | **UNKNOWN** | HIGH (concept), LOW (scores) |
| **Jane Street RTMDF (2024)** | MEDIUM | anonymized real market data | responder_6, fixed horizon | sample-weighted zero-mean R² | NN/GBDT ensembles, online update | NOT VERIFIED | weighted-R² optimization, online learning, careful CV | regression labels (forward returns) | **weighted-R² objective, online/temporal CV, sample weighting** | **NO** | **UNKNOWN** | MEDIUM |
| **Optiver Trading at the Close (2023)** | LOW | NASDAQ closing-auction order book | WAP move vs synthetic index, ~auction horizon | MAE | GBDT (LightGBM)+NN (LSTM/CNN) ensembles | 1st MAE ≈5.3070 (NOT VERIFIED); NN ~5.34 (VERIFIED) | imbalance features, online retraining, GBDT+NN blend | optional LightGBM/CatBoost adapters | **microstructure features (N/A to us), ensemble/blend, online retrain** | **NO** | **UNKNOWN** | LOW–MEDIUM |

"Compare now? = NO" for all rows because we have **no real strict-validated
results**; the comparison is **infrastructure-only**.

---

## 5. Current model vs competition-grade standards (Task 4)

Legend: **DONE / PARTIAL / BLOCKED / NOT STARTED / UNKNOWN**. "Blocks comp-grade
validation?" flags whether the gap prevents a credible real comparison.

### 5.1 Data quality / point-in-time correctness
| item | status | evidence | gap vs top | priority | blocks? |
| --- | --- | --- | --- | --- | --- |
| disclosure-axis no-look-ahead | DONE | `dataset.py` excludes `disclosure_date > decision_date`; `check-forward-readiness` | none | — | no |
| price-axis no-look-ahead | DONE | `forward_return` uses only rows strictly after decision date | none | — | no |
| adjusted close | PARTIAL | dataset prefers adjusted close; only 3 real tickers exist | breadth | P0 | **yes** |
| accounting basis | DONE | export + dataset filtering (consolidated/non_consolidated/unknown/mixed) | ahead of most public baselines | — | no |
| survivorship bias | UNKNOWN | universe = whatever is exported; delisted handling undocumented | top solutions use full point-in-time universe | P0 | **yes** |
| corporate actions | PARTIAL | adjusted close handles splits/dividends; no explicit CA ledger | — | P1 | partial |
| missing data | DONE | None-propagation + missing indicators, never fabricated | none | — | no |
| universe definition | PARTIAL | TOPIX1000-oriented, not a documented point-in-time membership | top use dated index membership | P0 | **yes** |

### 5.2 Feature engineering
| item | status | evidence |
| --- | --- | --- |
| value / quality / growth / momentum / volatility-risk | DONE | `factors.py` |
| sector-relative factors | DONE | `sector_zscore`, sector-neutral IC |
| market/sector neutralization (residualized) | PARTIAL | only z-score demeaning; no regression-residual neutralization |
| narrative / NLP features | PARTIAL | presence/keyword placeholders only; **no extractor** |
| interaction features | NOT STARTED | — |
| time-series lags | NOT STARTED | factors are point-in-time snapshots |
| cross-sectional ranks | DONE | ranks used in baseline + metrics |

### 5.3 Modeling
| item | status |
| --- | --- |
| baseline factor ranker | DONE |
| linear / ridge / elastic-net | NOT STARTED |
| LightGBM/CatBoost ranker+regressor | PARTIAL (adapters exist; untrained on real data; optional deps absent) |
| ensemble / stacking / blending | NOT STARTED |
| neural nets / transformers | NOT STARTED (correctly deferred until real time-series exists) |
| model diversity / seed stability | NOT STARTED |
| rank optimization / target neutralization | NOT STARTED |

### 5.4 Validation
| item | status |
| --- | --- |
| strict no-look-ahead | DONE (readiness check) |
| walk-forward | DONE |
| purged / embargo | DONE |
| multi-decision-date | DONE |
| Rank IC / ICIR / quantile spread / hit rate | DONE |
| Sharpe-like **long-short spread** (mean/std of strategy return) | **NOT STARTED** (only ICIR + mean spread) |
| decile/quintile spread | DONE |
| drawdown / transaction cost / liquidity / turnover | NOT STARTED |
| sector-neutral evaluation | DONE (sector-neutral IC) |
| feature/model stability | NOT STARTED |
| out-of-time holdout | PARTIAL (walk-forward folds; no dedicated final holdout) |
| **strict real validated results** | **BLOCKED** (data coverage) |

### 5.5 Reporting / commercial readiness
| item | status |
| --- | --- |
| disclaimers / research-only | DONE (every report) |
| reproducibility / determinism | DONE |
| synthetic-vs-real labeling | DONE |
| artifact tracking (JSON/CSV/MD) | DONE |
| config management | PARTIAL (CLI args; no config file/versioned run manifest) |
| model versioning | PARTIAL (`model_version` string only) |
| feature importance / explainability | NOT STARTED |
| risk warnings / limitations | DONE |
| audit logs | NOT STARTED |
| API / UI readiness | NOT STARTED (CLI only; out of scope per CLAUDE.md) |

---

## 6. Can we beat comparable winners? (Task 5)

**Direct answer: UNKNOWN / NOT EVIDENCED.** No real strict no-look-ahead
validated result exists, so no "beat the winners" claim is possible. Synthetic
fixture metrics are not evidence.

- **Already comparable to top solutions:** *(none on results.)* On
  *infrastructure*, our no-look-ahead rigor (disclosure + price axis, purged
  /embargo, readiness gate) and accounting-basis discipline are **at or above the
  hygiene of typical public baselines**.
- **Behind top solutions:** model sophistication (no ensembles/neutralization/
  trained GBDT-on-real-data), and the **JPX-style Sharpe-of-spread evaluator**
  the JPX metric is actually scored on.
- **Potentially stronger than basic public baselines:** point-in-time discipline
  and basis separation — but this is an *infrastructure* statement, not a score.
- **Not comparable yet:** Jane Street (anonymized market data) and Optiver
  (auction microstructure) are different problem classes.
- **Required evidence before any claim:** (1) real point-in-time fundamentals +
  adjusted-close prices extending ≥ `max_horizon+1` rows past each decision date;
  (2) `check-forward-readiness` = ELIGIBLE on a meaningful universe; (3) a
  JPX-style Sharpe-of-spread number computed on that real data across multiple
  decision dates; (4) walk-forward stability of that number.

---

## 7. Recommended improvement roadmap (Task 6)

### P0 — Strict real no-look-ahead data + validation *(BLOCKER; wait-for-data)*
- **Repo:** both. **Files:** topix exports; engine `validation/`, `modeling/dataset.py`.
- **Outline:** obtain point-in-time fundamentals + adjusted-close history past the
  disclosure date; run `check-forward-readiness`; only then run the pipeline.
- **No "beat winners" claim is possible without P0.** Implement nothing that
  depends on real labels until coverage exists.

### P1 — JPX-style ranking / long-short evaluator *(implement now, synthetic-tested)*
- **Repo:** engine. **Files:** new `modeling/portfolio_metrics.py`,
  `report.py`, `cli.py`, tests, `docs/`.
- **Outline:** per decision date, rank predictions; build **top-N long / bottom-N
  short** with rank-weighting; compute the **daily/period spread return series**,
  then **Sharpe = mean/std** of that series (the JPX metric type), plus
  **turnover** and a long-short **equity curve / max drawdown**. Adjusted close
  only. **No trading automation, no position sizing.**
- **Test plan:** deterministic synthetic — monotone signal ⇒ positive spread;
  reversed ⇒ negative; constant ⇒ Sharpe `None`; turnover bounds.
- **Artifact:** `portfolio_metrics.json/csv/md` + report section.
- **Risk:** must not be read as a profitability claim → keep research-only labels.

### P2 — Numerai-style rank + neutralization metrics *(implement now, synthetic-tested)*
- **Repo:** engine. **Files:** `modeling/neutralization.py`, `ranking_metrics.py`.
- **Outline:** regression-residual **feature/market/size/value neutralization**
  (residualize predictions on chosen exposures); **neutralized Rank IC**; a
  **meta-model-contribution-style** delta when ≥2 models exist. Numerai-Corr-style
  rank correlation as an additional metric.
- **Test plan:** neutralizing against an exposure removes that exposure’s IC;
  determinism.
- **Artifact:** neutralized metrics in the ranking report.

### P3 — Model improvements *(partially now; GBDT-on-real waits for P0)*
- **Repo:** engine. **Files:** `modeling/ml_models.py`, new linear adapter,
  `modeling/ensemble.py`.
- **Outline:** add **ridge/elastic-net** baseline; LightGBM/CatBoost **group-by
  decision_date** ranking (adapters already structured for it); **blend/ensemble**
  + **seed-stability** harness; **feature importance** export. Train on real data
  only after P0.
- **Test plan:** optional-dep skip preserved; ensemble of identical inputs is
  deterministic; importance sums sanely.

### P4 — Commercial validation *(after P0/P1)*
- **Repo:** engine. **Files:** `modeling/portfolio_metrics.py` (extend),
  `report.py`.
- **Outline:** **transaction-cost + turnover** drag, **liquidity/cap constraints**,
  **benchmark-relative** returns, **drift/stability monitoring**, audit manifest.
- **Test plan:** cost monotonically reduces net spread; constraints reduce
  universe deterministically.

---

## 8. External competition evidence — verification notes

Internet was available; verification used machine-readable sources (official
GitHub, docs sites, search). **Kaggle competition/leaderboard/writeup pages are
JavaScript-rendered and were not readable by the fetch tool**, so the following
remain **NOT VERIFIED** here and should be confirmed manually if a stronger claim
is ever needed:

1. JPX exact metric weighting (linear top/bottom-200 weights) and exact target
   day-offset — confirm on the official metric notebook
   (`kaggle.com/code/smeitoma/jpx-competition-metric-definition`).
2. Numerai MMC exact formula and current neutralization defaults — confirm on
   `docs.numer.ai`.
3. Jane Street RTMDF top-solution methods/scores.
4. Optiver 1st-place MAE (≈5.3070) and full ensemble composition — official
   writeup `kaggle.com/competitions/optiver-trading-at-the-close/writeups`.

**VERIFIED in this analysis:** JPX private-LB scores 0.381→0.280 (official
J-Quants repo); JPX/Numerai/Jane Street/Optiver metric *types*; Numerai targets
(20d/60d) and rank-only CORR; Optiver NN solution themes (LSTM/ConvNet) via a
public solution repo.

---

## 9. Bottom line

The system is **infrastructure-competitive on hygiene** (no-look-ahead,
purged/embargo, accounting-basis separation, reproducible research-only reports)
and **behind on the scoring surface that competitions actually grade** (a
Sharpe-of-spread long-short evaluator and neutralization) and on **trained-model
sophistication**. **No performance comparison or "beat the winners" claim is
possible until P0 real data and strict validation exist.** The highest-value,
safe next step is to build the **JPX-style ranking evaluator and Numerai-style
neutralized metrics as offline, synthetic-tested infrastructure** — closing the
metric-surface gap without making any real-data claim.

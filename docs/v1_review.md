# Japanese Stock Analysis Engine — v1 Review

**Date:** 2026-06-11
**Scope:** Strict review of the existing v1 implementation. No new features added.
**Verdict:** v1 is correct, reproducible, and safe across all three modes. One
schema-stability gap and a few minor design notes are documented below.

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

---

## 1. Current Architecture Summary

```
local CSV/TXT (providers/local_csv, local_json)
  → fundamentals  (analysis/fundamentals.py)
  → valuation     (analysis/valuation.py)
  → momentum      (analysis/momentum.py)
  → disclosure    (analysis/disclosure_nlp.py — deterministic JP rule-based)
  → risk          (analysis/risk.py)
  → scoring       (analysis/scoring.py — weighted, risk-adjusted, explainable)
  → screening     (analysis/screening.py — labels in screening/trade_signal modes)
  → signal_engine (analysis/signal_engine.py — opt-in trade_signal mode only)
  → reports       (reports/{json,csv,markdown}_report.py)
  → CLI           (cli.py — argparse `analyze` subcommand)
```

- **Typed models** (`schemas.py`, Pydantic v2): raw inputs (`PriceBar`,
  `FinancialStatement`, `CompanyMetadata`, `DisclosureDocument`), derived
  metrics, `ScoreBreakdown`, `ScreeningResult`, `SignalResult`, and the
  `StockAnalysisResult` bundle. Modes/labels are `Literal` types.
- **Config** (`config.py`): `DEFAULT_DISCLAIMER`, `ScoreWeights`,
  `SignalThresholds`, `AnalysisConfig` with validators (weights non-negative,
  thresholds 0–100). Default `signal_mode = analysis_only`.
- **Providers**: only local CSV/JSON are functional. `jquants_stub`,
  `edinet_stub`, `tdnet_stub`, `news_stub` are import-safe and raise
  `ProviderError` if invoked. Protocols in `base.py` define the future seam.
- **Errors** (`errors.py`): `JPStockAnalysisError` → `DataValidationError`,
  `ProviderError`, `InsufficientDataError`. Routine missing data uses warnings,
  not exceptions.

---

## 2. Confirmed Behavior

| Area | Result |
|---|---|
| `python -m pytest` | **83 passed** in ~0.18s |
| `ruff check .` | **All checks passed** |
| `analysis_only` (default) | Metrics/scores/risks only. No `screening_label`, no `signal` in JSON; no `## Screening` / `## Research Signal` markdown sections. Verified. |
| `screening` | Adds `screening_label` (CSV column + JSON field + `## Screening` section). No trade signals. Verified. |
| `trade_signal` (opt-in) | Adds `signal` with label, confidence, rationale, evidence, blocking_risks, supporting_factors, thresholds_used, disclaimer. Verified. |
| Reproducibility | Re-running the CLI produces byte-identical JSON to the prior `artifacts/v1_smoke` run. Verified by diff. |
| Disclaimer | Present in every JSON (top-level) and every Markdown report; trade_signal reports carry it twice (report footer + signal block). Verified. |
| No network in tests | No `requests`/`httpx`/`urllib`/`socket`/`aiohttp` imports or calls anywhere under `tests/`. Verified by grep. |
| Smoke fixtures | 7203 → 80.3 / strong_candidate / buy_signal; 6758 → 53.7 / watchlist / hold_signal; 9984 → 3.1 / avoid_candidate / sell_signal. Consistent across all modes. |

---

## 3. Risks or Bugs Found

No correctness bugs were found. Findings are ranked by severity.

**[MEDIUM — schema stability for future RAG] `document_type` is dropped; `fiscal_year` is only nested.**
`docs/future_rag_integration_separate_project.md` lists `document_type` and
`fiscal_year` among required export fields. In the current JSON:
- `document_type` exists on the input `DisclosureDocument` but is **not**
  propagated to `DisclosureAnalysisResult`, so it is absent from report output.
- `fiscal_year` is not at the result top level; it is reachable only under
  `results[].fundamentals.fiscal_year`.
This is not a v1 bug (RAG is explicitly out of scope), but it is the main thing
to close before any ingestion project depends on the schema. See §5 and §8.

**[LOW — scoring design] `final_score` can be driven by a single dimension.**
`score_stock` re-normalizes the weighted average over only the *available*
sub-scores. A prices-only ticker yields `final_score=66.0` from momentum alone.
This is currently safe because the confidence gate (`confidence = Σ component
confidences / 5`) caps such a result at ~18–20, well below
`min_confidence_for_signal=55`, so it cannot become a buy/sell signal or a
non-`insufficient_data` screening label. The risk is latent: if confidence
thresholds were ever loosened, single-dimension `final_score`s could leak into
labels/signals. Recommend keeping the confidence gate as a hard precondition.

**[LOW — output schema] CSV columns are dynamic across modes.**
`screening.csv` adds `screening_label` (screening/trade_signal) and
`trade_signal` (trade_signal only) columns conditionally. This matches the spec
("…if available") and is fine for humans, but a tabular ingester must tolerate a
variable header. Documented behavior, not a defect.

**[LOW — tooling] No coverage measurement available.**
The venv has `pytest` and `ruff` but not `pytest-cov`/`coverage`, and neither is
in `pyproject.toml` dev extras. Line coverage cannot be measured without
installing one. Tests are comprehensive by module (see §6), but the number is
unmeasured.

**[INFO] `artifacts/v1_smoke/README.md` paraphrases the disclaimer.**
That README says "These outputs are for analytical… They are not personalized
financial advice." The *reports themselves* use the exact required string; only
the human-written smoke README paraphrases. No action needed.

---

## 4. Signal-Safety Review

**Valuation alone cannot create a `buy_signal` — protected by two independent layers:**

1. **Construction:** `_supporting_factors()` draws only from quality, growth,
   momentum, and disclosure (plus disclosure positive-finding count). Valuation
   is deliberately excluded. `buy_signal` requires `len(supporting) >= 2`, so a
   valuation-only profile can never reach the buy branch — proven in isolation
   by `test_valuation_alone_never_creates_buy_signal` (valuation_score=95, all
   other sub-scores `None`, artificially high confidence → not buy).
2. **Confidence gate:** a valuation-only ticker has confidence ≈20 < 55 →
   `insufficient_data` before any directional branch is reached.

**Other signal-safety properties confirmed:**
- `analysis_only` and `screening` return `None` from `generate_signal` (mode
  guard at the top). No buy/sell/hold/watch/avoid labels can be emitted outside
  `trade_signal`.
- Branch ordering is safe: `insufficient_data` (low confidence / no final) →
  `sell_signal` (final ≤ 35, or critical flag + sub-candidate score) →
  `avoid_signal` (critical flag or risk ≥ 70) is evaluated **before** the buy
  branch, so a high-scoring but high-risk or critical-flag stock can never buy.
  Confirmed by `test_high_risk_score_blocks_buy`,
  `test_critical_risk_with_decent_score_yields_avoid_signal`.
- `buy_signal` requires all of: final ≥ 78, confidence ≥ 55, risk ≤ 45, no
  critical flag, ≥ 2 non-valuation supporting factors — multi-dimensional by
  construction.
- Missing risk assessment (`risks is None`) blocks buy and records
  "risk assessment unavailable" as a blocking risk
  (`test_missing_risk_assessment_blocks_buy`).
- Every `SignalResult` carries `thresholds_used`, `evidence`, `confidence`,
  `rationale`, `blocking_risks`, `supporting_factors`, and `disclaimer`.

**No position sizing, portfolio optimization, leverage, broker execution, or
auto-trading logic exists anywhere in the codebase.** Confirmed.

---

## 5. Output Schema Stability Review

**JSON** (most stable, recommended ingestion surface):
- Top level: `disclaimer`, `signal_mode`, `result_count`, `screening`,
  `results`. Stable across modes.
- `results[]`: `ticker`, `company_name`, `analysis_date`, `signal_mode`,
  `fundamentals`, `valuation`, `momentum`, `disclosure`, `risks`, `score`, and
  conditionally `screening_label` / `signal`. Missing analyses serialize as
  explicit `null`, never as fabricated zeros.
- `disclosure.findings[]`: `category`, `summary`, `evidence_text`, `severity`,
  `confidence`, `rule_id` — evidence-first and stable; good for RAG.
- `score`: all six sub-scores + `final_score` + `reasons` map + `confidence_score`.

**Stable enough for future J-Quants ingestion:** Yes. J-Quants feeds the *input*
side (prices/statements) behind existing provider protocols; it does not change
the result schema.

**Stable enough for future RAG ingestion:** Mostly, with one gap — `document_type`
is not propagated into `DisclosureAnalysisResult` and `fiscal_year` is only
available nested under `fundamentals`. Both are small additive fixes (see §8)
and should be made *before* a RAG project pins to the schema, not now.

**CSV:** intentionally dynamic columns (see §3). **Markdown:** human-facing;
stable section set, with Screening/Research-Signal sections gated by mode.

---

## 6. Test Quality Review

- **83 tests** across 11 modules (`fundamentals`, `valuation`, `momentum`,
  `disclosure_nlp`, `risk`, `scoring`, `screening`, `signal_engine`, `reports`,
  `cli`, `providers`). All deterministic, all offline.
- Long-history momentum uses closed-form generated series in `conftest.py`
  (`make_price_bars`), not large static fixtures — matches the spec.
- Edge cases covered: missing previous-year data, negative EPS, zero revenue,
  missing market price, PEG with non-positive growth, insufficient price
  history, empty/unmatched disclosure text, going-concern escalation, missing
  risk assessment, fundamentals-only ticker date determinism, provider
  `DataValidationError`/`ProviderError` paths, JSON round-trip.
- Mode-specific report assertions exist for all three modes in both unit
  (`test_reports.py`) and integration (`test_cli.py`) form.
- **Gaps:** (a) no line-coverage tooling (see §3); (b) the rule-based analyzer
  is tested at the category level but not exhaustively per individual rule_id —
  acceptable for v1.

Overall test quality is high and matches the project's testing rules.

---

## 7. Readiness Score

**88 / 100**

| Dimension | Score | Notes |
|---|---|---|
| Correctness (tests + ruff) | 19/20 | 83 green, lint clean |
| Mode behavior & signal safety | 25/25 | All modes + valuation-alone protection verified |
| Missing-data discipline | 14/15 | None + warnings + lowered confidence, no fabrication |
| Reproducibility | 10/10 | Byte-identical reruns |
| Output schema stability | 8/12 | RAG export gaps (`document_type`, nested `fiscal_year`) |
| Test quality / tooling | 7/10 | Strong tests; no coverage tooling |
| Docs | 5/8 | Good; RAG doc slightly overstates available fields |

Deductions are concentrated in forward-looking schema stability, not in v1
correctness or safety.

---

## 8. Exact Next Recommended Implementation Step

**Propagate disclosure provenance into the analysis result to lock the
RAG-export contract — a small, additive, non-breaking change.**

1. In `schemas.py`, add two optional fields to `DisclosureAnalysisResult`:
   `document_type: str | None = None` and `fiscal_year: int | None = None`.
2. In `disclosure_nlp.py` (`RuleBasedDisclosureAnalyzer.analyze` and
   `NoOpLLMDisclosureAnalyzer.analyze`), copy `document.document_type` and
   `document.fiscal_year` through to the result.
3. Add a regression test asserting `document_type` and `fiscal_year` survive
   into `screening.json`, and update
   `docs/future_rag_integration_separate_project.md` to state that all listed
   export fields are now present at a stable path.

This closes the only MEDIUM finding and makes the JSON schema safe for a future
`jp_stock_rag_service` to pin against. It adds no new feature, no dependency,
and no network. Do **not** start J-Quants/EDINET/TDnet/RAG until this is done.

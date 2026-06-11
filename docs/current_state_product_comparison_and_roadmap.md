# Current State, Product Comparison, and Roadmap

**Date:** 2026-06-12
**Scope:** Strict evidence-based analysis of the `jp-stock-analysis-engine` repository:
implementation status, comparison with similar tool categories, product/commercial
readiness, architecture evaluation, and a prioritized roadmap. No product features were
implemented in this step.

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

**Evidence convention used throughout:**
- **[VERIFIED]** — confirmed by reading code/tests/docs in this repo or by running a command in this session.
- **[REPORTED]** — stated in repo docs or by the owner (e.g. topix1000 platform state); not re-verified in this session.
- **[ASSUMPTION]** — explicitly marked inference or general category knowledge.

---

## 1. Executive Summary

`jp-stock-analysis-engine` is a small, well-tested, deterministic Japanese stock
analysis engine (~3,800 lines of `src/` Python, 131 passing tests, ruff-clean). The
full local pipeline — CSV/TXT loading → fundamentals / valuation / momentum /
rule-based Japanese disclosure NLP / risk → integrated scoring → sector-relative
percentiles → screening → opt-in trade signals → JSON/CSV/Markdown reports — is
**implemented and verified offline**. A cache-first J-Quants provider exists and is
validated in cache mode; **live J-Quants fetch is implemented but unverified**
(a real probe returned HTTP 403 "endpoint does not exist", documented in
`docs/v1_jquants_endpoint_diagnosis.md`).

The two biggest gaps are: (1) **no real data path** — every validated run uses
synthetic fixtures; and (2) **no validation of signal quality** — there is no backtest
or forward-return measurement anywhere in `src/` or `tests/` [VERIFIED by grep].

The verified-working `topix1000_disclosure_platform` (EDINET ingestion, PostgreSQL,
XBRL facts) is the natural real-data source. The recommended integration is
**file/export-based**: topix1000 exports deterministic JSON, this engine reads it as an
optional provider — exactly the pattern already proven by the `jquants-cache` provider.
PostgreSQL should **not** become a dependency of this engine.

**Recommended next step:** implement a `topix1000-export` reader provider in this repo
(fixture-driven, offline-tested), including a written export contract that the
topix1000 platform can then implement.

---

## 2. Repository Purpose

Per `CLAUDE.md` [VERIFIED]: a production-quality, modular Japanese stock analysis
engine for public-release-quality code and self-use trading research. Explicitly NOT a
SaaS, advisory service, RAG service, EDINET platform clone, broker integration, or
auto-trading bot. Three required modes are preserved: `analysis_only` (default),
`screening`, and `trade_signal` (explicit opt-in) — confirmed in
`src/jp_stock_analysis/cli.py` (default `--signal-mode analysis_only`),
`src/jp_stock_analysis/config.py` (`AnalysisConfig.signal_mode = "analysis_only"`),
and `src/jp_stock_analysis/analysis/signal_engine.py` (returns `None` unless mode is
`trade_signal`).

---

## 3. Current Verified Implementation Status

Validation run in this session (2026-06-12):

| Check | Result |
|---|---|
| `python -m pytest` | **131 passed** in 0.39s [VERIFIED] |
| `ruff check .` | **All checks passed** [VERIFIED] |
| `python -m jp_stock_analysis.cli --help` | Works; single `analyze` subcommand [VERIFIED] |
| Git state | Branch `main` at `8a8ed70`; uncommitted: RAG-export schema fix (disclosure metadata propagation), deletion of obsolete `providers/jquants_stub.py`, new `tests/fixtures/jquants_universe/`, `artifacts/`, three docs [VERIFIED] |

Feature-by-feature status:

| Capability | Status | Evidence |
|---|---|---|
| Local CSV/TXT loading | **Implemented** | `providers/local_csv.py` (prices, fundamentals, metadata CSVs; `<ticker>.txt` disclosures); `tests/test_providers.py` |
| Fundamentals analysis | **Implemented** | `analysis/fundamentals.py` (growth YoY, margins, ROE/ROA, equity ratio, FCF margin, payout); `tests/test_fundamentals.py` |
| Valuation analysis | **Implemented** | `analysis/valuation.py` (PER, PBR, PSR, dividend yield, PEG, market cap, classification); `tests/test_valuation.py` |
| Momentum analysis | **Implemented** | `analysis/momentum.py` (1/3/6/12m returns, 20/60/120/200d MAs, volatility, max drawdown, volume trend); `tests/test_momentum.py` |
| Disclosure NLP (Japanese) | **Implemented (rule-based)** | `analysis/disclosure_nlp.py` — deterministic keyword/pattern rules, evidence sentences, rule IDs, tone score; `NoOpLLMDisclosureAnalyzer` placeholder; `tests/test_disclosure_nlp.py` |
| Risk flags | **Implemented** | `analysis/risk.py` (severity-ranked flags with evidence and confidence); `tests/test_risk.py` |
| Integrated scoring | **Implemented** | `analysis/scoring.py` (weighted 0–100 sub-scores, risk adjustment, `reasons` dict, confidence); `tests/test_scoring.py` |
| Sector-relative scoring | **Implemented** | `analysis/sector_relative.py` (direction-aware percentiles vs same-sector peers; additive only — `final_score` unchanged, verified by `tests/test_sector_universe.py`) |
| Screening | **Implemented** | `analysis/screening.py` — the 5 required labels; `tests/test_screening.py` |
| trade_signal mode | **Implemented, opt-in** | `analysis/signal_engine.py` — 6 required labels; buy requires final score ≥ threshold AND confidence AND risk gate AND no critical flag AND ≥2 non-valuation core factors; valuation and sector-relative excluded from the buy gate by construction; `tests/test_signal_engine.py` (19 tests) |
| Configurable thresholds | **Implemented** | `config.py` — `ScoreWeights`, `SignalThresholds` (incl. `sector_support_*`), validated 0–100 |
| Reports JSON/CSV/Markdown | **Implemented** | `reports/{json,csv,markdown}_report.py`; disclaimer in every JSON and Markdown; mode-gated sections; `tests/test_reports.py`, `tests/test_cli.py` |
| J-Quants cache mode | **Implemented & validated offline** | `providers/jquants.py` cache-first; 12-code synthetic universe in `tests/fixtures/jquants_universe/`; `tests/test_jquants_provider.py` (18 tests, fake transport) |
| J-Quants live mode | **Implemented, UNVERIFIED live** | Real probe (2026-06-11, `tools/jquants_probe.py`) returned HTTP 403 "endpoint does not exist" for `/v2/prices/daily_quotes`; endpoints/auth made configurable via env vars; documented in `docs/v1_jquants_endpoint_diagnosis.md`, `docs/v1_jquants_live_pending.md` [VERIFIED docs; live status itself REPORTED, no live call made in this session] |
| EDINET / topix1000 integration | **Not implemented** | `providers/edinet_stub.py` raises `ProviderError` if used; no topix1000 reference anywhere in `src/` or `tests/` [VERIFIED by grep] |
| Backtest / forward returns | **Not implemented** | Zero occurrences of "backtest" in `src/` and `tests/`; only a mention in `docs/jp_stock_analysis_engine.md` [VERIFIED by grep] |
| README | **Empty (0 bytes)** | Real documentation lives in `docs/jp_stock_analysis_engine.md` (171 lines) [VERIFIED] |

---

## 4. What Is Complete

- **The entire offline analysis pipeline**, end to end, in all three modes, with
  byte-identical reproducibility confirmed twice (`docs/v1_review.md`,
  `docs/v1_sector_relative_universe_validation.md`) [VERIFIED docs + tests pass today].
- **Typed schemas** (`schemas.py`, Pydantic v2): strict separation of raw inputs,
  derived metrics, scores, screening labels, and trade signals, exactly as
  `CLAUDE.md` requires. `SchemaBase` tolerates partial data; missing values are
  `None` + warning, never fabricated.
- **Stable RAG-ready export paths**: `screening.json` field paths documented in
  `docs/future_rag_integration_separate_project.md`; the uncommitted working-tree
  change propagates `document_type`/`fiscal_year` through both disclosure analyzers
  with regression tests (`docs/v1_rag_export_schema_fix.md`) — all 131 tests pass with
  this change applied [VERIFIED].
- **CLI** with three providers (`local`, `jquants-cache`, `jquants-live`) and graceful
  degradation (missing statements/metadata → stderr warning, not failure).
- **Safety properties**: disclaimer everywhere, `analysis_only` default,
  `trade_signal` opt-in, valuation-alone cannot produce a buy signal
  (enforced in `_supporting_factors`, tested), no secrets in error messages
  (asserted by tests), no network in tests.
- **Smoke artifacts**: `artifacts/v1_smoke/` and `artifacts/v1_review/` hold real CLI
  outputs for all modes [VERIFIED present].

## 5. What Is Partial

- **J-Quants live provider** — code complete, auth header validated by probe
  (`x-api-key`, not Bearer), but the v2 endpoint paths are wrong or the plan lacks
  access; configurable via `JQUANTS_API_BASE_URL` / `JQUANTS_API_VERSION` /
  per-dataset path env vars. Blocked on consulting the official spec, not on code.
- **Disclosure text pipeline** — the analyzer is solid but input is limited to
  hand-placed `<ticker>.txt` files; no automated extraction from real filings.
- **Company/ticker mapping** — `CompanyMetadata` exists and J-Quants `listed_info`
  maps into it, but there is no validated real-universe code↔name↔sector mapping;
  EDINET codes (`edinetCode`/`secCode`) are not mapped at all.
- **Sector peer universe** — sector-relative logic is validated only on a 12-code
  synthetic universe with designed gradients; never on real market data.
- **Documentation** — `docs/` is thorough for development history, but `README.md`
  is empty and there is no user-facing quickstart at the repo root.
- **NoOpLLMDisclosureAnalyzer** — a placeholder seam only; no LLM analyzer exists
  (correct per project rules, noted for completeness).

## 6. What Is Missing

- **topix1000 export reader provider** — no code reads topix1000 output in any form.
- **A defined export contract** — no schema/spec for what topix1000 should emit for
  this engine to consume.
- **Production data pipeline** — no reproducible "update data → rerun analysis"
  workflow against real data.
- **Real historical backtest / forward-return validation** — scores and signals have
  never been measured against subsequent returns; signal quality is unknown.
- **Screener quality validation on a real universe** (e.g. TOPIX 1000 names).
- **EDINET→ticker mapping** (edinetCode/secCode ↔ 4/5-digit codes ↔ sector).
- **Data licensing notes in outputs** — `jquants.py` docs say raw data must not be
  redistributed, but reports don't carry data-source/licensing warnings.
- **Monitoring/observability** — acceptable absence for a local research tool;
  required before any service deployment.
- **UI/API, deployment story, benchmark vs. other products** — absent; see roadmap
  for why most of these should stay absent for now.

---

## 7. topix1000_disclosure_platform Integration Assessment

Verified platform state [REPORTED by owner, not re-verified here]: PostgreSQL running;
valid `EDINET_API_KEY`; list API HTTP 200 for 2026-06-04; 11 annual securities reports
fetched 11/11 with raw files (`original.zip`, `document.pdf`, `csv.zip`,
`list_response.json`); `ingest_zip` succeeded for S100Y7YT producing 9 documents,
4 contexts, 3 context dimensions, 2 units, **339 facts**.

Assessment:

- The platform already solves the hardest part this engine lacks: **real, structured
  EDINET/XBRL fundamentals**. 339 facts from one annual report is far richer than the
  ~13 fields of `FinancialStatement` — an export only needs to project a small,
  stable subset.
- **No integration exists today** in either direction [VERIFIED for this repo].
- The right contract mirrors the proven `jquants-cache` pattern: topix1000 writes
  deterministic JSON files per company (statements + metadata + optional disclosure
  text sections); this engine reads them via a new optional provider implementing the
  existing `base.py` protocols (`FundamentalsProvider`, `MetadataProvider`,
  `DisclosureProvider`). Tests use committed synthetic export fixtures — no Postgres,
  no network, exactly like `tests/fixtures/jquants_cache/`.
- Open question to resolve in the next step [ASSUMPTION]: whether topix1000 already
  has an export command. The engine-side provider should be written against a
  documented contract first; the contract doubles as the requirement spec for a
  topix1000 export CLI if one is needed.
- Prices are out of topix1000's scope (it is a disclosure platform), so the engine
  will still need J-Quants (or local CSVs) for prices — providers compose per
  dataset, which the current `analyze_data()` entry point already supports.

---

## 8. Similar Service/Tool Comparison (High-Level, Category Knowledge Only)

**[ASSUMPTION — all of section 8.]** No web browsing was performed. These are
general category characterizations, not claims about specific products, features, or
prices.

| Axis | Retail stock screeners (broker/web) | Financial data platforms (institutional) | JP equity research tools (kabu-oriented retail) | EDINET/XBRL analysis tools (mostly OSS/academic) | Quant/backtesting frameworks (OSS) | AI-assisted equity research tools | **This engine** |
|---|---|---|---|---|---|---|---|
| Data coverage | Broad, vendor-supplied | Very broad, multi-asset | JP-focused, broad | Filings only | BYO data | Varies, often US-first | Narrow: what you load (fixtures today) |
| Freshness | Daily/intraday | Real-time | Daily | Filing cadence | BYO | Varies | Manual/cached; no auto-refresh |
| JP market support | Native (JP brokers) | Good but costly | Native | Native | Generic | Often weak | **Native by design** (JP disclosure NLP, J-Quants/EDINET orientation) |
| EDINET/XBRL support | None/hidden | Indirect | Limited | **Strong** | None | Rare | Not yet — adjacent via topix1000 |
| Factor/scoring explainability | Low (black-box ranks) | Medium | Low–medium | N/A | High (you write it) | Often low (LLM opacity) | **High**: every score has reasons, evidence, warnings, thresholds |
| Backtesting | Rare | Yes (premium) | Rare | No | **Core strength** | Rare/unreliable | **Missing** |
| Portfolio/risk functionality | Basic | Strong | Basic | No | Strong | Varies | Intentionally out of scope |
| UI/UX | Polished web | Terminal/web | Polished | CLI/notebooks | Code-first | Chat/web | CLI + Markdown/CSV/JSON |
| Automation | Vendor-run | Vendor-run | Vendor-run | Scripted | Scripted | Vendor-run | Scriptable, fully local |
| Export/report quality | Limited export | Good | Limited | Raw data | DIY | Prose, weakly structured | **Strong**: typed JSON, CSV, evidence-rich Markdown |
| Cost/maintainability | Free–cheap (vendor lock) | Very expensive | Cheap | Free, high effort | Free, high effort | Subscription | Free; small, typed, tested codebase |
| Commercial defensibility | Brand/distribution | Data moats | Distribution | None | Community | Model+UX | Low as-is (see §10) |

Honest positioning: this engine's distinctive cell combination is **JP-native +
explainable + reproducible + evidence-first + local/free**. No mainstream category
occupies that exact spot, but each neighboring category beats it on at least one
axis it currently lacks (real data, backtesting, or UI).

---

## 9. Product Readiness Evaluation

- **Sellable today? No.** It analyzes only data the user supplies; the only validated
  runs use synthetic fixtures. There is no real-data path, no signal-quality
  evidence, and no packaging/distribution story. Selling analysis output without
  validation would also be ethically and possibly legally fraught.
- **Useful for the owner's own research today? Almost.** The analysis machinery is
  ready; it becomes genuinely useful the moment real fundamentals (topix1000 export)
  and/or real prices (working J-Quants live or manually exported CSVs) flow in.
  With local CSVs the owner could use it today, at manual-data-entry cost.
- **Useful as a portfolio/GitHub project? Yes, already.** Clean typed architecture,
  131 offline tests, deterministic outputs, documented safety constraints, honest
  handling of an external API failure (`v1_jquants_endpoint_diagnosis.md`) — these
  demonstrate engineering judgment well. The empty `README.md` is the one blocker;
  it is the first thing any visitor sees.
- **First user segment that could benefit [ASSUMPTION]:** technically capable
  self-directed Japanese retail investors / quant hobbyists who want explainable,
  reproducible screening over JP equities and are comfortable with a CLI.
- **Strongest differentiation:** evidence-first explainability (every score, label,
  and signal traces to metrics, rule-matched sentences, thresholds, and warnings)
  combined with native Japanese disclosure analysis and strict no-fabrication rules.
- **Weakest area:** no real data and no proof the scores predict anything.

## 10. Commercial Readiness Evaluation

**Score-relevant facts:** no real-data pipeline, no validation of predictive quality,
no UI, no packaging (PyPI), no licensing review of data redistribution, no user docs
at the root, no monitoring. The codebase quality is commercial-grade; the *product*
is not.

Before "commercial-grade" could honestly be claimed, at minimum:
1. A reliable real-data pipeline (topix1000 export + verified price source).
2. Backtest/forward-return evidence that screening adds value (or honest reframing
   as a "research workbench", not a ranking product).
3. Data licensing compliance (J-Quants terms prohibit redistribution of raw data;
   EDINET data has its own terms) [REPORTED in `jquants.py` docstring / ASSUMPTION
   for EDINET].
4. Distribution (package, versioning, changelog, README, examples).
5. Legal review of investment-advice boundaries for the target jurisdiction
   [ASSUMPTION: Japan's FIEA distinguishes tools from advisory services; this needs
   professional review, not code].

None of these are recommended for the immediate roadmap except #1 (and #4's README
sliver). The repo's own `CLAUDE.md` correctly scopes this as a self-use research
tool; commercial ambitions should not distort the next steps.

## 11. Technical Architecture Evaluation

Strengths [VERIFIED]:
- Clean layering: providers → typed schemas → analysis modules → scoring →
  screening/signals → reports → CLI. `analyze_data()` accepts pre-loaded typed data,
  so new providers need zero pipeline changes.
- Provider seam already formalized (`providers/base.py` runtime-checkable Protocols).
- Cache-first precedent (`jquants.py`) proves the file-based provider pattern works:
  deterministic JSON on disk, live fetch as explicit opt-in, offline tests with
  synthetic fixtures.
- Config is explicit and validated; no hidden state; errors never leak secrets.

Weaknesses / debts:
- `cli.py:_load_jquants_inputs` hardcodes provider wiring; a third provider will want
  a small provider-selection refactor (acceptable now, watch for sprawl).
- `run_analysis` requires prices; fundamentals-only analysis works (the code path
  handles missing bars) but the local CLI demands `--prices`. A topix1000-only run
  (fundamentals + disclosures, no prices) is already handled by `analyze_data` with
  warnings — good.
- Empty `README.md`.

## 12. Data Pipeline Evaluation

Current truth: **there is no data pipeline** — only loaders. Data freshness is
whatever the user last placed in CSVs or the J-Quants cache. That is acceptable and
even desirable for reproducible research (runs are snapshots), but it means:
- No reproducible update workflow (e.g. "export from topix1000 for date D, analyze").
- No provenance stamping of *when* data was fetched (only `source_metadata.source`
  and `disclosed_date` on statements).
- The future export contract should carry `as_of`/export-date metadata so reports can
  state data coverage honestly (CLAUDE.md report rules already require coverage).

## 13. Risk and Compliance Boundaries

All required boundaries are in place and test-enforced [VERIFIED]:
- `analysis_only` default; `trade_signal` explicit opt-in (`config.py`, `cli.py`).
- Valuation alone can never produce a buy signal (`signal_engine._supporting_factors`
  excludes valuation; sector-relative is post-label evidence only).
- Disclaimer on every JSON and Markdown report (`DEFAULT_DISCLAIMER` in `config.py`).
- No position sizing, broker execution, leverage, derivatives, or portfolio
  allocation anywhere in `src/` [VERIFIED by inspection].
- RAG boundary respected: JSON/Markdown export + stable metadata paths only
  (`.claude/rules/no-rag.md`; `docs/future_rag_integration_separate_project.md`).
- No secrets committed; no `.cache/` data committed; tests fully offline.

Gap: reports should eventually carry a data-source licensing note when real J-Quants
or EDINET-derived data flows through (do not redistribute raw vendor rows).

## 14. Recommended Architecture Boundary

**Keep `jp-stock-analysis-engine` file/cache-first. Do not add PostgreSQL.**

- topix1000_disclosure_platform = ingestion/storage platform (EDINET API, raw files,
  XBRL parsing, Postgres). It owns data acquisition and persistence.
- jp-stock-analysis-engine = analysis engine. It owns metrics, scores, screening,
  signals, reports. It reads **files**.
- Integration contract: topix1000 exports deterministic JSON per company
  (statements projected to the engine's `FinancialStatement` fields + metadata +
  optional disclosure text sections), under a versioned directory layout, e.g.
  `exports/<as_of_date>/{statements,metadata,disclosures}/<code>.json`.
- The engine adds one optional provider (`providers/topix1000_export.py`)
  implementing the existing protocols. Core dependencies stay `pandas/numpy/pydantic`.
- Why not direct Postgres: it would couple the engine's tests and portability to a
  running database, break the offline/deterministic test rule in `CLAUDE.md`, and
  duplicate topix1000's schema knowledge in a second codebase. File exports keep
  both repos independently testable and the contract explicit.

---

## 15. Recommended Next 1 Prompt

**Implement the topix1000 export reader provider in jp-stock-analysis-engine.**

Contents: (a) a written export contract doc (`docs/topix1000_export_contract.md`)
defining the JSON layout, required/optional fields, `as_of` metadata, and versioning;
(b) `providers/topix1000_export.py` implementing `FundamentalsProvider`,
`MetadataProvider`, `DisclosureProvider` against that contract, cache/file-only, no
network, no DB; (c) synthetic export fixtures + offline tests; (d) CLI wiring
(`--provider topix1000-export --topix1000-export-dir ...`), composable with price
sources; (e) graceful warnings for missing files, consistent with existing providers.

## 16. Recommended Next 3 Prompts

1. The export reader provider (above).
2. **Implement the matching export CLI in topix1000_disclosure_platform** (separate
   repo, separate prompt): project ingested XBRL facts to the contract's JSON, for
   the 11 already-ingested documents first. Then run the engine end-to-end on real
   exported data and record a smoke artifact.
3. **Minimal forward-return validation harness** in this engine: given a
   point-in-time analysis output and a later price CSV, compute realized N-day/N-month
   forward returns per screening label / score decile. Offline, deterministic,
   fixture-tested. This is measurement, not a trading backtest — no positions, no
   portfolio logic.

## 17. Recommended Next 5 Prompts

4. **Real-universe sector validation + ticker mapping**: build the
   edinetCode/secCode ↔ stock code ↔ sector mapping from topix1000 exports (+ J-Quants
   `listed_info` when available), validate sector-relative scoring on dozens of real
   companies, and document observed peer-count coverage.
5. **README + packaging pass**: fill the empty `README.md` (quickstart, modes,
   safety boundaries, data licensing note), add report-level data-source/licensing
   warnings, and tag a clean v1 release. (Resolve J-Quants live endpoints
   opportunistically when spec access is sorted — it is config-only now, not code.)

## 18. Explicitly Postponed Items

- J-Quants live endpoint verification (blocked on official spec/plan access; the
  provider is configurable via env vars, so no code change is expected).
- LLM-based disclosure analysis (seam exists: `NoOpLLMDisclosureAnalyzer`).
- Historical multi-year backtesting beyond the simple forward-return harness.
- Monitoring/observability, scheduling/automation of data updates.
- PyPI publication.
- Report-level data licensing text (do with README pass, #5).

## 19. Do-Not-Build-Yet List

These would add complexity without evidence of need, or violate scope:

- **PostgreSQL access from this engine** — breaks offline tests and the boundary.
- **UI / web app / API server** — no validated user need; CLI + Markdown suffices.
- **RAG service, embeddings, vector DB, chatbot** — out of scope per
  `.claude/rules/no-rag.md`.
- **Broker integration, auto-trading, position sizing, portfolio allocation,
  leverage/derivatives** — out of scope per `CLAUDE.md`.
- **Heavy ML scoring models** — the explainability of rule-based scoring is the
  product's main differentiation; don't trade it away before validation exists.
- **An EDINET ingestion clone inside this repo** — topix1000 already does it.
- **Real-time/intraday data** — research cadence is daily at most.

## 20. Exact Verification Commands

Run from `PROJECT_ROOT`:

```bash
python -m pytest                       # expect: 131 passed
ruff check .                           # expect: All checks passed
python -m jp_stock_analysis.cli --help

# Offline end-to-end smoke (local fixtures, default analysis_only mode):
python -m jp_stock_analysis.cli analyze \
  --prices tests/fixtures/prices_sample.csv \
  --fundamentals tests/fixtures/fundamentals_sample.csv \
  --metadata tests/fixtures/company_metadata_sample.csv \
  --disclosures tests/fixtures/disclosures \
  --output-dir /tmp/jp_out_analysis

# Offline 12-code sector universe (no API key, no network):
python -m jp_stock_analysis.cli analyze \
  --provider jquants-cache \
  --jquants-cache-dir tests/fixtures/jquants_universe \
  $(for c in 7001 7002 7003 7004 7005 6501 6502 6503 6504 9001 9002 9101; \
    do echo --jquants-code $c; done) \
  --output-dir /tmp/jp_out_universe
```

(Session results 2026-06-12: pytest 131 passed; ruff clean; CLI help OK.)

## 21. Final Recommendation

The engine side is done well and validated as far as synthetic data can take it.
Every further hour spent on analysis features has near-zero marginal value until real
data flows through the pipeline. The verified topix1000 platform makes real
fundamentals one export contract away, and the engine already has the exact pattern
(cache-first JSON provider) to consume it. Therefore: **define the export contract
and build the topix1000 export reader provider next**, then make topix1000 emit it,
then measure whether the scores mean anything (forward returns). Keep the engine
file-first, Postgres-free, `analysis_only` by default, and honest about what is
unvalidated.

---

## Next Claude Code Prompt

**Title: Implement topix1000 export reader provider in jp-stock-analysis-engine**

```
Implement a file-based topix1000 export reader provider in jp-stock-analysis-engine.

Context:
- topix1000_disclosure_platform (separate repo, do not modify) ingests EDINET
  documents into PostgreSQL and will later gain an export CLI. This engine must
  read its exports as deterministic local JSON files only — no PostgreSQL, no
  network, no new core dependencies.
- Follow the existing cache-first provider pattern in
  src/jp_stock_analysis/providers/jquants.py and the protocols in
  src/jp_stock_analysis/providers/base.py.

Tasks:
1. Write docs/topix1000_export_contract.md defining a versioned export layout:
   exports/<as_of_date>/statements/<code>.json (list of objects projecting to
   FinancialStatement fields; unknown fields ignored; missing fields null),
   exports/<as_of_date>/metadata/<code>.json (CompanyMetadata fields incl.
   edinet_code/sec_code in source_metadata), and optional
   exports/<as_of_date>/disclosures/<code>.json (document_type, fiscal_year,
   text sections). Include an export-level manifest with as_of date and source
   notes. State explicitly that this contract is the requirement spec for the
   future topix1000 export CLI.
2. Implement src/jp_stock_analysis/providers/topix1000_export.py:
   Topix1000ExportProvider(export_dir) implementing FundamentalsProvider,
   MetadataProvider, DisclosureProvider. Strictly read-only, file-based,
   deterministic. Missing files raise ProviderError with the expected path;
   the CLI degrades to warnings exactly like the J-Quants provider does.
   Never fabricate values; propagate source_metadata (docID, as_of, source).
3. Wire the CLI: --provider topix1000-export, --topix1000-export-dir,
   --topix1000-code (repeatable). Prices remain optional and may come from
   local CSV via existing flags; fundamentals-only runs must degrade with the
   existing warnings, not crash. Default provider stays local; default mode
   stays analysis_only; trade_signal stays explicit opt-in.
4. Add synthetic export fixtures under tests/fixtures/topix1000_export/
   (2-3 codes, 2 fiscal years, one code with missing metadata, one with
   disclosure text) and offline tests: provider mapping, missing-file
   behavior, CLI end-to-end in all three modes, determinism (byte-identical
   JSON on rerun), and confirmation that final_score and signal behavior are
   unchanged for non-topix1000 runs.
5. Document usage in docs/ and add a short section to README.md.
Constraints: no live API calls, no PostgreSQL, no RAG, no broker/trading
logic, tests fully offline, run python -m pytest and ruff check . before
finishing and report exact results.
```

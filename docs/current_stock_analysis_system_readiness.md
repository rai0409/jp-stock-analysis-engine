# Current Stock-Analysis System — Deployment-Readiness Report

**Date:** 2026-06-12
**Repos:** `PROJECT_ROOT` (engine, at `f341228`,
tag `jp-stock-analysis-engine-v1-confidence-guard`) and
`EXTERNAL_DISCLOSURE_PLATFORM_PATH` (platform, read-only, at `40f75e8`,
tag `topix1000-edinet-export-v1.1`).

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

**Evidence convention:** **[VERIFIED-NOW]** = re-run/inspected in this session.
**[VERIFIED-PRIOR]** = verified in an earlier documented run (e.g. live EDINET
fetch), not re-runnable now (no network allowed). **[SYNTHETIC]** = works, but
only with clearly-labeled synthetic data. **[ASSUMPTION]** marked inline.

---

## 1. Executive Summary

The two-repo system is **architecturally complete and mechanically sound for
the path it was designed for, but it has never analyzed a real Japanese stock
end-to-end.** Everything between "EDINET document on disk" and "ranked,
reliability-graded analysis report" exists, is tested (engine: 160 passed;
platform: 73 passed, 1 skipped; both ruff-clean [VERIFIED-NOW]), and is wired
together through a deterministic file contract that was exercised in this
session. The confidence-aware screening guard works exactly as designed: the
synthetic 9991 case (final_score 98.5 from 2 price bars) reports
`screening_score=4.8`, `screening_eligible=false`, `reliability_grade=low`
[VERIFIED-NOW].

What blocks real use is data, not code, and the blockers are narrow:

1. **The EDINET→listing-code mapping is synthetic.** The platform's import
   CLI works and is idempotent, but no real mapping CSV has been imported
   (`docs/export_linkage_smoke_result.md` explicitly records this).
2. **No real prices or fundamentals reach the engine.** J-Quants live is
   blocked on an endpoint mismatch (HTTP 403, documented); no real price CSV
   has been used; XBRL facts (339 for S100Y7YT) are ingested in the platform
   but not yet projected into the engine's `FinancialStatement` fields.
3. **Disclosure `text` is always `null`** — text extraction from the raw
   archives is not implemented, so disclosure NLP has never run on a real
   filing.
4. **No forward-return or backtest validation exists** in either repo
   (verified by grep), so even with real data, ranking quality is unproven.

**Verdict:** not deployable as a real stock-analysis workflow today; one
focused step (real mapping import → re-export → real-price smoke) away from
its first genuine end-to-end run.

## 2. System Purpose

A self-use Japanese stock analysis system: the platform ingests EDINET
disclosures into PostgreSQL and exports deterministic JSON; the engine reads
local files/exports and produces explainable metrics, scores,
reliability-graded screening, and opt-in research trade signals, as
JSON/CSV/Markdown. Explicitly not advisory, not a SaaS, no broker/trading
automation (`CLAUDE.md`, enforced in code and tests).

## 3. Repository Roles

| | topix1000_disclosure_platform | jp-stock-analysis-engine |
|---|---|---|
| Role | Ingestion/data platform | Analysis engine |
| Storage | PostgreSQL + raw archives | Files only (no DB dependency) [VERIFIED-NOW: `pyproject.toml` deps are pandas/numpy/pydantic] |
| Network | EDINET API (own key) | None in tests; J-Quants live is opt-in and currently non-functional |
| Coupling | None in either direction; only the `topix1000-export-v1` JSON file contract | reads exports via `providers/topix1000_export.py` |

This boundary is correct and should be preserved (§15).

## 4. Verified Completed Work

Status of every required item:

| Item | Status | Evidence |
|---|---|---|
| EDINET list fetch | **verified complete** [VERIFIED-PRIOR] | HTTP 200 for 2026-06-04, 11 annual reports; raw `list_response.json` files exist under `LOCAL_DISCLOSURE_RAW_DATA_DIR` [VERIFIED-NOW: directory present] |
| EDINET document fetch | **verified complete** [VERIFIED-PRIOR] | 11/11 documents fetched; raw files per docID |
| Raw storage | **verified complete** | `raw/`, `raw_backup_before_refetch/`, `derived/` present [VERIFIED-NOW] |
| XBRL/CSV ingest | **verified complete (≥1 document)** | S100Y7YT: 9 documents, 4 contexts, 2 units, 339 facts [VERIFIED-PRIOR]; ingest/parser tests pass [VERIFIED-NOW] |
| Disclosure export | **verified complete** | `export_disclosures --help` runs [VERIFIED-NOW]; deterministic 11-doc export at `/tmp/topix1000_disclosure_export_linked` (index.json + per-doc JSON) [VERIFIED-NOW] |
| Company linkage import | **verified complete (mechanism)** | `import_company_links --help` runs [VERIFIED-NOW]; idempotency documented in `docs/export_linkage_smoke_result.md` |
| Real company linkage availability | **missing** | Smoke doc states: no real mapping found; synthetic CSV used (3 of 5 EDINET codes, tickers 9991–9993) |
| Engine topix1000 provider | **verified complete** | `providers/topix1000_export.py`, commit `c9ae813`, 19 offline tests [VERIFIED-NOW] |
| Metadata-only disclosure handling | **verified complete** | `text: null` → empty-text document, disclosure confidence 0.0, warning; no fabrication [VERIFIED-NOW in smoke] |
| Provenance propagation | **verified complete** | `doc_id=S100Y7YT`, `edinet_code=E03627` in this session's smoke output JSON [VERIFIED-NOW] |
| Fundamentals analysis | **implemented; unverified with real data** | `analysis/fundamentals.py` + tests; only fixture statements so far |
| Valuation analysis | **implemented; unverified with real data** | `analysis/valuation.py` + tests |
| Momentum analysis | **implemented; unverified with real data** | `analysis/momentum.py` + tests |
| Disclosure NLP | **implemented; never run on a real filing** | rule-based, deterministic; real `text` is always null so far |
| Risk flags | **implemented; unverified with real data** | `analysis/risk.py` + tests |
| Integrated score | **implemented; unverified with real data** | `analysis/scoring.py` + tests |
| Confidence-aware screening | **verified complete** | `analysis/reliability.py`, commit `f341228`, 10 dedicated tests, smoke re-verified this session [VERIFIED-NOW] |
| Report outputs (JSON/CSV/Markdown) | **verified complete** | reports written and inspected this session; disclaimer present |
| Real data join (export ↔ prices by ticker) | **implemented but synthetic-only** | join works, but ticker 9991 and its prices are synthetic |
| Real price/fundamental ingestion | **missing (engine); partial tooling (platform)** | J-Quants live blocked (HTTP 403, `docs/v1_jquants_endpoint_diagnosis.md`); platform has deterministic `build_code_map` / `build_prices_dataset` CLIs [VERIFIED-NOW: both are CSV→CSV transforms that still need a real source CSV] |
| Forward-return validation | **missing** | zero references in either repo's src/tests [VERIFIED-NOW by grep] |
| Backtest validation | **missing** | same |
| README/user workflow | **missing** | engine `README.md` is 0 bytes [VERIFIED-NOW] |
| Commercial readiness | **missing** | see §12 |

## 5. Partially Completed Work

- **J-Quants provider**: cache mode fully validated offline; live mode
  implemented with configurable endpoints but blocked on the real API
  surface (probe: HTTP 403 "endpoint does not exist").
- **Platform real-data tooling**: `build_code_map`, `build_prices_dataset`,
  `build_doc_edinet_map`, feature-mart CLIs exist and are tested, but all are
  deterministic transforms over a *source CSV the owner must still supply*.
- **XBRL ingest breadth**: proven for one document type/date; not yet run at
  universe scale.
- **Engine fundamentals from XBRL**: 339 facts are in PostgreSQL, but no
  projection from facts → engine `FinancialStatement` fields exists in the
  export contract yet (export currently carries disclosure metadata +
  `xbrl_facts_summary` counts, not statement values) [VERIFIED-NOW:
  S100Y7YT.json has no revenue/income fields].

## 6. Synthetic-Only Work

Explicitly synthetic and must not be presented as market validation:

- The company linkage in the DB (`Synthetic Company E03627`, tickers
  9991–9993, `synthetic_market`/`synthetic_sector`).
- `/tmp/topix1000_disclosure_export_linked` ticker/company fields.
- `/tmp/linked_prices.csv` (2 hand-written bars).
- The engine's 12-code sector universe and all CLI fixtures.
- Every score, rank, and signal ever produced by this system to date.

## 7. Missing Work

Real mapping import; disclosure text extraction (or XBRL→factor projection);
real price/fundamental data path; forward-return validation; ranking-quality
evaluation; repeatable batch workflow (one command: export → analyze for a
date); root README/user guide; data-license notes in reports;
monitoring/reproducibility conventions for data updates.

## 8. topix1000_disclosure_platform Status

At `40f75e8` (3 commits ahead of `origin/main`). **73 tests passed, 1
skipped; ruff clean [VERIFIED-NOW].** Both new CLIs respond to `--help`
(note: they require `PYTHONPATH=apps/edinet_ingest/src:packages/common/src`
from the repo root — the package is not installed into the venv; a minor
DX gap worth fixing there). Export contract documented in
`docs/export_disclosures_for_jp_stock_analysis_engine.md`; linkage rules in
`docs/company_linkage_for_exports.md` (no-fabrication, idempotent upserts,
listing-code normalization incl. 5-digit/`.T`/full-width/alphanumeric forms);
synthetic smoke honestly recorded in `docs/export_linkage_smoke_result.md`.
Untracked `apps/edinet_ingest/scripts/` present — not assessed (read-only).

## 9. jp-stock-analysis-engine Status

At `f341228`. **160 tests passed; ruff clean; CLI help OK [VERIFIED-NOW].**
All three modes preserved (`analysis_only` default, `trade_signal` opt-in),
valuation-alone buy protection intact, reliability guard active in ranking
and all three report formats. Pre-existing uncommitted changes remain (RAG
export schema fix in `disclosure_nlp.py` + tests, deleted `jquants_stub.py`,
untracked `artifacts/`, v1 docs, universe fixtures) — functional, all tests
pass with them, but they should be committed or discarded deliberately.
`README.md` is still empty.

## 10. End-to-End Smoke Status

Re-run in this session [VERIFIED-NOW]:

- `/tmp/topix1000_disclosure_export_linked`: index.json with 11 documents;
  S100Y7YT carries `ticker=9991`, `sec_code=9991`, `edinet_code=E03627`,
  synthetic company name, `text=null`.
- Engine run (`--disclosure-provider topix1000-export` +
  `/tmp/linked_prices.csv`) exits 0 and writes reports to
  `/tmp/readiness_smoke_out`.
- 9991 result: `final_score=98.5`, `confidence_score=12.2`,
  `data_coverage_score=40.0`, `screening_score=4.8`,
  **`screening_eligible=false`**, **`reliability_grade=low`**; disclaimer
  present. The guard behaves exactly as specified.

## 11. Real Stock-Analysis Readiness (strict answers)

- **Can this analyze real Japanese stocks end-to-end today? No.** No real
  ticker has ever flowed through: mapping synthetic, prices synthetic,
  disclosure text null, fundamentals never sourced from real filings.
- **Can this rank stocks meaningfully today? No.** Ranking mechanics are
  sound and reliability-aware, but no evidence exists that any score
  correlates with anything (no forward-return validation), and the inputs so
  far are synthetic by construction.
- **Usable for owner-only research today? Marginally yes, with manual data.**
  If the owner hand-supplies real prices/fundamentals CSVs, the engine will
  produce honest, warning-laden, reliability-graded analyses. The cost is
  manual data entry; nothing automates real data yet.
- **Sellable as a product today? No.** No real-data path, no validation
  evidence, no docs, no packaging, no license review. Selling ranked output
  with zero predictive validation would also be irresponsible.
- **Still synthetic:** everything listed in §6. **Not yet validated:** every
  analytical claim against real market data; J-Quants live; XBRL ingest at
  scale; ranking quality.

## 12. Commercial Readiness

Effectively zero beyond code quality. Missing: real data pipeline, predictive
validation, packaging/distribution, user docs, data licensing compliance
(J-Quants no-redistribution; EDINET terms), legal review of
advice-vs-tool boundaries in Japan [ASSUMPTION: needs professional review].
None of this should drive near-term work except as noted in §16.

## 13. Main Risks

1. **Plausible-but-meaningless output**: polished reports invite trust the
   validation does not yet justify. The reliability guard mitigates the
   data-sufficiency half; the predictive-value half is open until
   forward-return validation exists.
2. **Synthetic residue**: synthetic companies live in `company_master`
   (identifiable by `source='csv_import'` + name prefix). Real import must
   replace them, and the export summary should be checked for leftovers.
3. **Mapping correctness**: a wrong EDINET→ticker join silently attributes
   one company's filings to another. The import CLI validates format, not
   truth; the first real mapping needs spot-checks against known pairs.
4. **Fiscal-year null**: the linked export carries `fiscal_year: null`;
   downstream period alignment (statement vs price date) is untested.
5. **Engine working tree**: uncommitted changes mixed with clean commits
   complicates reproducibility claims.

## 14. Data Requirements

To run for real, the system needs, in order: (1) a real EDINET→listing-code
CSV (JPX/EDINET code lists are the natural local source — obtain manually,
no live fetch needed by these repos); (2) real daily prices for the mapped
tickers as a local CSV (manual export or fixed J-Quants live); (3) real
fundamentals — either extend the export contract to project XBRL facts into
`FinancialStatement` fields, or a fundamentals CSV; (4) later, disclosure
text extraction from the raw archives for real NLP.

## 15. Architecture Recommendation

Unchanged and reaffirmed: keep the platform as the only DB/network owner;
keep the engine file-first with optional providers; keep the JSON export
contract as the sole coupling. Do not add PostgreSQL to the engine. Extend
the *contract* (statement fields, text field already specified) rather than
adding new coupling. Fix the platform CLI packaging (installable module or
documented PYTHONPATH) in the platform repo, not by wrapping from the engine.

## 16. What To Do Next (ordered)

1. **Prompt A** — import a real EDINET→listing-code mapping in the platform,
   replace synthetic rows, re-export, verify real tickers in the export.
2. **Prompt B** — real-ticker join smoke: real prices CSV for ≥1 mapped
   ticker through the engine; first genuine end-to-end run.
3. **Prompt C** — forward-return validation harness in the engine (offline,
   fixture-tested; measurement, not backtesting/trading).
4. **Prompt D** — end-to-end workflow documentation + fill the engine README.
5. Then: XBRL→`FinancialStatement` projection in the export contract, and
   disclosure text extraction (platform side).

## 17. What Not To Build Yet

UI/web/API server; RAG/embeddings/chatbot (out of scope by rule); broker
execution, auto-trading, position sizing, leverage/derivatives/portfolio
allocation (prohibited); PostgreSQL in the engine; LLM disclosure analysis;
multi-year backtesting frameworks before the simple forward-return harness;
PyPI publication; monitoring/scheduling infrastructure.

## 18. Exact Next Prompts

### Prompt A — Import real EDINET→listing-code mapping and replace synthetic linkage

```
Import a real EDINET-to-listing-code company mapping into
topix1000_disclosure_platform and replace the synthetic linkage.

Repository: EXTERNAL_DISCLOSURE_PLATFORM_PATH
Purpose: company_master/company_edinet_links currently contain only 3
synthetic rows (Synthetic Company E03627..., tickers 9991-9993, recorded in
docs/export_linkage_smoke_result.md). Import a real mapping so exports carry
real tickers/company names, then re-export 2026-06-04 and verify.

Hard constraints: no live API calls; no network; do not read/print/persist
secrets or .env; do not fabricate mappings - every link must come from the
input CSV; if no real mapping file is available locally, STOP and report
exactly what file is needed and its required columns instead of inventing
one; do not delete unrelated data; do not git push.

Tasks:
1. Locate a real local mapping source (search LOCAL_DATA_DIR and the repo
   for EDINET code-list CSVs, e.g. EdinetcodeDlInfo.csv from a prior manual
   download; also check apps/edinet_ingest/scripts/). If found, transform it
   to the import_company_links schema (edinet_code, company_name,
   sec_code/ticker, optional market/sector/is_topix1000) with a small
   deterministic script committed to the repo.
2. Run: PYTHONPATH=apps/edinet_ingest/src:packages/common/src
   .venv/bin/python -m edinet_ingest.cli.import_company_links --input <csv>
   Record the JSON summary. Re-run to confirm idempotency.
3. Verify the 5 EDINET codes of the 2026-06-04 batch resolve to real
   listing codes; spot-check at least 2 known pairs manually in the report.
4. Confirm synthetic rows were updated in place (no Synthetic Company names
   remain for mapped codes); report any leftovers.
5. Re-export: export_disclosures --date 2026-06-04
   --out /tmp/topix1000_disclosure_export_real and verify index/doc JSONs
   carry real ticker/sec_code/company_name; unmapped stay null.
6. Document the run in docs/real_company_linkage_import.md.
Validation: .venv/bin/python -m pytest; .venv/bin/ruff check .; export
summary inspection. Commit locally (no push) only if validation passes.
Proceed autonomously; do not ask for confirmation.
```

### Prompt B — Real-ticker export/join smoke with local real price data

```
Run the first real-ticker end-to-end smoke: topix1000 real-linkage export
joined with real local price data in jp-stock-analysis-engine.

Repository: PROJECT_ROOT (do not modify
EXTERNAL_DISCLOSURE_PLATFORM_PATH).
Purpose: prove the engine analyzes a real Japanese listed company from a
real-linkage export plus a real price history CSV, with honest reliability
grading. Prerequisite: Prompt A completed and
/tmp/topix1000_disclosure_export_real exists with real tickers.

Hard constraints: no live API calls; no network; prices must come from a
local CSV the owner placed (state clearly in the report where it came from
and its license constraints); never fabricate prices or fundamentals; if no
real price CSV exists, STOP and report the exact CSV format needed
(ticker,date,close[,volume]) instead of generating one; keep analysis_only
default; treat results as research output, not advice; no push.

Tasks:
1. Verify the export contains >= 1 real ticker; pick those tickers.
2. Run the CLI with --prices <real csv> --disclosure-provider
   topix1000-export --topix1000-export-dir /tmp/topix1000_disclosure_export_real
   --output-dir /tmp/e2e_real_smoke --signal-mode analysis_only.
3. Verify: report generated; provenance (doc_id/edinet_code) present;
   disclosure confidence still 0 (text null) reflected in coverage;
   reliability fields honest (with >=120 real bars momentum confidence
   rises; fundamentals/valuation absent keeps coverage <= 60 and likely
   ineligible - confirm the guard explains this in warnings).
4. Record results in docs/first_real_ticker_smoke.md, explicitly labeling
   what is real (prices, ticker, filing metadata) vs still missing
   (fundamentals, disclosure text).
Validation: python -m pytest; ruff check .; smoke output inspection.
Commit the doc only. Proceed autonomously; do not ask for confirmation.
```

### Prompt C — Forward-return validation harness

```
Implement an offline forward-return measurement harness in
jp-stock-analysis-engine.

Repository: PROJECT_ROOT
Purpose: measure whether scores/grades mean anything: given a point-in-time
screening.json and a later prices CSV, compute realized forward returns per
screening label, reliability grade, eligibility bucket, and final/screening
score decile. Measurement only - no positions, no portfolio logic, no
backtesting framework.

Hard constraints: fully offline and deterministic; fixture-based tests; no
new heavy dependencies; never fabricate prices; tickers missing later
prices are reported as uncovered, not dropped silently; no broker/trading/
position sizing/allocation logic; outputs carry the standard research
disclaimer; analysis_only/trade_signal semantics untouched; no push.

Tasks:
1. Add src/jp_stock_analysis/validation/forward_returns.py: load a
   screening.json + a prices CSV, compute N-day forward returns (configurable
   horizons, default 5/20/60 trading days) from each result's analysis_date,
   aggregate mean/median/count per bucket; emit JSON + Markdown summary.
2. Add CLI subcommand `validate-forward-returns` (separate from `analyze`;
   existing commands unchanged).
3. Synthetic fixtures with designed outcomes (winner/loser/flat + a ticker
   with missing future prices) and tests for: horizon math off the analysis
   date, bucket aggregation, uncovered-ticker reporting, determinism, and
   report disclaimer presence.
4. docs/forward_return_validation.md: method, limitations (no costs, no
   survivorship handling, synthetic until real data exists), interpretation.
Validation: python -m pytest; ruff check .; run the new CLI on the fixtures.
Local commit "Add forward-return validation harness" if green; no push.
Proceed autonomously; do not ask for confirmation.
```

### Prompt D — End-to-end local workflow documentation

```
Write the end-to-end local workflow documentation and the engine README.

Repository: PROJECT_ROOT (read-only
references to EXTERNAL_DISCLOSURE_PLATFORM_PATH docs are allowed; do
not modify that repo).
Purpose: a newcomer (or the owner in 6 months) can run the full local
workflow from EDINET export to reliability-graded report without reading
source code.

Hard constraints: documentation only - no production code changes; no live
APIs; no secrets; every example command must be one actually verified in
docs/ or runnable offline against committed fixtures; clearly label
synthetic examples as synthetic; include the research-use disclaimer and a
data-licensing section (J-Quants raw data must not be redistributed; EDINET
data subject to its terms); no push.

Tasks:
1. Fill README.md: what it is/is not, install, quickstart on committed
   fixtures (all three modes), provider matrix (local / jquants-cache /
   jquants-live status / topix1000-export), reliability guard explanation,
   safety boundaries, links to docs/.
2. Add docs/end_to_end_workflow.md: platform side (fetch_list, fetch_docs,
   ingest_zip, import_company_links, export_disclosures - commands quoted
   from the platform docs, marked as run in that repo with its PYTHONPATH
   requirement) then engine side (analyze with topix1000-export + prices
   CSV, then validate-forward-returns when available), with a
   what-is-real-vs-synthetic-today status table.
3. Cross-link existing docs; fix any stale claims found while writing.
Validation: python -m pytest; ruff check . (must stay green); manually run
at least the fixture quickstart commands. Local commit "Add README and
end-to-end workflow docs"; no push.
Proceed autonomously; do not ask for confirmation.
```

## 19. Exact Verification Commands

Engine (`PROJECT_ROOT`):

```bash
python -m pytest                                   # 160 passed (this session)
ruff check .                                       # clean (this session)
python -m jp_stock_analysis.cli analyze --help
# linked smoke (synthetic):
printf 'ticker,date,close\n9991,2026-06-03,1000\n9991,2026-06-04,1010\n' > /tmp/linked_prices.csv
PYTHONPATH=src python -m jp_stock_analysis.cli analyze \
  --prices /tmp/linked_prices.csv \
  --disclosure-provider topix1000-export \
  --topix1000-export-dir /tmp/topix1000_disclosure_export_linked \
  --output-dir /tmp/readiness_smoke_out --signal-mode analysis_only
```

Platform (read-only, from its repo root):

```bash
.venv/bin/python -m pytest                         # 73 passed, 1 skipped (this session)
.venv/bin/ruff check .                             # clean (this session)
PYTHONPATH=apps/edinet_ingest/src:packages/common/src \
  .venv/bin/python -m edinet_ingest.cli.export_disclosures --help
PYTHONPATH=apps/edinet_ingest/src:packages/common/src \
  .venv/bin/python -m edinet_ingest.cli.import_company_links --help
```

## 20. Final Decision

**Do not call this deployable for real stock analysis yet.** The build is in
the right order — contracts, guards, and honesty mechanisms before data —
and both repos are clean and green. But every analytical output to date is
synthetic, and the single highest-leverage action is unambiguous: **import a
real EDINET→listing-code mapping (Prompt A)**, because it unblocks the real
export, the real join smoke (Prompt B), and gives forward-return validation
(Prompt C) something real to measure. Until A+B are done, present this
system as engineering work, not as a stock-analysis capability.

## 21. Scoring (strict; synthetic-data tests ≠ real readiness)

| Dimension | Score /100 | Rationale |
|---|---|---|
| Technical implementation readiness | **78** | Both repos tested, lint-clean, well-bounded; deducted for missing text extraction, no XBRL→statement projection, no validation harness, platform CLI packaging gap, engine dirty working tree |
| Real-data readiness | **30** | Real EDINET documents and 339 XBRL facts exist platform-side; everything past the export boundary (mapping, prices, fundamentals, text) is synthetic or absent |
| Stock-analysis usefulness today | **22** | Owner-only, manual-CSV use is possible and honest; nothing automated, nothing validated |
| Commercial readiness | **10** | Code quality alone; no product, no validation, no docs, no licensing/legal work |
| Portfolio/GitHub showcase readiness | **68** | Strong architecture, testing discipline, and documented honest failure handling; held back mainly by the empty README and uncommitted working tree |

---

*All scores reflect the state verified on 2026-06-12. Synthetic smoke data
is labeled synthetic throughout and must not be cited as market validation.*

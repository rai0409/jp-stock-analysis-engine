# Current two-project stock-analysis status

_Evidence-based readiness report. Generated 2026-06-13. Every claim below is
backed by a command result captured during inspection; anything not directly
verified is explicitly labelled **(inferred)** or **(unverified)**._

## 1. Executive summary

Two cleanly separated projects exist and individually pass their own tests:
`topix1000_disclosure_platform` (EDINET acquisition → export bundles) at commit
`4122df7` / tag `topix1000-fable-engine-input-exports-20260613` (113 passed, 1
skipped; ruff clean), and `jp-stock-analysis-engine` (file/cache-first analysis
+ forward-return harness) at commit `5b73a6d` (176 passed; ruff clean). The
file-based integration works: the engine consumes the platform's metadata,
fundamentals, and disclosure export and produces real reports with real company
names, real XBRL-derived fundamentals, and real disclosure provenance for the 3
ingested tickers (3928, 4107, 4264). **Real predictive validation has not been
performed and is currently blocked**: no real `ticker,date,close` price history
for those tickers exists locally — only hand-made 2-row synthetic smoke CSVs —
so the forward-return harness has been exercised for plumbing only, never on
real prices. The system is a strong, honest research/showcase pipeline; it is
**not** commercially proven and makes no validated predictive claims.

## 2. Project 1: topix1000_disclosure_platform

**Role:** EDINET acquisition / storage / ingestion / export. Owns EDINET
document & list storage, company linkage, and the three export bundles
(metadata, fundamentals, disclosure) consumed by the analysis engine. It does
not perform analysis.

**Confirmed commits / tags** (`git log --oneline --decorate`):
- HEAD `4122df7` — *Export engine metadata and fundamentals from EDINET facts*,
  tagged **`topix1000-fable-engine-input-exports-20260613`**.
- `d1b6cfe` tag `topix1000-linked-export-v1`
- `6b9e08d` tag `topix1000-annual-export-v1`
- `1c09654` tag `topix1000-filing-type-v2`
- `bf558bb` tag `topix1000-edinet-export-v1`

**Tag content verified** (`git show --name-status topix1000-fable-engine-input-exports-20260613`):
exactly 8 intended files — 2 CLIs (`export_engine_metadata`,
`export_engine_fundamentals`), 2 export services (`engine_metadata.py`,
`engine_fundamentals.py`), modified `disclosures.py`, 2 test modules, and
`docs/jp_stock_analysis_engine_input_exports.md`. The stray J-Quants script is
**not** in the tag (verified: `git show --name-only … | grep
fetch_jquants_prices_source.py` → "OK - stray file NOT in tag").

**Completed capabilities (evidence-backed):**
- EDINET list/doc acquisition, storage, and filing-type classification (prior
  tagged work `topix1000-filing-type-v2`, `topix1000-annual-export-v1`).
- Real EDINET company linkage from official list responses
  (`topix1000-linked-export-v1`).
- Deterministic disclosure export (`index.json` + per-doc JSON).
- Deterministic metadata export CLI (`ticker,company_name,sector,market,edinet_code`).
- Deterministic fundamentals export from ingested XBRL facts
  (`ticker,fiscal_year,revenue,operating_income,net_income,equity,total_assets,doc_id,edinet_code`).

**Verified commands / results:**
- `.venv/bin/python -m pytest` → **113 passed, 1 skipped**.
- `.venv/bin/ruff check .` → **All checks passed**.
- `git status --short` → only `?? apps/edinet_ingest/scripts/` (untracked stray,
  not part of any tag).

**Generated artifacts (exact current outputs):**
- `/tmp/topix1000_engine_bundle/metadata.csv` — 3.5K, **75 data rows**.
- `/tmp/topix1000_engine_bundle/fundamentals.csv` — 336 B, **3 data rows**.
- `/tmp/topix1000_annual_report_export_linked/index.json` — 7.1K, **85
  documents**.
- Fundamentals rows (verbatim):
  - `3928,2025,7478296000,374476000,228133000,1521161000,4688878000,S100XRSE,E31991`
  - `4107,2025,39258000000,9484000000,6498000000,40070000000,51015000000,S100XT8H,E01028`
  - `4264,2025,6840816000,326122000,187586000,2926474000,5257475000,S100XTNP,E36859`

**Limitations:**
- Only **3 of 85** annual reports have ingested XBRL facts → fundamentals CSV
  has 3 rows; 72 linked docs are skipped as fact-less, 10 docs are unlisted
  filers with no listing code.
- `sector` / `market` columns are blank (no local source); never fabricated.
- Single fiscal year (2025) only — no prior-year rows.
- The platform does **not** yet produce a real price CSV as a validated
  pipeline output.

**Risks:**
- Bundle artifacts live in `/tmp` (ephemeral); regeneration requires the local
  DB. (inferred from paths)
- The untracked `apps/edinet_ingest/scripts/fetch_jquants_prices_source.py`
  calls live J-Quants and must never be confused with tagged work or run during
  offline tasks.

**Next tasks:** ingest XBRL facts for the remaining annual reports; add a
sector/market source; produce a real, validated price CSV export path; add
prior-year rows for growth metrics.

## 3. Project 2: jp-stock-analysis-engine

**Role:** Consumes local CSV/export bundles (file/cache-first), performs
analysis, scoring, reporting, confidence-aware screening, and offline
forward-return validation. Does **not** touch the topix1000 database.

**Confirmed commits / tags** (`git log --oneline --decorate`):
- HEAD **`5b73a6d` — *Add forward-return validation harness*** — **not tagged**
  (it is one commit ahead of the latest tag).
- `bfafe03` tag `jp-stock-analysis-engine-fable-current-20260613`
- `f341228` tag `jp-stock-analysis-engine-v1-confidence-guard`
- `c9ae813` tag `jp-stock-analysis-engine-v1-topix1000-export-provider`
- plus the `v1-sector-*` / `v1-jquants-*` tag series.

**Completed capabilities (evidence-backed):**
- `analyze` pipeline consuming `--prices`, `--metadata`, `--fundamentals`, and
  the `topix1000-export` disclosure provider.
- Confidence-aware screening guard (`analysis/reliability.py`): deterministic
  `data_coverage_score`, `screening_score`, `screening_eligible`,
  `reliability_grade`; scores are never mutated.
- Forward-return validation harness (`validation/forward_returns.py`, 668 LOC)
  with JSON/CSV/Markdown outputs.
- Safety defaults verified in `cli.py`: `--signal-mode` default
  **`analysis_only`**, choices `analysis_only|screening|trade_signal`
  (`trade_signal` explicit opt-in). Valuation-alone buy protection documented
  in `analysis/valuation.py` ("valuation alone must …").

**Verified commands / results:**
- `PYTHONPATH=src .venv/bin/python -m pytest` → **176 passed**.
- `.venv/bin/ruff check .` → **All checks passed**.
- `… cli --help` → subcommands `analyze`, `validate-forward-returns`.
- `… cli validate-forward-returns --help` → flags `--screening-json`,
  `--prices`, `--output-dir`, `--horizons` (default `5,20,60`),
  `--analysis-date`, `--no-markdown`.

**Forward-return harness status:**
- Functional smoke run this session (synthetic prices, plumbing only):
  `analyze` → `screening.json` → `validate-forward-returns --horizons 5`
  produced `forward_returns.{json,csv,md}`.
- **Look-ahead protection verified live**: for analysis_date `2026-03-27` the
  harness picked base_date `2026-03-30` (strictly after) and target_date
  `2026-04-06`, status `ok`. Source code docstring confirms "first row
  *strictly after* analysis_date … so there is no look-ahead."
- Reliability fields propagate into the harness output
  (`screening_eligible:false`, `reliability_grade:"low"` for 3928).
- Research-only disclaimer present in `forward_returns.md` ("no trading
  signals, no portfolio construction, no position sizing").
- `tests/test_forward_returns.py` (421 LOC) present; no-look-ahead behavior is
  covered (within the 176 passing tests).

**Limitations:**
- **No real future price CSV** for 3928/4107/4264. Only:
  `tests/fixtures/prices_sample.csv` (tickers 6758/7203/9984), and tiny
  synthetic `/tmp/annual_real_linked_prices*.csv` (2 dates each, round numbers).
- `docs/forward_return_validation.md` exists; **`docs/forward_return_validation_results.md`
  does NOT exist** (no real results doc).
- None of `/tmp/jstocks_topix1000_forward_input`,
  `/tmp/jstocks_forward_validation_topix1000`,
  `/tmp/jstocks_full_bundle_smoke` exist (verified MISSING) — no persisted real
  forward-return run.

**Risks:**
- Easy to mistake the synthetic `/tmp` "annual_real_linked_prices" filename for
  real data — it is **not** real (round-number smoke values).
- HEAD `5b73a6d` is untagged; the forward-return harness is not yet captured in
  a release tag. (inferred)

**Next tasks:** obtain/import a real `ticker,date,close` history; run real
forward-return validation; write `docs/forward_return_validation_results.md`;
tag the harness commit once a real run exists.

## 4. Integration status

**Flows successfully (file-based, verified this session):**
- topix1000 `metadata.csv` + `fundamentals.csv` + disclosure export
  (`index.json` + per-doc JSON) → engine `analyze` → per-ticker reports +
  `screening.json` + `screening.csv`.
- Engine reports carry **real** `company_name` from metadata (株式会社マイネット /
  伊勢化学工業株式会社 / 株式会社セキュア), **real** XBRL fundamentals (e.g. 3928 operating
  margin 5.01%, ROE 15.00%, equity ratio 32.44%, FY2025), and **real**
  disclosure provenance (`doc_id S100XRSE`, `xbrl.facts_count 1307`; 1488; 1317).
- The engine consumes only files — no topix1000 DB dependency. (verified: engine
  invoked from its own repo with `--metadata/--fundamentals/--topix1000-export-dir`)

**Tested with real EDINET-derived data:** company linkage, metadata,
fundamentals, and disclosure provenance for the 3 ingested tickers; analysis
report generation; confidence-aware screening (all 3 graded `low`, ineligible —
honestly reflecting thin coverage).

**Still synthetic / fixture-only:** all price inputs. Momentum and valuation
sub-scores are therefore meaningless beyond plumbing; forward-return validation
has only been run on synthetic prices.

**Blocked:** real forward-return / predictive validation — blocked on a real
price CSV (see §7).

## 5. Commercial / readiness scoring (strict, out of 100)

**Data pipeline readiness — 72/100**
- EDINET acquisition, classification, linkage, and 3 export bundles all tagged
  and tested (113 passed, ruff clean).
- Deterministic, no-fabrication exports verified byte-stable (prior session).
- –: only 3/85 reports have XBRL facts; sector/market blank.
- –: no validated price export path yet.

**Analysis engine readiness — 80/100**
- 176 tests pass, ruff clean; analyze + screening + forward-return harness all
  functional.
- Safety posture solid: analysis_only default, trade_signal opt-in,
  valuation-alone guard, reliability guard.
- –: forward-return results unproven on real data; HEAD untagged.

**Real-data coverage readiness — 30/100**
- Real company names + fundamentals + disclosure provenance for 3 tickers.
- –: only 3 fundamentals rows; 1 fiscal year; no prices at all.
- –: 72/85 annual reports lack ingested facts.

**Validation / predictive-usefulness readiness — 15/100**
- Harness exists, is deterministic, and has live-verified no-look-ahead logic.
- –: **zero** real forward-return runs; no statistical significance testing.
- –: no `forward_return_validation_results.md`.

**Commercial product readiness — 20/100**
- Strong engineering foundation and explicit research-only guardrails.
- –: no real predictive evidence, no UI/API, no real price feed, thin coverage.
- –: **not commercial-ready** on current evidence.

**Portfolio / GitHub showcase readiness — 82/100**
- Two clean, well-tested, well-documented repos with deterministic pipelines
  and honest limitation docs.
- Real EDINET→analysis flow demonstrable end-to-end on 3 real companies.
- –: a real forward-return chart/result would materially strengthen it.

## 6. What is definitely complete (evidence-backed)

- topix1000 at `4122df7` / tag `…-20260613`: 113 passed, 1 skipped; ruff clean;
  tag contains exactly the 8 intended files; stray script excluded.
- Metadata export: 75 linked listed companies; fundamentals export: 3 real
  XBRL-derived rows (3928, 4107, 4264); disclosure index: 85 documents.
- jp-stock-analysis-engine at `5b73a6d`: 176 passed; ruff clean; `analyze` +
  `validate-forward-returns` CLIs functional.
- Forward-return harness: deterministic, no-look-ahead verified live, JSON/CSV/MD
  output, research-only disclaimer present.
- File-based integration: engine produces real-name, real-fundamentals,
  real-provenance reports from the topix1000 bundle, with no DB coupling.
- Safety defaults: analysis_only default, trade_signal opt-in, valuation-alone
  protection, confidence-aware reliability guard.

## 7. What is not complete

- **No broad real price CSV** — only fixtures (6758/7203/9984) and tiny
  synthetic `/tmp` smoke files for 3928/4107/4264.
- **No real predictive validation** — harness never run on real prices; no
  results document.
- **Only 3 fundamentals rows** (3 of 85 reports ingested with XBRL facts).
- **No adjusted close** in any available real price source (fixture has raw
  OHLC only; synthetic files have `close` only).
- **No sector/market source** — columns blank.
- **No disclosure narrative text extraction** — disclosure `text` is null;
  engine disclosure findings are empty (still true).
- **No statistical significance testing** of screening fields vs. forward
  returns.
- **No production UI/API** — both projects are CLI/file tools only (still true).
- HEAD of the engine (`5b73a6d`) is **untagged**.

## 8. Immediate next steps

**Next 1 prompt** — Add a local real price CSV export/import path for
forward-return validation (the gating blocker). Source real `ticker,date,close`
history for 3928/4107/4264 from an already-permitted local/offline source (no
live J-Quants), or define a documented manual-import format; do **not** fabricate
prices. (Verified as the correct first step: every other validation gap depends
on having real prices.)

**Next 3 prompts:**
1. (above) real price CSV path.
2. Run real forward-return validation on 3928/4107/4264 and write
   `docs/forward_return_validation_results.md` with the real run; tag the engine
   harness commit once a real run exists.
3. Ingest XBRL facts for more of the 85 annual reports in topix1000 to grow the
   fundamentals CSV beyond 3 rows; re-export and re-run analysis.

**Next 5 prompts:** add 4. a sector/market source to populate metadata and
activate sector-relative analysis; 5. prior-year fundamentals rows so growth
metrics and a first significance check (sample-size caveated) become possible.

## 9. Exact recommended next prompt

```
Add a local real-price input path for forward-return validation in the
jp-stock-analysis-engine ↔ topix1000 pipeline.

Primary repository:
PROJECT_ROOT
Related (read-only unless explicitly needed):
EXTERNAL_DISCLOSURE_PLATFORM_PATH

Purpose:
Provide a real, offline, no-fabrication ticker,date,close price CSV for the
already-linked tickers 3928, 4107, 4264 so the existing forward-return
validation harness can be run on REAL prices instead of synthetic smoke data.
The harness already exists (validate-forward-returns); this task only supplies
and wires real price input — it does not change scoring or signal logic.

Autonomy policy:
- Proceed autonomously; do not ask for confirmation.
- Do not pause for repo-scoped inspection, tests, ruff, or local report runs.

Hard constraints:
- Do not use network. Do not call live EDINET. Do not call live J-Quants.
- Do not read, print, persist, or commit secrets or .env.
- Do not fabricate prices. Do not treat synthetic/fixture prices as real.
- Do not implement RAG, broker execution, auto-trading, position sizing,
  leverage, margin, derivatives, options, futures, FX execution, or portfolio
  allocation.
- Do not output personalized financial advice.
- Keep analysis_only the default; keep trade_signal explicit opt-in; keep
  valuation-alone buy-signal protection.
- Do not git push, git reset --hard, or git clean -fd. Do not delete local files.

Tasks:
1. Inspect for any already-permitted offline real price source under
   LOCAL_DATA_DIR (e.g. a J-Quants source CSV already fetched and stored, NOT
   a live call) and under topix1000 derived outputs. Report exactly what real
   price data, if any, already exists locally.
2. If a real source exists offline, add a deterministic, documented path that
   produces a ticker,date,close (and adj_close if available) CSV for 3928,
   4107, 4264 covering dates after 2026-03-27, written to a local path.
3. If NO real source exists offline, do NOT fabricate. Instead, document the
   exact required manual-import format and the precise gap, and add a
   validation that rejects synthetic/round-number placeholders.
4. If real prices were obtained, run:
   validate-forward-returns --screening-json <screening.json> --prices <real.csv>
   --output-dir <dir> --horizons 5,20,60
   and write docs/forward_return_validation_results.md with the real result,
   clearly labelled as a small-sample (n=3) descriptive run, not significance.

Validation:
- jp-stock-analysis-engine: PYTHONPATH=src python -m pytest; ruff check .
- topix1000 (if touched read-only only): note any inspection commands run.

Commit behavior:
- Do not push. Commit only intended code/docs after validation passes, message:
  "Add local real-price input path for forward-return validation".
- Do not tag unless the final response explicitly says it is safe.

Final response must include: summary; whether real prices were found/used or the
blocker remains; files changed; pytest result; ruff result; forward-return
result if run (labelled real vs synthetic); commit SHA if created; whether safe
to tag; and the exact next prompt.
```

## 10. Handoff summary (for another conversation)

- **Repos:** primary `jp-stock-analysis-engine` @ `5b73a6d` (untagged, 176 pass,
  ruff clean); platform `topix1000_disclosure_platform` @ `4122df7` / tag
  `topix1000-fable-engine-input-exports-20260613` (113 pass + 1 skip, ruff
  clean).
- **Bundle outputs (in /tmp, regenerable from the local DB):**
  `/tmp/topix1000_engine_bundle/metadata.csv` (75 rows),
  `/tmp/topix1000_engine_bundle/fundamentals.csv` (3 rows: 3928/S100XRSE/E31991,
  4107/S100XT8H/E01028, 4264/S100XTNP/E36859),
  `/tmp/topix1000_annual_report_export_linked/index.json` (85 docs).
- **Integration works** file-based: engine `analyze` yields real company names,
  real fundamentals, real disclosure provenance for the 3 tickers; engine has no
  DB coupling.
- **Forward-return harness** (`validate-forward-returns`) works, no-look-ahead
  verified, JSON/CSV/MD output — but **only on synthetic prices**.
- **Blocker:** no real `ticker,date,close` history for 3928/4107/4264 (only
  fixture 6758/7203/9984 and synthetic 2-row /tmp files). No
  `docs/forward_return_validation_results.md` yet.
- **Next action:** supply a real offline price CSV (no fabrication, no network),
  then run the harness for real and document results; see §9 for the full prompt.
- **Do not:** fabricate prices, run live J-Quants/EDINET, push, reset --hard,
  clean -fd, or call synthetic/fixture data "validation".

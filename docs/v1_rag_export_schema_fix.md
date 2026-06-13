# v1 RAG-Export Schema Fix — Validation Note

**Date:** 2026-06-11
**Scope:** Small schema/export compatibility fix only. No new providers, no RAG
implementation, no refactor.

## What changed

1. `src/jp_stock_analysis/schemas.py` — `DisclosureAnalysisResult` gained two
   optional provenance fields:
   - `document_type: str | None = None`
   - `fiscal_year: int | None = None`
2. `src/jp_stock_analysis/analysis/disclosure_nlp.py` — both analyzers now
   propagate `document.document_type` and `document.fiscal_year` into the
   result, on every return path:
   - `RuleBasedDisclosureAnalyzer.analyze` (normal and empty-text returns)
   - `NoOpLLMDisclosureAnalyzer.analyze`
3. `docs/future_rag_integration_separate_project.md` — the required-fields list
   was replaced with a confirmed stable-path table for `screening.json`.
4. Tests — four new regression tests (see below).

No changes were needed in `json_report.py`, `csv_report.py`,
`markdown_report.py`, or `cli.py`: reports serialize via `model_dump`, so the
new fields flow through automatically, and `load_disclosure_texts` already
sets `document_type="local_text"` on loaded documents.
`DisclosureDocument` already carried both fields; no input-schema change was
required.

## Why it was needed

The v1 review (`docs/v1_review.md`, finding MEDIUM) found that
`document_type` was dropped during analysis and `fiscal_year` had no stable
disclosure-level path, so a future `jp_stock_rag_service` could not pin to the
JSON schema. This fix closes that gap before any ingestion project depends on
the output format.

## Exact stable export fields (in `screening.json`)

| Field | Path |
|---|---|
| ticker | `results[].ticker` |
| company_name | `results[].company_name` |
| fiscal_year | `results[].fundamentals.fiscal_year`, `results[].disclosure.fiscal_year` |
| document_type | `results[].disclosure.document_type` |
| evidence_text | `results[].disclosure.findings[].evidence_text` |
| analysis summary | `results[].score.reasons`, `results[].disclosure.findings[].summary` |
| risks | `results[].risks.flags[]` |
| positive_factors | `results[].disclosure.findings[]` where `category == "positive_factor"` |
| score_breakdown | `results[].score` |
| signal | `results[].signal` — present **only** when `trade_signal` mode is enabled |

Fields are `null` when the source did not provide them (e.g. `fiscal_year` for
plain local text files), but the paths are always present and never fabricated.

## Tests added/updated

- `tests/test_disclosure_nlp.py::test_rule_based_analyzer_propagates_document_metadata`
  (covers the empty-text early-return path too)
- `tests/test_disclosure_nlp.py::test_noop_llm_analyzer_propagates_document_metadata`
- `tests/test_disclosure_nlp.py::test_metadata_defaults_to_none_when_absent`
- `tests/test_cli.py::test_rag_export_stable_paths_in_json` (pins every stable
  path above against real CLI output)

All pre-existing mode-behavior tests (analysis_only / screening / trade_signal,
valuation-alone protection, determinism) pass unchanged.

## Commands run

```
python -m pytest    → 87 passed
ruff check .        → All checks passed
python -m jp_stock_analysis.cli analyze ... --signal-mode analysis_only
                    → verified disclosure.document_type/"fiscal_year" present,
                      no signal/label keys in analysis_only output
```

## RAG remains out of scope

No vector DB, embeddings, retrieval API, chatbot, or any other RAG logic was
added to this repository. This change only stabilizes the JSON export contract
that a **separate** future project (`jp_stock_rag_service`) may consume.

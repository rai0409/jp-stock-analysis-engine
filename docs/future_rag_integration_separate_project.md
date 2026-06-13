# Future RAG Integration (Separate Project)

RAG integration is **out of scope for this repository**. This engine will not
implement a vector database, embedding pipeline, retrieval API, chatbot, or
any RAG UI.

## What this repo provides instead

The JSON and Markdown reports are designed to be ingestible by a future,
separate project. Recommended name: **`jp_stock_rag_service`**.

That separate project may consume the exports of this engine as plain files.
Nothing in this repository should ever import from or depend on it.

## Required export fields — confirmed stable paths

All fields a RAG ingestion layer needs are now available at stable paths in
`screening.json` (guarded by `tests/test_cli.py::test_rag_export_stable_paths_in_json`):

| Field | Stable JSON path |
|---|---|
| `ticker` | `results[].ticker` |
| `company_name` | `results[].company_name` |
| `fiscal_year` | `results[].fundamentals.fiscal_year` and `results[].disclosure.fiscal_year` |
| `document_type` | `results[].disclosure.document_type` |
| `evidence_text` | `results[].disclosure.findings[].evidence_text` |
| `analysis_summary` | `results[].score.reasons` (per-sub-score explanations) and `results[].disclosure.findings[].summary`; the Markdown report provides the prose executive summary |
| `risks` | `results[].risks.flags[]` (severity, explanation, evidence, confidence) |
| `positive_factors` | `results[].disclosure.findings[]` filtered by `category == "positive_factor"` |
| `score_breakdown` | `results[].score` (all sub-scores, final score, reasons, confidence) |
| `signal` | `results[].signal` — present only when `trade_signal` mode was explicitly enabled |

`document_type` and `fiscal_year` are propagated from the source
`DisclosureDocument` through both the rule-based and no-op LLM analyzers; they
are `null` when the source did not provide them (e.g. fiscal year for plain
local text files), but the paths themselves are always present.

## Boundary rules

Allowed in this repo:

- JSON export
- Markdown export
- evidence blocks
- metadata fields that ease future ingestion

Not allowed in this repo:

- chatbot implementation
- vector DB integration
- embedding pipeline
- retrieval API
- RAG UI

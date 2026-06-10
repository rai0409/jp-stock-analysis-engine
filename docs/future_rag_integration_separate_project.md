# Future RAG Integration (Separate Project)

RAG integration is **out of scope for this repository**. This engine will not
implement a vector database, embedding pipeline, retrieval API, chatbot, or
any RAG UI.

## What this repo provides instead

The JSON and Markdown reports are designed to be ingestible by a future,
separate project. Recommended name: **`jp_stock_rag_service`**.

That separate project may consume the exports of this engine as plain files.
Nothing in this repository should ever import from or depend on it.

## Required export fields for future ingestion

The JSON report already carries (or has schema room for) the fields a RAG
ingestion layer needs:

- `ticker`
- `company_name`
- `fiscal_year`
- `document_type`
- `evidence_text` (per disclosure finding)
- `analysis_summary` (markdown executive summary / score reasons)
- `risks` (risk flags with severity, explanation, evidence)
- `positive_factors` (disclosure findings with `category=positive_factor`)
- `score_breakdown` (all sub-scores, reasons, confidence)
- `signal` (only when `trade_signal` mode was explicitly enabled)

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
